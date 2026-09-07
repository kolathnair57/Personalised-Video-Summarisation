
import argparse
import sys
from pathlib import Path

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))
sys.path.insert(0, str(ROOT / "src"))

from anchor_free.dsnet_af import DSNetAF          
from helpers import vsumm_helper, bbox_helper     
from kts.cpd_auto import cpd_auto                 
from query_head import QueryConditioner           


def decode_sampled(path, sample_rate):
    """Sequentially decode, keeping every `sample_rate`-th frame. -> (n_frames, [PIL])."""
    frames, n = [], 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i % sample_rate == 0:
                frames.append(fr.to_image())
            n = i + 1
    return n, frames


def clip_features(frames, model, preprocess, device, batch=128):
    """CLIP image embeddings. NOT normalised -- matches src/extract_clip.py, which is
    how the training features were produced."""
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            x = torch.stack([preprocess(f) for f in frames[i:i + batch]]).to(device)
            out.append(model.encode_image(x).float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def run_kts(n_frames, features, sample_rate, max_shots=None, vmax=0.5):
    """Shot boundaries for a video that has no annotations (DSNet's procedure).

    The kernel is built from L2-NORMALISED features. DSNet normalises its GoogLeNet
    descriptors before this step; our CLIP features are deliberately un-normalised to
    match training, and feeding those in directly makes kernel values ~120x larger, which
    swamps cpd_auto's complexity penalty and returns the maximum number of change points
    (one "shot" per sampled frame). Normalising only for the kernel fixes that and leaves
    the model input untouched.

    `max_shots` also caps ncp; TVSum videos have a median of ~38 shots, so allowing
    seq_len-1 candidates is far more than the data ever contains.
    """
    seq_len = len(features)
    picks = np.arange(0, seq_len) * sample_rate
    F = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    kernel = np.matmul(F, F.T)

    ncp = min(seq_len - 1, max_shots if max_shots else max(int(seq_len / 2), 2))
    cps, _ = cpd_auto(kernel, ncp, vmax, verbose=False)
    cps = cps * sample_rate
    cps = np.hstack((0, cps, n_frames))
    begin, end = cps[:-1], cps[1:]
    change_points = np.vstack((begin, end - 1)).T
    return change_points, end - begin, picks


def predict_summary(model, feats, qemb, cps, nfps, picks, n_frames, device, nms_thresh):
    seq = torch.from_numpy(feats).unsqueeze(0).to(device)
    q = torch.from_numpy(qemb).to(device) if qemb is not None else None
    with torch.no_grad():
        pred_cls, pred_bboxes = model.predict(seq, q)
    seq_len = len(feats)
    pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
    pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
    return vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps,
                                     n_frames, nfps, picks)


def write_summary(src, dst, keep_mask):
    """Re-decode the source and write only the kept frames.

    Kept frames are non-contiguous in the source, so their original timestamps must be
    replaced with sequential ones. Leaving them produces a summary that inherits the gaps
    and plays with long freezes (1425 frames reported as 376 s instead of ~59 s); simply
    clearing pts makes the muxer reject non-monotonic DTS. So pts is assigned explicitly
    from the output frame index.
    """
    from fractions import Fraction
    with av.open(str(src)) as ic:
        istream = ic.streams.video[0]
        istream.thread_type = "AUTO"
        fps = int(round(float(istream.average_rate or 30)))
        with av.open(str(dst), "w") as oc:
            ostream = oc.add_stream("libx264", rate=fps)
            ostream.width = istream.codec_context.width
            ostream.height = istream.codec_context.height
            ostream.pix_fmt = "yuv420p"
            tb = Fraction(1, fps)
            ostream.time_base = tb
            kept = 0
            for i, frame in enumerate(ic.decode(istream)):
                if i < len(keep_mask) and keep_mask[i]:
                    out_frame = frame.reformat(format="yuv420p")
                    out_frame.pts = kept
                    out_frame.time_base = tb
                    for pkt in ostream.encode(out_frame):
                        oc.mux(pkt)
                    kept += 1
            for pkt in ostream.encode():
                oc.mux(pkt)
    return kept


def selected_shots(mask, cps):
    return {s for s, (a, b) in enumerate(cps) if mask[a:b + 1].any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--query2", default=None,
                    help="second preference; writes a second summary and reports overlap")
    ap.add_argument("--save", default="summary.mp4")
    ap.add_argument("--ckpt", default=str(ROOT / "models/p9_persona_film/split0.pt"))
    ap.add_argument("--qcond", default="film")
    ap.add_argument("--sample-rate", type=int, default=15)
    ap.add_argument("--nms-thresh", type=float, default=0.4)
    ap.add_argument("--kts-vmax", type=float, default=0.5,
                    help="KTS complexity penalty; 0.5 matches dataset shot density for CLIP")
    ap.add_argument("--max-shots", type=int, default=None,
                    help="cap on KTS change points (default seq_len/4)")
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import open_clip
    dev = args.device
    print(f"loading CLIP {args.clip_model}/{args.clip_pretrained} ...")
    clip, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained)
    clip = clip.eval().to(dev)
    tok = open_clip.get_tokenizer(args.clip_model)

    print(f"decoding {args.source} (every {args.sample_rate}th frame) ...")
    n_frames, frames = decode_sampled(args.source, args.sample_rate)
    feats = clip_features(frames, clip, preprocess, dev)
    print(f"  {n_frames} frames -> {len(frames)} sampled, features {feats.shape}")

    print("detecting shots with KTS ...")
    cps, nfps, picks = run_kts(n_frames, feats, args.sample_rate, args.max_shots, args.kts_vmax)
    print(f"  {len(cps)} shots")

    print(f"loading model {args.ckpt} ...")
    model = DSNetAF(base_model="attention", num_feature=feats.shape[1],
                    num_hidden=128, num_head=8)
    if args.qcond != "none":
        model.qcond = QueryConditioner(feats.shape[1], 512, mode=args.qcond)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.eval().to(dev)

    def embed(text):
        with torch.no_grad():
            e = clip.encode_text(tok([text]).to(dev)).float()
            e = e / e.norm(dim=-1, keepdim=True)      # matches query_head.precompute
        return e[0].cpu().numpy().astype(np.float32)

    outs = []
    queries = [(args.query, args.save)]
    if args.query2:
        stem = Path(args.save).with_suffix("")
        queries = [(args.query, f"{stem}_A.mp4"), (args.query2, f"{stem}_B.mp4")]
    for text, dst in queries:
        mask = predict_summary(model, feats, embed(text), cps, nfps, picks,
                               n_frames, dev, args.nms_thresh)
        kept = write_summary(args.source, dst, mask)
        sel = selected_shots(mask, cps)
        print(f"\nquery: {text!r}")
        print(f"  kept {kept}/{n_frames} frames ({100*kept/n_frames:.1f}%), "
              f"{len(sel)}/{len(cps)} shots -> {dst}")
        outs.append((text, sel))

    if len(outs) == 2:
        a, b = outs[0][1], outs[1][1]
        j = len(a & b) / len(a | b) if (a | b) else 1.0
        print(f"\n=== personalisation check ===")
        print(f"  shots only in A : {sorted(a - b)}")
        print(f"  shots only in B : {sorted(b - a)}")
        print(f"  shared          : {len(a & b)}")
        print(f"  Jaccard overlap : {j:.3f}   (1.0 = identical summaries)")


if __name__ == "__main__":
    main()
