"""Re-extract frame features with CLIP, keeping the eccv16 h5 schema (Phase 4, step 2).

Replaces ONLY `features` (GoogLeNet pool5, 1024-d) with CLIP image embeddings (512-d for
ViT-B-32). `gtscore`, `change_points`, `picks`, `n_frames`, `n_steps`, `user_summary` and
`gtsummary` are copied through untouched, so KTS, the knapsack selection and the whole
evaluation harness keep working unchanged.

Why CLIP: the persona query is encoded by CLIP's *text* encoder in Phase 8. CLIP is
trained so image and text embeddings share one space, so query and frame become directly
comparable. GoogLeNet has no such text counterpart.

Frames are taken at the `picks` positions, which index the ORIGINAL video. Decoding is
strictly sequential: frame-seek APIs (e.g. OpenCV's CAP_PROP_POS_FRAMES) can silently
return a neighbouring frame on some codecs, which would misalign every feature with its
label. Sequential decode is slower but exact.

Requires data/video_map.json from src/map_videos.py (which verifies the key->file
mapping; see that file for why that matters).

Usage:
    python src/extract_clip.py --dataset tvsum --out data/clip_h5
    python src/extract_clip.py --dataset summe --out data/clip_h5
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import av
import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE_H5 = {"tvsum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
           "summe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
OUT_NAME = {"tvsum": "tvsum_clip.h5", "summe": "summe_clip.h5"}


def read_frames_at(video_path, wanted):
    """Decode sequentially, returning {frame_index: PIL.Image} for indices in `wanted`.

    Only the requested frames are converted to RGB images; the rest are decoded and
    discarded, which is the unavoidable cost of exact indexing.
    """
    want = set(int(w) for w in wanted)
    last = max(want)
    out = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"          # multi-threaded decode
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                out[i] = frame.to_image()
            if i >= last:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tvsum", "summe"], required=True)
    ap.add_argument("--map", default=str(ROOT / "data/video_map.json"))
    ap.add_argument("--out", default=str(ROOT / "data/clip_h5"))
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only process N videos (smoke test)")
    args = ap.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.eval().to(device)
    dim = model.visual.output_dim
    print(f"CLIP {args.model}/{args.pretrained} on {device}, output dim {dim}")

    vmap = json.load(open(args.map))[args.dataset]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / OUT_NAME[args.dataset]

    # start from a copy so every non-feature dataset is preserved exactly
    if not out_h5.exists():
        shutil.copy(BASE_H5[args.dataset], out_h5)
        print(f"copied {BASE_H5[args.dataset].name} -> {out_h5}")

    keys = sorted(vmap, key=lambda k: int(k.split("_")[1]))
    if args.limit:
        keys = keys[:args.limit]

    t0 = time.time()
    with h5py.File(out_h5, "r+") as f:
        for n, key in enumerate(keys, 1):
            g = f[key]
            picks = g["picks"][...].astype(np.int64)
            n_steps = int(g["n_steps"][()])
            assert len(picks) == n_steps, f"{key}: picks {len(picks)} != n_steps {n_steps}"

            if g["features"].shape == (n_steps, dim) and g.attrs.get("clip_model"):
                print(f"[{n}/{len(keys)}] {key}: already done, skipping")
                continue

            ts = time.time()
            frames = read_frames_at(vmap[key]["path"], picks)
            missing = [int(p) for p in picks if int(p) not in frames]
            if missing:
                raise RuntimeError(
                    f"{key}: {len(missing)} picked frames could not be decoded "
                    f"(first few: {missing[:5]}) from {vmap[key]['path']}")

            imgs = [frames[int(p)] for p in picks]          # strict picks order
            feats = []
            with torch.no_grad():
                for i in range(0, len(imgs), args.batch_size):
                    batch = torch.stack([preprocess(im) for im in imgs[i:i + args.batch_size]])
                    feats.append(model.encode_image(batch.to(device)).float().cpu().numpy())
            feats = np.concatenate(feats, 0).astype(np.float32)
            assert feats.shape == (n_steps, dim), f"{key}: got {feats.shape}, want {(n_steps, dim)}"

            del g["features"]
            g.create_dataset("features", data=feats)
            g.attrs["clip_model"] = f"{args.model}/{args.pretrained}"
            g.attrs["source_video"] = Path(vmap[key]["path"]).name
            print(f"[{n}/{len(keys)}] {key}: {feats.shape} from "
                  f"{Path(vmap[key]['path']).name}  ({time.time()-ts:.1f}s)", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out_h5}")


if __name__ == "__main__":
    main()
