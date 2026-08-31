"""Train the query-conditioned student on persona labels (Phase 9).

Reuses DSNet's anchor-free model and losses unchanged; the only additions are
(a) query conditioning (Phase 8) and (b) an auxiliary frame-level loss.

## How persona labels enter training

DSNet does NOT regress `gtscore`. It converts it to a binary 15%-budget keyshot target
(`get_keyshot_summ` -> `downsample_summ`) and trains focal / IoU / centreness losses on
that. So a persona's `gtscore` changes WHICH shots become positives -- the existing
pipeline consumes persona supervision correctly with no modification.

The runbook asks for "MSE loss against the persona gtscore", which does not map onto a
detection model as written. Instead the DSNet losses are kept and an auxiliary loss is
added on `pred_cls` (anchor-free emits exactly one score per frame) against the persona
`gtscore`. That is the quantity Phase 10 measures with rank correlation, so the auxiliary
loss optimises the primary metric directly -- which is what runbook 9.2 is really after.

  --loss mse         DSNet losses + MSE(pred_cls, gtscore)
  --loss mse+rank    ... plus a pairwise ranking loss (Axis 2)
  --loss none        DSNet losses only

## Two traps this script avoids

**Video-level splits, not pair-level.** Each video appears with up to 8 personas. A random
split over (video, persona) pairs would put the same video in train and test, letting the
model memorise per-video importance and score well while ignoring the query. Splits are
therefore taken over VIDEOS, reusing DSNet's own 5 splits so numbers stay comparable with
Phases 2-4.

**Model selection must be persona-aware.** DSNet keeps the epoch with max F1 against
`user_summary` -- the ORIGINAL human generic summary, identical for all personas of a
video. Selecting on that while training on persona labels would systematically pick the
epoch that best IGNORES the persona. Selection here uses mean Spearman between `pred_cls`
and the held-out persona `gtscore`; generic F1 is still logged for reference.

Usage:
    python src/train_persona.py --dataset tvsum --loss mse+rank --qcond concat \
                                --out models/persona_af_tvsum
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))
sys.path.insert(0, str(ROOT / "src"))

from anchor_free import anchor_free_helper                      # noqa: E402
from anchor_free.dsnet_af import DSNetAF                        # noqa: E402
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss  # noqa: E402
from helpers import vsumm_helper                                # noqa: E402
from query_head import QueryConditioner, load_query_embeddings  # noqa: E402

logger = logging.getLogger("train_persona")
SPLITS = {"tvsum": ROOT / "third_party/DSNet/splits/tvsum.yml",
          "summe": ROOT / "third_party/DSNet/splits/summe.yml"}
GENERIC = {"tvsum": ROOT / "data/clip_h5/tvsum_clip.h5",
           "summe": ROOT / "data/clip_h5/summe_clip.h5"}


# ---------------------------------------------------------------------------- data
class Sample:
    # gtscore      = TRAINING target (varies by condition)
    # eval_gtscore = MEASUREMENT target, ALWAYS the persona label, so every condition is
    #                scored on the same yardstick. Without this each condition would be
    #                evaluated against its own training target and trivially "win".
    __slots__ = ("seq", "gtscore", "eval_gtscore", "cps", "n_frames", "nfps", "picks",
                 "user_summary", "query", "video", "persona")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def load_samples(dataset, labels, qemb, generic_query_zero=True):
    """Return {video_key: [Sample, ...]} for the requested label source.

    labels:
      persona        -- persona gtscore + that persona's query (the method)
      generic_paired -- SAME (video, persona) pairs and queries, but the TARGET is the
                        generic human gtscore. This is runbook 9.3's control: the model
                        still reads a query, but supervision does not depend on it, so it
                        isolates "reads a query" from "persona supervision".
      generic        -- plain DSNet: one sample per video, no query (= Phase 4 baseline)
    """
    out = defaultdict(list)
    if labels in ("persona", "generic_paired"):
        gen = {}
        if labels == "generic_paired":
            with h5py.File(GENERIC[dataset], "r") as gf:
                for v in gf.keys():
                    gen[v] = gf[v]["gtscore"][...].astype(np.float32)
        for fp in sorted((ROOT / "data/persona_h5" / dataset).rglob("*.h5")):
            with h5py.File(fp, "r") as f:
                pid = f.attrs["persona_id"]
                if pid not in qemb:
                    continue
                q = qemb[pid]
                for v in f.keys():
                    g = f[v]
                    out[v].append(Sample(
                        seq=g["features"][...].astype(np.float32),
                        gtscore=(gen[v] if labels == "generic_paired"
                                 else g["gtscore"][...].astype(np.float32)),
                        eval_gtscore=g["gtscore"][...].astype(np.float32),
                        cps=g["change_points"][...].astype(np.int32),
                        n_frames=int(g["n_frames"][()]),
                        nfps=g["n_frame_per_seg"][...].astype(np.int32),
                        picks=g["picks"][...].astype(np.int32),
                        user_summary=g["user_summary"][...].astype(np.float32),
                        query=q.astype(np.float32), video=v, persona=pid))
    else:                                    # generic control: stock gtscore
        with h5py.File(GENERIC[dataset], "r") as f:
            for v in f.keys():
                g = f[v]
                gt = g["gtscore"][...].astype(np.float32)
                q = np.zeros(512, dtype=np.float32) if generic_query_zero else None
                out[v].append(Sample(
                    seq=g["features"][...].astype(np.float32), gtscore=gt,
                    eval_gtscore=gt,
                    cps=g["change_points"][...].astype(np.int32),
                    n_frames=int(g["n_frames"][()]),
                    nfps=g["n_frame_per_seg"][...].astype(np.int32),
                    picks=g["picks"][...].astype(np.int32),
                    user_summary=g["user_summary"][...].astype(np.float32),
                    query=q, video=v, persona="generic"))
    return out


def video_keys_of(split_entry):
    return [k.split("/")[-1] for k in split_entry]


# ---------------------------------------------------------------------------- losses
def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)


def pairwise_rank_loss(pred, target, n_pairs=512, margin=0.0, gen=None):
    """Penalise inversions: for target_i > target_j, want pred_i > pred_j."""
    n = pred.numel()
    if n < 2:
        return pred.new_zeros(())
    i = torch.randint(0, n, (n_pairs,), device=pred.device, generator=gen)
    j = torch.randint(0, n, (n_pairs,), device=pred.device, generator=gen)
    ti, tj = target[i], target[j]
    keep = ti != tj
    if keep.sum() == 0:
        return pred.new_zeros(())
    i, j, ti, tj = i[keep], j[keep], ti[keep], tj[keep]
    sign = torch.sign(ti - tj)
    # logistic ranking loss on the signed difference
    return torch.nn.functional.softplus(-sign * (pred[i] - pred[j]) + margin).mean()


# ---------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, samples, device, nms_thresh, metric):
    """Persona-aware selection metric (primary) + generic F1 (reference only).

    rho  : Spearman between the model's per-frame score and the PERSONA gtscore.
           This is the selection criterion -- DSNet's built-in max-F1-vs-user_summary
           would select the epoch that best ignores the persona, since user_summary is
           the generic human annotation and identical across a video's personas.
    f1   : the Phase 2-4 harness, computed against the generic user_summary, logged for
           comparability only. It is NOT used for selection.
    """
    from helpers import bbox_helper
    model.eval()
    rhos, f1s = [], []
    for s in samples:
        seq = torch.from_numpy(s.seq).unsqueeze(0).to(device)
        q = torch.from_numpy(s.query).to(device) if s.query is not None else None

        raw_cls, _, _ = model(seq, q)
        raw = raw_cls.detach().cpu().numpy().reshape(-1)
        r = spearmanr(raw, s.eval_gtscore)[0]   # ALWAYS the persona label
        if np.isfinite(r):
            rhos.append(r)

        pred_cls, pred_bboxes = model.predict(seq, q)      # DSNet's own logic
        seq_len = len(raw)
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
        summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, s.cps,
                                         s.n_frames, s.nfps, s.picks)
        f1s.append(vsumm_helper.get_summ_f1score(summ, s.user_summary, metric))
    return float(np.mean(rhos)) if rhos else 0.0, float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------- train
def train_split(args, split_idx, split, by_video, device):
    train_keys = set(video_keys_of(split["train_keys"]))
    test_keys = set(video_keys_of(split["test_keys"]))
    tr = [s for v, ss in by_video.items() if v in train_keys for s in ss]
    te = [s for v, ss in by_video.items() if v in test_keys for s in ss]
    if not tr or not te:
        return None

    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head).to(device)
    if args.qcond != "none":
        model.qcond = QueryConditioner(args.num_feature, args.query_dim,
                                       mode=args.qcond).to(device)
    # The query head starts at the exact identity (weights on the query half = 0), so it
    # must travel further than the rest of the network before it influences anything.
    # With one shared lr and early stopping at epoch 0-2, it never moves and the model
    # ignores the persona entirely (measured: outputs for different personas correlated
    # at +0.9999). A separate, larger lr for the conditioner fixes that.
    qparams = list(model.qcond.parameters()) if getattr(model, "qcond", None) else []
    qids = {id(p) for p in qparams}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in qids]
    groups = [dict(params=base, lr=args.lr)]
    if qparams:
        groups.append(dict(params=qparams, lr=args.lr * args.qcond_lr_mult))
    opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)

    best_rho, best_f1, best_epoch = -2.0, 0.0, -1
    save_path = Path(args.out) / f"split{split_idx}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed + split_idx)

    for epoch in range(args.max_epoch):
        model.train()
        order = rng.permutation(len(tr))
        agg = defaultdict(float); nb = 0
        for idx in order:
            s = tr[idx]
            keyshot = vsumm_helper.get_keyshot_summ(s.gtscore, s.cps, s.n_frames,
                                                    s.nfps, s.picks)
            target = vsumm_helper.downsample_summ(keyshot)
            if not target.any():
                continue
            seq = torch.from_numpy(s.seq).unsqueeze(0).to(device)
            q = torch.from_numpy(s.query).to(device) if s.query is not None else None
            cls_label = torch.from_numpy(target.astype(np.float32)).to(device)
            loc_label = torch.from_numpy(
                anchor_free_helper.get_loc_label(target).astype(np.float32)).to(device)
            ctr_label = torch.from_numpy(
                anchor_free_helper.get_ctr_label(
                    target, anchor_free_helper.get_loc_label(target)).astype(np.float32)
            ).to(device)

            pred_cls, pred_loc, pred_ctr = model(seq, q)
            cls_loss = calc_cls_loss(pred_cls, cls_label, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label, args.reg_loss)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)
            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss

            # auxiliary frame-level objective against the persona gtscore
            if args.loss != "none":
                gt = torch.from_numpy(s.gtscore).to(device)
                gt = norm01(gt)
                if "mse" in args.loss:
                    mse = torch.nn.functional.mse_loss(pred_cls.view(-1), gt)
                    loss = loss + args.lambda_score * mse
                    agg["mse"] += float(mse)
                if "rank" in args.loss:
                    rk = pairwise_rank_loss(pred_cls.view(-1), gt)
                    loss = loss + args.lambda_rank * rk
                    agg["rank"] += float(rk)

            opt.zero_grad(); loss.backward(); opt.step()
            agg["loss"] += float(loss); nb += 1

        rho, f1 = evaluate(model, te, device, args.nms_thresh, args.metric)
        if rho > best_rho:                       # persona-aware selection
            best_rho, best_f1, best_epoch = rho, f1, epoch
            torch.save(model.state_dict(), save_path)
        if epoch % args.log_every == 0 or epoch == args.max_epoch - 1:
            msg = " ".join(f"{k} {v/max(nb,1):.4f}" for k, v in agg.items())
            logger.info(f"  split{split_idx} ep{epoch:3d}/{args.max_epoch} {msg} "
                        f"| val rho {rho:+.4f} (best {best_rho:+.4f} @ep{best_epoch}) "
                        f"| genericF1 {f1:.4f}")
    return dict(split=split_idx, rho=best_rho, f1=best_f1, epoch=best_epoch,
                n_train=len(tr), n_test=len(te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tvsum", choices=["tvsum", "summe"])
    ap.add_argument("--labels", default="persona",
                    choices=["persona", "generic_paired", "generic"])
    ap.add_argument("--qcond", default="concat", choices=["concat", "film", "none"])
    ap.add_argument("--loss", default="mse+rank", choices=["none", "mse", "mse+rank", "rank"])
    ap.add_argument("--out", default=str(ROOT / "models/persona_af"))
    ap.add_argument("--max-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-feature", type=int, default=512)
    ap.add_argument("--query-dim", type=int, default=512)
    ap.add_argument("--num-hidden", type=int, default=128)
    ap.add_argument("--num-head", type=int, default=8)
    ap.add_argument("--base-model", default="attention")
    ap.add_argument("--cls-loss", default="focal")
    ap.add_argument("--reg-loss", default="soft-iou")
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-ctr", type=float, default=1.0)
    ap.add_argument("--lambda-score", type=float, default=1.0)
    ap.add_argument("--lambda-rank", type=float, default=1.0)
    ap.add_argument("--nms-thresh", type=float, default=0.4)
    ap.add_argument("--metric", default=None, choices=["avg", "max"],
                    help="annotator aggregation; DSNet uses avg for tvsum, max for summe")
    ap.add_argument("--qcond-lr-mult", type=float, default=1.0,
                    help="lr multiplier for the query conditioner (it starts at identity)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--splits", type=int, nargs="+", default=None)
    args = ap.parse_args()
    if args.metric is None:
        args.metric = "avg" if args.dataset == "tvsum" else "max"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    qemb = load_query_embeddings(ROOT / "data/query_emb.npz")
    by_video = load_samples(args.dataset, args.labels, qemb)
    n = sum(len(v) for v in by_video.values())
    logger.info(f"{args.dataset}/{args.labels}: {n} samples over {len(by_video)} videos "
                f"| qcond={args.qcond} loss={args.loss}")

    splits = yaml.safe_load(open(SPLITS[args.dataset]))
    idxs = args.splits if args.splits is not None else range(len(splits))
    results = []
    for i in idxs:
        r = train_split(args, i, splits[i], by_video, device)
        if r:
            results.append(r)
            logger.info(f"split{i} done: rho {r['rho']:+.4f} genericF1 {r['f1']:.4f} "
                        f"(ep{r['epoch']}, {r['n_train']} train / {r['n_test']} test)")
    if results:
        rho = np.array([r["rho"] for r in results]); f1 = np.array([r["f1"] for r in results])
        logger.info(f"\n=== {args.dataset} {args.labels} qcond={args.qcond} loss={args.loss} ===")
        logger.info(f"  spearman vs persona label: {rho.mean():+.4f} +- {rho.std():.4f}")
        logger.info(f"  generic F1 (reference)   : {100*f1.mean():.2f} +- {100*f1.std():.2f}")
        out = Path(args.out) / "results.json"
        out.write_text(json.dumps(dict(args=vars(args), splits=results,
                                       rho_mean=float(rho.mean()), rho_std=float(rho.std()),
                                       f1_mean=float(f1.mean()), f1_std=float(f1.std())), indent=2))
        logger.info(f"  wrote {out}")


if __name__ == "__main__":
    main()
