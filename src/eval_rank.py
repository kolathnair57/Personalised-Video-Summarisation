"""Rank-correlation evaluation, Otani et al. (2019) protocol (Phase 10.1).

Why this is the PRIMARY metric: Phase 3 measured a random-score model at 56.42 F1 on
TVSum versus published DSNet's 62.09. F1 is dominated by KTS segmentation plus the 15%
knapsack budget, so it largely measures the pipeline rather than the model. Rank
correlation compares the predicted ORDERING against human orderings and has no such
shortcut.

## The protocol, and its asymmetry

  TVSum : correlate the prediction against EACH annotator separately, then average the
          correlations. TVSum ships 20 per-frame Likert (1-5) scores per video in
          ydata-tvsum50-anno.tsv. (Verified: that file's length == n_frames, and the h5
          gtscore is exactly its mean sampled at `picks`, Spearman 1.0000.)
  SumMe : average the annotators FIRST, then correlate once. SumMe's annotations are
          binary keyshot selections (15-18 users), so correlating against one user's
          binary vector is noise-dominated; the mean is a selection frequency.

Getting this backwards is the runbook's listed failure mode ("Rank correlation ~ 0 --
protocol mismatch"), so it is implemented explicitly per dataset.

## The two rows that make a number interpretable

  random floor  : tau/rho ~ 0 by construction.
  human ceiling : leave-one-annotator-out -- each annotator against the mean of the
                  others, averaged. TVSum's inter-annotator agreement is famously low
                  (~0.2), so a model at 0.15 is doing well, not badly. Reporting a model
                  score without the ceiling is uninterpretable.

Usage:
    python src/eval_rank.py --dataset tvsum --out results/phase10/rank_tvsum.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))
sys.path.insert(0, str(ROOT / "src"))

ANNO = Path("/var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv")
BASE = {"tvsum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
        "summe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
CLIP = {"tvsum": ROOT / "data/clip_h5/tvsum_clip.h5",
        "summe": ROOT / "data/clip_h5/summe_clip.h5"}


def tau_rho(pred, ref):
    if np.std(pred) == 0 or np.std(ref) == 0:
        return np.nan, np.nan
    return kendalltau(pred, ref)[0], spearmanr(pred, ref)[0]


def load_references(dataset):
    """-> {video_key: (refs, picks)} where refs is (n_annotators, n_steps)."""
    out = {}
    if dataset == "tvsum":
        rows = {}
        with open(ANNO) as f:
            for r in csv.reader(f, delimiter="\t"):
                rows.setdefault(r[0], []).append([int(x) for x in r[2].split(",")])
        vm = json.loads((ROOT / "data/video_map.json").read_text())["tvsum"]
        stem2key = {Path(v["path"]).stem: k for k, v in vm.items()}
        with h5py.File(BASE["tvsum"], "r") as h:
            for stem, ann in rows.items():
                k = stem2key[stem]
                picks = h[k]["picks"][...]
                A = np.asarray(ann, dtype=float)             # (20, n_frames)
                out[k] = (A[:, picks], picks)                # (20, n_steps)
    else:
        with h5py.File(BASE["summe"], "r") as h:
            for k in h.keys():
                picks = h[k]["picks"][...]
                U = h[k]["user_summary"][...].astype(float)  # (n_users, n_frames)
                out[k] = (U[:, picks], picks)
    return out


def human_ceiling(refs, dataset):
    """Leave-one-annotator-out agreement -- the interpretive ceiling."""
    taus, rhos = [], []
    for k, (R, _) in refs.items():
        n = R.shape[0]
        for i in range(n):
            others = np.delete(R, i, axis=0).mean(0)
            t, r = tau_rho(R[i], others)
            if np.isfinite(t):
                taus.append(t); rhos.append(r)
    return float(np.mean(taus)), float(np.mean(rhos))


def score_against(pred_by_video, refs, dataset):
    """Apply the dataset's protocol. pred_by_video: {video: (n_steps,) or list of them}."""
    taus, rhos = [], []
    for k, (R, _) in refs.items():
        if k not in pred_by_video:
            continue
        preds = pred_by_video[k]
        preds = preds if isinstance(preds, list) else [preds]
        for p in preds:
            if dataset == "tvsum":                 # per-annotator, then average
                ts, rs = [], []
                for i in range(R.shape[0]):
                    t, r = tau_rho(p, R[i])
                    if np.isfinite(t):
                        ts.append(t); rs.append(r)
                if ts:
                    taus.append(np.mean(ts)); rhos.append(np.mean(rs))
            else:                                   # average annotators, then correlate
                t, r = tau_rho(p, R.mean(0))
                if np.isfinite(t):
                    taus.append(t); rhos.append(r)
    return float(np.mean(taus)), float(np.mean(rhos)), len(taus)


# ------------------------------------------------------------------ predictors
def pred_random(refs, seed=0):
    rng = np.random.RandomState(seed)
    return {k: rng.rand(R.shape[1]) for k, (R, _) in refs.items()}


def pred_generic_label(dataset, refs):
    """The stock human gtscore -- a strong reference predictor, not a model."""
    out = {}
    with h5py.File(CLIP[dataset], "r") as f:
        for k in refs:
            if k in f:
                out[k] = f[k]["gtscore"][...].astype(float)
    return out


def pred_dsnet(ckpt_dir, dataset, refs, num_feature, model_type="anchor-free",
               qcond=None, qcond_mode="film", personas=None, device="cuda"):
    """Run a trained DSNet over every video; average over splits' checkpoints.

    For a persona model, `personas` maps video -> list of query embeddings; each video is
    scored under EVERY persona and all predictions are returned, so the protocol averages
    over personas as well as annotators.
    """
    from modules.model_zoo import get_model
    from query_head import QueryConditioner
    h5p = CLIP[dataset] if num_feature == 512 else BASE[dataset]
    ckpts = sorted(Path(ckpt_dir).glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    out = {k: [] for k in refs}
    with h5py.File(h5p, "r") as f:
        for ck in ckpts:
            kw = dict(base_model="attention", num_feature=num_feature, num_hidden=128,
                      num_head=8)
            if model_type == "anchor-based":
                kw["anchor_scales"] = [4, 8, 16, 32]
            m = get_model(model_type, **kw)
            if qcond:
                m.qcond = QueryConditioner(num_feature, 512, mode=qcond_mode)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m = m.eval().to(device)
            with torch.no_grad():
                for k in refs:
                    if k not in f:
                        continue
                    seq = torch.from_numpy(f[k]["features"][...].astype(np.float32))
                    seq = seq.unsqueeze(0).to(device)
                    qs = personas.get(k, [None]) if personas else [None]
                    for q in qs:
                        qt = torch.from_numpy(q).to(device) if q is not None else None
                        cls = m(seq, qt)[0] if qcond else m(seq)[0]
                        out[k].append(cls.detach().cpu().numpy().reshape(-1))
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tvsum", choices=["tvsum", "summe"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    refs = load_references(args.dataset)
    print(f"{args.dataset}: {len(refs)} videos, "
          f"{list(refs.values())[0][0].shape[0]} annotators each\n")

    rows = []
    ht, hr = human_ceiling(refs, args.dataset)
    rows.append(("Human ceiling (leave-one-out)", ht, hr, "-"))

    t, r, n = score_against(pred_random(refs), refs, args.dataset)
    rows.append(("Random floor", t, r, n))

    t, r, n = score_against(pred_generic_label(args.dataset, refs), refs, args.dataset)
    rows.append(("Generic human gtscore (reference)", t, r, n))

    # persona query embeddings per video (for the persona model)
    personas = None
    ppath = ROOT / "data/query_emb.npz"
    if ppath.exists():
        from query_head import load_query_embeddings
        Q = load_query_embeddings(ppath)
        personas = {}
        for fp in sorted((ROOT / "data/persona_h5" / args.dataset).rglob("*.h5")):
            with h5py.File(fp, "r") as f:
                pid = f.attrs["persona_id"]
                if pid not in Q:
                    continue
                for v in f.keys():
                    personas.setdefault(v, []).append(Q[pid])

    models = [
        ("DSNet AF (GoogLeNet-1024)", ROOT / "models/af_basic/checkpoint", 1024, None, None),
        ("DSNet AF (CLIP-512)", ROOT / "models/af_clip/checkpoint", 512, None, None),
        ("Ours: persona+FiLM", ROOT / "models/p9_persona_film", 512, "film", personas),
        ("Control: generic+FiLM", ROOT / "models/p9_generic_film", 512, "film", personas),
    ]
    for name, ck, nf, qc, per in models:
        try:
            preds = pred_dsnet(ck, args.dataset, refs, nf, qcond=qc, personas=per,
                               device=args.device)
            t, r, n = score_against(preds, refs, args.dataset)
            rows.append((name, t, r, n))
        except Exception as e:
            rows.append((name, np.nan, np.nan, f"ERR {type(e).__name__}"))

    print(f"{'method':36s}{'tau':>9s}{'rho':>9s}{'n':>8s}")
    for nm, t, r, n in rows:
        print(f"  {nm:34s}{t:9.4f}{r:9.4f}{str(n):>8s}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            [dict(method=nm, tau=None if np.isnan(t) else t,
                  rho=None if np.isnan(r) else r, n=str(n)) for nm, t, r, n in rows],
            indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
