
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = {"tvsum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
        "summe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
CLIP = {"tvsum": ROOT / "data/clip_h5/tvsum_clip.h5",
        "summe": ROOT / "data/clip_h5/summe_clip.h5"}
PRESERVE = ("gtscore", "change_points", "picks", "n_frames", "n_steps",
            "user_summary", "gtsummary", "n_frame_per_seg")


def profile(F):
    """Cosine similarity between consecutive frames -- a video's temporal fingerprint."""
    F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-8, None)
    return np.einsum("ij,ij->i", F[:-1], F[1:])


def verify(name, thresh):
    b, c = h5py.File(BASE[name], "r"), h5py.File(CLIP[name], "r")
    problems, corrs, ranks = [], {}, {}
    bkeys, ckeys = set(b.keys()), set(c.keys())
    if bkeys != ckeys:
        problems.append(f"key mismatch: missing {bkeys - ckeys}, extra {ckeys - bkeys}")

    for k in sorted(bkeys & ckeys, key=lambda x: int(x.split("_")[1])):
        n_steps = int(b[k]["n_steps"][()])
        # 1. schema preserved
        for ds in PRESERVE:
            if ds in b[k] and not np.array_equal(b[k][ds][()], c[k][ds][()]):
                problems.append(f"{k}/{ds} differs from base")
        # 2. shape + sanity
        F = c[k]["features"][...]
        if F.ndim != 2 or F.shape[0] != n_steps:
            problems.append(f"{k}: features {F.shape}, expected ({n_steps}, D)")
            continue
        if not np.isfinite(F).all():
            problems.append(f"{k}: features contain NaN/Inf")
        if (np.abs(F).sum(1) == 0).any():
            problems.append(f"{k}: {(np.abs(F).sum(1)==0).sum()} all-zero feature rows")
        # 3. mapping evidence -- RANK test, not an absolute threshold.
        #    An absolute cutoff wrongly flags near-static videos: if consecutive frames
        #    are ~97% identical (SumMe video_1: profile mean 0.975, std 0.015) there is
        #    almost no temporal variation to correlate, so the coefficient is small even
        #    when the mapping is correct. What actually matters is whether this CLIP
        #    profile matches its OWN base profile better than any other video's.
        pc = profile(F)
        scores = {}
        for k2 in sorted(bkeys & ckeys):
            pg = profile(b[k2]["features"][...])
            n = min(len(pg), len(pc))
            if n > 10:
                scores[k2] = spearmanr(pg[:n], pc[:n])[0]
        rank = sorted(scores, key=lambda x: -scores[x])
        corrs[k] = float(scores.get(k, np.nan))
        pos = rank.index(k) + 1 if k in rank else -1
        ranks[k] = pos
        if pos != 1:
            runner = rank[0]
            problems.append(
                f"{k}: its CLIP profile matches '{runner}' better than itself "
                f"(rank #{pos}/{len(rank)}; own {scores.get(k, float('nan')):.3f} vs "
                f"{scores[runner]:.3f}) -- LIKELY WRONG VIDEO "
                f"({c[k].attrs.get('source_video','?')})")
        elif corrs[k] < thresh:
            print(f"    note: {k} corr {corrs[k]:.3f} is low but still rank #1 "
                  f"(near-static video); mapping OK")

    # 4. control: how does a deliberately wrong pairing score?
    ks = sorted(bkeys & ckeys, key=lambda x: int(x.split("_")[1]))
    ctrl = []
    for i in range(min(10, len(ks) - 1)):
        pg, pc = profile(b[ks[i]]["features"][...]), profile(c[ks[i + 1]]["features"][...])
        n = min(len(pg), len(pc))
        if n > 10:
            ctrl.append(spearmanr(pg[:n], pc[:n])[0])

    dims = {c[k]["features"].shape[1] for k in ckeys}
    print(f"\n=== {name} ===")
    print(f"  videos            : {len(ckeys)}")
    print(f"  feature dim       : {dims}")
    print(f"  clip_model attr   : {set(c[k].attrs.get('clip_model','?') for k in ckeys)}")
    if corrs:
        v = np.array([x for x in corrs.values() if np.isfinite(x)])
        print(f"  mapping corr      : mean {v.mean():.3f}  min {v.min():.3f}  "
              f"(worst: {min(corrs, key=corrs.get)})")
    if ranks:
        ok = sum(1 for r in ranks.values() if r == 1)
        print(f"  rank-#1 matches   : {ok}/{len(ranks)}  <- each video matches ITSELF best")
    if ctrl:
        print(f"  control (wrong)   : mean {np.mean(ctrl):+.3f}  <- what a mismapping looks like")
    print(f"  problems          : {len(problems)}")
    for p in problems[:20]:
        print(f"    !! {p}")
    b.close(); c.close()
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["tvsum", "summe"])
    ap.add_argument("--corr-threshold", type=float, default=0.30)
    args = ap.parse_args()
    allp = []
    for name in args.dataset:
        allp += verify(name, args.corr_threshold)
    print("\n" + ("ALL CHECKS PASSED" if not allp else f"{len(allp)} PROBLEM(S) FOUND"))
    sys.exit(1 if allp else 0)


if __name__ == "__main__":
    main()
