"""Evaluate PGL-SUM's predicted scores with DSNet's evaluation harness.

Why this exists (runbook Phase 3 checkpoint: "all under one evaluation harness"):
PGL-SUM ships its own F1 code, which differs from DSNet's in annotator aggregation and
summary generation. Putting a number from PGL-SUM's harness next to a number from
DSNet's harness compares measurement instruments, not models. So we take PGL-SUM's raw
per-frame scores and push them through the *same* KTS->knapsack selection and the *same*
F1 as Phase 2.

Model selection: DSNet keeps the epoch with max F-score on the test split
(anchor_based/train.py:102-104). That is optimistic, but to compare fairly PGL-SUM gets
the identical rule -- max over epochs, same harness. Both numbers are therefore
optimistic in the same way. Epoch -1 (the untrained model) is excluded.

Usage (cwd = repo root):
    python src/eval_pglsum.py --exp third_party/PGL-SUM/Summaries/PGL-SUM/exp1 \
                              --out results/phase3/pglsum_dsnet_harness.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))
from helpers import vsumm_helper  # noqa: E402

H5 = {"TVSum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
      "SumMe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
# DSNet's rule: tvsum averages over annotators, summe takes the max
METRIC = {"TVSum": "avg", "SumMe": "max"}


def load_meta(h5_path):
    """Preload the per-video arrays the selection/eval needs."""
    meta = {}
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            g = f[k]
            meta[k] = dict(cps=g["change_points"][...].astype(np.int32),
                           n_frames=g["n_frames"][()].astype(np.int32),
                           nfps=g["n_frame_per_seg"][...].astype(np.int32),
                           picks=g["picks"][...].astype(np.int32),
                           user=g["user_summary"][...].astype(np.float32))
    return meta


def score_epoch(scores_json, meta, metric):
    """F1 of one epoch's predictions, averaged over that split's test videos."""
    f1s = []
    for vid, scores in scores_json.items():
        m = meta[vid]
        pred = np.asarray(scores, dtype=np.float32)
        summ = vsumm_helper.get_keyshot_summ(
            pred, m["cps"], m["n_frames"], m["nfps"], m["picks"])
        f1s.append(vsumm_helper.get_summ_f1score(summ, m["user"], metric))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="PGL-SUM Summaries/.../exp1 dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for ds in ("TVSum", "SumMe"):
        meta = load_meta(H5[ds])
        per_split, best_epochs = [], []
        for si in range(5):
            d = Path(args.exp) / ds / "results" / f"split{si}"
            files = [p for p in d.glob(f"{ds}_*.json")
                     if not p.stem.endswith("_-1")]          # drop untrained epoch
            if not files:
                print(f"  !! no score files in {d}")
                continue
            per_epoch = []
            for p in sorted(files, key=lambda x: int(x.stem.split("_")[-1])):
                per_epoch.append((int(p.stem.split("_")[-1]),
                                  score_epoch(json.load(open(p)), meta, METRIC[ds])))
            ep, f1 = max(per_epoch, key=lambda t: t[1])      # DSNet's max-F1 rule
            per_split.append(f1)
            best_epochs.append(ep)
            print(f"  {ds} split{si}: F1 {100*f1:.2f} (best epoch {ep} of {len(per_epoch)})")
        mean = 100 * st.mean(per_split)
        sd = 100 * st.stdev(per_split) if len(per_split) > 1 else 0.0
        results[ds] = dict(f1_mean=round(mean, 2), f1_std=round(sd, 2),
                           per_split=[round(100 * v, 2) for v in per_split],
                           best_epochs=best_epochs)
        print(f"{ds}: F1 {mean:.2f} +- {sd:.2f}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("written:", args.out)


if __name__ == "__main__":
    main()
