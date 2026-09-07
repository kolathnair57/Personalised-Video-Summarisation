
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))  # absolute: works from any cwd
from helpers import data_helper, vsumm_helper  # noqa: E402


def eval_random(splits_path: str, seeds: int, proportion: float):
    """Return (per_seed_means, per_split_means_of_last_seed) for one splits file."""
    splits = data_helper.load_yaml(splits_path)
    per_seed = []
    per_split_all = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        split_scores = []
        for split in splits:
            dataset = data_helper.VideoDataset(split["test_keys"])
            f1s = []
            for key, seq, _gt, cps, n_frames, nfps, picks, user_summary in dataset:
                # the ONLY difference from a real model: scores are noise
                pred = rng.rand(len(seq)).astype(np.float32)
                pred_summ = vsumm_helper.get_keyshot_summ(
                    pred, cps, n_frames, nfps, picks, proportion=proportion)
                # identical annotator-aggregation rule as DSNet's evaluate.py
                metric = "avg" if "tvsum" in key else "max"
                f1s.append(vsumm_helper.get_summ_f1score(
                    pred_summ, user_summary, metric))
            split_scores.append(float(np.mean(f1s)))
        per_seed.append(float(np.mean(split_scores)))
        per_split_all = split_scores
    return per_seed, per_split_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=str, nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--proportion", type=float, default=0.15)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # Resolve user-supplied paths BEFORE chdir, then move to DSNet's src: the split files
    # reference the h5 datasets by paths relative to that directory.
    args.splits = [str(Path(sp).resolve()) for sp in args.splits]
    if args.out:
        args.out = str(Path(args.out).resolve())
    os.chdir(ROOT / "third_party/DSNet/src")

    results = {}
    for sp in args.splits:
        name = Path(sp).stem
        per_seed, per_split = eval_random(sp, args.seeds, args.proportion)
        mean, sd = 100 * st.mean(per_seed), 100 * (st.stdev(per_seed) if len(per_seed) > 1 else 0.0)
        results[name] = {"f1_mean": round(mean, 2), "f1_std_over_seeds": round(sd, 2),
                         "per_seed": [round(100 * v, 2) for v in per_seed],
                         "per_split_last_seed": [round(100 * v, 2) for v in per_split],
                         "seeds": args.seeds, "proportion": args.proportion}
        print(f"{name:8s} random floor: F1 {mean:.2f} +- {sd:.2f} "
              f"(over {args.seeds} seeds)  per-seed {results[name]['per_seed']}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("written:", args.out)


if __name__ == "__main__":
    main()
