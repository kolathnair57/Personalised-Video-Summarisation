# Phase 2 — DSNet reproduction (baseline #1)

Date: 2026-08-11. Machine: `nitt`, RTX 3090 (GPU 0 = anchor-based, GPU 1 = anchor-free).
DSNet base commit: `1804176e2e8b57846beb063667448982273fca89`.

## Result

F1 (%), mean +- std over the repo's own 5 splits, evaluated with `evaluate.py`:

| Variant | Dataset | **Ours** | Published | Delta |
|---|---|---|---|---|
| Anchor-based | TVSum | **62.09** +- 2.01 | 62.05 | **+0.04** |
| Anchor-based | SumMe | **47.23** +- 2.95 | 50.19 | **-2.96** |
| Anchor-free  | TVSum | **61.16** +- 2.71 | 61.86 | -0.70 |
| Anchor-free  | SumMe | **51.31** +- 3.73 | 51.18 | **+0.13** |

Three of four land within 0.7 points. The one gap is anchor-based on SumMe (-2.96), and
it is **within one standard deviation of the split-to-split spread** (+-2.95). SumMe has
only 25 videos, so each of the 5 folds tests on ~5 videos; per-split F1 ranges from 43.24
to 51.12 for this cell. A 3-point difference at that sample size is noise, not a defect.

Per runbook 2.3, **these local numbers are the reference baseline** for all later
comparisons, not the published ones. Full per-split values are in `../baselines.csv`;
raw logs in `train_*.log` and `eval_*.log`.

`evaluate.py` reproduced the training-time F-scores exactly, which confirms checkpoints
round-trip correctly through save/load.

## Changes made to DSNet

The runbook says "clone, unmodified". DSNet targets Python 3.6-3.8, torch 1.1.0,
numpy 1.19.5, ortools 8.0; this machine runs Python 3.12, torch 2.6, numpy 2.4,
ortools 9.15. Two library APIs had been removed in the interval.

Complete diff: **14 insertions, 14 deletions across 3 files** — saved as
`dsnet_compat_fixes.patch`, applies cleanly to base commit `1804176`.

| Fix | Files | Cause |
|---|---|---|
| `np.bool` -> `bool` (8 sites) | `vsumm_helper.py`, `bbox_helper.py`, `anchor_free_helper.py` | numpy 2.0 removed the deprecated alias |
| `ortools.algorithms.pywrapknapsack_solver` -> `ortools.algorithms.python.knapsack_solver`; `Init/Solve/BestSolutionContains` -> `init/solve/best_solution_contains` | `vsumm_helper.py` | ortools 9.x moved the module and renamed methods to snake_case |

**Every change is a library API rename. No algorithm, hyperparameter, loss, split, or
data path was altered.** For the dissertation this is defensible as "DSNet unmodified
apart from two library-compatibility renames forced by a five-year dependency gap."

Verified before training, against known-correct values:

```
knapsack([60,100,120],[10,20,30], cap=50) -> items [1,2], value 220   (optimal)
f1_score([1,1,0,0],[1,0,0,0])             -> 0.6667                   (correct)
get_keyshot_summ(...)                     -> selects 15.0% of frames  (standard budget)
```

`torch.load` was deliberately **not** changed. torch 2.6 flipped `weights_only` to
`True` by default, but these files are plain tensor `state_dict`s, which load fine —
confirmed by `evaluate.py` reproducing the training F-scores from disk.

## Notes for later phases

- **`--num-feature 1024`** matches GoogLeNet pool5. Phase 4 replaces features with
  512-d CLIP, so training must pass `--num-feature 512` or it fails on a shape mismatch.
- **`torch_geometric` is never needed** with the default `--base-model attention`; the
  import lives inside `GCNExtractor.__init__`. Do not use `--base-model gcn` — it would
  drag in torch-scatter/sparse/cluster, which usually need compiling from source.
- Training cost: ~2h for 10 splits (5 folds x 2 datasets) at 300 epochs, per variant.
  The two variants ran concurrently on separate GPUs. GPU memory ~1 GB, utilisation
  <20% — the bottleneck is data loading, not compute, so parallel runs are near-free.
- Seed is fixed at 12345 in the repo defaults; runs are reproducible.

## Reproduce

```bash
source env/activate-train.sh
cd third_party/DSNet/src
export CUDA_VISIBLE_DEVICES=0
python train.py    anchor-based --model-dir ../../../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
python evaluate.py anchor-based --model-dir ../../../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
python train.py    anchor-free  --model-dir ../../../models/af_basic --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
python evaluate.py anchor-free  --model-dir ../../../models/af_basic --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

Datasets downloaded fresh from the authors' Dropbox link (`dsnet_datasets.zip`, 178 MB):

```
a108d05ba7a9369ef6c891df475e6a2b  eccv16_dataset_summe_google_pool5.h5
b56ae0ae124d5b3f8e7c0f641bb5c059  eccv16_dataset_tvsum_google_pool5.h5
```
