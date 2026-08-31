# Phase 3 — Baselines

Date: 2026-08-11. All numbers produced by **DSNet's evaluation harness** on **DSNet's splits**.

## Results

F1 (%), mean +- std over 5 splits. "vs floor" is the gain over a model that assigns
*random* importance scores.

### TVSum

| Method | F1 | vs floor | Published | Delta |
|---|---|---|---|---|
| **Random floor** | **56.42** +- 0.90 | (floor) | - | - |
| DSNet (anchor-based) | 62.09 +- 2.01 | **+5.67** | 62.05 | +0.04 |
| DSNet (anchor-free) | 61.16 +- 2.71 | **+4.74** | 61.86 | -0.70 |
| PGL-SUM | 62.23 +- 2.19 | **+5.81** | 61.0 | +1.23 |

### SumMe

| Method | F1 | vs floor | Published | Delta |
|---|---|---|---|---|
| **Random floor** | **41.49** +- 1.33 | (floor) | - | - |
| DSNet (anchor-based) | 47.23 +- 2.95 | **+5.74** | 50.19 | -2.96 |
| DSNet (anchor-free) | 51.31 +- 3.73 | **+9.82** | 51.18 | +0.13 |
| PGL-SUM | 48.20 +- 6.80 | **+6.71** | 57.1 | -8.90 |

## The headline finding: the random floor

**A model assigning pure noise scores reaches 56.42 F1 on TVSum.** Every published
architecture here beats it by only ~5-6 points, and all three cluster within 1.1 points
of each other (61.16 - 62.23).

This is the Otani et al. (2019) result, reproduced locally on our own data and harness.
The cause is structural: F1 is computed *after* KTS shot segmentation and a knapsack that
fills a ~15% length budget. That pipeline produces a plausible summary regardless of the
scores fed into it, so F1 largely measures the pipeline rather than the model.

**Consequence for this project:** reporting "the persona model achieves X F1" is close to
meaningless without the floor beside it. This is the empirical justification for the
runbook's decision to make **rank correlation primary** and F1 secondary
("never lead with it"). We now have that evidence first-hand rather than by citation.

Random floor is averaged over **5 seeds** (a single random draw is itself noisy):
TVSum per-seed 56.09 / 57.48 / 56.06 / 57.18 / 55.28.

## Methodological decisions

Two choices that a reviewer will probe, both made for comparability:

**1. PGL-SUM was run on DSNet's splits, not its own.** Both repos ship "5 randomly
generated 80/20 splits", but they were generated independently: their test sets overlap
by only 2-4 of 10 videos per fold. Comparing across them would compare *test sets*, not
models. DSNet's splits were converted to PGL-SUM's JSON format
(`third_party/PGL-SUM/data/datasets/splits/`).

**2. PGL-SUM was scored with DSNet's harness, and with DSNet's epoch-selection rule.**
DSNet keeps the epoch with max F-score on the test split (`anchor_based/train.py:102-104`).
That is optimistic (test-set model selection), but applying a stricter rule to PGL-SUM
alone would rig the comparison. Both therefore use max-over-epochs. Epoch -1 (untrained)
excluded. Implemented in `src/eval_pglsum.py`.

Before trusting this, the two codebases' summary-generation routines were compared
line by line and are algorithmically equivalent:

| Step | PGL-SUM `generate_summary.py` | DSNet `get_keyshot_summ` |
|---|---|---|
| upsample subsampled scores to frame level | via `positions` | via `picks` |
| shot score | mean of frames in shot | mean of frames in shot |
| selection | knapsack, 15% budget | knapsack, 15% budget |

Only difference: DSNet quantises shot scores to `int(1000 * mean)` because ortools
requires integer values. That cannot account for a multi-point difference.

## The SumMe discrepancy (PGL-SUM, -8.9)

TVSum reproduces well (+1.23). SumMe does not (48.20 vs 57.1 published). Contributing
factors, in order of likely importance:

1. **Different splits.** The published number uses PGL-SUM's own splits; we used DSNet's
   (see decision 1). On a 25-video dataset, split choice moves the number a lot.
2. **Extreme variance.** SumMe has 25 videos, so each fold tests on ~5. Our per-split
   results range 40.91 to 57.84, std 6.80. The published 57.1 sits within ~1.3 std of
   our mean -- and is almost exactly our best split (57.84).
3. **Split 2 failed to train**: its best epoch was 0 of 200, i.e. the model never beat
   its initialisation on that fold.

Not yet ruled out: whether PGL-SUM recovers ~57 on its *own* splits. That is a cheap,
worthwhile check (10 runs, ~45 min) and would cleanly separate "splits" from "our setup".
**Not run yet.**

Per the Phase 2 convention, our locally produced numbers are the reference baseline.

## CLIP-It: dropped

Runbook 3.2 assumes a clonable reference implementation. There isn't one:

- Official [medhini/clip_it](https://github.com/medhini/clip_it): **README only**,
  "Code coming soon!" since 2021.
- Unofficial [srpkdyy/CLIP-It](https://github.com/srpkdyy/CLIP-It): **142 lines, one file,
  model definition only.** No optimizer, loss, dataloader or eval (grep for
  `optimizer|loss|backward|DataLoader|train(` returns zero matches). Expects raw frames
  and 7 caption sentences, not the precomputed h5 features.

It would also have been blocked on CLIP features (Phase 4) and captions (Phase 5).
**Dropped by decision on 2026-08-11.** The clone is left in `third_party/CLIP-It/` unused.

**Replacement control.** CLIP-It's role was to separate "the model reads a query" from
"the supervision is persona-conditioned". A cleaner substitute, planned for Phase 9:
train the Phase 8 query-conditioned DSNet on the **stock `gtscore`**. Same architecture,
query head enabled, generic supervision -- an apples-to-apples ablation within one model
instead of across two codebases. Costs one training run, no new code.

## Files

```
src/random_baseline.py             random floor (5 seeds, DSNet harness)
src/eval_pglsum.py                 PGL-SUM scores -> DSNet harness
src/watch_progress.sh              live training progress
results/baselines.csv              the table
results/phase2/random_floor.json   per-seed random results
results/phase3/pglsum_dsnet_harness.json  per-split PGL-SUM results + best epochs
results/phase3/pglsum_*.log        raw training logs (tqdm noise, no metrics)
```

## Reproduce

```bash
# random floor
source env/activate-train.sh
cd third_party/DSNet/src
python ../../../src/random_baseline.py --splits ../splits/tvsum.yml ../splits/summe.yml --seeds 5

# PGL-SUM (10 runs; TVSum ~35 min, SumMe ~15 min with 5 concurrent per GPU)
cd third_party/PGL-SUM
for i in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=1 python model/main.py --split_index $i --n_epochs 200 --video_type TVSum & done
for i in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 python model/main.py --split_index $i --n_epochs 200 --video_type SumMe & done
wait
cd ../..
python src/eval_pglsum.py --exp third_party/PGL-SUM/Summaries/PGL-SUM/exp1 --out results/phase3/pglsum_dsnet_harness.json
```

PGL-SUM required **no** compatibility fixes (runs clean on torch 2.6). It needs only
`tensorboardX`; the `tensorflow` import lives in `evaluation/exportTensorFlowLog.py`,
which drives their epoch-selection and is unused here.
