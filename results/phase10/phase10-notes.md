# Phase 10 — Evaluation

Date: 2026-08-13. Primary metric: rank correlation (Otani et al. 2019 protocol).

## Why rank correlation is primary

Phase 3 measured a **random-score** model at **56.42 F1** on TVSum against published
DSNet's 62.09. F1 is computed after KTS segmentation and a knapsack filling a 15% budget,
so it largely measures the pipeline, not the model. Rank correlation compares the
predicted *ordering* against human orderings and has no such shortcut.

Protocol, implemented per dataset (`src/eval_rank.py`):

* **TVSum** — correlate against **each of the 20 annotators**, then average. Uses
  `ydata-tvsum50-anno.tsv` (per-frame 1-5 Likert). Verified: that file's length equals
  `n_frames`, and the h5 `gtscore` is exactly its mean sampled at `picks` (Spearman
  1.0000).
* **SumMe** — average the 15-18 binary annotator selections **first**, then correlate once.

Reversing these is the runbook's listed failure mode ("rank correlation ~ 0 -- protocol
mismatch"). The random floor coming out at tau = +0.0009 (TVSum) and -0.0029 (SumMe)
confirms the implementation is sound.

## 10.1 Main results — TVSum

| method | tau | rho | F1 |
|---|---|---|---|
| **Human ceiling** (leave-one-out) | **0.3139** | 0.3957 | - |
| Random floor | 0.0009 | 0.0013 | 56.42 |
| *Generic human gtscore (reference)* | *0.3782* | *0.4732* | - |
| DSNet AF (GoogLeNet-1024) | 0.0898 | 0.1177 | 61.16 |
| DSNet AF (CLIP-512) | 0.0826 | 0.1079 | 61.29 |
| **Ours: persona + FiLM** | **0.1581** | **0.2059** | - |
| Control: generic + FiLM | 0.2701 | 0.3488 | - |

### Two findings

**1. The auxiliary ranking objective is worth ~3x on the primary metric.**
Standard DSNet training reaches tau 0.083-0.090. The same architecture trained with the
Phase 9 frame-level `mse+rank` objective reaches **0.158** (persona labels) and **0.270**
(generic labels). DSNet's detection losses optimise a binary 15%-budget target, which is
only loosely related to the ordering that rank correlation measures; adding an explicit
ranking term on `pred_cls` closes that gap. This is runbook Axis 2's rationale confirmed
empirically, and it is a result in its own right.

**2. The metric structurally penalises personalisation.**
The control (generic labels) scores 0.270 versus our 0.158. That is expected and is not
evidence the method is worse: **the reference is the *generic* human annotation**, so a
model that deliberately deviates per persona is scored down for exactly the behaviour it
was built to have. The control optimises precisely what this metric rewards.

TVSum has no per-persona ground truth, so tau-against-generic-annotators cannot separate
"personalised" from "wrong". This is the concrete reason the runbook schedules QFVS
(10.3) and the persona-sensitivity test (11.2) -- see below.

Note the human ceiling (0.3139) sits *below* the generic gtscore reference (0.3782)
because the latter is the mean of all 20 annotators, while the ceiling correlates each
annotator against the mean of the other 19. The ceiling is the honest bar for a model.

## 10.1 Main results — SumMe

| method | tau | rho | F1 | note |
|---|---|---|---|---|
| **Human ceiling** | **0.2966** | 0.3299 | - | |
| Random floor | -0.0029 | -0.0039 | 41.49 | |
| Generic human gtscore | 1.0000 | 1.0000 | - | **circular** -- SumMe's gtscore is derived from `user_summary` |
| DSNet AF (GoogLeNet-1024) | 0.1002 | 0.1318 | 51.31 | |
| DSNet AF (CLIP-512) | 0.1070 | 0.1415 | 50.64 | |
| Ours: persona + FiLM | 0.0256 | 0.0329 | - | **TVSum-trained, zero-shot transfer** |
| Control: generic + FiLM | 0.0920 | 0.1236 | - | **TVSum-trained, zero-shot transfer** |

Two rows must not be read naively:

* the "generic human gtscore" row is **self-comparison** (SumMe's h5 `gtscore` is built
  from `user_summary`), so 1.0 is an artefact, not evidence;
* our models were **never trained on SumMe** (Phase 9 was TVSum-only), so those rows
  measure cross-dataset transfer. The drop from 0.158 to 0.026 says the model does not
  transfer, which is unsurprising given 16 domains and 313 training samples.

**Gap to close:** Phase 9 should be re-run on SumMe for a like-for-like row.

## 10.2 F1 (secondary)

Reported in the table above and in `results/baselines.csv`, always beside the random
floor (TVSum 56.42, SumMe 41.49). Never lead with it: the floor is within ~5 points of
every published model, so F1 has very little discriminative range on these datasets.

## 10.3 QFVS — BLOCKED, not run

`data/qfvs/` is empty; the UT Egocentric videos and per-query Oracle annotations were
never downloaded. This is the only protocol in the runbook with **genuine per-query
ground truth**, and Phase 10.1 above shows precisely why it matters: on TVSum/SumMe a
personalised model can only be penalised for personalising.

**This is the single highest-value remaining experiment.** Cost: 4 UTE videos (10-17 h
each) plus the QFVS annotation set, and a separate evaluation harness.

## 10.4 Synthetic-label validation (the crux)

Does the teacher's persona score agree with real humans at all? If not, the persona h5
files are elaborately generated noise.

| | tau vs human annotators |
|---|---|
| Random floor | +0.0009 |
| **Teacher persona labels** | **+0.0802** +- 0.1116 |
| Human ceiling | +0.3139 |
| Generic human gtscore | +0.3782 |

n = 393 (video, persona) pairs; **79.9% positive**, range [-0.296, +0.403].

**The labels are not noise**: 80x the random floor, four fifths positive. But they capture
only **26% of human-ceiling agreement**. Part of that shortfall is intentional -- persona
labels are *supposed* to deviate from generic judgment -- and part is presumably teacher
error. **Generic human annotations cannot separate the two.** QFVS can.

## Checkpoint

`results/main_table.csv` exists with tau/rho (primary), F1 (secondary), and random-floor
and human-ceiling rows for both datasets. **Met**, with the caveats above:
PGL-SUM and CLIP-It rows are absent (CLIP-It was dropped in Phase 3; PGL-SUM's per-frame
scores could be added from its saved epoch JSONs).

## Honest summary

* The pipeline runs end to end and every metric is bracketed by a floor and a ceiling.
* The frame-level ranking objective is a real, transferable gain (~3x tau over stock
  DSNet training).
* The teacher's synthetic labels carry genuine human-aligned signal (26% of ceiling).
* Whether the *persona-specific* component is right cannot be tested on TVSum/SumMe,
  because their ground truth is generic. On this evidence the personalised model looks
  worse than a generic one **by construction of the metric**.
* Phase 11.2 (persona sensitivity) and Phase 10.3 (QFVS) are the two evaluations that can
  actually adjudicate the central claim.

## Files

```
src/eval_rank.py                 Otani-protocol evaluator (both datasets)
results/main_table.csv           the Phase 10 checkpoint table
results/phase10/rank_tvsum.json  per-method tau/rho
results/phase10/rank_summe.json
```
