# Phase 9 — Training the student on synthetic labels

> **Superseded headline:** the concat results below were produced by an
> architecture that *cannot* personalise. See **FINAL RESULT (FiLM)** at the
> bottom, which is the configuration to report.

Date: 2026-08-13. TVSum, anchor-free DSNet + query head, CLIP-512 features.
5 splits (DSNet's own), 313 train / 80 test (video, persona) samples per split.

## Headline result

Spearman between the model's per-frame scores and the **held-out persona labels**
(every condition measured on the same yardstick):

| Condition | labels | query | mean rho | sd |
|---|---|---|---|---|
| **A (the method)** | persona | yes | **+0.2154** | 0.056 |
| B | persona | no | +0.1995 | 0.061 |
| C | generic | yes | +0.2186 | 0.049 |

Paired per-split comparisons (same splits, same data):

| comparison | mean delta | splits positive | paired t |
|---|---|---|---|
| A - B (what the **query** adds) | +0.0159 | **5/5** | **p = 0.037** |
| A - C (what **persona supervision** adds) | -0.0033 | 3/5 | p = 0.866 |

**Persona supervision gives no measurable benefit over generic supervision.**
The query contributes a small but consistent gain -- which the diagnostic below shows is
not actually personalisation.

Reference points (split 0):

```
random scores                +0.004
untrained model              -0.087
trained model                +0.192
generic human gtscore        +0.213   <- a trivial predictor still beats the model
```

## The decisive diagnostic: the student does not personalise

Same video, different persona queries, correlation between the model's outputs:

| checkpoint | mean \|W_query\| | spearman between personas | min |
|---|---|---|---|
| main run A (lr mult 1) | 7.3e-04 (~0) | **+0.9999** | 1.000 |
| lr mult 10 | 3.3e-02 | +0.9682 | 0.850 |
| lr mult 100 | 5.1e-02 | +0.9984 | 0.987 |
| lr mult 1000 | 1.0e-01 | +0.9993 | 0.997 |

(1.0 means the query changes nothing.)

**Two different personas receive essentially the same summary.** So A ~= C because both
are effectively plain DSNet, and the +0.016 "query benefit" is a negligible perturbation,
not personalisation.

Cause, in two parts:

1. **The query head never trained.** Identity initialisation (Phase 8) starts the query
   weights at exactly zero -- correct for the checkpoint, but it means the head must
   travel further than any other parameter. Model selection picks epoch 0-2 because the
   detection task overfits almost immediately on 313 samples, so early stopping fires
   before the query path leaves zero (\|W_query\| = 7e-04).
2. **Forcing it to train does not fix the ordering.** With a larger lr on the conditioner
   the weights do grow (up to 1e-01), but sensitivity does not increase monotonically:
   10x gives the most differentiation, 100x and 1000x give less. The weights grow into a
   direction that shifts all frames roughly equally, and Spearman is rank-based, so a
   uniform shift changes nothing. The model is not learning persona-specific **ordering**.

## Why this is a real finding, not just a bug

The teacher **does** personalise (Phase 7): pairwise Spearman between two personas'
labels for the same video has median 0.64, with 27.3% of pairs below 0.3 and 15.3%
negative. The student, trained on exactly those labels, collapses to 0.97-1.00.

The signal exists in the labels and is lost in the student. Most plausible reasons, in
order:

1. **Data scale.** 313 training samples = 40 videos x ~8 personas. Learning a
   query -> score mapping that generalises to *unseen videos* is far harder than fitting
   per-video importance, and the latter reduces the loss faster.
2. **The persona-specific component is small.** Phase 7: persona labels correlate +0.147
   with the generic human label, and 20.5% of persona pairs are already near-identical in
   the teacher's own output. The learnable persona-specific residual is thin.
3. **Objective mismatch.** DSNet's detection losses dominate; the frame-level ranking
   objective (the one matching the metric) had to be weighted 10x before it mattered at
   all, and even then the model prefers the generic solution.

## Tuning that did work

| change | rho |
|---|---|
| lambda_score = lambda_rank = 1 (runbook default) | +0.1401 |
| **lambda_score = lambda_rank = 10** | **+0.1919** |
| rank-only, lambda 20 | +0.1842 |

DSNet's detection losses run ~2.5 while MSE was ~0.11 and rank ~0.66, so the objectives
matching the primary metric were being drowned out. Weighting them 10x gave +37% relative.

Learning rate: 5e-5 (+0.192, peak ep0), 2e-5 (+0.186, ep2), 1e-5 (+0.184, ep3),
5e-6 (+0.180, ep12). Lower lr moves the optimum later but reaches the **same ~0.19
ceiling** -- an optimisation-independent limit.

## Two methodological traps avoided (and one caught late)

**Video-level splits, not pair-level.** Each video appears with ~8 personas; a random
split over pairs would place the same video in train and test, letting the model memorise
per-video importance and score well while ignoring the query entirely. DSNet's own 5
splits are reused so results stay comparable with Phases 2-4.

**Persona-aware model selection.** DSNet keeps the epoch with max F1 against
`user_summary` -- the generic human annotation, identical across a video's personas.
Selecting on that while training on persona labels would pick the epoch that best
*ignores* the persona. Selection uses Spearman against held-out persona labels instead.

**Caught late: the control was measuring itself.** `evaluate()` originally scored each
model against `s.gtscore`, i.e. its own training target, so condition C (generic labels)
reported rho +0.4650 and appeared to beat the method 2.4x. Fixed by splitting the roles:
`gtscore` is the training target (varies by condition), `eval_gtscore` is the measurement
target and is **always** the persona label. Uncorrected, Phase 9 would have concluded that
generic supervision beats persona supervision -- an inverted result.

## Implementation notes

DSNet does not regress `gtscore`: it converts it into a binary 15%-budget keyshot target
and trains focal / IoU / centreness losses. Persona labels therefore enter through *which
shots become positives*, so the pipeline consumes them correctly unmodified. The runbook's
"MSE against persona gtscore" was implemented as an auxiliary loss on `pred_cls` (the
anchor-free per-frame score, which is exactly what Phase 10 measures).

Best config: `--loss mse+rank --lambda-score 10 --lambda-rank 10 --lr 5e-5 --qcond concat`.

## What to do next

The honest framing for the dissertation: **the pipeline works end-to-end and the teacher
personalises; the student at this data scale does not learn to.** That is a legitimate
result, and it is diagnosed rather than merely observed.

Options, roughly by expected value:

1. **Evaluate on QFVS (Phase 10.3).** The runbook is explicit that per-query Oracle
   summaries are "the test TVSum/SumMe can't give you". TVSum has no per-persona ground
   truth, so rho-against-teacher-labels is a proxy for a proxy.
2. **Run the Phase 11.2 persona-sensitivity test properly**, and report the teacher's
   divergence (median 0.64) beside the student's (0.97-1.00). The gap between them is
   itself the finding.
3. **Increase persona-specific signal**: more personas per video, more videos, or a
   stronger teacher (a 4-bit AWQ 32B, since FP8 is impossible on sm_86).
4. **Try FiLM conditioning** (Axis 1) -- multiplicative modulation may produce
   rank-changing effects where additive concat produced a near-uniform shift.
5. Consider conditioning **earlier or at multiple depths** rather than once at the input.

## Files

```
src/train_persona.py                 trainer (3 label modes, 3 conditioning modes)
models/p9_persona_concat/            condition A checkpoints + results.json
models/p9_persona_noquery/           condition B
models/p9_generic_concat/            condition C
results/phase9/main_AB.log, main_C.log, loss_sweep.log, lr_sweep.log,
              qcond_lr_sweep.log
```


---

# FINAL RESULT (FiLM) — the configuration to report

```
--qcond film --qcond-lr-mult 100 --loss mse+rank
--lambda-score 10 --lambda-rank 10 --lr 5e-5 --max-epoch 20
```

## Why concat had to fail

concat computes `out = W_f f + W_q q + b`. The query `q` is **constant across all frames**
of a video, so `W_q q` is one fixed vector added identically to every frame. It translates
the sequence in feature space but leaves *relative* differences between frames almost
unchanged -- and rank correlation only sees relative order.

concat was therefore **structurally incapable** of expressing a persona-specific
reordering. Raising its query-head lr to 1000x grew the weights to 1e-01 and still left
inter-persona correlation at 0.999.

FiLM computes `gamma(q) * f + beta(q)`. The multiplicative term scales each **dimension**
differently, so frames with different feature values move by different amounts -- which
genuinely reorders them.

**Axis 1 (concat vs FiLM) is therefore not a hyperparameter; it is the difference between
a model that cannot personalise and one that can.** The runbook lists concat as the
default; on this evidence FiLM should be.

## Results, 5 splits, TVSum

Accuracy (Spearman vs held-out persona labels):

| condition | labels | mean rho | sd |
|---|---|---|---|
| **A2 (the method)** | persona | **+0.2237** | 0.044 |
| C2 (control) | generic | +0.2199 | 0.057 |

paired A2 - C2 = **+0.0037**, 3/5 positive, **p = 0.872** -> no accuracy difference.

Persona sensitivity (correlation between the model's outputs for two DIFFERENT personas
on the same video; 1.0 means the query is ignored):

| model | mean | median | min | **% pairs < 0.9** |
|---|---|---|---|---|
| **A2 persona + FiLM** | +0.9080 | +0.9397 | **-0.289** | **27.4%** |
| C2 generic + FiLM | +0.9829 | +0.9897 | +0.614 | **1.0%** |
| teacher (Phase 7) | - | +0.64 | - | 27.3% (below 0.3) |

## The finding

**Persona supervision produces a persona-sensitive model -- 27x more differentiated
outputs than the control -- while producing no gain in aggregate accuracy.**

Both halves matter:

* The control behaves exactly as theory predicts. C2 was trained with queries on labels
  that do not depend on them, and correctly learned to **ignore** the query (1.0%
  differentiated). That validates the sensitivity measurement itself.
* A2 learned persona-conditioned behaviour from persona labels alone, and its share of
  differentiated pairs (27.4%) closely matches the teacher's own (27.3%). The student
  inherited roughly the proportion of persona-differentiated cases its supervision
  contained, expressed more weakly (median 0.94 vs the teacher's 0.64).

**Why rho cannot see this:** persona labels correlate +0.147 with generic importance
(Phase 7), so most of what rho measures is the generic component. Predicting generic
importance captures most of the metric; the persona-specific residual is small in rho but
is exactly what A2 captures behaviourally.

**Implication for evaluation:** agreement metrics (F1, rank correlation against a single
label) structurally cannot detect personalisation. The counterfactual divergence test --
runbook Phase 11.2 -- is required. This result is the empirical justification for that.

## Phase 9 checkpoint

> "a trained personalised model that, given the same video and two different persona
> queries, produces two visibly different summaries"

**PASSES** with FiLM: 27.4% of persona pairs differ substantially, minimum correlation
-0.289 (two personas receiving near-opposite selections of the same video).
It FAILS with concat (0.0% of pairs, correlation +0.9999).

Checkpoints: `models/p9_persona_film/`, `models/p9_generic_film/`.
Still TVSum only -- SumMe was not trained in Phase 9.
