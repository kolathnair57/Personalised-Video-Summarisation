# Phase 11 — Ablations and the persona-sensitivity test

Date: 2026-08-13. TVSum, 5 splits, persona labels, lr 5e-5, lambda_score = lambda_rank = 10,
20 epochs. Each axis varies exactly one thing from that default.

Two metrics per row:

* **rho** — Spearman against held-out persona labels (accuracy).
* **persona sensitivity** — mean Spearman between the model's outputs for two DIFFERENT
  personas on the same video. **1.0 means the query is ignored**; lower means the model
  actually personalises. Accuracy alone cannot tell these apart.

## 11.1 Axis 1 — query mechanism (the decisive axis)

| variant | rho | sd | persona sens. | % pairs < 0.9 |
|---|---|---|---|---|
| **FiLM** (default) | **+0.2237** | 0.044 | **+0.9080** | **27.4%** |
| concat | +0.2036 | 0.059 | +0.9923 | 0.4% |
| none (no query at all) | +0.2025 | 0.061 | n/a | n/a |

**concat is functionally equivalent to having no query.** It scores +0.2036 against
no-query's +0.2025 — a difference of 0.001 — and changes its output for only 0.4% of
persona pairs.

This is structural, not a tuning failure. concat computes `W_f f + W_q q + b`; the query
`q` is **constant across all frames** of a video, so `W_q q` is a single fixed vector added
identically to every frame. It translates the sequence in feature space and leaves the
*relative* ordering essentially untouched — and rank metrics only see relative order.
FiLM's `gamma(q) * f` scales each dimension, so frames with different feature values move
by different amounts, which genuinely reorders them.

**The runbook specifies concat as the default (Axis 1, "start with concatenation").
On this evidence that default cannot personalise and FiLM should be the default.**

## 11.1 Axis 2 — loss (all with FiLM)

| variant | rho | sd | persona sens. | % pairs < 0.9 |
|---|---|---|---|---|
| **mse+rank** (default) | **+0.2237** | 0.044 | +0.9080 | 27.4% |
| rank only | +0.2209 | 0.052 | +0.9351 | 16.6% |
| mse only | +0.2117 | 0.039 | **+0.9151** | **32.2%** |
| none (DSNet losses only) | +0.1648 | 0.036 | +0.9499 | 10.6% |

The auxiliary frame-level objective is worth **+0.059 rho (+36% relative)** over DSNet's
detection losses alone. DSNet's losses optimise a binary 15%-budget target, which is only
loosely coupled to the ordering the primary metric measures.

Note the accuracy/sensitivity trade-off: `mse+rank` is the most accurate, but `mse` alone
is the most persona-sensitive (32.2% of pairs). If personalisation is the goal rather than
agreement with teacher labels, `mse` is arguably the better choice.

## 11.1 Axis 3 — teacher self-consistency (k=1 vs k=3)

Measured directly on **label quality** (Otani tau against the 20 human annotators), which
isolates the teacher without needing a training run. k=1 is recoverable for free from the
cached k=3 calls (repeat 0).

| | tau vs human annotators |
|---|---|
| k=1 | **+0.0823** +- 0.109 |
| k=3 | +0.0802 +- 0.111 |
| paired k=3 - k=1 | **-0.0021**, 50% positive, **p = 0.382** |

k=1 and k=3 labels agree at tau +0.799 — averaging changes the labels but not their human
alignment.

**Self-consistency produced no measurable benefit at 3x the cost** (6,576 teacher calls
versus 2,192). The runbook lists it as the quality lever to apply *first*; on this evidence
it should be dropped, and the ~4,400 saved calls spent on more personas or more videos.

## 11.1 — axes NOT run, with reasons

| axis | why not |
|---|---|
| teacher 8B vs 32B | FP8 is impossible on sm_86 (Phase 1); would need a 4-bit AWQ build |
| modality visual+ASR | no transcript pipeline exists in this project |
| cross-attention conditioning | not implemented |
| KL loss | not implemented |
| frames vs captions teacher input | not implemented |

## 11.2 Persona sensitivity — the counterfactual test

Phase 10 showed TVSum/SumMe ground truth is *generic*, so agreement metrics penalise a
model for personalising. This test needs no per-persona ground truth: fix the video, change
only the persona, measure whether the **summary** changes — using Jaccard over the actually
selected shot sets (the real KTS -> knapsack pipeline).

Crucially, changing is not enough. The claim requires that divergence track persona
**meaning**: similar personas -> similar summaries.

    persona distance   = 1 - cosine(query_emb_i, query_emb_j)   (CLIP text space)
    summary divergence = 1 - Jaccard(selected shots)

| condition | n | mean divergence | % pairs differ | corr(distance, divergence) | p |
|---|---|---|---|---|---|
| **TEACHER labels** (upper bound) | 1372 | **0.4811** | 92.3% | **+0.0222** | **0.410** |
| **Ours: persona+FiLM** | 1372 | 0.1451 | 57.0% | **+0.0898** | **0.001** |
| Control: generic+FiLM | 1372 | 0.1082 | 45.3% | +0.1564 | 0.000 |
| Ours + **shuffled** queries | 1372 | 0.1787 | 71.4% | +0.0333 | 0.218 |

### The central finding

**The teacher's labels diverge enormously but not semantically.** Divergence is 0.48 and
92% of persona pairs differ — yet the correlation with persona semantic distance is
+0.022, **not significant (p = 0.41)**. Qwen gives different personas different scores, but
semantically similar personas do **not** receive more similar scores.

The supervision is *differentiated* but not *semantically organised*. A student cannot
learn a distance -> divergence relationship that its labels do not contain. This is the
precise, upstream explanation for the weak Phase 9 and Phase 10 personalisation results,
and it locates the limitation in the **teacher**, not the student or the architecture.

Plausible cause: the Phase 7 scoring prompt rates shots for **one persona in isolation**.
Nothing in it encourages the model to place similar personas near each other in score
space. A prompt that scores several personas jointly, or an explicit consistency
constraint across personas, is the obvious next experiment.

### The positive signal

The shuffled-query condition (not in the runbook, added because a reviewer will ask
whether the model responds to persona *meaning* or merely to *some vector changing*):

* correct personas: r = +0.0898, **p = 0.001**
* shuffled personas: r = +0.0333, **p = 0.218**

The distance-divergence relationship exists **only when the model receives the matching
persona**. The model does respond to persona meaning — weakly, but measurably, and the
effect is not an artefact of perturbation.

### Two caveats stated rather than hidden

1. **The control correlates higher (+0.156) than the method (+0.090).** This is probably
   not in its favour: its divergence magnitude is lower (0.108 vs 0.145), and a model
   trained on query-independent labels plausibly learned a *smooth* function of the query
   embedding. Smooth responses correlate well with distance while carrying no persona
   information, so this metric partly rewards smoothness rather than correctness.
2. **The control diverges at all (45.3% of pairs)** despite its score-level correlation
   being 0.983. Jaccard over selected shots **amplifies** small score differences, because
   knapsack selection is discrete: a tiny score change can flip which shot is chosen.
   Score-level and summary-level sensitivity are different measurements and should be
   reported separately.

## 11.3 User study — not run

Requires 10-15 human participants; outside what can be executed here. The generated
material (persona description + resulting summary) is available from the persona h5 files
and trained checkpoints.

## Summary for the write-up

1. **Axis 1 is not a hyperparameter.** concat cannot personalise (0.4% of pairs); FiLM can
   (27.4%). The runbook's default is the wrong one.
2. **The auxiliary ranking objective is worth +36% rho** over stock DSNet training.
3. **Self-consistency (k=3) is not worth 3x the teacher cost.**
4. **The teacher's persona scores are differentiated but not semantically organised**
   (p = 0.41). This is the key limitation of the current method and the clearest direction
   for future work.
5. **The student does respond to persona meaning** (p = 0.001, and it vanishes under
   shuffled queries) — weakly, bounded by (4).

## Files

```
src/persona_sensitivity.py           the counterfactual test
results/phase11/sensitivity.json     11.2 results
results/phase11/axis1.log, axis2.log ablation training logs
models/ab_qcond_concat, ab_qcond_none, ab_loss_{none,mse,rank}
```
