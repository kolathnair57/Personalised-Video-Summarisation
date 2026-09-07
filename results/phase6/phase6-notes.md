# Phase 6 — Domain-conditioned persona pool

Date: 2026-08-12. Generator: `Qwen3-VL-8B-Instruct` via the Phase 5 server, temperature 0.9.

## Output

```
personas/seeds.json   16 domains + attribute axes + video->domain map for all 75 videos
personas/pool.json    128 personas (8 per domain), each with persona_id, biography,
                      occupation, preference_query, attributes, domain, dataset
```

## Domains (gap G2)

A single global persona distribution is wrong, and Phase 5 proved it: a pastry-chef
persona rating a tyre-changing tutorial returned a **constant all-zero** score vector —
zero learning signal, undefined rank correlation. So personas are generated per domain
and Phase 7 pairs a persona only with videos of its own domain.

- **TVSum** — its own 10 native categories, 5 videos each, read from
  `ydata-tvsum50-info.tsv` and joined to h5 keys via the mp4 stem (== TVSum `video_id`):
  VT tyre repair, VU vehicles stuck, BK beekeeping, BT bike/motorcycle tricks,
  DS dog shows, GA animal grooming, MS making sandwiches, PK parkour, PR parades,
  FM flash mobs.
- **SumMe** — no category field, so its 25 videos were assigned explicitly from their
  official titles into 6 domains: action_sports (10), aviation (4), vehicles (4),
  landmarks_travel (3), everyday_life (3), animals_nature (1). Hand-assigned but
  version-controlled in `src/build_seeds.py` and auditable.

`animals_nature` has only **one** video (Saving dolphines) — too thin to support a
per-domain claim on its own; treat it as merged or excluded in any SumMe-domain analysis.

## Sampling: stride, not random

Random sampling of attribute combinations clumps and leaves holes. Each attribute axis is
instead walked with a stride **co-prime with the axis length**, offset per domain, which
spreads N personas evenly and is fully deterministic (so the pool regenerates identically).

Axes: `age_band` (5), `attention_budget` (4), `expertise` (4), `watching_context` (5),
`gender_presentation` (3), plus per-domain `viewing_goal` and `interest_focus`.
1,200 global combinations before the domain-specific axes.

Temperature contrast worth remembering:

| | temperature | why |
|---|---|---|
| persona generation (Phase 6) | **0.9** | diversity is the goal |
| shot scoring (Phase 7) | **0.2** | reproducible labels; variance is noise |

## The audit found three real defects

A pool can look perfect persona-by-persona and still be useless, so diversity was
**measured**, not assumed (`src/audit_personas.py`).

**1. A stride bug collapsed an axis.** `watching_context` used only 3 of 5 values, and
was *constant within any single domain*. Cause: stride 5 on a 5-element list gives
`(i*5) % 5 == 0` for all i. `attention_budget` (4 values, stride 2) had the same defect.
Fixed by selecting a stride co-prime with each axis length. All axes now reach full
coverage in 8 draws.

**2. Severe gender skew: 84% she/her vs 9% he/him** when the model was free to choose.
Fixed by sampling `gender_presentation` explicitly as an audited attribute — now
34% / 37% / 29% (he / she / they). This controls the distribution rather than leaving it
to the model's defaults, and does not affect what a persona wants to *see*.

**3. Occupation repetition** — "graphic designer" x20, only 56% unique. Now 79% unique
(101 distinct across 128), largest repeat "barista" x9.

## Final pool quality

| check | result |
|---|---|
| CLIP text length (limit **77**, truncates silently) | mean 31.9, **max 48** — all 128 fit |
| pairwise cosine similarity (CLIP space) | mean 0.352, p95 0.580, max 0.849 |
| near-duplicates (>= 0.95) | **0** |
| worst within-domain mean similarity | 0.618 (tvsum:VT) |
| best within-domain | 0.376 (summe:everyday_life) |
| age bands | 5 distinct, 22-32 each |
| expertise / attention budget | perfectly balanced, 32 each |
| occupations | 101 distinct / 128 (79%) |

The CLIP-token check is not in the runbook but matters: Phase 8 encodes
`preference_query` with CLIP's **text** encoder, which truncates at 77 tokens **without
error**. Over-long queries would silently lose information and could collapse two
personas to the same embedding. Max is 48, so there is comfortable headroom — but this
must be re-checked if the query wording is ever expanded.

Similarity is measured in **CLIP embedding space**, not by string overlap, because that
is the space Phase 8 actually conditions on.

## Reproduce

```bash
python src/build_seeds.py                                    # seeds.json (deterministic)
source /var/tmp/akn57/envs/pvs-vllm/bin/activate             # needs the Phase 5 server up
python src/gen_personas.py --n-per-domain 8 --out personas/pool.json
source env/activate-train.sh
python src/audit_personas.py --pool personas/pool.json       # needs open_clip => train env
```

Note the two-environment split: generation talks to the vLLM server, the audit needs
`open_clip`, so they run in different venvs.

## For Phase 7

- Pair a persona **only** with videos where `seeds.json:video_domain[dataset][key]`
  equals the persona's `domain`.
- Keep the Phase 5 discrimination system prompt and assert `scores.std() > 0` before
  writing any persona h5 — a constant vector is a silent data-quality failure.
- Workload if every persona scores every video in its domain:
  TVSum 10 domains x 8 personas x 5 videos = **400 (video, persona) pairs**;
  SumMe 6 domains x 8 personas x (its video counts) = **200 pairs**. Videos with more
  than 16 shots need chunking, so the real call count is several times that — hence the
  `labels_cache/` requirement.
