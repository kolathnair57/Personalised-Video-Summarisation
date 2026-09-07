# Phase 7 — Teacher scoring and persona h5 files

Date: 2026-08-12. Teacher `Qwen3-VL-8B-Instruct` (bf16) via the Phase 5 server.

This is the core of the method: (video, persona) -> a `gtscore` vector.

## Output

```
labels_cache/*.json              6,576 cached chunk calls (the sacred cache)
labels_cache/scores/<ds>/<pid>/<video>.json   551 per-pair shot-score vectors
data/persona_h5/<ds>/<domain>/<persona_id>.h5 118 persona datasets, 721 MB
/var/tmp/akn57/data/shot_frames/  3,117 cached shot frames (84 MB)
```

547 (video, persona) `gtscore` vectors across 118 persona files, built on the **CLIP**
h5 (Phase 8 conditions on CLIP text embeddings, so the features must be CLIP).

## Scale and cost

| | |
|---|---|
| (video, persona) pairs attempted | 600 |
| pairs scored | **551** (48 rejected, see below) |
| chunk calls (k=3 self-consistency) | **6,576** |
| teacher throughput | 3.47 calls/s at concurrency 6; ~10 workers to saturate |
| wall time | ~75 min scoring + 6 min frame extraction |

**Frames are decoded once per video, not per pair.** The persona changes the prompt, not
the pixels; decoding inside the scoring loop would have meant ~600 decodes (~3 h) instead
of 75 (~6 min).

## The chunking problem (not addressed by the runbook)

The server allows 16 images per request; videos have up to **130 shots** (TVSum median 38,
max 130). The runbook says "chunk the shots and stitch". Naive stitching is wrong:

> the model scores each chunk **relative to that chunk**, and the Phase 5 discrimination
> prompt — which is required to avoid constant vectors — forces every chunk to span the
> full 0–1 range.

Measured on a real 32-shot video, the three chunks each spanned ~0.1–0.9 while their true
levels differed nearly 3x:

| chunk | local range | mean |
|---|---|---|
| shots 0–15 | 0.10–0.92 | 0.49 |
| shots 12–27 | 0.10–0.83 | 0.41 |
| shots 24–31 | 0.07–0.90 | **0.18** |

Concatenating would rank the last chunk's local 0.90 alongside the first chunk's 0.92 —
corrupting the **global ordering**, which is exactly what the primary metric (rank
correlation) measures.

**Fix: overlapping chunks with anchor alignment.** Windows of 16 shots with 4-shot
overlap; each chunk is mapped onto the previous chunk's scale by a least-squares linear
fit `a*x + b` over the shots they share, with a mean-matching fallback when the fit is
degenerate. Overlapping shots average the two estimates. Implemented in
`align_and_merge()` in `src/score_teacher.py`.

Note this makes absolute values comparable *within* a video. DSNet normalises `gtscore`
per video at load time (`data_helper.py:33-34`), so only the ordering matters downstream —
which is precisely what alignment protects.

## 48 rejected pairs (8%) — the guard working

`assert std > 0` rejects any constant score vector: it carries **zero learning signal**
and makes rank correlation undefined. 48 pairs were dropped, and they cluster by content,
not by shot count:

| video | shots | degenerate personas |
|---|---|---|
| Scuba | 15 | 8 |
| Bearpark_climbing | 23 | 7 |
| Excavators river crossing | 65 | 7 |
| Saving dolphines | 45 | 6 |
| Fire Domino | 11 | 6 |

Not a shot-count effect (video_12 has 7 shots and zero rejects). These are **continuous
single-scene home videos** where every shot genuinely looks alike, so the teacher cannot
discriminate. SumMe is unedited home footage; TVSum is edited YouTube content with real
scene changes — which is why rejects concentrate in SumMe.

Consequence: some personas cover fewer videos than their domain contains. Two
`summe:animals_nature` personas were dropped entirely (that domain has only **one** video,
already flagged in Phase 6).

## Does it actually personalise? (honest answer)

Pairwise Spearman between two personas' `gtscore` for the **same video**, 1,857 pairs:

| | |
|---|---|
| median | **0.64** |
| p10 / p90 | -0.12 / 0.95 |
| pairs with r < 0.3 (clearly different) | **27.3%** |
| pairs with r < 0 (opposed) | 15.3% |
| pairs with r >= 0.90 (nearly the same) | **20.5%** |
| pairs with r >= 0.999 (identical) | 3.7% (45 pairs) |

**Read this honestly: personalisation is real but not uniform.** A quarter of persona
pairs rank a video's shots very differently, and 15% are actively opposed — that is real
signal for Phase 9. But a fifth of pairs are near-identical, meaning the teacher collapses
two distinct personas onto one ranking for a meaningful minority of cases. Phase 11's
persona-sensitivity test should quantify whether divergence tracks persona semantic
distance, and this table is the baseline it must beat.

Against the **generic** human `gtscore` (n=547): mean Spearman **+0.147**, sd 0.277,
range [-0.67, +0.78]; **65.6%** of persona labels have |r| < 0.3. So the persona labels
are largely independent of the generic label rather than a relabelling of it — which is
the point of the method. Mildly positive on average, as expected: some shots are
uninteresting to everyone.

Within-vector spread: `gtscore` std mean 0.266, min 0.064 — no near-constant vectors
survived.

## Implementation decisions

- **Domain pairing**: a persona only scores videos of its own domain (Phase 6, gap G2).
- **`response_format`**, not `guided_json` (silently ignored on vLLM 0.19.1).
- **Flat numeric array with `minItems`/`maxItems`** — prevents empty arrays, ~15x cheaper
  in tokens than array-of-objects.
- **k=3 self-consistency**, averaged per chunk (runbook's first-listed quality lever).
- **Cache key** = md5(dataset, video, persona, model, PROMPT_VERSION, chunk span, repeat).
  Bump `PROMPT_VERSION` to invalidate. Corrupt entries are deleted and recomputed.
- **Persona files contain only their domain's videos** — 721 MB instead of ~11 GB, which
  matters given ~19 GB of home quota.

## Reproduce

```bash
python src/extract_shot_frames.py --dataset tvsum summe        # ~6 min, cached
source /var/tmp/akn57/envs/pvs-vllm/bin/activate                # server must be up
python src/score_teacher.py --k 3 --workers 10                  # ~75 min, resumable
source env/activate-train.sh
python src/build_persona_h5.py                                  # ~2 min
```

Re-running costs nothing: every chunk call is cached and every completed pair is skipped.

## For Phase 8/9

- Persona files are keyed `data/persona_h5/<dataset>/<domain>/<persona_id>.h5`; each
  group carries `query`, `persona_id` and `domain` attrs, and the file carries them too.
- Features are **512-d CLIP** -> DSNet must run with `--num-feature 512`.
- 547 (video, persona) training vectors is a modest dataset; expect the query head to
  need regularisation.
- The 20.5% of near-identical persona pairs is the ceiling on how much a query-conditioned
  model can differentiate. Worth stating in the dissertation as a limitation of the
  teacher, separable from the student's ability to learn.
