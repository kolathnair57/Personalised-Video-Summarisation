# Phase 13 — Raw video -> personalised summary (not in the runbook)

Date: 2026-08-13. `src/infer_personalised.py`.

The runbook stops at evaluation; it never produces a usable system. This phase closes
that gap: an arbitrary `.mp4` plus a free-text preference in, a summary `.mp4` out.

```bash
python src/infer_personalised.py \
  --source video.mp4 \
  --query  "close-ups of hands working with tools, and the finished result" \
  --save   summary.mp4

# demonstrate personalisation: two preferences, same video, reports the overlap
python src/infer_personalised.py --source video.mp4 \
  --query "the athlete mid-jump and landing" \
  --query2 "people being interviewed and talking to camera" \
  --save demo.mp4
```

**The query is not restricted to the 128 generated personas.** CLIP maps arbitrary text
into the same space, so any sentence works; the model never sees a persona id at inference.

## What it does

```
raw mp4 -> sample every 15th frame -> CLIP-512 features
        -> KTS shot boundaries (no annotations needed)
        -> FiLM query conditioning + DSNet anchor-free
        -> knapsack at 15% budget -> summary mp4
```

DSNet's own `infer.py` already had this shape. Three changes were required: CLIP-512
features instead of GoogLeNet-1024, threading `query_emb` into `model.predict`, and
encoding the user's text with CLIP's text encoder. PyAV is used for both decode and
encode, so OpenCV is not a dependency.

## Working demo

Source: `cjibtmSLxQ4.mp4` (TVSum video_21, parkour) -- **in split 0's test set, unseen by
the model**. 647 s, 19,406 frames.

| query | shots | frames | duration |
|---|---|---|---|
| "the athlete mid-jump and landing, the body in motion during a vault" | 45/175 | 2,910 | 97.0 s |
| "people being interviewed and talking to camera, the city background" | 44/175 | 2,910 | 97.0 s |

**Jaccard overlap 0.679** — 9 shots unique to A, 8 unique to B, 36 shared. Both summaries
are exactly 15.0% of the source, as intended.

## Three real bugs found and fixed

**1. KTS on un-normalised CLIP features degenerates.** DSNet L2-normalises its GoogLeNet
descriptors before building the KTS kernel; our CLIP features are deliberately
un-normalised (to match training), which makes kernel values ~120x larger and swamps
`cpd_auto`'s complexity penalty. Result: 635 "shots" for 636 sampled frames -- one shot per
frame. Fixed by normalising **for the kernel only**, leaving the model input untouched.

**2. KTS `vmax` needed retuning for CLIP.** With `vmax=1.0` (DSNet's value) the same video
gave 22 shots against the dataset's 64, and the coarse selection was *identical for both
queries*. Measured sweep:

| vmax | shots found (dataset truth: 64) |
|---|---|
| 1.00 | 22 |
| **0.50** | **67** |
| <= 0.20 | 319 (maximum) |

`vmax=0.5` is now the default and reproduces the dataset's ~1 shot per 10 sampled frames.

**3. Non-contiguous frames inherit their source timestamps.** Kept frames are scattered
through the source, so writing them with original PTS produced a summary that reported
376 s for 1,425 frames (~59 s of content) and would play with long freezes. Clearing
`pts` instead makes the muxer reject non-monotonic DTS. Fixed by assigning `pts` from the
output frame index with an explicit `time_base`.

## How reliably does it personalise?

Two contrasting, domain-appropriate queries per video, on five **unseen** split-0 test
videos, using the dataset's own shot boundaries:

| video | domain | shots | A-only | B-only | Jaccard |
|---|---|---|---|---|---|
| video_16 | making sandwich | 64 | 3 | 4 | 0.682 |
| video_43 | bike tricks | 33 | 2 | 2 | 0.714 |
| video_21 | parkour | 130 | 10 | 10 | 0.500 |
| video_35 | flash mob | 30 | 1 | 1 | 0.818 |
| video_38 | beekeeping | 20 | 2 | 2 | 0.500 |

**5/5 produced different summaries, mean Jaccard 0.643.**

## Caveats to state in a demo

* **Shot boundaries matter more than expected.** The persona signal is strong enough to
  reorder scores but only marginally strong enough to flip a *discrete* selection. On
  video_16, the dataset's 64 shots gave Jaccard 0.682 while our KTS's 67 shots gave 1.000
  (identical) -- same count, different boundary placement. This is a direct consequence of
  the weak persona signal measured in Phase 11, and it means demo quality varies by video.
* **Generalisation is limited.** Trained on 40 TVSum videos across 16 domains. Expect
  sensible behaviour on similar content (how-to, sports, events) and unpredictable
  behaviour elsewhere. Phase 10 measured zero-shot SumMe transfer at tau 0.026 vs 0.158
  in-domain.
* **A demo is not evidence.** The results chapter rests on Phases 10-11. A compelling
  side-by-side is persuasive in a viva, but is a single sample.
* The checkpoint used is `models/p9_persona_film/split0.pt`, trained on split 0's 40
  training videos. For a deployed demo, train one model on all 50.

## Files

```
src/infer_personalised.py            the pipeline
results/phase13/demo_parkour_A.mp4   "athlete mid-jump and landing"
results/phase13/demo_parkour_B.mp4   "people being interviewed"
```
