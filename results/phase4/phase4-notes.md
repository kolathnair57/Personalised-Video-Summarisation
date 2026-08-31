# Phase 4 — CLIP feature re-extraction

Date: 2026-08-12. Model: `open_clip ViT-B-32 / laion2b_s34b_b79k`, 512-d.

## Why

Phase 8 conditions DSNet on a persona query encoded by CLIP's **text** encoder. CLIP is
trained so image and text embeddings share one space, so a query and a frame become
directly comparable. GoogLeNet pool5 has no text counterpart, so there would be nothing
to compare the query against.

Only `features` changes (1024-d GoogLeNet -> 512-d CLIP). `gtscore`, `change_points`,
`picks`, `n_frames`, `n_steps`, `user_summary`, `gtsummary` and `n_frame_per_seg` are
copied through unchanged, so KTS, knapsack selection and the evaluation harness are
untouched.

## Does switching features cost accuracy?

Same splits, same harness as Phase 2 — the **only** variable is the features.

| Variant | Dataset | CLIP-512 | GoogLeNet-1024 | Delta |
|---|---|---|---|---|
| anchor-based | TVSum | 61.16 | 62.09 | -0.93 |
| anchor-based | SumMe | 49.86 | 47.23 | **+2.63** |
| anchor-free | TVSum | 61.29 | 61.16 | +0.13 |
| anchor-free | SumMe | 50.64 | 51.31 | -0.67 |

**Conclusion: CLIP features are equivalent to GoogLeNet for this task** — mean absolute
change ~1.1 F1, in both directions, well inside the split-to-split std (2-4 points).
This matters for the thesis: the move to CLIP is required for query-conditioning and it
is **not** bought at the cost of feature quality, so Phase 9 gains cannot be dismissed
as an artefact of changing the backbone. Both remain ~5 points above the random floor
(TVSum 56.42 / SumMe 41.49), the same margin as GoogLeNet.

**Caveat — ignore the `diversity` column when comparing across features.** It jumps from
~0.47 (GoogLeNet) to ~75 (CLIP) simply because it is computed from raw feature vectors
and CLIP embeddings have a much larger norm (L2 ~11). It is not comparable across
feature spaces. F1 is unaffected.

## The mapping problem (the real risk of this phase)

`picks` index the **original video**, and the h5 gives no filename for TVSum
(`video_name` exists for SumMe only). Pairing the wrong mp4 with the right `gtscore` is
**silent**: training runs, loss falls, every number is wrong. The previous attempt at
this project left a file named `segment_captions_MISMAPPED.json.bak`.

Mapping is therefore derived from evidence and verified before any encoding
(`src/map_videos.py`, kept deliberately separate from extraction):

- **SumMe** — match `video_name` to filename, then confirm exact frame count.
- **TVSum** — no names stored, so match on `n_frames` from `ffprobe -count_frames`
  (decoded count, not the unreliable container header).
- Hard check for every video: `max(picks) < decoded_frame_count`.

Result: **TVSum 50/50, SumMe 25/25.**

### Three problems it caught

1. **SumMe ships every video twice** (`.mp4` and `.webm`, identical stems). Name matching
   alone picks one arbitrarily. Now the candidate whose decoded frame count equals the
   h5's is chosen — `video_25` legitimately resolves to `playing_ball.webm`.
2. **Two TVSum videos had no exact match**: `video_16` -> `WG0MBPpPC6I.mp4` and
   `video_39` -> `Se3oxnaPsz0.mp4`, each **+1 frame** (2019-vs-2026 decoder difference).
   The alternative pairing was off by ~5,000 frames, so unambiguous. Handled by a
   documented `+-2` tolerance pass that still refuses genuinely ambiguous cases.
3. **The verification itself was wrong** (see below).

## Verification: the rank test

`src/verify_clip_h5.py` checks schema preservation, shape/sanity, and mapping.

The mapping check exploits the fact that the base GoogLeNet features were computed from
the *correct* video at the *same* picks. So the frame-to-frame cosine similarity profile
of the CLIP features should track the GoogLeNet one.

**First version used an absolute correlation threshold, and it was wrong.** It flagged
SumMe `video_1` (Air_Force_One) at 0.246. Investigation: that video is nearly static —
its GoogLeNet profile has mean 0.975 and std 0.015, so consecutive frames are ~97.5%
identical and there is almost no temporal variation to correlate. The coefficient is
small even though the mapping is correct.

Replaced with a **rank test**, which is scale-free: does this CLIP profile match its
*own* base profile better than every other video's? For `video_1` the answer was yes —
rank #1 of 25 (0.246 vs next best 0.180) — and `video_name` plus an exact frame-count
match agreed independently.

Final result:

| | TVSum | SumMe |
|---|---|---|
| videos | 50 | 25 |
| feature dim | 512 | 512 |
| non-feature datasets bit-identical | yes | yes |
| **rank-#1 matches** | **50/50** | **25/25** |
| mean profile corr | 0.786 | 0.599 |
| control (deliberately wrong pairing) | +0.005 | +0.055 |

## Decisions

- **Sequential decode (PyAV), never frame seeking.** OpenCV's `CAP_PROP_POS_FRAMES` can
  silently return a neighbouring frame on some codecs, misaligning every feature with its
  label while looking normal. Sequential decode is slower and exact.
- **Copy-then-replace**, so non-feature datasets are provably untouched.
- **Resumable** — already-extracted videos are skipped.
- Each video group records `clip_model` and `source_video` attrs for provenance.

## Files

```
data/clip_h5/tvsum_clip.h5     120 MB, 50 videos, features (n_steps, 512)
data/clip_h5/summe_clip.h5      36 MB, 25 videos
data/video_map.json            verified h5 key -> mp4 mapping (+ match type)
src/map_videos.py              builds & verifies the mapping (run FIRST)
src/extract_clip.py            extraction
src/verify_clip_h5.py          schema + sanity + rank test
third_party/DSNet/splits/{tvsum,summe}_clip.yml
models/{ab_clip,af_clip}/      trained on CLIP features
results/phase4/*.log
```

Raw videos live at `/var/tmp/akn57/data/raw/` (not redistributed, not in git):
TVSum `tvsum50_ver_1_1.tgz` (641 MB) from people.csail.mit.edu/yalesong/tvsum,
SumMe `SumMe.zip` (2.4 GB) from data.vision.ee.ethz.ch/cvl/SumMe.

## Reproduce

```bash
source env/activate-train.sh
python src/map_videos.py --tvsum-dir /var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/video \
                         --summe-dir /var/tmp/akn57/data/raw/videos --out data/video_map.json
python src/extract_clip.py --dataset tvsum        # ~15 min
python src/extract_clip.py --dataset summe        # ~14 min
python src/verify_clip_h5.py --dataset tvsum summe
cd third_party/DSNet/src
python train.py anchor-based --model-dir ../../../models/ab_clip --num-feature 512 \
    --splits ../splits/tvsum_clip.yml ../splits/summe_clip.yml
```

## For later phases

- **`--num-feature 512` is required** on every DSNet run using the CLIP h5.
- `ydata-tvsum50-info.tsv` (from the TVSum download, at
  `/var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/data/`) gives each video's **category and
  title** — directly usable for the Phase 6 domain-conditioned persona seeds (gap G2),
  avoiding hand-labelling 50 videos.
- The `diversity` metric is not comparable across feature spaces (see caveat above).
