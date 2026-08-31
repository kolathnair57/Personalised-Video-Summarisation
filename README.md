# Personalised Video Summarisation

Query-conditioned video summarisation trained on **synthetic persona supervision**: a
vision-language teacher (Qwen3-VL-8B) rescores every shot of a video from a generated
viewer's point of view, those scores replace `gtscore` in the standard eccv16 HDF5 file,
and a query-conditioned DSNet is trained on the result.

Full build log: **`results/report/PVS_project_log.pdf`** (45 pages).

## Status

Phases 0-11 and 13 complete. Not run: QFVS evaluation (10.3), user study (11.3).
Phase 9 trained on TVSum only.

## Quick start

```bash
source env/activate-train.sh     # DSNet / CLIP / evaluation
source env/activate-vllm.sh      # vLLM teacher (only needed for Phases 6-7)
```

Summarise any video for any free-text preference:

```bash
python src/infer_personalised.py \
    --source video.mp4 \
    --query  "close-ups of hands working with tools, and the finished result" \
    --save   summary.mp4
```

The query is arbitrary text -- CLIP maps it into the same space as the frame features, so
it is not limited to the 128 generated personas.

## Headline results (TVSum, 5 splits)

Rank correlation is the primary metric. A **random-score** model reaches 56.42 F1 against
published DSNet's 62.09, so F1 has almost no discriminative range on this benchmark.

| Method | tau | rho | F1 |
|---|---|---|---|
| Human ceiling (leave-one-out) | 0.3139 | 0.3957 | - |
| Random floor | 0.0009 | 0.0013 | 56.42 |
| DSNet AF (GoogLeNet-1024) | 0.0898 | 0.1177 | 61.16 |
| DSNet AF (CLIP-512) | 0.0826 | 0.1079 | 61.29 |
| **Ours (persona + FiLM)** | **0.1581** | **0.2059** | - |
| Control (generic labels + FiLM) | 0.2701 | 0.3488 | - |

Note the control "wins" because the reference is the **generic** human annotation: a model
that deviates per persona is scored down for the behaviour it was built to have. That is
why the counterfactual test below matters, and why QFVS is the outstanding experiment.

**Persona sensitivity** -- same video, different persona, does the summary change?

| Model | mean corr. between personas | % pairs differing |
|---|---|---|
| Ours (persona labels) | +0.9080 | **27.4%** |
| Control (generic labels) | +0.9829 | 1.0% |

Persona supervision produces a 27x more persona-sensitive model while giving no gain in
aggregate accuracy.

## Four findings worth knowing

1. **Concat conditioning cannot personalise.** Its query term is constant across frames,
   so it translates the feature sequence without reordering it. The runbook's default is
   concat; FiLM should be used (0.4% vs 27.4% of pairs differing).
2. **An auxiliary frame-level ranking loss is worth ~3x tau** over stock DSNet training,
   whose detection losses optimise a binary 15%-budget target only loosely related to
   ordering.
3. **The teacher personalises but not *semantically*.** Its labels diverge (0.48, 92% of
   pairs) yet the divergence does not track persona semantic distance (r=+0.022, p=0.41).
   A student cannot learn a relationship its labels do not contain. This is the central
   limitation and the clearest direction for future work.
4. **Teacher self-consistency (k=3) is not worth 3x the cost** (paired p=0.382).

## Layout

```
src/                 all project code (17 scripts, see below)
configs/             default.yaml (final config) + ablations.yaml (Phase 11 grid)
personas/            seeds.json (16 domains) + pool.json (128 personas)
data/                base_h5, clip_h5, persona_h5, video_map.json   [gitignored]
labels_cache/        6,576 cached teacher calls, keyed and never recomputed [gitignored]
models/              trained checkpoints [gitignored]
results/             per-phase notes, metrics, main_table.csv, the PDF log
third_party/DSNet    cloned baseline + a 32-line compatibility/query patch [gitignored]
third_party/PGL-SUM  cloned baseline [gitignored]
env/                 lockfiles, constraints, activation scripts, environment notes
```

## Pipeline

| Script | Phase | Purpose |
|---|---|---|
| `map_videos.py` | 4 | h5 key -> mp4 mapping, **verified** (mismapping is silent) |
| `extract_clip.py` | 4 | GoogLeNet-1024 -> CLIP-512 features |
| `verify_clip_h5.py` | 4 | schema + rank test (75/75 videos matched themselves) |
| `build_seeds.py` | 6 | domain-conditioned seed library |
| `gen_personas.py` | 6 | 128 personas via the teacher |
| `audit_personas.py` | 6 | diversity + bias audit |
| `extract_shot_frames.py` | 7 | one frame per shot, cached once per video |
| `score_teacher.py` | 7 | persona-conditioned shot scoring, overlap-aligned |
| `build_persona_h5.py` | 7 | broadcast scores -> persona datasets |
| `query_head.py` | 8 | FiLM/concat conditioner + CLIP text embeddings |
| `train_persona.py` | 9 | trains the student |
| `eval_rank.py` | 10 | Otani-protocol tau/rho |
| `eval_pglsum.py` | 3 | PGL-SUM scores through DSNet's harness |
| `random_baseline.py` | 3 | the random floor |
| `persona_sensitivity.py` | 11 | counterfactual personalisation test |
| `infer_personalised.py` | 13 | raw video + text -> summary video |
| `md2pdf.py` | - | builds the PDF log |

## Reproducibility

Every teacher call is cached under a key covering (video, persona, model, prompt version,
chunk span, repeat), so re-running Phase 7 costs nothing. Environments are pinned in
`env/requirements-*.txt` with `constraints-*.txt` to stop a later install silently
swapping torch for a build compiled against the wrong CUDA.

Hardware note: this ran on 3x RTX 3090 (24 GB, sm_86), not the HEX A100/H100s the runbook
assumes. FP8 is unavailable, so the 32B teacher is out of reach; vLLM is pinned to 0.19.1
because 0.20.0+ requires CUDA 13 (driver >= 580; this machine has 555). See
`env/environment-notes.md`.

## Data

TVSum and SumMe are downloaded from their original sources and **not redistributed**.
Raw videos live outside the repo on local scratch.
