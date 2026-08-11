# Personalised Video Summarisation

Persona-conditioned video summarisation: a VLM teacher (Qwen3-VL, served locally
via vLLM) produces persona-conditioned importance scores, which supervise a
query-conditioned DSNet student.

**The core idea:** DSNet never reads raw video at train time — it reads an
eccv16-schema HDF5 file in which `gtscore` is the supervision signal. The method
is therefore a *data transformation*: for each generated persona, write a new h5
where `gtscore` is replaced by the teacher's persona-conditioned scores and a
`query` string is attached. KTS, knapsack selection, and the training loop are
reused unchanged.

## Stack

| Role | Component |
|------|-----------|
| Student | DSNet (anchor-free variant is the one extended) |
| Baselines | PGL-SUM, CLIP-It, random-summary floor, generic-label DSNet |
| Frame encoder | CLIP ViT-B-32 (`laion2b_s34b_b79k`), 512-d |
| Teacher | Qwen3-VL-8B-Instruct (32B-FP8 for ablations) via vLLM |
| Data | TVSum, SumMe, QFVS/UTE |
| Compute | HEX cluster |

## eccv16 h5 schema

```
<video_key>/
  features        (n_steps, D)            # subsampled frame features
  gtscore         (n_steps,)              # <-- the supervision; this is what we swap
  change_points   (num_segments, 2)       # shot boundaries (KTS)
  n_frame_per_seg (num_segments,)
  picks           (n_steps,)              # which original frames were kept
  n_frames        scalar
  n_steps         scalar
  user_summary    (num_users, n_frames)   # test time only
  gtsummary       (n_steps,)
  video_name      (SumMe only)
```

## Layout

```
env/            environment specs / lockfiles
third_party/    cloned baselines (DSNet, PGL-SUM, CLIP-It) — not vendored into git
data/
  base_h5/      stock GoogLeNet h5 (tvsum, summe)
  clip_h5/      re-extracted CLIP-feature h5
  persona_h5/   generated persona datasets (the deliverable data)
  qfvs/         QFVS/UTE for personalisation eval
personas/       seed library + generated persona pools (JSON)
labels_cache/   cached Qwen3-VL outputs (keyed, never recomputed)
src/            pipeline scripts (see below)
configs/        yaml experiment configs
results/        metric tables, logs
```

## Two environments (do not merge them)

vLLM pins aggressive `torch` / `transformers` versions that conflict with the
DSNet/CLIP training environment.

- `~/envs/pvs-train` — DSNet, CLIP, evaluation → `env/requirements-train.txt`
- `~/envs/pvs-vllm` — vLLM, Qwen3-VL serving → `env/requirements-vllm.txt`

## Pipeline

```bash
# serve the teacher (pvs-vllm env)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --limit-mm-per-prompt image=16

# generate persona labels (cached, resumable)
python src/gen_personas.py     --seeds personas/seeds.json --out personas/pool.json
python src/score_teacher.py    --pool personas/pool.json --videos data/clip_h5 --out labels_cache
python src/build_persona_h5.py --base data/clip_h5 --scores labels_cache --out data/persona_h5

# train + evaluate the student (pvs-train env)
python src/train.py    --data data/persona_h5 --qcond concat --loss mse+rank
python src/eval_rank.py --pred results/pred --data data/clip_h5
```

## Scripts

| File | Phase | Purpose |
|------|-------|---------|
| `src/extract_clip.py` | 4 | Re-extract CLIP features into the eccv16 schema |
| `src/serve_notes.md` | 5 | vLLM serving commands + run log |
| `src/gen_personas.py` | 6 | Expand domain seeds into a persona pool |
| `src/score_teacher.py` | 7.1 | Per-shot persona importances from Qwen3-VL |
| `src/build_persona_h5.py` | 7.2 | Swap `gtscore`, attach `query` |
| `src/query_head.py` | 8 | Query-conditioning for the DSNet head |
| `src/eval_rank.py` | 10.1 | Kendall tau / Spearman rho (Otani protocol) |
| `src/persona_sensitivity.py` | 11.2 | Counterfactual persona-divergence test |

## Build status

- [x] **Phase 0** — repository layout, version control
- [ ] **Phase 1** — environments on HEX
- [ ] **Phase 2** — replicate DSNet unchanged (baseline #1)
- [ ] **Phase 3** — PGL-SUM, CLIP-It, sanity baselines
- [ ] **Phase 4** — re-extract CLIP features
- [ ] **Phase 5** — serve Qwen3-VL via vLLM
- [ ] **Phase 6** — generate personas
- [ ] **Phase 7** — teacher scoring → persona h5 files
- [ ] **Phase 8** — query-conditioning in the DSNet head
- [ ] **Phase 9** — train the student on synthetic labels
- [ ] **Phase 10** — evaluation (rank correlation primary)
- [ ] **Phase 11** — ablations + persona-sensitivity test
- [ ] **Phase 12** — reproducibility hygiene (throughout)

Critical path: everything downstream depends on two artefacts being right — the
**CLIP-feature h5 files** and the **persona h5 files with swapped `gtscore` +
`query`**.

## Reproducibility rules

- Version everything that generates data: seed library, persona pool, prompts,
  persona h5 files. Every experiment regenerable from a commit hash.
- Pin versions: both requirements files, model IDs, and vLLM sampling params
  recorded with each run.
- `labels_cache/` is sacred — keyed by (video, persona, model, prompt-hash), so a
  pre-emption or re-run costs nothing.
- Do not redistribute videos. Release code + synthetic labels only; TVSum,
  SumMe, and UTE stay under their own terms.
