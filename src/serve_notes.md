# Phase 5 — serving Qwen3-VL locally with vLLM

Operational notes for the teacher model. Keep this file updated with the exact
commands and flags that actually worked on HEX, plus the node/GPU they ran on.

## Workhorse (8B dense)

```bash
source ~/envs/pvs-vllm/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # needed for offline multimodal
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --limit-mm-per-prompt image=16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

## Ablation teacher (32B FP8, H100-class node)

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --tensor-parallel-size 2 --port 8000 \
    --limit-mm-per-prompt image=16 --max-model-len 32768
```

## Health check

```bash
curl http://localhost:8000/v1/models
```

## Practical notes

- Prefer `Instruct` over `Thinking` for scoring: faster and more deterministic.
- On bf16 OOM: lower `--max-model-len` or send images only (no video input).
- The 235B MoE flagship needs 8x80 GB — out of scope for this project.

## Run log

| Date | Model | Node / GPUs | Flags changed | Outcome |
|------|-------|-------------|---------------|---------|
|      |       |             |               |         |
