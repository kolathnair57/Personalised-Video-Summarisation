# Phase 5 — Serving Qwen3-VL as the teacher

Date: 2026-08-12. Model `Qwen/Qwen3-VL-8B-Instruct` (bf16), vLLM 0.19.1, one RTX 3090.

## Working launch command

```bash
source env/activate-vllm.sh          # sets HF_HOME + VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=1
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": {"count": 16, "width": 512, "height": 512}}' \
  --served-model-name qwen3vl-8b
```

Startup ~5 min (196 s of it is engine init + CUDA graph capture). Measured:

| | |
|---|---|
| model weights | 16.64 GiB |
| available KV cache | 3.9 GiB = 28,400 tokens |
| **max concurrency at 8192 ctx** | **3.47x** |
| 16 images @ 512px | 3,003 prompt tokens (~185/image) |

## Deviations from the runbook (all necessary)

**1. `--max-model-len 8192`, not 32768.** Predicted from the config before downloading:
36 layers x 8 KV heads x 128 head_dim x 2 (K,V) x 2 bytes = **144 KB/token**. With 16 GB
of weights on a 24 GB card only ~4 GB remains for KV cache, so 32768 tokens = 4.83 GB
would allow **0.87 concurrent sequences** -- less than a single request, i.e. thrash or
OOM. 8192 gives 3.47x concurrency and comfortably fits the real workload (16 images +
prompt + JSON ~= 3.5k tokens).

**2. `--limit-mm-per-prompt` takes JSON in 0.19.1**, not the runbook's `image=16`. The
configurable form also caps resolution, which is what keeps image tokens bounded:
`'{"image": {"count": 16, "width": 512, "height": 512}}'`.

**3. FP8 is impossible** (Phase 1 finding): sm_86 has no FP8 tensor cores, so the
runbook's `Qwen3-VL-32B-Instruct-FP8` cannot run here at all. 8B bf16 is the workhorse.

## Structured output: the runbook's method silently fails

| Method | Result on vLLM 0.19.1 |
|---|---|
| `extra_body={"guided_json": schema}` (runbook 7.1) | **`'0.5, 0.8, 0.3'` -- plain text. Schema ignored, no error, no warning** |
| `extra_body={"structured_outputs": {"json": schema}}` | valid JSON |
| `response_format={"type":"json_schema", ...}` | valid JSON |

`guided_json` was removed in newer vLLM and is now silently ignored. In Phase 7 this
would yield thousands of unparseable responses, failing intermittently depending on what
the model happened to emit.

**Use `response_format`** -- it is the OpenAI standard, so it survives a move to a hosted
API, which is the whole reason for using the OpenAI-compatible server.

## Schema design (two traps found)

**Trap 1 -- an unconstrained array can come back empty.** With the runbook's schema (no
`minItems`), the model returned `[ ]` for a persona it judged irrelevant. `finish_reason`
was `stop` and the JSON was valid, so nothing errored -- the video would simply have had
no scores. Fix: `"minItems": N, "maxItems": N`.

**Trap 2 -- array-of-objects wastes the token budget.** The model pads JSON with large
runs of whitespace; `[{"shot_id":0,"importance":0.9}, ...]` for 16 shots blew past 1200
tokens and truncated mid-array. A flat array of numbers, with position implying shot id
(safe because the length is now pinned), costs **81 completion tokens** for the same
information.

Working schema:

```python
N = len(shots)
schema = {"type": "array", "minItems": N, "maxItems": N,
          "items": {"type": "number", "minimum": 0, "maximum": 1}}
```

## The core question: does the teacher actually personalise?

This is the assumption the whole project rests on, so it was tested directly on real
TVSum shots (`video_1` = "How to change tires for off road vehicles", category VT).

**First attempt failed** -- and the failure was informative:

| persona | scores |
|---|---|
| pastry chef | all 0.00 |
| sports fan | all 0.10 |
| political journalist | 0.00 ... 0.80 ... 0.80 (differentiated) |

Constant scores are the runbook's "teacher scores all ~=0.5" failure mode. A constant
`gtscore` carries **zero learning signal** and makes rank correlation undefined.

Two causes, both fixable:

1. **Domain mismatch.** A pastry chef genuinely has no interest in any shot of a
   tyre-changing tutorial, so all-zero is arguably *correct* and useless. This is exactly
   why Phase 6 must key persona seeds by video domain (gap **G2**) -- personas must be
   drawn from the video's domain, never globally.
2. **The prompt did not demand discrimination.** Adding a system message requiring
   relative ranking ("rank shots RELATIVE to each other; at least one above 0.7 and one
   below 0.3; never return the same score for every shot") fixed the spread.

**With a domain-matched persona pair and the discrimination prompt:**

| shot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mechanic (wants exact steps) | .80 | .70 | .60 | .30 | .40 | .20 | .10 | .50 | .40 | .30 | .20 | .10 | .30 | .90 | .80 | .70 |
| impatient (wants the result) | .10 | .10 | .20 | .10 | .20 | .10 | .30 | .10 | .10 | .10 | .10 | .10 | .40 | .70 | .10 | .10 |

- spearman(mechanic, impatient) = **+0.046** -- effectively uncorrelated
- std 0.255 / 0.159 -- real spread, no degeneracy
- vs the human **generic** gtscore: mechanic **+0.109**, impatient **-0.512** -- the
  persona scores deviate from the generic label, which is the point of the method

**Conclusion: the teacher does personalise, conditional on (a) domain-matched personas
and (b) a prompt that demands relative discrimination.** Both are now requirements on
Phase 6 and Phase 7, not optional quality levers.

## Bonus for Phase 6

`/var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv` (from the Phase 4
download) gives every TVSum video's **category, title and URL**. TVSum is exactly
**10 categories x 5 videos**, which is the domain-seed structure Phase 6 needs -- no
hand-labelling of 50 videos required.

## Environment note

`h5py` and `scipy` were added to the vLLM env (constrained, torch/vllm unchanged): the
scoring script must read `change_points` from the h5 *and* call the server.

## Checkpoint (runbook Phase 5)

- `curl http://localhost:8000/v1/models` -> lists `qwen3vl-8b`, max_model_len 8192.
- A chat request with a real TVSum frame returns an accurate description
  ("A man with gray hair ... against a background of horizontal blinds").
- 16-image structured-output request returns a valid 16-number array.

The server is still running on GPU 1 (PID 3423990) for Phases 6-7.
Log: `results/phase5/vllm_server.log`.
