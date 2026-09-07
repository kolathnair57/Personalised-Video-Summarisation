# Serving the Qwen3-VL teacher (quick reference)

Full rationale and measurements: `results/phase5/phase5-notes.md`.

## Start the server

```bash
source env/activate-vllm.sh
export CUDA_VISIBLE_DEVICES=1          # pick a FREE gpu - shared machine, check nvidia-smi first
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": {"count": 16, "width": 512, "height": 512}}' \
  --served-model-name qwen3vl-8b
```

Takes ~5 minutes to become ready. Check with:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Hard constraints on this hardware

- **No FP8.** RTX 3090 is sm_86; `*-FP8` checkpoints cannot run. bf16 only.
- **8B fills the card.** 16.6 GB weights + 3.9 GB KV cache on 24 GB.
  `--max-model-len 32768` leaves room for <1 sequence -> do not raise it.
- One server per GPU. Do not run two.

## Calling it (the pattern Phase 7 uses)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

N = len(shot_frames)
schema = {"type": "array", "minItems": N, "maxItems": N,          # minItems is REQUIRED
          "items": {"type": "number", "minimum": 0, "maximum": 1}}

r = client.chat.completions.create(
    model="qwen3vl-8b",
    messages=[{"role": "system", "content": DISCRIMINATION_PROMPT},
              {"role": "user", "content": content}],
    response_format={"type": "json_schema",                        # NOT guided_json
                     "json_schema": {"name": "shot_scores", "schema": schema}},
    temperature=0.2, max_tokens=1500)
scores = json.loads(r.choices[0].message.content)
```

### Three things that will silently break it

1. **`extra_body={"guided_json": ...}` is ignored in vLLM 0.19.1.** No error - it just
   returns unconstrained text. Use `response_format`.
2. **Without `minItems`/`maxItems` the model may return `[]`** and it counts as valid.
3. **Array-of-objects burns the token budget** on whitespace padding (>1200 tokens and
   truncates). A flat numeric array does the same job in ~81 tokens.

### And one that quietly ruins the labels

The teacher returns a **constant** score vector if the persona does not match the video's
domain (e.g. a pastry chef rating a tyre-changing tutorial -> all 0.0). Constant labels
give zero learning signal. Two requirements:

- draw personas from the **video's own domain** (TVSum category, see
  `ydata-tvsum50-info.tsv`);
- keep the system prompt that demands relative discrimination:

```
You rate video shots for a specific viewer. You must DISCRIMINATE between shots:
rank them RELATIVE to each other. Give at least one shot a score above 0.7 and at
least one below 0.3. Never return the same score for every shot.
```

Always assert `scores.std() > 0` before writing a persona h5.
