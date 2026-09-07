"""Score every shot of every video for every matching persona (Phase 7.1).

Produces, for each (video, persona) pair, one importance value per shot. Phase 7.2
broadcasts those onto `picks` to become the new `gtscore`.

Design points that matter:

**Domain pairing.** A persona only scores videos of its own domain. Phase 5 showed a
mismatched persona (pastry chef vs a tyre-repair tutorial) returns a CONSTANT all-zero
vector -- zero learning signal, undefined rank correlation.

**Chunking with anchor alignment.** The server allows 16 images per request but videos
have up to 130 shots. The runbook says "chunk and stitch", but naive stitching is wrong:
the model scores each chunk RELATIVE TO THAT CHUNK, and the Phase 5 discrimination prompt
(needed to avoid constant vectors) forces every chunk to span the full 0-1 range. Two
chunks would therefore both use 0..1 even if one is globally far less interesting, and
concatenating them corrupts the global ordering -- which is exactly what the primary
metric (rank correlation) measures.

So chunks OVERLAP by `--overlap` shots, and each chunk is rescaled onto the previous
chunk's scale by a least-squares linear fit (a*x+b) over the shots they share. The
overlap is the calibration evidence. Where chunks overlap, the two estimates are averaged.

**Self-consistency.** Each chunk is scored k times (different seeds) and averaged --
the runbook's first-listed quality lever. Cheap here: ~3.5 calls/s at concurrency 6.

**The cache is sacred.** Every chunk call is cached under a key covering dataset, video,
persona, model, prompt version, chunk span and repeat index. Re-runs and crashes cost
nothing.

**Structured output** uses `response_format` with minItems/maxItems and a flat numeric
array -- `guided_json` is silently ignored by vLLM 0.19.1, and an unconstrained array can
come back empty (see results/phase5/phase5-notes.md).

Usage:
    python src/score_teacher.py --k 3 --workers 6
"""
import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
FRAMES = Path("/var/tmp/akn57/data/shot_frames")
CACHE = ROOT / "labels_cache"
SCORES = CACHE / "scores"

PROMPT_VERSION = "v2"          # bump to invalidate the cache when the prompt changes

# ---------------------------------------------------------------------------------------
# Two scoring methods.
#
# "chunked" (v1, the runbook's approach): 16 shot images per call, model returns a score
# per shot. MEASURED TO FAIL: the resulting labels select shots that match the persona's
# own query no better than another persona's (alignment margin -0.002, own-query wins for
# 4/8 personas = chance), against a CLIP-oracle upper bound of +0.011. Two causes, both of
# which had to be fixed:
#   1. with 16 images in one prompt the query is not attended to per shot;
#   2. asking for a bare number per frame collapses to a constant (5/8 personas).
#
# "pershot" (v2, default): one image per call, and instead of asking for a number we ask a
# yes/no question and read P(yes) from the logprobs -- continuous by construction.
# Alignment margin +0.014 with 6/8 personas correct, matching the CLIP oracle, and every
# persona yields a non-constant vector. It is also faster (~15 calls/s: one image, one
# output token) and removes the need for chunking and overlap alignment entirely, since
# each shot is scored independently.
# ---------------------------------------------------------------------------------------

SYS_PERSHOT = ("You answer a single yes/no question about one video frame. Reply with "
               "exactly one word: Yes or No. Judge only what is literally visible in the "
               "frame, not how exciting or well-shot it is.")

SYSTEM = (
    "You rate video shots for a specific viewer. You must DISCRIMINATE between shots: "
    "rank them RELATIVE to each other. Give at least one shot a score above 0.7 and at "
    "least one below 0.3. Never return the same score for every shot."
)

_print_lock = threading.Lock()


@lru_cache(maxsize=4096)
def b64_of(path):
    """Cached: with k=3 self-consistency the same frames are sent repeatedly, and
    re-reading + base64-encoding 16 JPEGs per call is pure overhead."""
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode()


def chunk_spans(n_shots, size, overlap):
    """Overlapping windows covering 0..n_shots-1."""
    if n_shots <= size:
        return [(0, n_shots)]
    stride = size - overlap
    spans, start = [], 0
    while start < n_shots:
        end = min(start + size, n_shots)
        spans.append((start, end))
        if end == n_shots:
            break
        start += stride
    return spans


def cache_key(ds, vkey, pid, model, span, rep):
    raw = f"{ds}|{vkey}|{pid}|{model}|{PROMPT_VERSION}|{span[0]}-{span[1]}|{rep}"
    return hashlib.md5(raw.encode()).hexdigest()


def score_shot_pyes(client, model, ds, vkey, persona, frame_dir, index, shot,
                    temperature=0.0):
    """One cached call: P(yes) that this frame shows what the persona asked for."""
    import math
    key = cache_key(ds, vkey, persona["persona_id"], model, (shot, shot + 1), "pyes")
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        try:
            return float(json.loads(cpath.read_text())["p"])
        except Exception:
            cpath.unlink(missing_ok=True)
    q = persona["preference_query"]
    content = [{"type": "text",
                "text": f'Does this frame show: "{q}"? Answer Yes or No.'},
               {"type": "image_url", "image_url":
                   {"url": f"data:image/jpeg;base64,{b64_of(str(frame_dir / index['shots'][shot]))}"}}]
    err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS_PERSHOT},
                          {"role": "user", "content": content}],
                temperature=temperature, max_tokens=1,
                logprobs=True, top_logprobs=20, seed=3 + attempt)
            top = r.choices[0].logprobs.content[0].top_logprobs
            py = pn = 0.0
            for t in top:
                w = t.token.strip().lower()
                if w.startswith("yes"):
                    py += math.exp(t.logprob)
                elif w.startswith("no"):
                    pn += math.exp(t.logprob)
            if py + pn <= 0:
                raise ValueError("no yes/no mass in top logprobs")
            p = py / (py + pn)
            cpath.write_text(json.dumps({"p": p, "shot": shot}))
            return p
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:100]}"
    raise RuntimeError(f"{ds}/{vkey}/{persona['persona_id']} shot {shot}: {err}")


def score_chunk(client, model, ds, vkey, persona, dom_label, frame_dir, index, span, rep,
                temperature, max_retries=3):
    """One cached teacher call for shots [span[0], span[1])."""
    key = cache_key(ds, vkey, persona["persona_id"], model, span, rep)
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        try:
            return np.asarray(json.loads(cpath.read_text())["scores"], dtype=float)
        except Exception:
            cpath.unlink(missing_ok=True)      # corrupt cache entry -> recompute

    a, b = span
    n = b - a
    schema = {"type": "array", "minItems": n, "maxItems": n,
              "items": {"type": "number", "minimum": 0, "maximum": 1}}
    content = [{"type": "text", "text":
                f"VIEWER: {persona['biography']}\n"
                f"WHAT THEY WANT TO SEE: {persona['preference_query']}\n\n"
                f"Video domain: {dom_label}.\n"
                f"These are shots {a}..{b-1} of the video, in order. "
                f"Rate all {n} shots 0-1 by importance TO THIS VIEWER. "
                f"Return a JSON array of exactly {n} numbers."}]
    for j in range(a, b):
        content += [{"type": "text", "text": f"Shot {j}:"},
                    {"type": "image_url", "image_url":
                        {"url": f"data:image/jpeg;base64,{b64_of(str(frame_dir / index['shots'][j]))}"}}]

    err = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "shot_scores", "schema": schema}},
                temperature=temperature, max_tokens=64 + 12 * n,
                seed=7000 + 131 * rep + attempt)
            v = np.asarray(json.loads(r.choices[0].message.content), dtype=float)
            if v.shape != (n,) or not np.isfinite(v).all():
                raise ValueError(f"bad shape/values {v.shape}")
            cpath.write_text(json.dumps({"scores": v.tolist(), "span": list(span), "rep": rep}))
            return v
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
    raise RuntimeError(f"{ds}/{vkey}/{persona['persona_id']} span {span} rep {rep}: {err}")


def align_and_merge(spans, chunk_scores, n_shots):
    """Stitch overlapping chunks onto one global scale.

    Chunk k+1 is mapped by a*x+b, fitted by least squares on the shots it shares with the
    already-merged prefix. Without this, each chunk's independent 0-1 range would corrupt
    the global ordering.
    """
    merged = np.full(n_shots, np.nan)
    counts = np.zeros(n_shots)
    acc = np.zeros(n_shots)

    for idx, ((a, b), v) in enumerate(zip(spans, chunk_scores)):
        v = np.asarray(v, dtype=float)
        if idx == 0:
            acc[a:b] += v
            counts[a:b] += 1
            continue
        # shots this chunk shares with what is already merged
        prev = acc[a:b] / np.maximum(counts[a:b], 1)
        mask = counts[a:b] > 0
        if mask.sum() >= 2 and np.std(v[mask]) > 1e-6:
            A = np.vstack([v[mask], np.ones(mask.sum())]).T
            coef, *_ = np.linalg.lstsq(A, prev[mask], rcond=None)
            scale, off = float(coef[0]), float(coef[1])
            if not np.isfinite(scale) or scale <= 0:      # degenerate fit -> match means
                scale, off = 1.0, float(prev[mask].mean() - v[mask].mean())
        elif mask.sum() >= 1:
            scale, off = 1.0, float(prev[mask].mean() - v[mask].mean())
        else:
            scale, off = 1.0, 0.0
        acc[a:b] += scale * v + off
        counts[a:b] += 1

    merged = acc / np.maximum(counts, 1)
    if not np.isfinite(merged).all():
        merged = np.nan_to_num(merged, nan=float(np.nanmean(merged)))
    lo, hi = merged.min(), merged.max()
    return (merged - lo) / (hi - lo) if hi > lo else merged * 0.0


def do_pair(args, client, ds, vkey, persona, dom_label):
    out = SCORES / ds / persona["persona_id"] / f"{vkey}.json"
    if out.exists():
        return "cached"
    frame_dir = FRAMES / ds / vkey
    index = json.loads((frame_dir / "index.json").read_text())
    n_shots = index["n_shots"]

    if args.method == "pershot":
        spans = [(i, i + 1) for i in range(n_shots)]
        v = np.array([score_shot_pyes(client, args.model, ds, vkey, persona,
                                      frame_dir, index, s) for s in range(n_shots)])
        lo, hi = v.min(), v.max()
        v = (v - lo) / (hi - lo) if hi > lo else v * 0.0
    else:
        spans = chunk_spans(n_shots, args.chunk_size, args.overlap)
        chunk_scores = []
        for span in spans:
            reps = [score_chunk(client, args.model, ds, vkey, persona, dom_label, frame_dir,
                                index, span, r, args.temperature) for r in range(args.k)]
            chunk_scores.append(np.mean(reps, axis=0))  # self-consistency
        v = align_and_merge(spans, chunk_scores, n_shots)
    if float(np.std(v)) <= 1e-6:
        return f"DEGENERATE (std={np.std(v):.2e})"      # constant vector = unusable label
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        dataset=ds, video=vkey, persona_id=persona["persona_id"], domain=persona["domain"],
        n_shots=n_shots, k=args.k, chunks=[list(s) for s in spans],
        prompt_version=PROMPT_VERSION, scores=[round(float(x), 4) for x in v])))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "personas/pool.json"))
    ap.add_argument("--seeds", default=str(ROOT / "personas/seeds.json"))
    ap.add_argument("--model", default="qwen3vl-8b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--method", default="pershot", choices=["pershot", "chunked"],
                    help="pershot = P(yes) per frame (aligned, default); "
                         "chunked = runbook v1 (measured to be semantically unaligned)")
    ap.add_argument("--k", type=int, default=3, help="self-consistency repeats (chunked only)")
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset", nargs="+", default=["tvsum", "summe"])
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(args.pool).read_text())
    seeds = json.loads(Path(args.seeds).read_text())
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180.0, max_retries=0)

    tasks = []
    for ds in args.dataset:
        vd = seeds["video_domain"][ds]
        for vkey, domain in vd.items():
            label = seeds["domains"][domain]["label"]
            for persona in pool.get(domain, []):
                tasks.append((ds, vkey, persona, label))
    if args.limit_pairs:
        tasks = tasks[:args.limit_pairs]
    print(f"{len(tasks)} (video, persona) pairs; k={args.k}, chunk={args.chunk_size}, "
          f"overlap={args.overlap}, workers={args.workers}")

    done = {"ok": 0, "cached": 0}
    problems = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(do_pair, args, client, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            ds, vkey, persona, _ = futs[f]
            try:
                r = f.result()
                done[r] = done.get(r, 0) + 1
                if r not in ("ok", "cached"):
                    problems.append(f"{ds}/{vkey}/{persona['persona_id']}: {r}")
            except Exception as e:
                problems.append(f"{ds}/{vkey}/{persona['persona_id']}: {type(e).__name__} {str(e)[:110]}")
            if i % 25 == 0 or i == len(futs):
                with _print_lock:
                    print(f"  {i}/{len(futs)}  ok={done.get('ok',0)} cached={done.get('cached',0)} "
                          f"problems={len(problems)}", flush=True)

    print(f"\ndone: {done.get('ok',0)} scored, {done.get('cached',0)} already cached, "
          f"{len(problems)} problems")
    for p in problems[:15]:
        print("   !!", p)
    if problems:
        (ROOT / "results/phase7/score_problems.txt").write_text("\n".join(problems))


if __name__ == "__main__":
    main()
