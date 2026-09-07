"""Diversity check and bias audit of the persona pool (Phase 6.2, second half).

Why this exists: a pool can look fine persona-by-persona and still be useless. If the
`preference_query` strings are near-duplicates, every persona h5 gets nearly the same
gtscore, the student cannot learn to differentiate, and the Phase 11 persona-sensitivity
test measures noise. That failure is invisible in the JSON -- it has to be measured.

Checks:
  1. CLIP token length -- Phase 8 encodes preference_query with CLIP's text encoder,
     which has a HARD 77-token limit and truncates SILENTLY. Queries longer than that
     lose information, and two personas can collapse to the same embedding.
  2. Semantic diversity -- pairwise cosine similarity of CLIP text embeddings, within
     each domain and across the whole pool. This measures the space Phase 8 actually
     operates in, not string overlap.
  3. Attribute coverage -- is the sampled seed space actually being covered?
  4. Bias audit -- distribution of age bands, gendered pronouns and occupations, so
     skew can be reported rather than discovered later.

Usage:  python src/audit_personas.py --pool personas/pool.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "personas/pool.json"))
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "results/phase6/persona_audit.json"))
    args = ap.parse_args()

    import open_clip
    pool = json.loads(Path(args.pool).read_text())
    people = [p for v in pool.values() for p in v]
    queries = [p["preference_query"] for p in people]
    ids = [p["persona_id"] for p in people]
    print(f"pool: {len(people)} personas, {len(pool)} domains\n")

    tokenizer = open_clip.get_tokenizer(args.model)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.eval().cuda()

    # ---- 1. CLIP token length (77 limit, silent truncation) ----------------------
    ctx = model.context_length if hasattr(model, "context_length") else 77
    raw = [tokenizer.encode(q) if hasattr(tokenizer, "encode") else None for q in queries]
    if raw[0] is None:                       # fall back: count non-pad tokens
        toks = tokenizer(queries)
        lens = (toks != 0).sum(1).tolist()
    else:
        lens = [len(r) + 2 for r in raw]     # +2 for SOT/EOT
    over = [(i, l) for i, l in enumerate(lens) if l >= ctx]
    print("=== 1. CLIP text length (limit %d tokens, truncation is SILENT) ===" % ctx)
    print(f"  tokens: mean {np.mean(lens):.1f}  median {np.median(lens):.0f}  max {max(lens)}")
    print(f"  queries at/over the limit: {len(over)}/{len(queries)}"
          + ("  <-- these lose information in Phase 8" if over else "  (all fit)"))
    for i, l in over[:5]:
        print(f"     {ids[i]} ({l} tok): {queries[i][:90]}...")

    # ---- 2. semantic diversity ---------------------------------------------------
    with torch.no_grad():
        emb = model.encode_text(tokenizer(queries).cuda()).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    E = emb.cpu().numpy()
    sim = E @ E.T
    iu = np.triu_indices(len(E), 1)
    allpairs = sim[iu]
    print("\n=== 2. semantic diversity of preference_query (CLIP space) ===")
    print(f"  pairwise cosine: mean {allpairs.mean():.3f}  p95 {np.percentile(allpairs,95):.3f}  max {allpairs.max():.3f}")
    dups = [(ids[i], ids[j], float(sim[i, j]))
            for i, j in zip(*iu) if sim[i, j] >= args.dup_threshold]
    print(f"  near-duplicates (>= {args.dup_threshold}): {len(dups)}")
    for a, b, s in sorted(dups, key=lambda t: -t[2])[:8]:
        print(f"     {s:.3f}  {a}  <->  {b}")

    per_dom = {}
    print("\n  within-domain mean similarity (high = personas in a domain are too alike):")
    idx = 0
    for dk, v in pool.items():
        n = len(v)
        sub = sim[idx:idx + n, idx:idx + n]
        m = sub[np.triu_indices(n, 1)].mean() if n > 1 else float("nan")
        per_dom[dk] = float(m)
        idx += n
    for dk, m in sorted(per_dom.items(), key=lambda t: -t[1]):
        flag = "  <-- least diverse" if m == max(per_dom.values()) else ""
        print(f"     {dk:26s} {m:.3f}{flag}")

    # ---- 3. attribute coverage ---------------------------------------------------
    print("\n=== 3. attribute coverage ===")
    axes = ["age_band", "expertise", "attention_budget", "watching_context"]
    cov = {}
    for a in axes:
        c = Counter(p["attributes"][a] for p in people)
        cov[a] = dict(c)
        print(f"  {a:18s} {len(c)} distinct  {dict(c)}")

    # ---- 4. bias audit -----------------------------------------------------------
    print("\n=== 4. bias audit ===")
    bios = " ".join(p["biography"].lower() for p in people)
    pron = {g: len(re.findall(rf"\b{g}\b", bios)) for g in ["he", "him", "his", "she", "her", "they", "them"]}
    male = pron["he"] + pron["him"] + pron["his"]
    female = pron["she"] + pron["her"]
    neutral = pron["they"] + pron["them"]
    tot = max(male + female + neutral, 1)
    print(f"  gendered pronouns: he/him {male} ({100*male/tot:.0f}%), "
          f"she/her {female} ({100*female/tot:.0f}%), they/them {neutral} ({100*neutral/tot:.0f}%)")
    occ = Counter(p["occupation"].lower().strip() for p in people)
    print(f"  occupations: {len(occ)} distinct across {len(people)} personas "
          f"({100*len(occ)/len(people):.0f}% unique)")
    print(f"  most repeated: {occ.most_common(5)}")
    ages = Counter(p["attributes"]["age_band"] for p in people)
    print(f"  age bands: {dict(ages)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n_personas=len(people), n_domains=len(pool),
        clip_tokens=dict(mean=float(np.mean(lens)), max=int(max(lens)),
                         limit=int(ctx), n_over=len(over)),
        similarity=dict(mean=float(allpairs.mean()), p95=float(np.percentile(allpairs, 95)),
                        max=float(allpairs.max()), n_near_duplicates=len(dups),
                        within_domain=per_dom),
        coverage=cov,
        bias=dict(pronouns=pron, n_distinct_occupations=len(occ), age_bands=dict(ages)),
    ), indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
