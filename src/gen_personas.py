
import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        # generous maxLength: a tight cap makes the constrained decoder stop mid-word
        "biography": {"type": "string", "minLength": 40, "maxLength": 600},
        "occupation": {"type": "string", "minLength": 3, "maxLength": 60},
        "preference_query": {"type": "string", "minLength": 25, "maxLength": 320},
    },
    "required": ["biography", "occupation", "preference_query"],
    "additionalProperties": False,
}

SYSTEM = (
    "You invent realistic, specific video viewers. Each viewer must be a plausible "
    "individual, not a stereotype or a marketing segment. The preference_query must "
    "describe, in concrete visual terms, what this viewer wants to SEE in the video -- "
    "it will be used to search video frames, so name visible things (objects, actions, "
    "people, moments), not abstract feelings. Keep preference_query to ONE sentence of "
    "at most 30 words, and finish the sentence. Give each viewer a distinct first name "
    "and background; do not reuse common placeholder names such as Alex, Sam or Jordan. "
    "Vary occupations widely -- avoid defaulting to graphic designer, librarian or teacher."
)

STRIDES = [1, 2, 3, 5, 7, 11, 13]


def _stride_for(n, axis):
    """A stride co-prime with n, so walking it visits EVERY value before repeating.

    This matters: a stride sharing a factor with len(values) collapses the axis. With
    5 values and stride 5, (i*5) % 5 == 0 for every i, so the attribute never varies
    within a domain. The first audit caught exactly that on `watching_context`.
    """
    from math import gcd
    cands = STRIDES[axis % len(STRIDES):] + STRIDES[:axis % len(STRIDES)]
    for s in cands:
        if gcd(s, n) == 1:
            return s
    return 1


def stride_pick(values, i, axis, offset=0):
    """Deterministic spread across `values` for persona i on a given axis.

    `offset` (derived from the domain) is essential: without it every axis picks
    index 0 at i=0, so persona #0 would be attribute-identical in every domain.
    """
    n = len(values)
    return values[(i * _stride_for(n, axis) + offset + axis) % n]


def build_prompt(domain_key, dom, attrs):
    return (
        f"Domain: {dom['label']}.\n"
        f"Seed attributes for this viewer:\n"
        f"  age band          : {attrs['age_band']}\n"
        f"  gender            : {attrs['gender_presentation']} (use matching pronouns)\n"
        f"  expertise         : {attrs['expertise']}\n"
        f"  attention budget  : {attrs['attention_budget']} (how much video they will watch)\n"
        f"  watching context  : {attrs['watching_context']}\n"
        f"  viewing goal      : {attrs['viewing_goal']}\n"
        f"  interest focus    : {attrs['interest_focus']}\n\n"
        "Expand this into ONE coherent viewer. The biography is 2-3 sentences. "
        "The preference_query states what they want to see, in concrete visual terms, "
        "consistent with their interest focus and viewing goal."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(ROOT / "personas/seeds.json"))
    ap.add_argument("--out", default=str(ROOT / "personas/pool.json"))
    ap.add_argument("--n-per-domain", type=int, default=8)
    ap.add_argument("--model", default="qwen3vl-8b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    seeds = json.loads(Path(args.seeds).read_text())
    G = seeds["global_attributes"]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    pool, failures = {}, []
    for domain_key, dom in seeds["domains"].items():
        # per-domain offset so different domains start at different points in the space
        off = sum(ord(c) for c in domain_key)
        people = []
        for i in range(args.n_per_domain):
            attrs = {
                "age_band": stride_pick(G["age_band"], i, 0, off),
                "attention_budget": stride_pick(G["attention_budget"], i, 1, off),
                "expertise": stride_pick(G["expertise"], i, 2, off),
                "watching_context": stride_pick(G["watching_context"], i, 3, off),
                "gender_presentation": stride_pick(G["gender_presentation"], i, 4, off),
                "viewing_goal": stride_pick(dom["viewing_goal"], i, 5, off),
                "interest_focus": stride_pick(dom["interest_focus"], i, 6, off),
            }
            p = None
            for attempt in range(3):          # constrained decoding occasionally emits bad JSON
                try:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": build_prompt(domain_key, dom, attrs)}],
                        response_format={"type": "json_schema",
                                         "json_schema": {"name": "persona",
                                                         "schema": PERSONA_SCHEMA}},
                        temperature=args.temperature, max_tokens=600,
                        seed=1000 + i + 977 * attempt)
                    p = json.loads(r.choices[0].message.content)
                    break
                except Exception as e:
                    err = f"{type(e).__name__} {str(e)[:100]}"
            if p is None:
                failures.append(f"{domain_key}#{i}: {err} (after 3 attempts)")
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", domain_key.lower()).strip("-")
            p["persona_id"] = f"{slug}-{i:02d}"
            p["domain"] = domain_key
            p["dataset"] = dom["dataset"]
            p["attributes"] = attrs
            people.append(p)
        pool[domain_key] = people
        print(f"  {domain_key:26s} {len(people)}/{args.n_per_domain} personas", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2))
    total = sum(len(v) for v in pool.values())
    print(f"\nwrote {out}: {total} personas across {len(pool)} domains")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures[:10]:
            print("   ", f)


if __name__ == "__main__":
    main()
