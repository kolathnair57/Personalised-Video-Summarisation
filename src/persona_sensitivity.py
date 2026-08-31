"""Persona-sensitivity: the counterfactual test (Phase 11.2).

Phase 10 showed that TVSum/SumMe ground truth is *generic*, so agreement metrics penalise
a model for personalising. This test needs no per-persona ground truth: fix the video,
change only the persona, and measure whether the SUMMARY changes.

**Changing is not enough.** A model that perturbs its output randomly with the query would
also "change". The scientific claim requires that divergence track persona MEANING:

    semantically similar personas  -> similar summaries
    semantically distant personas  -> different summaries

so the headline number is the correlation between

    persona distance   = 1 - cosine(query_emb_i, query_emb_j)     (CLIP text space,
                                                                   the space the model
                                                                   actually conditions on)
    summary divergence = 1 - Jaccard(selected shots_i, selected shots_j)

Divergence is computed on the ACTUAL selected shot sets -- the real KTS -> knapsack
pipeline -- so this compares summaries, not raw scores.

Three references are reported alongside:

  teacher      -- divergence of the supervision itself (the realistic upper bound)
  control      -- model trained on generic labels; it learned to ignore the query, so it
                  should show ~0 divergence and ~0 correlation
  shuffled-query -- the same model fed MISMATCHED persona embeddings. This separates
                  "responds to this persona's meaning" from "responds to any vector".
                  Not in the runbook, but without it a reviewer can ask exactly that.

Usage:
    python src/persona_sensitivity.py --out results/phase11/sensitivity.json
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DSNet/src"))
sys.path.insert(0, str(ROOT / "src"))

from anchor_free.dsnet_af import DSNetAF          # noqa: E402
from helpers import vsumm_helper                  # noqa: E402
from query_head import QueryConditioner, load_query_embeddings  # noqa: E402


def jaccard(a, b):
    """Runbook 11.2: a, b are sets of selected shot indices."""
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def selected_shots(score, cps, n_frames, nfps, picks):
    """Run the real selection pipeline and return the chosen shot indices."""
    summ = vsumm_helper.get_keyshot_summ(score, cps, n_frames, nfps, picks)
    sel = set()
    for s, (a, b) in enumerate(cps):
        if summ[a:b + 1].any():
            sel.add(s)
    return sel


def load_persona_samples(dataset):
    """{video: [(persona_id, gtscore, meta), ...]} from the persona h5 files."""
    out = defaultdict(list)
    for fp in sorted((ROOT / "data/persona_h5" / dataset).rglob("*.h5")):
        with h5py.File(fp, "r") as f:
            pid = f.attrs["persona_id"]
            for v in f.keys():
                g = f[v]
                meta = dict(cps=g["change_points"][...].astype(np.int32),
                            n_frames=int(g["n_frames"][()]),
                            nfps=g["n_frame_per_seg"][...].astype(np.int32),
                            picks=g["picks"][...].astype(np.int32),
                            feats=g["features"][...].astype(np.float32))
                out[v].append((pid, g["gtscore"][...].astype(np.float32), meta))
    return out


def pairs_for(scores_by_persona, meta, Q, shuffle_map=None):
    """-> list of (distance, divergence) over all persona pairs of one video."""
    sel = {pid: selected_shots(sc, meta["cps"], meta["n_frames"], meta["nfps"],
                               meta["picks"])
           for pid, sc in scores_by_persona.items()}
    res = []
    for a, b in combinations(sorted(sel), 2):
        qa, qb = Q[shuffle_map[a] if shuffle_map else a], Q[shuffle_map[b] if shuffle_map else b]
        dist = 1.0 - float(np.dot(qa, qb) / (np.linalg.norm(qa) * np.linalg.norm(qb)))
        div = 1.0 - jaccard(sel[a], sel[b])
        res.append((dist, div))
    return res


@torch.no_grad()
def model_scores(ckpt_dir, split_idx, videos, samples, Q, device, shuffle_map=None):
    """Per-video, per-persona model predictions for one split's checkpoint."""
    m = DSNetAF(base_model="attention", num_feature=512, num_hidden=128, num_head=8)
    m.qcond = QueryConditioner(512, 512, "film")
    m.load_state_dict(torch.load(Path(ckpt_dir) / f"split{split_idx}.pt",
                                 map_location="cpu"))
    m = m.eval().to(device)
    out = {}
    for v in videos:
        if v not in samples:
            continue
        per = {}
        seq = torch.from_numpy(samples[v][0][2]["feats"]).unsqueeze(0).to(device)
        for pid, _, _ in samples[v]:
            key = shuffle_map[pid] if shuffle_map else pid
            q = torch.from_numpy(Q[key]).to(device)
            per[pid] = m(seq, q)[0].cpu().numpy().reshape(-1)
        out[v] = per
    return out


def summarise(name, pairs):
    if not pairs:
        return dict(name=name, n=0)
    d = np.array([p[0] for p in pairs]); g = np.array([p[1] for p in pairs])
    r, p = spearmanr(d, g) if np.std(d) > 0 and np.std(g) > 0 else (np.nan, np.nan)
    return dict(name=name, n=len(pairs), mean_divergence=float(g.mean()),
                median_divergence=float(np.median(g)), max_divergence=float(g.max()),
                frac_div_gt_0=float((g > 0).mean()),
                corr_distance_divergence=None if np.isnan(r) else float(r),
                corr_p=None if np.isnan(p) else float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tvsum")
    ap.add_argument("--out", default=str(ROOT / "results/phase11/sensitivity.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Q = load_query_embeddings(ROOT / "data/query_emb.npz")
    samples = load_persona_samples(args.dataset)
    splits = yaml.safe_load(open(ROOT / f"third_party/DSNet/splits/{args.dataset}.yml"))
    rng = np.random.RandomState(0)

    results = []

    # --- reference 1: the TEACHER's own labels (upper bound) ---------------------
    tpairs = []
    for v, lst in samples.items():
        if len(lst) < 2:
            continue
        tpairs += pairs_for({pid: sc for pid, sc, _ in lst}, lst[0][2], Q)
    results.append(summarise("TEACHER labels (upper bound)", tpairs))

    # --- models, evaluated only on each split's HELD-OUT videos ------------------
    for label, ckdir in (("Ours: persona+FiLM", ROOT / "models/p9_persona_film"),
                         ("Control: generic+FiLM", ROOT / "models/p9_generic_film")):
        allp = []
        for si, sp in enumerate(splits):
            te = [k.split("/")[-1] for k in sp["test_keys"]]
            preds = model_scores(ckdir, si, te, samples, Q, args.device)
            for v, per in preds.items():
                if len(per) >= 2:
                    allp += pairs_for(per, samples[v][0][2], Q)
        results.append(summarise(label, allp))

    # --- reference 2: SHUFFLED queries (does it respond to MEANING?) -------------
    pids = sorted(Q)
    perm = list(pids); rng.shuffle(perm)
    smap = dict(zip(pids, perm))
    allp = []
    for si, sp in enumerate(splits):
        te = [k.split("/")[-1] for k in sp["test_keys"]]
        preds = model_scores(ROOT / "models/p9_persona_film", si, te, samples, Q,
                             args.device, shuffle_map=smap)
        for v, per in preds.items():
            if len(per) >= 2:
                # distance still measured with the TRUE persona embeddings
                allp += pairs_for(per, samples[v][0][2], Q)
    results.append(summarise("Ours + SHUFFLED queries", allp))

    print(f"{'condition':32s}{'n':>7s}{'mean div':>10s}{'%>0':>7s}{'corr(dist,div)':>16s}{'p':>8s}")
    for r in results:
        if not r.get("n"):
            print(f"  {r['name']:30s}  (no pairs)"); continue
        c = r["corr_distance_divergence"]
        print(f"  {r['name']:30s}{r['n']:7d}{r['mean_divergence']:10.4f}"
              f"{100*r['frac_div_gt_0']:6.1f}%"
              f"{(f'{c:+.4f}' if c is not None else 'n/a'):>16s}"
              f"{(f'{r['corr_p']:.3f}' if r['corr_p'] is not None else ''):>8s}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
