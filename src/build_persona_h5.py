"""Broadcast shot scores to `gtscore` and write the persona h5 files (Phase 7.2).

This is the deliverable data of the project: for each persona, a dataset identical in
structure to the CLIP h5 except that `gtscore` carries that persona's importance and a
`query` attribute carries their preference_query.

Built from **data/clip_h5/** (not base_h5): Phase 8 conditions on the persona query
encoded by CLIP's text encoder, which only shares a space with CLIP image features.

Broadcast rule (runbook 7.2): each shot's score fills every subsampled frame (`picks`)
inside that shot's `change_points` span.

A persona only covers videos of its own domain, so its file contains just those videos --
which also keeps the whole persona_h5 tree small (~1 GB rather than ~11 GB).

Guards:
  * every pick must fall inside some shot, else the video is reported;
  * `gtscore.std() > 0`, since a constant vector is a silent data-quality failure;
  * all non-gtscore datasets are copied unchanged.

Usage:
    python src/build_persona_h5.py --base data/clip_h5 --scores labels_cache/scores \
                                   --out data/persona_h5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLIP_NAME = {"tvsum": "tvsum_clip.h5", "summe": "summe_clip.h5"}


def broadcast(scores, cps, picks, n_steps):
    """Shot scores -> per-subsampled-frame gtscore."""
    out = np.zeros(n_steps, dtype=np.float32)
    covered = np.zeros(n_steps, dtype=bool)
    for s, (a, b) in enumerate(cps):
        m = (picks >= a) & (picks <= b)
        out[m] = scores[s]
        covered |= m
    return out, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/clip_h5"))
    ap.add_argument("--scores", default=str(ROOT / "labels_cache/scores"))
    ap.add_argument("--out", default=str(ROOT / "data/persona_h5"))
    ap.add_argument("--pool", default=str(ROOT / "personas/pool.json"))
    ap.add_argument("--seeds", default=str(ROOT / "personas/seeds.json"))
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    seeds = json.loads(Path(args.seeds).read_text())
    by_id = {p["persona_id"]: p for v in pool.values() for p in v}

    # which (persona, video) score files exist
    have = defaultdict(dict)          # (dataset, persona_id) -> {video: path}
    for p in Path(args.scores).rglob("*.json"):
        ds, pid, vkey = p.parts[-3], p.parts[-2], p.stem
        have[(ds, pid)][vkey] = p
    print(f"score files: {sum(len(v) for v in have.values())} across {len(have)} (dataset, persona)")

    problems, written = [], 0
    for (ds, pid), vids in sorted(have.items()):
        persona = by_id.get(pid)
        if persona is None:
            problems.append(f"{pid}: not in pool.json")
            continue
        domain = persona["domain"]
        expected = [k for k, d in seeds["video_domain"][ds].items() if d == domain]
        # A pair is legitimately dropped upstream when the teacher returns a constant
        # vector (nothing in that video interests this persona). Dropping the whole
        # persona for one missing video would waste the rest, so write what exists and
        # record the shortfall -- but require at least 2 videos to be a usable file.
        if len(vids) < len(expected):
            problems.append(f"{ds}/{pid}: {len(vids)}/{len(expected)} videos scored "
                            f"(missing {sorted(set(expected) - set(vids))})")
        if len(vids) < 2:
            problems.append(f"{ds}/{pid}: only {len(vids)} usable video(s) -- skipped")
            continue

        out_h5 = Path(args.out) / ds / domain.replace(":", "_") / f"{pid}.h5"
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        src = Path(args.base) / CLIP_NAME[ds]

        # copy only this persona's domain videos out of the clip h5
        with h5py.File(src, "r") as fin, h5py.File(out_h5, "w") as fout:
            for vkey in sorted(vids, key=lambda x: int(x.split("_")[1])):
                gin = fin[vkey]
                cps = gin["change_points"][...]
                picks = gin["picks"][...].astype(np.int64)
                n_steps = int(gin["n_steps"][()])
                sc = np.asarray(json.loads(vids[vkey].read_text())["scores"], dtype=np.float32)
                if len(sc) != len(cps):
                    problems.append(f"{ds}/{pid}/{vkey}: {len(sc)} scores vs {len(cps)} shots")
                    continue

                new_gt, covered = broadcast(sc, cps, picks, n_steps)
                if not covered.all():
                    problems.append(f"{ds}/{pid}/{vkey}: {(~covered).sum()}/{n_steps} picks "
                                    f"outside every shot span")
                if float(new_gt.std()) <= 1e-6:
                    problems.append(f"{ds}/{pid}/{vkey}: constant gtscore (std=0) -- unusable")
                    continue

                gout = fout.create_group(vkey)
                for name in gin.keys():
                    if name == "gtscore":
                        continue
                    gout.create_dataset(name, data=gin[name][()])
                gout.create_dataset("gtscore", data=new_gt)          # <-- the swap
                for k, v in gin.attrs.items():
                    gout.attrs[k] = v
                gout.attrs["query"] = persona["preference_query"]     # for the query head
                gout.attrs["persona_id"] = pid
                gout.attrs["domain"] = domain
            fout.attrs["persona_id"] = pid
            fout.attrs["domain"] = domain
            fout.attrs["query"] = persona["preference_query"]
            fout.attrs["source"] = str(src.name)
        written += 1
        if written % 20 == 0:
            print(f"  wrote {written} persona files...", flush=True)

    print(f"\nwrote {written} persona h5 files under {args.out}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:15]:
            print("   !!", p)
        (ROOT / "results/phase7/build_problems.txt").write_text("\n".join(problems))


if __name__ == "__main__":
    main()
