"""Cache one representative frame per shot, once per video (Phase 7, step 0).

Phase 7 scores 600 (video, persona) pairs. Decoding the video inside the scoring loop
would mean ~600 decodes at ~20 s each (~3 hours) to produce frames that are IDENTICAL
across personas -- the persona changes the prompt, not the pixels. So frames are decoded
once per video (75 decodes) and cached as JPEGs.

One frame per shot, taken at the midpoint of each `change_points` segment, resized so its
long side is <= 512 px (matching the server's --limit-mm-per-prompt width/height, ~185
image tokens each).

Decoding is sequential, not seek-based -- see src/extract_clip.py for why.

Usage:
    python src/extract_shot_frames.py --dataset tvsum summe
"""
import argparse
import json
from pathlib import Path

import av
import h5py

ROOT = Path(__file__).resolve().parents[1]
BASE = {"tvsum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
        "summe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
OUT_ROOT = Path("/var/tmp/akn57/data/shot_frames")


def shot_midpoints(cps, n_file_frames):
    """Midpoint frame index of each shot, clamped inside the file."""
    return [min(int((a + b) // 2), n_file_frames - 1) for a, b in cps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["tvsum", "summe"])
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    vmap_all = json.loads((ROOT / "data/video_map.json").read_text())
    for ds in args.dataset:
        vmap = vmap_all[ds]
        with h5py.File(BASE[ds], "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
            for n, key in enumerate(keys, 1):
                cps = f[key]["change_points"][...]
                meta = vmap[key]
                out_dir = OUT_ROOT / ds / key
                done = out_dir / "index.json"
                if done.exists() and len(json.loads(done.read_text())["shots"]) == len(cps):
                    print(f"[{n}/{len(keys)}] {key}: cached ({len(cps)} shots)", flush=True)
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                mids = shot_midpoints(cps, meta["file_frames"])
                want = {m: i for i, m in enumerate(mids)}   # frame idx -> shot idx
                saved = {}
                with av.open(meta["path"]) as c:
                    st = c.streams.video[0]
                    st.thread_type = "AUTO"
                    last = max(want)
                    for i, fr in enumerate(c.decode(st)):
                        if i in want:
                            im = fr.to_image()
                            im.thumbnail((args.max_side, args.max_side))
                            p = out_dir / f"shot_{want[i]:04d}.jpg"
                            im.save(p, format="JPEG", quality=args.quality)
                            saved[want[i]] = p.name
                        if i >= last:
                            break
                missing = [s for s in range(len(cps)) if s not in saved]
                if missing:
                    raise RuntimeError(f"{ds}/{key}: could not decode shots {missing[:5]} "
                                       f"(video {meta['path']})")
                done.write_text(json.dumps(dict(
                    video=Path(meta["path"]).name, n_shots=len(cps),
                    shots=[saved[s] for s in range(len(cps))],
                    midpoints=mids), indent=1))
                print(f"[{n}/{len(keys)}] {key}: {len(cps)} shots -> {out_dir}", flush=True)

    tot = sum(1 for _ in OUT_ROOT.rglob("*.jpg"))
    print(f"\ncached {tot} shot frames under {OUT_ROOT}")


if __name__ == "__main__":
    main()
