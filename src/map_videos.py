"""Build and VERIFY the h5-key -> raw-video-file mapping (Phase 4, step 1).

Why this is a separate script: getting this wrong is the single most dangerous error in
the whole project. If `video_7` is paired with the wrong mp4, CLIP features are computed
for one video and combined with another video's gtscore/change_points. Training still
runs, loss still falls, and every downstream number is silently wrong. (The previous
attempt at this project left a file literally named `segment_captions_MISMAPPED.json.bak`.)

So the mapping is derived from *evidence*, never from assumed ordering:

  SumMe : the h5 stores `video_name` -> match the filename, then CONFIRM n_frames agrees.
  TVSum : the h5 stores no name -> match on `n_frames`, which is a near-unique fingerprint.

Frame counts come from `ffprobe -count_frames`, which decodes and counts exactly, rather
than the container's metadata header (which is frequently wrong for these files).

Any video whose count is ambiguous or unmatched is reported and left OUT of the mapping;
the script exits non-zero so extraction cannot silently proceed on a partial mapping.

Usage:
    python src/map_videos.py --tvsum-dir <dir> --summe-dir <dir> --out data/video_map.json
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
H5 = {"tvsum": ROOT / "data/base_h5/eccv16_dataset_tvsum_google_pool5.h5",
      "summe": ROOT / "data/base_h5/eccv16_dataset_summe_google_pool5.h5"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def count_frames(path):
    """Exact frame count by decoding (container metadata is unreliable here)."""
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout.strip()
        return str(path), int(out.split(",")[0])
    except Exception as e:
        return str(path), -1


def scan(video_dir):
    vids = sorted(p for p in Path(video_dir).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    print(f"  found {len(vids)} video files under {video_dir}")
    counts = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, (p, n) in enumerate(ex.map(count_frames, vids), 1):
            counts[p] = n
            if i % 10 == 0 or i == len(vids):
                print(f"    counted {i}/{len(vids)}", flush=True)
    bad = [p for p, n in counts.items() if n <= 0]
    for p in bad:
        print(f"  !! could not count frames: {p}")
    return {p: n for p, n in counts.items() if n > 0}


def h5_info(name):
    with h5py.File(H5[name], "r") as f:
        info = {}
        for k in f.keys():
            vn = None
            if "video_name" in f[k]:
                v = f[k]["video_name"][()]
                vn = v.decode() if isinstance(v, bytes) else str(v)
            info[k] = dict(n_frames=int(f[k]["n_frames"][()]),
                           n_steps=int(f[k]["n_steps"][()]),
                           video_name=vn)
    return info


def map_by_name(info, counts):
    """SumMe: filename == video_name. Confirm with n_frames.

    SumMe ships every video twice (.mp4 and .webm) with identical stems, and the two
    encodings do not necessarily have the same frame count. So collect ALL files with a
    matching stem and keep the one whose decoded frame count equals the h5's n_frames --
    evidence, not file-extension preference.
    """
    by_stem = defaultdict(list)
    for p in counts:
        by_stem[Path(p).stem].append(p)
    mapping, problems = {}, []
    for k, d in info.items():
        stem = d["video_name"]
        cands = by_stem.get(stem, [])
        if not cands:
            problems.append(f"{k}: no file named '{stem}'")
            continue
        exact = [p for p in cands if counts[p] == d["n_frames"]]
        if len(exact) >= 1:
            mapping[k] = sorted(exact)[0]          # deterministic if both encodings agree
        else:
            got = ", ".join(f"{Path(p).name}={counts[p]}" for p in cands)
            problems.append(f"{k} ('{stem}'): h5 n_frames={d['n_frames']}, files have {got}")
    return mapping, problems, {k: "exact" for k in mapping}


def map_by_frames(info, counts, tol=2):
    """TVSum: no names stored -> match on frame count.

    Pass 1 is an exact match. Pass 2 handles a known artefact: the h5 files were built
    in 2019 with a different decoder, and for a couple of videos ffprobe now counts one
    extra frame (typically a trailing/duplicate frame at EOF). Those are matched within
    +-`tol` frames, but ONLY when exactly one unused candidate is that close -- otherwise
    it stays a reported problem. Every match is additionally required to satisfy
    max(picks) < decoded_frame_count, since picks index the raw video directly.
    """
    by_count = defaultdict(list)
    for p, n in counts.items():
        by_count[n].append(p)
    mapping, problems, how = {}, [], {}
    used = set()

    # pass 1: exact
    for k, d in info.items():
        cands = [p for p in by_count.get(d["n_frames"], []) if p not in used]
        if len(cands) == 1:
            mapping[k], how[k] = cands[0], "exact"
            used.add(cands[0])
        elif len(cands) > 1:
            problems.append(f"{k}: AMBIGUOUS, {len(cands)} videos have {d['n_frames']} frames: "
                            + ", ".join(Path(c).name for c in cands))

    # pass 2: near-match within tolerance, unique candidate only
    for k, d in info.items():
        if k in mapping or any(k in p for p in problems):
            continue
        near = [(p, n) for p, n in counts.items()
                if p not in used and abs(n - d["n_frames"]) <= tol]
        if len(near) == 1:
            p, n = near[0]
            mapping[k], how[k] = p, f"tolerance({n - d['n_frames']:+d})"
            used.add(p)
            print(f"  note: {k} matched {Path(p).name} within tolerance "
                  f"(h5={d['n_frames']}, file={n}, diff={n - d['n_frames']:+d})")
        elif not near:
            problems.append(f"{k}: no video within +-{tol} of {d['n_frames']} frames")
        else:
            problems.append(f"{k}: AMBIGUOUS within +-{tol}: "
                            + ", ".join(f"{Path(p).name}={n}" for p, n in near))
    return mapping, problems, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tvsum-dir")
    ap.add_argument("--summe-dir")
    ap.add_argument("--out", default=str(ROOT / "data/video_map.json"))
    args = ap.parse_args()

    result, all_problems = {}, []
    for name, d in (("tvsum", args.tvsum_dir), ("summe", args.summe_dir)):
        if not d:
            continue
        print(f"\n=== {name} ===")
        info = h5_info(name)
        counts = scan(d)
        mapping, problems, how = (map_by_name if name == "summe" else map_by_frames)(info, counts)

        # hard safety check: picks index the RAW video, so every pick must exist in the file
        with h5py.File(H5[name], "r") as f:
            for k, p in list(mapping.items()):
                mx = int(f[k]["picks"][...].max())
                if mx >= counts[p]:
                    problems.append(f"{k} -> {Path(p).name}: max(picks)={mx} but file has "
                                    f"only {counts[p]} frames -- would read past the end")
                    del mapping[k]

        print(f"  mapped {len(mapping)}/{len(info)} videos "
              f"({sum(1 for v in how.values() if v == 'exact')} exact)")
        for p in problems:
            print(f"  PROBLEM  {p}")
        all_problems += [f"{name}: {p}" for p in problems]
        result[name] = {k: dict(path=v, n_frames=info[k]["n_frames"],
                                n_steps=info[k]["n_steps"],
                                video_name=info[k]["video_name"],
                                file_frames=counts[v], match=how.get(k, "?"))
                        for k, v in mapping.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwritten: {args.out}")
    for name in result:
        print(f"  {name}: {len(result[name])} mapped")
    if all_problems:
        print(f"\n{len(all_problems)} PROBLEM(S) -- extraction must not proceed until resolved.")
        sys.exit(1)
    print("\nAll videos mapped and frame counts verified.")


if __name__ == "__main__":
    main()
