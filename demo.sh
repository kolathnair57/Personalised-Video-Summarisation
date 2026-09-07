#!/usr/bin/env bash
# Demo: same video, same model, two different viewer preferences -> two different summaries.
#
#   ./demo.sh                 # default video + query pair (verified to diverge)
#   ./demo.sh --sheets        # also write side-by-side contact sheets you can eyeball
#   ./demo.sh --video dogshow # a second verified pair
#
# Override anything:
#   ./demo.sh --source /path/to/my.mp4 --query "..." --query2 "..."
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TVSUM=/var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/video
OUTDIR="$ROOT/demo_out"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SHEETS=0

# --- video presets -----------------------------------------------------------
# Both pairs below were run end-to-end and produce genuinely different summaries.
# Jaccard is shot overlap between the two summaries: 1.000 = identical.
preset_tires() {          # Jaccard 0.692 -- 2 shots swapped each way
  SOURCE="$TVSUM/AwmHb44_ouw.mp4"
  QUERY="close-ups of hands working with tools on the wheel"
  QUERY2="wide landscape scenery and the vehicle driving"
  NAME=tires
}
preset_dogshow() {        # Jaccard 0.756 -- 7 vs 4 unique shots
  SOURCE="$TVSUM/E11zDS9XGzg.mp4"
  QUERY="close-up portraits of individual dogs"
  QUERY2="the crowd, the judges and the award ceremony"
  NAME=dogshow
}
preset_tires

# --- args --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)   "preset_$2"; shift 2 ;;
    --source)  SOURCE="$2"; NAME=$(basename "${2%.*}"); shift 2 ;;
    --query)   QUERY="$2";  shift 2 ;;
    --query2)  QUERY2="$2"; shift 2 ;;
    --out)     OUTDIR="$2"; shift 2 ;;
    --gpu)     GPU="$2";    shift 2 ;;
    --sheets)  SHEETS=1;    shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$SOURCE" ]] || { echo "source video not found: $SOURCE" >&2; exit 1; }

# --- run ---------------------------------------------------------------------
# shellcheck disable=SC1091
source env/activate-train.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUTDIR"

echo "video : $SOURCE"
echo "A     : $QUERY"
echo "B     : $QUERY2"
echo

python src/infer_personalised.py \
  --source "$SOURCE" \
  --query  "$QUERY" \
  --query2 "$QUERY2" \
  --save   "$OUTDIR/$NAME.mp4"

echo
echo "wrote $OUTDIR/${NAME}_A.mp4  and  $OUTDIR/${NAME}_B.mp4"

# --- optional contact sheets -------------------------------------------------
# 12 evenly-spaced frames per summary, tiled 6x2. Quicker to compare than playing both.
if [[ $SHEETS -eq 1 ]]; then
  for s in A B; do
    f="$OUTDIR/${NAME}_$s.mp4"
    n=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$f")
    step=$(( n / 12 )); [[ $step -lt 1 ]] && step=1
    ffmpeg -v error -y -i "$f" \
      -vf "select='not(mod(n\,$step))',scale=200:-1,tile=6x2" \
      -frames:v 1 "$OUTDIR/sheet_${NAME}_$s.png"
  done
  echo "wrote $OUTDIR/sheet_${NAME}_A.png and $OUTDIR/sheet_${NAME}_B.png"
fi
