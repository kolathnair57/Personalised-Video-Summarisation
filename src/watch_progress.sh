#!/usr/bin/env bash
# Live progress of the PGL-SUM runs.  Usage:  bash src/watch_progress.sh
# (Ctrl-C to stop. Add `watch -n 30 bash src/watch_progress.sh` for auto-refresh.)
cd "$(dirname "$0")/.." || exit 1
EXP=third_party/PGL-SUM/Summaries/PGL-SUM/exp1
echo "PGL-SUM progress @ $(date '+%H:%M:%S')"
for d in TVSum SumMe; do
  for i in 0 1 2 3 4; do
    n=$(ls $EXP/$d/results/split$i/*.json 2>/dev/null | wc -l)
    pct=$((100 * n / 201))
    bar=$(printf '%*s' $((pct / 5)) '' | tr ' ' '#')
    printf "  %-6s split%d  %3d/201  %-20s %3d%%\n" "$d" "$i" "$n" "$bar" "$pct"
  done
done
echo "  running processes: $(pgrep -fc 'model/main.py --split_index')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'
