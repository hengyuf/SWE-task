#!/usr/bin/env bash
# Keep the given runs' round-<N> probes at most at CAP concurrent rollouts per model (ATP's admission doubles limits under
# healthy API load, and sandbox PTY interruptions do not count as pressure). Every 10 min: if a probe's concurrency.json
# exceeds CAP, re-apply run_deepswe.py set-concurrency CAP. A run drops out when its build root has summary.json.
# Stop all: touch $S/synth/queue/release-probe-hold   (main does this when the sandbox platform has recovered)
# Usage: tools/hold_probe_concurrency.sh ROUND CAP RUN [RUN ...]
set -u
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
round=$1; cap=$2; shift 2; runs=("$@")
while true; do
  [ -e "$S/synth/queue/release-probe-hold" ] && { echo "$(date +%T) released"; exit 0; }
  active=0
  for run in "${runs[@]}"; do
    root=$(ls -d "$S/synth/runs/$run"/build/*/* 2>/dev/null | head -1)
    [ -n "$root" ] && [ -f "$root/summary.json" ] && continue
    for probe in "$root"/round-"$round"/probe-*; do
      [ -f "$probe/concurrency.json" ] || continue
      active=1
      over=$(/usr/bin/python3 -I -c "import json,sys; print(int(any(v > int(sys.argv[2]) for v in json.load(open(sys.argv[1])).values())))" "$probe/concurrency.json" "$cap")
      if [ "$over" = 1 ]; then
        (cd "$S/atp/scripts" && timeout 120 ../runtime/controller/bin/python3 -B run_deepswe.py set-concurrency --campaign "$probe" --value "$cap" >/dev/null 2>&1) \
          && echo "$(date +%T) $run: re-applied cap $cap"
      fi
    done
  done
  [ "$active" = 0 ] && { echo "$(date +%T) no active probes left"; exit 0; }
  sleep "${HOLD_INTERVAL:-600}"
done
