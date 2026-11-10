#!/usr/bin/env bash
# Run tools/strip_gate_timeouts.py --apply every 30 s for the given runs (round 1) until each run's round-1 probe
# exists (the tool then reports "skipped"), so tasks packaged just before the probe starts are still fixed.
# Usage: tools/strip_gate_timeouts_loop.sh RUN [RUN ...]
set -uo pipefail
S=$(cd "$(dirname "$0")/.." && pwd)
pending=("$@")
while ((${#pending[@]})); do
  next=()
  for run in "${pending[@]}"; do
    out=$(/usr/bin/python3 -I -B "$S/tools/strip_gate_timeouts.py" --run "$run" --round 1 --apply 2>&1)
    case "$out" in
      *'"skipped"'*) echo "$(date +%T) $run: probe started, done";;
      *'"changes": []'*) next+=("$run");;
      *) echo "$(date +%T) $run: $out"; next+=("$run");;
    esac
  done
  pending=("${next[@]}")
  ((${#pending[@]})) && sleep 30
done
echo "$(date +%T) all done"
