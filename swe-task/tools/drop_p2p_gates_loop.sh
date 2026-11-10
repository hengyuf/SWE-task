#!/usr/bin/env bash
# Run tools/drop_p2p_gates.py --apply every 30 s for one run/round until that round's probe exists (then the tool
# reports "skipped"). Usage: tools/drop_p2p_gates_loop.sh RUN ROUND GATE [GATE ...]
set -u
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
run=$1; round=$2; shift 2
gates=(); for g in "$@"; do gates+=(--gate "$g"); done
while true; do
  out=$(/usr/bin/python3 -I -B "$S/tools/drop_p2p_gates.py" --run "$run" --round "$round" "${gates[@]}" --apply 2>&1)
  case "$out" in *'"changes": []'*) ;; *) echo "$(date +%T) $out" ;; esac
  case "$out" in *skipped*) echo "$(date +%T) $run round $round probe exists; stopping"; exit 0 ;; esac
  sleep 30
done
