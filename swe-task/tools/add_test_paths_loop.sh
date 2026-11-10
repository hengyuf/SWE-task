#!/usr/bin/env bash
# Re-apply tools/add_test_paths.py every 60 s until the round's probe dir exists (then the probe has copied the
# task dirs and later edits would not matter). Usage: tools/add_test_paths_loop.sh RUN ROUND GLOB [GLOB ...]
set -u
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
run=$1; round=$2; shift 2
globs=(); for g in "$@"; do globs+=(--glob "$g"); done
while true; do
  out=$(/usr/bin/python3 -I -B "$S/tools/add_test_paths.py" --run "$run" --round "$round" "${globs[@]}" --apply 2>&1)
  echo "$(date +%T) $(echo "$out" | /usr/bin/python3 -I -c 'import json,sys
t=sys.stdin.read()
try:
    d=json.loads(t.strip().splitlines()[-1])
except Exception:
    print("ERROR", t[-300:]); sys.exit()
if d.get("skipped"): print("STOP", d["skipped"]); sys.exit()
print("changed", len(d["changes"]), [c["cand"] for c in d["changes"]], "refused", d["refused"], "waiting", len(d["not_gated_yet"]))')"
  case "$out" in *'"skipped"'*) exit 0 ;; esac
  sleep 60
done
