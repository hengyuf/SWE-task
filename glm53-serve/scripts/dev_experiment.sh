#!/bin/bash
# One serving experiment on the 2-node debug hold: start the engine, wait for /v1/models, run correctness gates,
# then the replay load, then stop the engine. Results under runs/exp/<name>/.
# Usage: dev_experiment.sh <name> <mode mp|dp> <replay-concurrency list e.g. "16 32"> -- <vllm args>
# Env: JOB NODE0 NODE1 VENV EXTRA_ENV (as dev_launch2.sh), REPLAY_DURATION (s, default 600), KEEP=1 to leave it up.
set -uo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
name="$1"; mode="$2"; concs="$3"; shift 3; [[ "${1:-}" == "--" ]] && shift
: "${JOB:?}" "${NODE0:?}" "${NODE1:?}"
OUT="$G/runs/exp/$name"; mkdir -p "$OUT"
IP0=$(getent hosts "$NODE0" | awk '{print $1; exit}')
URL="http://$IP0:8000/v1"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/experiment.log"; }
printf '%q ' "$@" > "$OUT/vllm_args.txt"; echo "${EXTRA_ENV:-}" > "$OUT/extra_env.txt"

stop_engine() {
  for s in $(squeue -h -s -j "$JOB" -o "%i %j" | awk '$2=="env"{print $1}'); do scancel "$s" 2>/dev/null; done
  for _ in $(seq 60); do
    [[ -z "$(squeue -h -s -j "$JOB" -o "%i %j" | awk '$2=="env"{print $1}')" ]] && break; sleep 5
  done
  sleep 10
}

log "start $name ($mode) url=$URL"
t0=$(date +%s)
bash "$G/scripts/dev_launch2.sh" "$name" "$mode" -- "$@" >> "$OUT/experiment.log" 2>&1
ready=0
for _ in $(seq 240); do
  if curl -s -m 5 "$URL/models" | grep -q '"id"'; then ready=1; break; fi
  sleep 15
  if [[ -z "$(squeue -h -s -j "$JOB" -o "%j" | grep '^env$')" ]]; then log "engine steps exited"; break; fi
done
startup=$(( $(date +%s) - t0 ))
grep -a -E "GPU KV cache size" "$G/logs/$name-rank0.log" | tail -1 >> "$OUT/experiment.log"
if (( ! ready )); then log "NOT READY after ${startup}s"; grep -a -E "Error|error" "$G/logs/$name-rank0.log" | grep -v import_utils | tail -20 >> "$OUT/experiment.log"; stop_engine; exit 1; fi
log "ready after ${startup}s"

cd "$G/hsg-ref/serve_scripts"
for s in tools roundtrip; do
  timeout 900 python3 bench_endpoint.py --url "$URL" --model glm-5.3 --scenario $s -c 8 -n 16 --max-tokens 2048 \
    > "$OUT/gate-$s.txt" 2>&1; log "gate $s: $(grep -E 'CORRECT' "$OUT/gate-$s.txt" | tr -s ' ')"
done
timeout 1200 python3 bench_endpoint.py --url "$URL" --model glm-5.3 --scenario longctx --prompt-tokens 140000 -c 4 -n 4 \
  --max-tokens 1024 --stream > "$OUT/gate-longctx.txt" 2>&1
log "gate longctx: $(grep -E 'CORRECT|TTFT' "$OUT/gate-longctx.txt" | tr -s ' ' | tr '\n' ' ')"

cd "$G/tools/replay"
for c in $concs; do
  timeout $(( ${REPLAY_DURATION:-600} + 900 )) python3 replay_load.py --replay data/deepswe_glm_64.jsonl.gz --base-url "$URL" \
    --model glm-5.3 --concurrency "$c" --duration "${REPLAY_DURATION:-600}" --drain-timeout 300 --warmup 120 --ramp 60 \
    --think-scale 1 --think-cap 60 --max-context 262144 --metrics-url "http://$IP0:8000/metrics" --metrics-interval 10 \
    --print-every 60 --out "$OUT/replay-c$c" > "$OUT/replay-c$c.log" 2>&1
  log "replay c$c: $(tail -c 1500 "$OUT/replay-c$c/summary.json" 2>/dev/null | tr -d '\n' | cut -c1-600)"
done
if [[ "${KEEP:-0}" != 1 ]]; then stop_engine; log "stopped"; fi
log "done $name"
