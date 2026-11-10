#!/usr/bin/env bash
# One node rank of the GLM-5.3 engine, run by serve_glm53_dfw.sbatch INSIDE the container (srun task per node).
# Node rank 0 runs the API server (port $API_PORT) and pipeline stage 0; node rank r>0 runs stage r --headless.
set -euo pipefail
NODE_RANK=${SLURM_PROCID:?}
ulimit -c 0
ulimit -Sn 65535 2>/dev/null || true
unset VLLM_PORT HF_HUB_ENABLE_HF_TRANSFER
export PATH="$VENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" TRITON_CACHE_DIR="$CACHE_ROOT/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export DG_JIT_CACHE_DIR="$CACHE_ROOT/deep_gemm" FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer" CUDA_CACHE_PATH="$CACHE_ROOT/nv"
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if (( PP > 1 )); then export VLLM_PP_LAYER_PARTITION="$PP_PARTITION"; else unset VLLM_PP_LAYER_PARTITION; fi
# GLM's tool parser constrains every tool-enabled request with an xgrammar structural tag (tool strict level
# FUNCTION) unless this is 0. On stock vLLM 0.33 / env vllm-0.33.0-pr60477 under PP that zeroes the MTP acceptance
# (1.1 vs 3.0; fixed in env vllm-0.33.0-glm53, see notes/divergence/upstream/mtp-so-pp-fix). Default 0 = no grammar
# on tool calls, as the hosted GLM-5.3 endpoint (0 unparsable tool calls in 82k hub replies).
export VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}

if (( NODE_RANK == 0 )); then LOG="$RANK0_LOG"; else LOG="${RANK0_LOG%_rank0.log}_rank$NODE_RANK.log"; fi

# Exit path (normal end, crash, scancel / time limit = SIGTERM, then SIGKILL after Slurm's KillWait of 120 s):
# remove the node-local weight copy (704 GB on a shared node's /raid) unless STAGE_KEEP=1. The note goes to the
# rank's engine log on Lustre: srun's stdout forwarding is already gone when a job is cancelled.
VLLM_PID=""
STAGED=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true; fi
  if [[ -n "$STAGED" && "${STAGE_KEEP:-0}" != 1 ]]; then
    if rm -rf "$STAGED" 2>/dev/null; then
      echo "[rank$NODE_RANK $(date -Is)] $(hostname) removed staged weights $STAGED" >> "$LOG" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

# NCCL/GLOO bootstrap over the node's routed Ethernet interface; NCCL data goes over the 8 IB HCAs the image presets
# (NCCL_IB_HCA / UCX_NET_DEVICES = mlx5_0,1,2,4,5,6,7,8; mlx5_3 is the 100 GbE port).
# The container has no `ip` binary: find the interface that carries this node's DNS address via /sys + python.
IFACE=${IFACE:-}
if [[ -z "$IFACE" ]]; then
  IFACE=$(python3 - <<'PY' 2>/dev/null || true
import fcntl, os, socket, struct
ip = socket.gethostbyname(socket.gethostname())
for name in sorted(os.listdir("/sys/class/net")):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))[20:24])
    except OSError:
        continue
    if addr == ip:
        print(name)
        break
PY
)
fi
if [[ -z "$IFACE" || "$IFACE" == lo ]]; then
  if [[ -e /sys/class/net/enp90s0np0 ]]; then IFACE=enp90s0np0; else IFACE=""; fi
fi
if [[ -n "$IFACE" ]]; then export GLOO_SOCKET_IFNAME="$IFACE" NCCL_SOCKET_IFNAME="$IFACE"; fi

WEIGHTS="$MODEL"
if [[ "$STAGE_LOCAL" == 1 ]]; then
  dst=/raid/scratch/${SLURM_JOB_USER:-${USER:-glm53}}-glm53/$(basename "$MODEL")
  mkdir -p "$(dirname "$dst")" 2>/dev/null || true
  STAGED="$dst"
  echo "[rank$NODE_RANK $(date +%T)] staging $MODEL -> $dst"
  bash "$G/tools/stage_to_shm.sh" "$MODEL" "$dst" 2>&1 | grep -v bashrc | tail -1 || true
  [[ -f "$dst/config.json" ]] || { echo "[rank$NODE_RANK] staging failed; loading from $MODEL"; dst="$MODEL"; }
  WEIGHTS="$dst"
fi

args=(
  "$WEIGHTS"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$HEAD_IP"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enable-prefix-caching
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
  --enable-prompt-tokens-details
  --disable-uvicorn-access-log
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}'
)
if (( DCP > 1 )); then args+=(--decode-context-parallel-size "$DCP" --dcp-comm-backend ag_rs); fi
if (( MTP > 0 )); then args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
if [[ -n "$DEFAULT_REASONING_EFFORT" ]]; then
  args+=(--default-chat-template-kwargs "{\"reasoning_effort\":\"$DEFAULT_REASONING_EFFORT\"}")
fi
if (( NODE_RANK == 0 )); then args+=(--host 0.0.0.0 --port "$API_PORT"); else args+=(--headless); fi
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then read -r -a extra <<< "$EXTRA_VLLM_ARGS"; args+=("${extra[@]}"); fi

echo "[rank$NODE_RANK $(date +%T)] $(hostname) iface=$IFACE log=$LOG"

# Startup with in-job restarts: vLLM 0.33 sometimes hangs in CUDA-graph capture ("direct DCP q-gather multimem
# timeout", 2 of ~12 starts on 2026-11-07, different nodes). Rank 0 detects a hang (that message in a rank log since
# the current start, vLLM exiting before ready, or no readiness after READY_TIMEOUT_S) and bumps the start generation
# in logs/serve/<job>-<restart>.gen; every rank then kills its vLLM process group and starts again on
# MASTER_PORT + generation, keeping the allocation and the staged weights. At most MAX_STARTS starts per job.
GEN_FILE="$LOG_DIR/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.gen"
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1500}
MAX_STARTS=${MAX_STARTS:-3}
HANG_PATTERN="multimem timeout"
gen=0
if (( NODE_RANK == 0 )); then echo 0 > "$GEN_FILE.tmp" && mv -f "$GEN_FILE.tmp" "$GEN_FILE"; fi
read_gen() { local g; g=$(cat "$GEN_FILE" 2>/dev/null || true); [[ "$g" =~ ^[0-9]+$ ]] && echo "$g" || echo 0; }

start_vllm() {
  local port=$(( ${MASTER_PORT:-8476} + gen ))
  echo "[$(date -Is)] start $gen: vllm serve ${args[*]} --master-port $port" >> "$LOG"
  # own process group (setsid), so a restart can kill the engine core and every worker
  setsid vllm serve "${args[@]}" --master-port "$port" >> "$LOG" 2>&1 &
  VLLM_PID=$!
}

stop_vllm() {
  [[ -n "$VLLM_PID" ]] || return 0
  kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  sleep 10  # let the driver release the GPU memory
}

rank_logs() { ls "${RANK0_LOG%_rank0.log}"_rank*.log 2>/dev/null || true; }

start_vllm

# Sleeps run as `sleep N & wait $!` so a SIGTERM runs the exit trap at once (bash defers traps during a foreground
# command). Every curl runs under `timeout`: a curl hung for 12 min despite -m 5 on 2026-11-07 (overloaded node) and
# blocked the readiness loop and its restart check.
if (( NODE_RANK == 0 )); then
  while true; do
    t0=$(date +%s)
    declare -A offset=()
    for f in $(rank_logs); do offset[$f]=$(stat -c %s "$f" 2>/dev/null || echo 0); done
    reason=""
    until timeout -k 3 15 curl -s -m 5 "http://127.0.0.1:$API_PORT/v1/models" | grep -q "\"$SERVED_MODEL_NAME\""; do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then reason="vLLM exited before becoming ready"; break; fi
      if (( $(date +%s) - t0 > READY_TIMEOUT_S )); then reason="not ready after ${READY_TIMEOUT_S} s"; break; fi
      for f in $(rank_logs); do
        if grep -a -q "$HANG_PATTERN" <(tail -c +$(( ${offset[$f]:-0} + 1 )) "$f" 2>/dev/null); then
          reason="startup hang ($HANG_PATTERN in $(basename "$f"))"; break
        fi
      done
      [[ -n "$reason" ]] && break
      sleep 10 & wait $!
    done
    if [[ -z "$reason" ]]; then break; fi
    echo "[rank0 $(date +%T)] start $gen failed: $reason; tail of $LOG:"; tail -20 "$LOG"
    if (( gen + 1 >= MAX_STARTS )); then
      echo "failed $(date -Is) ($reason; $MAX_STARTS starts)" >> "$STATUS_FILE"; exit 1
    fi
    gen=$(( gen + 1 ))
    echo "$gen" > "$GEN_FILE.tmp" && mv -f "$GEN_FILE.tmp" "$GEN_FILE"
    echo "restart $(date -Is) start $gen ($reason)" >> "$STATUS_FILE"
    stop_vllm
    sleep 15  # the other ranks stop their engines too
    start_vllm
  done
  kv=$(grep -a "GPU KV cache size" "$LOG" 2>/dev/null | tail -1 | sed 's/.*GPU KV cache size/GPU KV cache size/' || true)
  echo "[rank0 $(date +%T)] READY after $(( $(date +%s) - t0 )) s (start $gen): http://$HEAD_IP:$API_PORT/v1 ($kv)"
  echo "ready $(date -Is) http://$HEAD_IP:$API_PORT/v1 $kv" >> "$STATUS_FILE"
  # Health watchdog: 4 consecutive failed /health probes (30 s apart, 20 s timeout) -> fail the step so Slurm tears
  # down both ranks (srun --kill-on-bad-exit) instead of leaving a wedged endpoint behind a live URL file.
  # Stall watchdog: /health only checks that the engine process lives; requests running but no prompt/generation token
  # progress for STALL_S (/metrics, every 60 s) also fails the step, so the routers move the sessions elsewhere.
  STALL_S=${STALL_S:-600}
  fails=0; tick=0; last_count=-1; last_progress=$(date +%s)
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep 30 & wait $!
    if timeout -k 3 30 curl -s -m 20 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$API_PORT/health" | grep -q '^200$'; then
      fails=0
    else
      fails=$((fails + 1))
      echo "[rank0 $(date +%T)] /health failed ($fails/4)"
      if (( fails >= 4 )); then
        echo "unhealthy $(date -Is)" >> "$STATUS_FILE"; kill -TERM -- "-$VLLM_PID" 2>/dev/null || true; sleep 20; exit 1
      fi
    fi
    tick=$((tick + 1))
    if (( tick % 2 == 0 )); then
      running=0; count=-1
      read -r running count < <(timeout -k 3 30 curl -s -m 20 "http://127.0.0.1:$API_PORT/metrics" 2>/dev/null | awk '
        /^vllm:num_requests_running[{]/ {r += $NF} /^vllm:(generation|prompt)_tokens_total[{]/ {c += $NF}
        END {printf "%d %.0f\n", r, c}') || true
      now=$(date +%s)
      if [[ "$count" != "$last_count" || "${running:-0}" == 0 ]]; then
        last_count=$count; last_progress=$now
      elif (( now - last_progress > STALL_S )); then
        echo "[rank0 $(date +%T)] stalled: $running requests running, no token progress for $(( now - last_progress )) s"
        echo "unhealthy $(date -Is) (stalled: $running running, no progress for $(( now - last_progress )) s)" >> "$STATUS_FILE"
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true; sleep 20; exit 1
      fi
    fi
  done
  wait "$VLLM_PID"
else
  # Follow rank 0's restart requests; exit only when vLLM ends without one.
  while true; do
    while kill -0 "$VLLM_PID" 2>/dev/null && (( $(read_gen) <= gen )); do sleep 10 & wait $!; done
    if (( $(read_gen) <= gen )); then  # vLLM ended by itself: rank 0 may be about to request a restart
      for _ in $(seq 9); do (( $(read_gen) > gen )) && break; sleep 10 & wait $!; done
    fi
    if (( $(read_gen) > gen )); then
      gen=$(read_gen)
      echo "[rank$NODE_RANK $(date +%T)] restart requested: start $gen"
      stop_vllm
      start_vllm
      continue
    fi
    wait "$VLLM_PID"
    exit $?
  done
fi
