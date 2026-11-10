#!/usr/bin/env bash
# train-acc-v2 (2026-10-24) copy of swe-task/student/serve_student.sh for the SFT training-accuracy test: the same
# worker, except (1) STUDENT_ROOT is required (each fleet has its own registry/, logs/ and cache/, so the synth
# pipeline's routers never see these workers), (2) the lenient-think reasoning parser (eval/deepswe-v1.1/assets/
# nano_v3_lenient_think_parser.py: in an unclosed think block the first <tool_call> ends the reasoning), the same parser
# as the DeepSWE lenient evals, for both the SFT model and the base model, (3) process tag ta-worker-<job>.
# 2026-10-25: TA_REASONING_PARSER / TA_REASONING_PARSER_PLUGIN (env from submit.sh) select the parser; the 10ep and
# 20ep fleets use assets/nano_v3_lenient_think_counted_parser.py (the same parse, plus TA_THINK_* log lines).
# Persistent student inference worker, INSIDE the production container (worker.sbatch: srun --container-image ...):
# one TP8 vLLM replica of the local student checkpoint (alias sc14, served as "policy") with the DeepSWE eval serving
# configuration, published for the routers in registry/<job>.json. worker.sbatch is frozen at submit time; all worker
# logic lives here and is read when the job starts, so a fix reaches jobs that are already queued.
#
# vLLM engine args and runtime env = EVAL/scripts/serve_vllm.sh verbatim (RL r2 policy.generation); every start is
# diffed against the RL engine log by EVAL/tools/check_serving_parity.py (recorded, never fatal). Differences, none of
# which changes what is served: per-job log logs/vllm-<job>.log (appended, never truncated), seed = job id mod 2^31
# (distinct per replica), compile caches in cache/compile/<node> and HF_HOME in cache/hf_home (EVAL's and the RL
# runtime's are read-only for us), PYTHONDONTWRITEBYTECODE=1, STUDENT_WORKER_TAG (marks this job's vLLM processes).
#
# Registry record (atomic replace; schema in README.md): starting -> ready -> draining (last STUDENT_DRAIN_S of the
# job) -> stopped; or failed (startup; exclude_node says whether the node is suspect) / unhealthy (while serving).
# heartbeat_at = last time this worker saw vLLM alive: every ~30 s during startup, and while serving only after a
# passing GET /health.
#
# Env from worker.sbatch: STUDENT_END_EPOCH, STUDENT_JOB_START (unix seconds). Optional knobs (default):
#   STUDENT_MODEL_PATH (swonlysc-step0014/hf), STUDENT_READY_TIMEOUT_S (1200, counted from the job start),
#   STUDENT_LOAD_TIMEOUT_S (600: vLLM launch -> first "Loading weights took"; the known init hang never gets there),
#   STUDENT_DRAIN_S (2700), STUDENT_STOP_MARGIN_S (30), STUDENT_HEALTH_INTERVAL_S (30),
#   STUDENT_REG_TIMEOUT_S (15: one registry write attempt; a failed attempt is retried once).
#   Test hooks only (tests/test_fleet.py runs a fake server on the login node): STUDENT_ROOT, STUDENT_PORT,
#   STUDENT_TEST_SERVER.
# Exit status: 0 stopped at the walltime, 2 bad environment, 3 vLLM exited during startup (or port taken),
#   4 startup hang / not ready in time, 6 unhealthy while serving, 128+N stopped by signal N.
set -uo pipefail
set -m  # background jobs get their own process group: vLLM and all its workers are stopped as one group

STUDENT="${STUDENT_ROOT:?STUDENT_ROOT (the fleet directory) is required}"
MIG=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration
EVAL="$MIG/eval/deepswe-v1.1"
RUNTIME="$MIG/runtime"
VLLM_PY="$RUNTIME/envs/vllm/bin/python"
MODEL="${STUDENT_MODEL_PATH:-$MIG/models/swonlysc-step0014/hf}"
SERVED_MODEL_NAME=policy
MAX_MODEL_LEN=262144
PER_CALL_CAP=131072  # enforced by the routers (contract); recorded for them
PORT="${STUDENT_PORT:-8000}"
JOB="${SLURM_JOB_ID:?serve_student.sh must run inside a Slurm job}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
SLOT="${SLURM_JOB_NAME:-}"; SLOT="${SLOT##*-w}"
[[ "$SLOT" =~ ^[0-9][0-9]$ ]] || SLOT="${SLURM_JOB_NAME:-manual}"
LOGS="$STUDENT/logs"
REGISTRY="$STUDENT/registry"
REG="$REGISTRY/$JOB.json"
VLLM_LOG="$LOGS/vllm-$JOB.log"
SEED=$(( JOB % 2147483648 ))
TAG="ta-worker-$JOB-${SLURM_RESTART_COUNT:-0}"
JOB_START="${STUDENT_JOB_START:-$(date +%s)}"
END_EPOCH="${STUDENT_END_EPOCH:-}"
READY_TIMEOUT_S="${STUDENT_READY_TIMEOUT_S:-1200}"
LOAD_TIMEOUT_S="${STUDENT_LOAD_TIMEOUT_S:-600}"
DRAIN_S="${STUDENT_DRAIN_S:-2700}"
STOP_MARGIN_S="${STUDENT_STOP_MARGIN_S:-30}"
HEALTH_INTERVAL_S="${STUDENT_HEALTH_INTERVAL_S:-30}"
REG_TIMEOUT_S="${STUDENT_REG_TIMEOUT_S:-15}"
PY="$(command -v python3 || echo "$VLLM_PY")"  # the image's /opt/nemo_rl_venv python; stdlib only below

STATUS="" REASON="" READY_AT="" DRAIN_AT="" HEARTBEAT="" PARITY="" KV_TOKENS="" EXCLUDE_NODE=""
FINAL="" STOP_REASON="" VLLM_PID="" LOG_OFFSET=0 LAUNCHED_AT=""

log() { echo "[$(date -Is)] [$NODE job $JOB] $*"; }

# Atomic writer of registry/<job>.json: temp file in the same directory (dot name, never *.json) + os.replace.
REGISTRY_PY=$(cat <<'PYEOF'
import json, os, sys
env = os.environ
def number(name):
    value = env.get(name, "")
    return int(value) if value.isdigit() else None
record = {
    "job_id": env["W_JOB"], "slot": env["W_SLOT"], "node": env["W_NODE"], "ip": env["W_IP"],
    "port": int(env["W_PORT"]), "base_url": "http://%s:%s/v1" % (env["W_IP"], env["W_PORT"]),
    "served_model_name": env["W_SERVED"], "model_path": env["W_MODEL"], "max_model_len": int(env["W_MAX_LEN"]),
    "per_call_cap": int(env["W_CAP"]), "seed": int(env["W_SEED"]), "started_at": number("W_STARTED"),
    "ready_at": number("W_READY_AT"), "end_epoch": number("W_END"), "drain_at": number("W_DRAIN_AT"),
    "heartbeat_at": number("W_HEARTBEAT"), "status": env["W_STATUS"], "reason": env.get("W_REASON") or None,
    "parity": env.get("W_PARITY") or None, "kv_cache_tokens": number("W_KV_TOKENS"),
    "partition": env.get("W_PARTITION") or None, "vllm_log": env["W_VLLM_LOG"],
}
if env.get("W_EXCLUDE_NODE"):
    record["exclude_node"] = env["W_EXCLUDE_NODE"] == "1"
path = sys.argv[1]
tmp = os.path.join(os.path.dirname(path), ".%s.%d.tmp" % (os.path.basename(path), os.getpid()))
with open(tmp, "w") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(tmp, path)
PYEOF
)

# The address other nodes reach this worker at: what Gym advertises (gethostbyname(gethostname())), unless loopback.
IP_PY=$(cat <<'PYEOF'
import socket
ip = ""
try:
    ip = socket.gethostbyname(socket.gethostname())
except OSError:
    pass
if not ip or ip.startswith("127."):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # sends nothing; selects the outbound interface
        ip = probe.getsockname()[0]
    except OSError:
        pass
    finally:
        probe.close()
print(ip)
PYEOF
)

reg_write() {  # reg_write <status> [reason]: atomically replace this job's registry record
  # A Lustre stall can hang the write (2026-10-20 15:22-15:26, every bridge at once). Each attempt gets REG_TIMEOUT_S
  # and a failed attempt is retried once, so one heartbeat costs at most ~31 s, about one HEALTH_INTERVAL_S. A writer
  # stuck in uninterruptible I/O cannot be killed before the filesystem answers; routers then keep a worker with a
  # stale heartbeat on their own /health checks (router.py).
  local attempt
  STATUS="$1"; REASON="${2:-}"
  for attempt in 1 2; do
    W_JOB="$JOB" W_SLOT="$SLOT" W_NODE="$NODE" W_IP="$IP" W_PORT="$PORT" W_SERVED="$SERVED_MODEL_NAME" \
    W_MODEL="$MODEL" W_MAX_LEN="$MAX_MODEL_LEN" W_CAP="$PER_CALL_CAP" W_SEED="$SEED" W_STARTED="$JOB_START" \
    W_READY_AT="$READY_AT" W_END="$END_EPOCH" W_DRAIN_AT="$DRAIN_AT" W_HEARTBEAT="$HEARTBEAT" W_STATUS="$STATUS" \
    W_REASON="$REASON" W_PARITY="$PARITY" W_KV_TOKENS="$KV_TOKENS" W_PARTITION="${SLURM_JOB_PARTITION:-}" \
    W_VLLM_LOG="$VLLM_LOG" W_EXCLUDE_NODE="$EXCLUDE_NODE" \
      timeout "$REG_TIMEOUT_S" "$PY" -I -S -c "$REGISTRY_PY" "$REG" && return 0
    log "WARNING: registry write failed (status $STATUS, attempt $attempt/2)"
    (( attempt < 2 )) && pause 1
  done
  return 0
}

fail() {  # fail <exit status> <node suspect: 1|0> <reason>: record a startup failure and exit (cleanup keeps it)
  FINAL=failed; EXCLUDE_NODE="$2"; log "FAILED: $3"; reg_write failed "$3"; exit "$1"
}

unhealthy() {  # unhealthy <reason>: vLLM stopped serving; the routers drop this worker at once
  FINAL=unhealthy; EXCLUDE_NODE=1; log "UNHEALTHY: $1"; reg_write unhealthy "$1"; exit 6
}

pause() { sleep "$1" & wait $!; }  # interruptible: a foreground sleep would hold off the TERM/INT traps
run_bg() { "$@" & wait $!; }       # same for curl

this_launch_log() { tail -c +"$((LOG_OFFSET + 1))" "$VLLM_LOG" 2>/dev/null; }
# Not a pipe: with pipefail, grep -q exiting early would SIGPIPE tail and turn a match into a failure.
launch_log_has() { grep -q -- "$1" < <(this_launch_log); }

last_error() {  # last error-looking line of this launch's vLLM output (for the registry reason)
  this_launch_log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E 'Error|ERROR|Exception|error:' | tail -n 1 | cut -c1-300
}

tagged_pids() {  # processes started by this job's vLLM (environment carries the tag); never this shell
  grep -lzxF "STUDENT_WORKER_TAG=$TAG" /proc/[0-9]*/environ 2>/dev/null \
    | sed -nE 's#^/proc/([0-9]+)/environ$#\1#p' | grep -vxF -e "$$" -e "$BASHPID"
}

hang_evidence() {  # keep what a startup hang looked like (GPU/CPU busy pattern, stacks) before it is killed
  local out="$LOGS/hang-$JOB.txt" pid pids
  pids="$(tagged_pids | head -n 16 | paste -sd, -)"  # API server, engine core, 8 workers, ...: bounded
  {
    echo "== $(date -Is) $NODE job $JOB: vLLM startup did not progress"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
    [[ -n "$pids" ]] && ps -o pid,ppid,stat,pcpu,etime,wchan:24,args -p "$pids"
    # py-spy needs ptrace rights (kernel.yama.ptrace_scope=1 may refuse it); the ps/nvidia-smi pattern stays
    for pid in ${pids//,/ }; do
      echo "== pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)"
      timeout 10 "$RUNTIME/envs/vllm/bin/py-spy" dump --pid "$pid" 2>&1 | head -n 60
    done
    echo "== last vLLM output"
    this_launch_log | tail -n 60
  } > "$out" 2>&1
  log "startup hang evidence: $out"
}

stop_vllm() {  # TERM the vLLM process group, KILL what is left, then any orphan still carrying this job's tag
  local left
  [[ -n "$VLLM_PID" ]] || return 0
  kill -TERM -- "-$VLLM_PID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$VLLM_PID" 2>/dev/null
  left="$(tagged_pids | tr '\n' ' ')"
  if [[ -n "${left// /}" ]]; then
    log "killing leftover vLLM processes: $left"
    # shellcheck disable=SC2086  # word splitting of the pid list is intended
    kill -KILL $left 2>/dev/null
  fi
  VLLM_PID=""
}

cleanup() {
  local rc=$?
  trap '' TERM INT
  if [[ "$FINAL" != failed && "$FINAL" != unhealthy ]]; then  # first: routers stop picking it before vLLM goes
    reg_write stopped "${STOP_REASON:-exit status $rc}"
  fi
  stop_vllm
  log "worker exits with status $rc (registry status ${FINAL:-stopped})"
}

hang() {  # hang <reason>: record 'failed' (node suspect) at once, then keep the evidence (bounded), exit 4
  FINAL=failed; EXCLUDE_NODE=1; log "FAILED: $1"; reg_write failed "$1"
  hang_evidence
  exit 4
}

IP="$(timeout 30 "$PY" -I -S -c "$IP_PY" 2>/dev/null)"
[[ -n "$IP" ]] || IP="$NODE"
mkdir -p "$LOGS" "$REGISTRY" || { echo "cannot create $LOGS or $REGISTRY" >&2; exit 2; }
trap cleanup EXIT
trap 'STOP_REASON="SIGTERM (scancel or walltime)"; exit 143' TERM
trap 'STOP_REASON="SIGINT"; exit 130' INT

# ---- 1. preflight -------------------------------------------------------------------------------------------------
HEARTBEAT=$(date +%s)
[[ "$END_EPOCH" =~ ^[0-9]+$ ]] || fail 2 0 "unknown job end time (STUDENT_END_EPOCH='$END_EPOCH')"
DRAIN_AT=$(( END_EPOCH - DRAIN_S ))
reg_write starting
log "student worker (slot $SLOT, ip $IP, port $PORT, seed $SEED): job ends $(date -d "@$END_EPOCH" -Is)," \
  "drains from $(date -d "@$DRAIN_AT" -Is); model $MODEL"
[[ -f "$MODEL/config.json" ]] || fail 2 0 "not an HF checkpoint: $MODEL"
[[ -x "$VLLM_PY" ]] || fail 2 0 "missing vLLM runtime: $VLLM_PY"
curl -s -o /dev/null -m 5 "http://127.0.0.1:$PORT/"
rc=$?
(( rc == 7 )) || fail 3 1 "port $PORT already answers on this node before launch (curl status $rc): leftover server"

# ---- 2. vLLM with the eval/RL serving configuration (EVAL/scripts/serve_vllm.sh) ------------------------------------
CACHE="$STUDENT/cache/compile/$(hostname)"  # per-node compile caches (one writer per node: --exclusive)
mkdir -p "$CACHE" "$STUDENT/cache/hf_home"
VLLM_ARGS=(
  --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code
  --dtype bfloat16 --tensor-parallel-size 8 --pipeline-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 64 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.75 --kv-cache-dtype auto
  --enable-prefix-caching --mamba-cache-mode align --mamba-ssm-cache-dtype float32
  --moe-backend triton --logprobs-mode processed_logprobs
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
  --reasoning-parser-plugin "${TA_REASONING_PARSER_PLUGIN:-$EVAL/assets/nano_v3_lenient_think_parser.py}"
  --reasoning-parser "${TA_REASONING_PARSER:-nano_v3_lenient_think}" --tool-call-parser qwen3_coder --enable-auto-tool-choice
  --enable-prompt-tokens-details --seed "$SEED" --host 0.0.0.0 --port "$PORT"
)
VLLM_CMD=("$VLLM_PY" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}")
[[ -n "${STUDENT_TEST_SERVER:-}" ]] && VLLM_CMD=("$PY" "$STUDENT_TEST_SERVER" "$PORT")
LOG_OFFSET=$(stat -c %s "$VLLM_LOG" 2>/dev/null || echo 0)
{
  echo "== $(date -Is) host=$NODE job=$JOB restart=${SLURM_RESTART_COUNT:-0} port=$PORT model=$MODEL" \
    "served_model_name=$SERVED_MODEL_NAME seed=$SEED"
  printf '%q ' "${VLLM_CMD[@]}"; echo
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
} >> "$LOGS/vllm-$JOB.cmd.txt" 2>&1
run_vllm() {  # runs as a background job (its own subshell and process group), so these exports stay out of this shell
  # Same runtime environment as the RL generation workers (serve_vllm.sh / launch_recipe.sh).
  export VLLM_USE_V2_MODEL_RUNNER=0 NRL_SC_IID_LOGPROBS=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1
  export HF_HOME="$STUDENT/cache/hf_home" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
  export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:$RUNTIME/envs/driver/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
  export PATH="$RUNTIME/bin:$RUNTIME/envs/vllm/bin:$PATH"
  export VLLM_CACHE_ROOT="$CACHE/vllm" TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
  export PYTHONDONTWRITEBYTECODE=1 STUDENT_WORKER_TAG="$TAG"
  cd /tmp || exit 2
  exec "${VLLM_CMD[@]}" >> "$VLLM_LOG" 2>&1
}
run_vllm &
VLLM_PID=$!
LAUNCHED_AT=$(date +%s)
log "vLLM launched (pid $VLLM_PID, log $VLLM_LOG)"

# ---- 3. readiness, with a progress watchdog for the known init hang (7 GPUs busy-waiting in NCCL, 1 idle) ----------
loaded=0
last_beat=$LAUNCHED_AT
until run_bg curl -sf -m 10 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    wait "$VLLM_PID"
    rc=$?  # VLLM_PID stays set: cleanup still kills the rest of its process group and tagged orphans
    fail 3 1 "vLLM exited during startup (status $rc): $(last_error)"
  fi
  now=$(date +%s)
  if (( !loaded )) && launch_log_has 'Loading weights took'; then
    loaded=1
    log "weights loaded $((now - LAUNCHED_AT)) s after launch"
  fi
  if (( !loaded && now - LAUNCHED_AT > LOAD_TIMEOUT_S )); then
    hang "startup hang: no 'Loading weights took' ${LOAD_TIMEOUT_S} s after vLLM launch"
  fi
  if (( now - JOB_START > READY_TIMEOUT_S )); then
    hang "vLLM not ready ${READY_TIMEOUT_S} s after the job start"
  fi
  if (( now >= END_EPOCH - STOP_MARGIN_S )); then STOP_REASON="walltime before ready"; exit 0; fi
  if (( now - last_beat >= 30 )); then HEARTBEAT=$now; reg_write starting; last_beat=$now; fi
  pause 5
done
READY_AT=$(date +%s)
log "vLLM ready $((READY_AT - JOB_START)) s after the job start ($((READY_AT - LAUNCHED_AT)) s after launch)"
KV_TOKENS="$(this_launch_log | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' | tr -cd '0-9')"
run_bg curl -sf -m 10 -o /dev/null "http://$IP:$PORT/health" \
  || log "WARNING: the advertised address http://$IP:$PORT does not answer /health from this node"
# Read-only reuse of the eval's parity check (vs the RL r2 engine log + MasterConfig); a mismatch is recorded only.
if timeout 120 "$PY" -I -B "$EVAL/tools/check_serving_parity.py" "$VLLM_LOG" --json="$LOGS/parity-$JOB.json" \
     > "$LOGS/parity-$JOB.txt" 2>&1; then
  PARITY=PASS
elif grep -q '^serving parity: FAIL' "$LOGS/parity-$JOB.txt"; then
  PARITY=FAIL
else
  PARITY=ERROR
fi
log "serving parity: $PARITY ($LOGS/parity-$JOB.txt); KV cache ${KV_TOKENS:-?} tokens"

# ---- 4. serve: heartbeat + health watchdog; draining for the last DRAIN_S; clean stop just before the walltime ------
fails=0
while :; do
  now=$(date +%s)
  if (( now >= END_EPOCH - STOP_MARGIN_S )); then STOP_REASON="walltime"; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || unhealthy "vLLM exited while serving: $(last_error)"
  if run_bg curl -sf -m 20 -o /dev/null "http://127.0.0.1:$PORT/health"; then
    fails=0
    HEARTBEAT=$(date +%s)
    # Drain on demand: an operator creates registry/<job>.drain to move calls off this worker before an early scancel
    # (the router prefers ready workers and moves conversations off draining ones); this job stays the only writer.
    if [[ -e "$REGISTRY/$JOB.drain" ]] && (( DRAIN_AT > HEARTBEAT )); then
      DRAIN_AT=$HEARTBEAT; log "drain requested ($REGISTRY/$JOB.drain): draining from now"
    fi
    state=ready
    (( HEARTBEAT >= DRAIN_AT )) && state=draining
    [[ "$state" != "$STATUS" ]] && log "status $state"
    reg_write "$state"
  elif (( ++fails >= 4 )); then
    unhealthy "vLLM /health failed 4 consecutive checks"
  else
    log "WARNING: /health check failed ($fails/4)"
  fi
  left=$(( END_EPOCH - STOP_MARGIN_S - $(date +%s) ))
  pause $(( left < HEALTH_INTERVAL_S ? (left > 1 ? left : 1) : HEALTH_INTERVAL_S ))
done
exit 0
