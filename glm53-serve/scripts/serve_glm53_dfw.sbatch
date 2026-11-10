#!/usr/bin/env bash
#SBATCH --account=nemotron_agents_dev
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=glm53-serve
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"inference_server","description":"GLM-5.3 vLLM inference server; idle between client requests"}}'
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/logs/serve/%j_%x.out
#SBATCH --exclude=pool0-00058,pool0-00064,pool0-00070,pool0-00094,pool0-00128,pool0-00133,pool0-00180,pool0-00182,pool0-00196,pool0-00207,pool0-00402,pool0-00484,pool0-00490,pool0-00502,pool0-00525,pool0-00560,pool0-00613,pool0-00632,pool0-00825,pool0-00826,pool0-00861,pool0-00862,pool0-00899,pool0-00955,pool0-00965,pool0-00989,pool0-01012,pool0-01017,pool0-01239,pool0-01282,pool0-01283,pool0-01290,pool0-01447,pool0-01460,pool0-01598,pool0-01689,pool0-01764,pool0-01813,pool0-01856,pool0-01884,pool0-01888,pool0-01941,pool0-01970
# =====================================================================================================================
# GLM-5.3 (zai-org/GLM-5.3, FP8, 756 GB) vLLM server on dfw: 2 nodes x 8 H100 80GB, one engine
#   TP8 inside each node x PP2 across the nodes + DCP8 (KV sharded inside each node) + MTP3, fp8_ds_mla KV, 262k
#   context, 5.9M KV tokens, on vLLM 0.33.0 + two fixes (env/vllm-0.33.0-glm53, env/build_glm53_env.sh): a backport of
#   vLLM PR #60477 (TP x PP corruption) and the MTP + structured-output fix (notes/divergence/upstream/mtp-so-pp-fix).
#
#   sbatch scripts/serve_glm53_dfw.sbatch                       # endpoint -> logs/serve/<jobid>.url (http://IP:8000/v1)
#   sbatch --partition=interactive scripts/serve_glm53_dfw.sbatch
#   TP=16 PP=1 VENV=$G/env/vllm-0.33.0 sbatch ...               # stock-vLLM fallback (no patch; half the KV, ~25% slower)
#   DCP=1 MTP=0 sbatch scripts/serve_glm53_dfw.sbatch            # knobs are environment variables (see below)
#
# Why this shape (measured on dfw H100, see notes/EXPERIMENTS.md and README.md):
#   * 704 GiB of weights do not fit one 8x80GB node -> 2 nodes.
#   * TP x PP is only correct with the PR #60477 fix: GLM-5.3's MoE output is a TP-partial sum at the last layer of a
#     pipeline stage and the stock PP send path re-gathers it as if replicated (every release <= 0.33.0). Unpatched,
#     the model stays fluent but loops ("Hmm - wait...") up to the 131k output cap in OpenCode sessions (safe14 0/11)
#     and its teacher-forced NLL of hosted-GLM tokens is 0.789 instead of 0.514. This script refuses PP>1 on a venv
#     without the fix. Stages must also start on an indexer-owning layer (vllm #63170; fix #59961 only on main):
#     VLLM_PP_LAYER_PARTITION=42,36.
#   * TP8xPP2 (patched) vs TP16 (stock): same NLL (0.514) and greedy decode/prefill agreement (96-97%), but TP16 pays a
#     cross-node all-reduce in every layer. Replay of 64 real agent sessions: 983 vs 823 gen tok/s at 16 sessions,
#     1,087 vs 626 at 32; swe-task OpenCode E2E at 16 sessions: 849 vs 641 gen tok/s, 14/16 vs 12/16 tasks passed
#     (hosted GLM-5.3 on the same tasks 70/80).
#   * DCP8 (decode context parallel inside each node) shards the MLA KV cache instead of replicating it on all 8 ranks
#     of a stage: 0.76M -> 5.94M KV tokens (TP16: 0.36M -> 2.89M; hosted-GLM OpenCode sessions keep ~125k live tokens
#     each). DCP16 hung at load.
#   * MTP 3: ~2x decode; acceptance length 2.4-3.3 on agent traffic.
#   * VLLM_ENFORCE_STRICT_TOOL_CALLING=0 (set in serve_glm53_rank.sh): no grammar on tool calls, as the hosted GLM-5.3.
#     With the old envs (vllm-0.33.0, vllm-0.33.0-pr60477) every grammar-constrained request (strict tool calls,
#     response_format json_schema) lost all MTP drafts under PP (acceptance length 1.1); vllm-0.33.0-glm53 fixes that
#     (3.1-3.3), so =1 is now an option (~8% fewer tok/s on tool traffic at c16, grammar-valid tool calls).
#   * vLLM 0.33.0: >= 0.32 clamps max_tokens to the context (<= 0.31 returns 400 when prompt + max_tokens > max_model_len,
#     which kills OpenCode slots); 0.33 adds GLM / spec-decode fixes over 0.32.
#   * --max-num-batched-tokens 4096 (prefill chunk): shorter mixed prefill+decode steps (+13% tok/s at 32 sessions vs
#     8192 under PP2). Keep <= 8192 and --gpu-memory-utilization <= 0.90 (vLLM #59305: FlashMLA-sparse workspace for
#     mixed batches is allocated outside the profiled budget; 0.925 is reported to OOM on H100).
#
# Knobs (environment, defaults in brackets):
#   MODEL [/lustre/.../models/GLM-5.3]   SERVED_MODEL_NAME [glm-5.3]   API_PORT [8000]   MAX_MODEL_LEN [262144]
#   TP [8] PP [8 x nodes / TP] PP_PARTITION [42,36, only used if PP>1] DCP [8] MTP [3] KV_CACHE_DTYPE [fp8]
#   GPU_MEMORY_UTILIZATION [0.90]
#   MAX_NUM_SEQS [64] MAX_NUM_BATCHED_TOKENS [4096]
#   STAGE_LOCAL [1: copy the weights to /raid/scratch/<user>-glm53 first (80-100 s; vLLM loads them in ~20 s instead of
#     ~6 min from Lustre) and delete the copy when the job ends; 0: load from Lustre]  STAGE_KEEP [0; 1 keeps the copy]
#   VENV [glm53-serve/env/vllm-0.33.0-glm53] IMAGE [nemo-rl cuda13.2 sqsh] CACHE_ROOT [glm53-serve/cache-0.33]
#   EXTRA_VLLM_ARGS []
#   VLLM_ENFORCE_STRICT_TOOL_CALLING [0: no xgrammar structural tags on tool calls; they zero the MTP acceptance]
#   DEFAULT_REASONING_EFFORT [unset = template default "max"; low|high to set a server-side default]
# Known-bad nodes (vLLM startup hangs, IB/NVLS errors, XID drains; union of the dfw exclusion lists, 2026-11-06) are
# excluded above; extend with sbatch --exclude=... on the command line (a command-line --exclude replaces this one).
# Files: logs/serve/<jobid>.url (endpoint, written at start, removed at exit), logs/serve/<jobid>.status (state),
#        logs/serve/<jobid>_vllm_rank{0,1}.log (engine logs).
# Readiness: curl -s $(cat logs/serve/<jobid>.url)/models | grep glm-5.3, or "ready" in the .status file (7-10 min
#   after the job starts with warm JIT caches: container ~1.5 min, staging ~1.5 min, vLLM 4-7 min; ~25 min on the
#   very first start of a new vLLM version / layout while JIT caches are built)
# =====================================================================================================================
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin:$PATH
# Nothing secret-like from the submitting shell reaches the server (srun --export=ALL; Slurm also stores job_env).
for name in $(compgen -e); do
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;; esac
done
# Submitted through submit_serve.sh (dual-account babysit pair)? Then only the winning candidate continues.
if [[ -n "${BABYSIT_LEG:-}" ]]; then
  source /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
  babysit_rc=0
  babysit_enter || babysit_rc=$?
  [[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
  [[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
fi

G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
MODEL=${MODEL:-/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.3}
API_PORT=${API_PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
TP=${TP:-8}; PP=${PP:-$(( 8 * SLURM_JOB_NUM_NODES / TP ))}  # TP=16 alone -> PP1
PP_PARTITION=${PP_PARTITION:-42,36}; DCP=${DCP:-8}; MTP=${MTP:-3}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
STAGE_LOCAL=${STAGE_LOCAL:-1}
STAGE_KEEP=${STAGE_KEEP:-0}
VENV=${VENV:-$G/env/vllm-0.33.0-glm53}
IMAGE=${IMAGE:-/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/images/nemo-rl-cuda13.2-base-20260903-x86_64.sqsh}
CACHE_ROOT=${CACHE_ROOT:-$G/cache-0.33}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}
DEFAULT_REASONING_EFFORT=${DEFAULT_REASONING_EFFORT:-}
LOG_DIR=$G/logs/serve
mkdir -p "$LOG_DIR"
URL_FILE=$LOG_DIR/${SLURM_JOB_ID}.url
STATUS_FILE=$LOG_DIR/${SLURM_JOB_ID}.status
RANK0_LOG=$LOG_DIR/${SLURM_JOB_ID}_vllm_rank0.log
NNODES=$SLURM_JOB_NUM_NODES

(( TP * PP == 8 * NNODES )) || { echo "TP ($TP) x PP ($PP) must equal 8 GPUs x $NNODES nodes" >&2; exit 2; }
if (( PP > 1 )); then
  # PP is only correct with the upstream PR #60477 fix (all-reduce the TP-partial MoE output before the PP send).
  if ! grep -q "residual + tensor_model_parallel_all_reduce(hidden_states)" \
       "$(ls -d "$VENV"/lib/python3*/site-packages)/vllm/models/deepseek_v32/nvidia/model.py"; then
    echo "refusing PP=$PP: $VENV lacks the vLLM PR #60477 fix (TP x PP corrupts GLM-5.3); use VENV=$G/env/vllm-0.33.0-glm53" >&2
    exit 2
  fi
fi
[[ -f "$IMAGE" ]] || { echo "container image not found: $IMAGE" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
[[ -x "$VENV/bin/vllm" ]] || { echo "vLLM venv not found: $VENV" >&2; exit 2; }

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1; exit}')
[[ -n "$HEAD_IP" ]] || { echo "cannot resolve head node $HEAD_NODE" >&2; exit 2; }
printf 'http://%s:%s/v1\n' "$HEAD_IP" "$API_PORT" > "$URL_FILE"
echo "starting $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}" > "$STATUS_FILE"
trap 'rm -f "$URL_FILE"; echo "exited $(date -Is)" >> "$STATUS_FILE"' EXIT
echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST head=$HEAD_NODE ($HEAD_IP) endpoint=http://$HEAD_IP:$API_PORT/v1"
echo "layout: TP$TP x PP$PP DCP$DCP MTP$MTP kv=$KV_CACHE_DTYPE len=$MAX_MODEL_LEN util=$GPU_MEMORY_UTILIZATION chunk=$MAX_NUM_BATCHED_TOKENS"

export G MODEL SERVED_MODEL_NAME API_PORT MAX_MODEL_LEN TP PP PP_PARTITION DCP MTP KV_CACHE_DTYPE
export GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS STAGE_LOCAL STAGE_KEEP VENV CACHE_ROOT EXTRA_VLLM_ARGS
export DEFAULT_REASONING_EFFORT LOG_DIR STATUS_FILE RANK0_LOG HEAD_IP NNODES

export SLURM_EXPORT_ENV=ALL
# A job submitted from inside another job (e.g. the glm-rollouts keeper: 1 CPU, 2 GB) inherits that job's
# SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK through --export=ALL, and srun then confined the engine step to 1 CPU and
# 2 GB per node (2026-11-07: 30-min staging, 40 s per weight shard, failed starts). Drop them and ask for the whole node.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_CPUS_PER_TASK SLURM_CPUS_PER_GPU SLURM_TRES_PER_TASK
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --mem=0 --cpus-per-task="${SLURM_CPUS_ON_NODE:-128}" \
  --container-image="$IMAGE" --container-name="glm53-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" \
  --container-mounts=/lustre/fsw:/lustre/fsw,/raid:/raid --no-container-mount-home \
  --container-workdir=/tmp --export=ALL \
  bash "$G/scripts/serve_glm53_rank.sh"
