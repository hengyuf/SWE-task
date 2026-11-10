#!/bin/bash
# Dev: run the PRODUCTION rank script (serve_glm53_rank.sh) inside an existing 2-node allocation (srun --overlap), with
# serve_glm53_dfw.sbatch's defaults unless overridden in the environment. Weights are staged to /raid/scratch and KEPT
# (STAGE_KEEP=1) so restarts are fast; delete them before releasing the hold.
#   JOB=<hold job id> TAG=<name> [TP= PP= DCP= MTP= VENV= VLLM_ENFORCE_STRICT_TOOL_CALLING= EXTRA_VLLM_ARGS=] \
#     bash scripts/dev_prod.sh          -> logs/dev/<TAG>.status, logs/dev/<TAG>_vllm_rank{0,1}.log, logs/dev/<TAG>.srun.log
# Stop: scancel <JOB>.<step> (squeue -s -j JOB), or kill -INT the printed srun pid twice.
set -euo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
: "${JOB:?}" "${TAG:?}"
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
nodelist=$(squeue -j "$JOB" -h -o %N)
HEAD_NODE=$(scontrol show hostnames "$nodelist" | head -1)
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1; exit}')
export G HEAD_IP NNODES=2
export MODEL=${MODEL:-/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3} SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.3}
export API_PORT=${API_PORT:-8000} MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
export TP=${TP:-8}; export PP=${PP:-$(( 16 / TP ))}; export PP_PARTITION=${PP_PARTITION:-42,36}
export DCP=${DCP:-8} MTP=${MTP:-3} KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-64} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export STAGE_LOCAL=${STAGE_LOCAL:-1} STAGE_KEEP=1
export VENV=${VENV:-$G/env/vllm-0.33.0-pr60477} CACHE_ROOT=${CACHE_ROOT:-$G/cache-0.33}
export EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-} DEFAULT_REASONING_EFFORT=${DEFAULT_REASONING_EFFORT:-}
export VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}
mkdir -p "$G/logs/dev"
export LOG_DIR=$G/logs/dev STATUS_FILE=$G/logs/dev/$TAG.status RANK0_LOG=$G/logs/dev/${TAG}_vllm_rank0.log
IMAGE=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/images/nemo-rl-cuda13.2-base-20260903-x86_64.sqsh
echo "starting $(date -Is) dev $TAG in job $JOB" > "$STATUS_FILE"
echo "layout TP$TP x PP$PP DCP$DCP MTP$MTP strict_tools=$VLLM_ENFORCE_STRICT_TOOL_CALLING venv=$VENV extra='$EXTRA_VLLM_ARGS' endpoint http://$HEAD_IP:$API_PORT/v1"
export SLURM_EXPORT_ENV=ALL
cd /tmp
nohup srun --jobid="$JOB" --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-image="$IMAGE" --container-name=glm53dev \
  --container-mounts=/lustre/fsw:/lustre/fsw,/raid:/raid --no-container-mount-home --container-workdir=/tmp \
  --export=ALL bash "$G/scripts/serve_glm53_rank.sh" > "$G/logs/dev/$TAG.srun.log" 2>&1 &
echo "srun pid $!"
