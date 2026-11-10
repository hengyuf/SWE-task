#!/bin/bash
# Dev launcher: runs `vllm serve` INSIDE the nemo-rl container on one node (one node rank of a multi-node engine).
# Env: VENV (vllm venv dir), LOGF (log file), plus any VLLM_*/NCCL_* overrides; args are passed to vllm serve.
set -uo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
VENV="${VENV:-$G/env/vllm-0.32.0}"
LOGF="${LOGF:-$G/logs/dev-$(hostname)-$(date +%H%M%S).log}"
export PATH="$VENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_ENABLE_HF_TRANSFER
CR="${CACHE_ROOT:-$G/cache}"
export VLLM_CACHE_ROOT="$CR/vllm" TRITON_CACHE_DIR="$CR/triton" TORCHINDUCTOR_CACHE_DIR="$CR/inductor"
export DG_JIT_CACHE_DIR="$CR/deep_gemm" FLASHINFER_WORKSPACE_BASE="$CR/flashinfer" CUDA_CACHE_PATH="$CR/nv"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -c 0
mkdir -p "$(dirname "$LOGF")"
echo "[$(date -Is)] $(hostname) vllm serve $*" | tee -a "$LOGF"
exec vllm serve "$@" >> "$LOGF" 2>&1
