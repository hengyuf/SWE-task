#!/bin/bash
# Install FlashInfer's precompiled JIT cache into a vLLM venv (run inside the container). Usage: install_fi_kernels.sh <venv>
set -euo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
VENV="${1:?venv}"
export PATH="$VENV/bin:/root/.local/bin:$PATH" UV_CACHE_DIR="$G/env/.uv_cache" UV_LINK_MODE=hardlink
vllm download-kernels 2>&1 | tail -8
"$VENV/bin/python" -m pip list 2>/dev/null | grep -i flashinfer
