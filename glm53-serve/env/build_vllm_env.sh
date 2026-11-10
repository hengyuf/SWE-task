#!/bin/bash
# Build a standalone vLLM venv for GLM-5.3 serving on dfw, INSIDE the nemo-rl cuda13.2 container (python 3.13 + uv).
# Usage (inside the container): bash build_vllm_env.sh <vllm-version> [extra uv pip args...]
set -euo pipefail
VER="${1:?vllm version}"; shift || true
E=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/env
V="$E/vllm-$VER"
export UV_CACHE_DIR="$E/.uv_cache" UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=300 UV_PYTHON_DOWNLOADS=never
echo "[$(date -Is)] host $(hostname) building $V"
if [[ ! -x "$V/bin/python" ]]; then
  /root/.local/bin/uv venv --python /opt/nemo_rl_venv/bin/python3 --seed "$V"
fi
/root/.local/bin/uv pip install --python "$V/bin/python" "vllm==$VER" "$@"
"$V/bin/python" - <<'PY'
import importlib, sys
for m in ("vllm", "torch", "flashinfer", "transformers", "deep_gemm", "nixl", "triton"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, str(e)[:200])
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "n", torch.cuda.device_count())
PY
"$V/bin/python" -m pip list 2>/dev/null | grep -i -E "^(vllm|torch|flashinfer|deep|nixl|triton|transformers|nvidia-nccl|xgrammar) " || true
ls "$V"/lib/python3*/site-packages/vllm/third_party/ 2>/dev/null || true
echo "[$(date -Is)] done $V"
