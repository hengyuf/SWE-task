#!/bin/bash
# Build env/vllm-0.33.0-pr60477 = stock vLLM 0.33.0 + backport of upstream PR #60477 ("DSA: all-reduce partial
# hidden_states before PP send", merged on main 2026-10-29 as 1aea05364, not in any release <= 0.33.0).
# Without it, TP>1 x PP>1 corrupts GLM-5.3 / DeepSeek-V3.2 activations at every pipeline-stage boundary.
# Run inside the container. Fresh venv (uv venv shebangs are absolute, so a copied venv would run the old code).
set -euo pipefail
E=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/env
bash "$E/build_vllm_env.sh" 0.33.0 >/dev/null 2>&1 || true   # makes sure the stock env exists (no-op if present)
V="$E/vllm-0.33.0-pr60477"
export UV_CACHE_DIR="$E/.uv_cache" UV_LINK_MODE=hardlink UV_PYTHON_DOWNLOADS=never
if [[ ! -x "$V/bin/python" ]]; then /root/.local/bin/uv venv --python /opt/nemo_rl_venv/bin/python3 --seed "$V"; fi
/root/.local/bin/uv pip install --python "$V/bin/python" "vllm==0.33.0"
SP=$(ls -d "$V"/lib/python3*/site-packages)
patch -p1 -N -d "$SP" < /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/divergence/upstream/pr60477-nvidia-model.patch
grep -n "tensor_model_parallel_all_reduce(hidden_states)" "$SP/vllm/models/deepseek_v32/nvidia/model.py"
"$V/bin/python" -c "import vllm.models.deepseek_v32.nvidia.model as m; print('import ok', m.__file__)"
echo "[$(date -Is)] done $V"
