#!/bin/bash
# Build env/vllm-0.33.0-glm53 = stock vLLM 0.33.0 + two GLM-5.3 fixes:
#   1. notes/divergence/upstream/pr60477-nvidia-model.patch: backport of vLLM PR #60477 (TP x PP corrupts GLM-5.3:
#      the TP-partial MoE output is re-gathered at the pipeline-stage send; main 1aea05364, in no release <= 0.33.0).
#   2. notes/divergence/upstream/mtp-so-pp-fix/mtp-so-pp-fix.patch: MTP drafts of structured-output requests (every
#      tool-enabled chat call with the glm47 parser's structural tag) were all rejected under PP + async scheduling
#      (acceptance length 1.0): per-request draft retention (backport of vLLM PR #63219, main 322f90a, WITHOUT its
#      revert of the #54442 fail-closed guard, see vllm issue #65430) + the engine fills a structured-output request's
#      placeholder drafts from the retained ones before building the grammar bitmask (glm53-serve fix).
# Run inside the container. Fresh venv (uv venv shebangs are absolute, so a copied venv would run the old code).
set -euo pipefail
E=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/env
U=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/divergence/upstream
V="$E/vllm-0.33.0-glm53"
export UV_CACHE_DIR="$E/.uv_cache" UV_LINK_MODE=hardlink UV_PYTHON_DOWNLOADS=never
if [[ ! -x "$V/bin/python" ]]; then /root/.local/bin/uv venv --python /opt/nemo_rl_venv/bin/python3 --seed "$V"; fi
/root/.local/bin/uv pip install --python "$V/bin/python" "vllm==0.33.0"
SP=$(ls -d "$V"/lib/python3*/site-packages)
patch -p1 -N -d "$SP" < "$U/pr60477-nvidia-model.patch"
patch -p1 -N -d "$SP" < "$U/mtp-so-pp-fix/mtp-so-pp-fix.patch"
grep -n "tensor_model_parallel_all_reduce(hidden_states)" "$SP/vllm/models/deepseek_v32/nvidia/model.py"
grep -n "_needs_retained_drafts\|def begin_step" "$SP/vllm/v1/engine/core.py" "$SP/vllm/v1/worker/gpu/spec_decode/utils.py"
"$V/bin/python" -c "import vllm.models.deepseek_v32.nvidia.model, vllm.v1.engine.core, vllm.v1.worker.gpu.model_runner, vllm.v1.worker.gpu.spec_decode.utils as u; print('import ok', u.DraftTokensHandler.begin_step)"
echo "[$(date -Is)] done $V"
