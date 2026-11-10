#!/usr/bin/env bash
# Submit one GLM-5.3 server (serve_glm53_dfw.sbatch) as a dual-account babysit pair (nemotron_agents_dev +
# nemotron_rl_algo race; the first to start cancels the other), per the user's schedule-helper rule.
#   bash scripts/submit_serve.sh <slot-name> [extra sbatch args, e.g. --time=02:00:00 --partition=batch_short]
# Server knobs (DCP, MTP, MAX_MODEL_LEN, STAGE_LOCAL, ...) pass through the environment (--export=ALL).
# For a continuous service submit two slots ~2 h apart, or a second leg: BABYSIT_LEG=2 ... --dependency=afterany:A:B
# For quick debugging on the interactive partition just use: sbatch --partition=interactive scripts/serve_glm53_dfw.sbatch
set -euo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh  # JSON-safe reaper tag
slot="${1:?slot name, e.g. glm53-a}"; shift
EXCLUDE=$(sed -n 's/^#SBATCH --exclude=//p' "$G/scripts/serve_glm53_dfw.sbatch")
REAPER='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"inference_server","description":"GLM-5.3 vLLM inference server; idle between client requests"}}'
export RESULTS_DIR="$G/runs/serve/$slot" BATCH_SCRIPT="$G/scripts/serve_glm53_dfw.sbatch" EXP_NAME="glm53-serve-$slot"
mkdir -p "$RESULTS_DIR" "$G/logs/serve"
ids=$(source "$HELPER" && babysit_submit --partition="${PARTITION:-batch}" --exclude="$EXCLUDE" --comment="$REAPER" "$@")
echo "$EXP_NAME $ids   (endpoint: $G/logs/serve/<winning job id>.url)"
