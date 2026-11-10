#!/usr/bin/env bash
# glm-rollouts-sftv3: submit keeper.sbatch as a schedule-helper babysit pair (CPU partitions, 23 h). The keeper keeps
# servers-target.txt GLM-5.3 server slots alive; touch keeper.STOP (or write 0 to servers-target.txt) to stop it.
set -euo pipefail
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107
export RESULTS_DIR="$GR/babysit/keeper" BATCH_SCRIPT="$GR/keeper.sbatch" EXP_NAME="gr-keeper"
mkdir -p "$RESULTS_DIR"
ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
      babysit_submit --partition="${GR_PARTITION:-cpu_datamover,cpu_interactive,cpu}" --cpus-per-task=1 --mem=2G --time=23:00:00)
echo "$EXP_NAME $ids"
