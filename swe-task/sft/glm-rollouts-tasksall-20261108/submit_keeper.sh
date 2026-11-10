#!/usr/bin/env bash
# glm-rollouts-tasksall: submit keeper.sbatch as a schedule-helper babysit pair (CPU partitions, 23 h), each keeper job
# with a fresh babysit state dir (babysit/keeper/<tag>). The keeper keeps the GLM-5.3 server slots alive and schedules
# the campaigns; touch keeper.STOP to stop it. A running keeper chains its successor itself: with GR_KEEPER_AFTER=<job>
# the new pair waits for that job to end (--dependency=afterany:<job>).
set -euo pipefail
unset BABYSIT_LEG BABYSIT_INDEX BABYSIT_ENABLED BABYSIT_DEFER_RELEASE SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU \
  SLURM_MEM_PER_GPU SLURM_CPUS_PER_TASK SLURM_CPUS_PER_GPU SLURM_TRES_PER_TASK
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108
after=()
if [[ -n "${GR_KEEPER_AFTER:-}" ]]; then
  [[ "$GR_KEEPER_AFTER" =~ ^[0-9]+$ ]] || { echo "bad GR_KEEPER_AFTER" >&2; exit 2; }
  after=(--dependency="afterany:$GR_KEEPER_AFTER")
fi
export RESULTS_DIR="$GR/babysit/keeper/k$(date +%s)" BATCH_SCRIPT="$GR/keeper.sbatch" EXP_NAME="gta-keeper"
mkdir -p "$RESULTS_DIR"
ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
      babysit_submit --partition="${GR_PARTITION:-cpu_datamover,cpu_interactive,cpu}" --cpus-per-task=1 --mem=2G \
      --time=23:00:00 ${after[@]+"${after[@]}"})
echo "$EXP_NAME $ids"
