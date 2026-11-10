#!/bin/bash
# teacher-deepswe (2026-10-30): one hub-teacher probe campaign leg (driver.py) in a CPU job, as kimi-rollouts-v4's
# driver.sbatch (the CPU node only orchestrates: rollouts run in OpenSandbox, the model on the NVIDIA hub). Submitted
# only through the schedule helper (babysit pair, submit.sh); the losing candidate exits at once.
# Env from the submitter: TDW_MODEL (kimi|glm|deepseek), TDW_TASKSET (pilot|deepswe-w1|deepswe-w2), TDW_GROUP,
# TDW_MAX_CONCURRENCY, RESULTS_DIR, EXP_NAME.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # the code reads every credential from its file when it needs it (as run.sbatch)
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
HELPER_SHA256=62056f42a6368949aa3ad22a7b12079b565666c471a7b6fcf317239f2deb6978
[[ "$(sha256sum "$HELPER" | cut -d' ' -f1)" == "$HELPER_SHA256" ]] || { echo "schedule helper changed (sha256); refusing"; exit 1; }
source "$HELPER"
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${TDW_MODEL:?}" "${TDW_TASKSET:?}" "${TDW_GROUP:=0}" "${TDW_MAX_CONCURRENCY:=8}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}: model $TDW_MODEL taskset $TDW_TASKSET group $TDW_GROUP max-concurrency $TDW_MAX_CONCURRENCY"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/driver.py \
  --model "$TDW_MODEL" --taskset "$TDW_TASKSET" --group "$TDW_GROUP" --max-concurrency "$TDW_MAX_CONCURRENCY"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
