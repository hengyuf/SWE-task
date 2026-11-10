#!/bin/bash
# kimi-rollouts-v3 (2026-10-26): one Kimi-K3 probe campaign leg (driver.py) in a CPU job, like teacher-rollouts-v2's
# driver.sbatch (the CPU node only orchestrates: rollouts run in OpenSandbox, the model on the NVIDIA hub). Submitted
# only through the schedule helper (babysit pair, submit.sh); the losing candidate exits at once.
# Env from the submitter: KR_TASKSET (pilot|sel50|rest47|baihe-pilot|baihe), KR_GROUP, KR_MAX_CONCURRENCY,
# RESULTS_DIR, EXP_NAME.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # the code reads every credential from its file when it needs it (as run.sbatch)
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${KR_TASKSET:?}" "${KR_GROUP:=0}" "${KR_MAX_CONCURRENCY:=8}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}: taskset $KR_TASKSET group $KR_GROUP max-concurrency $KR_MAX_CONCURRENCY"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026/driver.py \
  --taskset "$KR_TASKSET" --group "$KR_GROUP" --max-concurrency "$KR_MAX_CONCURRENCY"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
