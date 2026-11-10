#!/bin/bash
# teacher-rollouts-v2 (2026-10-25): one hub-teacher probe campaign (driver.py) in a CPU job, like train-acc-v2's
# driver.sbatch (the CPU node only orchestrates: rollouts run in OpenSandbox, the model on the NVIDIA hub). Submitted
# only through the schedule helper (babysit pair); the losing candidate exits at once.
# Env from the submitter: TR_MODEL (glm|deepseek), TR_TASKSET (pilot|full), TR_GROUP, TR_MIN_REMAINING, TR_MAX_CONCURRENCY,
# TR_HUB_KEY_NAMES (2026-10-26: ordered hub key NAMES, e.g. NVAPI_KEY_NEW_2,NVAPI_KEY_NEW_3 -> driver.py --key-names;
# captured before the credential scrub below, which unsets every variable named *KEY*), RESULTS_DIR, EXP_NAME.
# 2026-10-26: replaced by write + mv, never edited in place (running jobs keep the old inode).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
hub_key_names="${TR_HUB_KEY_NAMES:-}"  # key NAMES only (no values); kept as an unexported shell variable
for name in $(compgen -e); do  # the code reads every credential from its file when it needs it (as run.sbatch)
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${TR_MODEL:?}" "${TR_TASKSET:?}" "${TR_GROUP:=0}" "${TR_MIN_REMAINING:=0}" "${TR_MAX_CONCURRENCY:=32}"
[[ -n "$hub_key_names" ]] || { echo "TR_HUB_KEY_NAMES missing: submit through submit.sh"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname): model $TR_MODEL taskset $TR_TASKSET group $TR_GROUP min-remaining $TR_MIN_REMAINING max-concurrency $TR_MAX_CONCURRENCY key-names $hub_key_names"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/driver.py \
  --model "$TR_MODEL" --taskset "$TR_TASKSET" --group "$TR_GROUP" --min-remaining "$TR_MIN_REMAINING" \
  --max-concurrency "$TR_MAX_CONCURRENCY" --key-names "$hub_key_names"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
