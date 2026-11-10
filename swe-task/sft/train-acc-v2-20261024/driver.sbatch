#!/bin/bash
# train-acc-v2 (2026-10-24): one probe campaign (driver.py) in a CPU job, like synth/run.sbatch (the CPU node only
# orchestrates: rollouts run in OpenSandbox, the model over HTTP). Submitted only through the schedule helper (babysit
# pair); the losing candidate exits at once. Env from the submitter: TA_MODEL, TA_GROUP, RESULTS_DIR, EXP_NAME.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # the code reads every credential from its file when it needs it (as run.sbatch)
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${TA_MODEL:?}" "${TA_GROUP:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname): model $TA_MODEL group $TA_GROUP"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/driver.py \
  --model "$TA_MODEL" --group "$TA_GROUP"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
