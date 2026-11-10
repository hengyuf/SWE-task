#!/bin/bash
# glm-rollouts-tasksall (2026-11-08): keeper.py in a small CPU job (GLM-5.3 server slots + campaign scheduling).
# Submitted through the schedule helper (babysit pair): bash submit_keeper.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
# the keeper's own babysit identity must not reach the jobs it submits (keeper.py also scrubs child environments)
unset BABYSIT_LEG BABYSIT_INDEX BABYSIT_ENABLED BABYSIT_DEFER_RELEASE RESULTS_DIR EXP_NAME BATCH_SCRIPT
echo "[$(date -Is)] keeper job $SLURM_JOB_ID on $(hostname)"
/usr/bin/python3 -I -B /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/keeper.py
rc=$?
echo "[$(date -Is)] keeper exit $rc"
exit $rc
