#!/bin/bash
# glm-rollouts-sftv3 (2026-11-07): keeper.py in a small CPU job (keeps servers-target.txt GLM-5.3 server slots alive).
# Submitted through the schedule helper (babysit pair): bash submit_keeper.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/logs/%x-%j.out
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
echo "[$(date -Is)] keeper job $SLURM_JOB_ID on $(hostname)"
/usr/bin/python3 -I -B /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/keeper.py
rc=$?
echo "[$(date -Is)] keeper exit $rc"
exit $rc
