#!/bin/bash
# glm-rollouts-tasksall (2026-11-09): wait for one campaign's results, then plan a dip re-run round for it (rerun_dip.py
# --plan-only) and submit one rerun_dip.sbatch job per planned campaign (parallel mode). Env: GR_WAIT_CAMPAIGN (e.g.
# t1-h00), GR_DIP_NAME (e.g. dip3), GR_DIP_ARGS (extra planning args, e.g. "--include-ungraded --tiers 1").
# Submitted through the schedule helper babysit pair (state babysit/<GR_DIP_NAME>-launcher). While it waits it is a
# gta-dip* job, so the keeper keeps the GPU servers (tier-1 work is not finished until these re-runs are).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=16:00:00
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
unset BABYSIT_LEG BABYSIT_INDEX BABYSIT_ENABLED BABYSIT_DEFER_RELEASE RESULTS_DIR EXP_NAME BATCH_SCRIPT
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_CPUS_PER_TASK SLURM_CPUS_PER_GPU SLURM_TRES_PER_TASK
: "${GR_WAIT_CAMPAIGN:?}" "${GR_DIP_NAME:?}" "${GR_DIP_ARGS:=}"
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3
echo "[$(date -Is)] waiting for $GR/results/glmlocal-$GR_WAIT_CAMPAIGN.json"
until [[ -f "$GR/results/glmlocal-$GR_WAIT_CAMPAIGN.json" ]]; do sleep 120; done
echo "[$(date -Is)] $GR_WAIT_CAMPAIGN done; planning $GR_DIP_NAME ($GR_DIP_ARGS)"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
plan=$($PY -B "$GR/rerun_dip.py" --base-url http://127.0.0.1:1/v1 --name "$GR_DIP_NAME" --plan-only $GR_DIP_ARGS) || { echo "planning failed"; exit 1; }
echo "$plan"
campaigns=$(printf '%s' "$plan" | $PY -c 'import json,sys; print(" ".join(json.load(sys.stdin)["plan"]))')
[[ -n "$campaigns" ]] || { echo "[$(date -Is)] nothing to re-run"; exit 0; }
for c in $campaigns; do
  mkdir -p "$GR/babysit/$GR_DIP_NAME/$c"
  ids=$(export RESULTS_DIR="$GR/babysit/$GR_DIP_NAME/$c" BATCH_SCRIPT="$GR/rerun_dip.sbatch" EXP_NAME="gta-$c" \
               GR_RERUN_ARGS="--name $GR_DIP_NAME --campaign $c"; \
        source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
        babysit_submit --partition=cpu_datamover,cpu_interactive,cpu --cpus-per-task=4 --mem=32G --time=12:00:00)
  echo "[$(date -Is)] submitted gta-$c $ids"
done
