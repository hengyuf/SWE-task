#!/bin/bash
# TEMPLATE, NOT RUN (2026-11-06, swe-task-harness subagent). One local-GLM probe campaign leg (glm_local_driver.py) in a
# CPU job, patterned on sft/teacher-deepswe-20261030/driver.sbatch. The CPU node only orchestrates: OpenCode runs in
# OpenSandbox Cell-3, the model on OUR vLLM server (a separate GPU job). Submit through the schedule helper (babysit
# pair, one candidate per account), e.g.:
#   export GLM_ROOT=<new campaign root> GLM_BASE_URL=http://<server-ip>:8000/v1 GLM_TASKSET=pilot GLM_GROUP=0 \
#          GLM_REPEATS=1 GLM_MAX_CONCURRENCY=2 GLM_MIN_MODEL_LEN=262144
#   export RESULTS_DIR=$GLM_ROOT/babysit/pilot BATCH_SCRIPT=<this file> EXP_NAME=glmlocal-pilot
#   (source <schedule_helper.sh> && babysit_submit --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover \
#       --cpus-per-task=2 --mem=16G --time=12:00:00)
# ATP admits a slot only while >= 10,800 + 1,800 + 600 s = 3.67 h are left, so use >= 8 h (TR2 used 12 h, TDW 23 h).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # every credential is read from its file by the code that needs it
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
source "$HELPER"
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${GLM_ROOT:?}" "${GLM_BASE_URL:?}" "${GLM_TASKSET:?}" "${GLM_GROUP:=0}" "${GLM_MAX_CONCURRENCY:=8}" "${GLM_MIN_MODEL_LEN:=262144}"
repeats=()
[[ -n "${GLM_REPEATS:-}" ]] && repeats=(--repeats "$GLM_REPEATS")
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}: $GLM_TASKSET group $GLM_GROUP via $GLM_BASE_URL"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/swe-task-harness/template/glm_local_driver.py \
  --root "$GLM_ROOT" --base-url "$GLM_BASE_URL" --taskset "$GLM_TASKSET" --group "$GLM_GROUP" "${repeats[@]}" \
  --max-concurrency "$GLM_MAX_CONCURRENCY" --min-model-len "$GLM_MIN_MODEL_LEN"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
