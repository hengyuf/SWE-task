#!/bin/bash
# GLM-5.3 local-endpoint E2E on the synthetic SWE tasks: one probe campaign leg (glm_local_driver.py) in a CPU job.
# The CPU node only orchestrates (OpenCode runs in OpenSandbox Cell-3, the model on our vLLM GPU job). Submitted only
# through submit.sh (schedule helper babysit pair: one candidate per account; the loser exits at once).
# Env from submit.sh: GLM_ROOT GLM_BASE_URL GLM_TASKSET GLM_GROUP GLM_REPEATS GLM_MAX_CONCURRENCY GLM_MIN_MODEL_LEN,
# RESULTS_DIR, EXP_NAME. Install changes with write + mv, never edit in place (running jobs read this file).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/e2e/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # every credential is read from its file by the code that needs it
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${GLM_ROOT:?}" "${GLM_BASE_URL:?}" "${GLM_TASKSET:?}" "${GLM_GROUP:=0}" "${GLM_MAX_CONCURRENCY:=8}" "${GLM_MIN_MODEL_LEN:=262144}"
repeats=()
[[ -n "${GLM_REPEATS:-}" ]] && repeats=(--repeats "$GLM_REPEATS")
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?} account ${SLURM_JOB_ACCOUNT:-?}: $GLM_TASKSET group $GLM_GROUP repeats ${GLM_REPEATS:-default} conc $GLM_MAX_CONCURRENCY via $GLM_BASE_URL"
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
  /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/e2e/glm_local_driver.py \
  --root "$GLM_ROOT" --base-url "$GLM_BASE_URL" --taskset "$GLM_TASKSET" --group "$GLM_GROUP" "${repeats[@]}" \
  --max-concurrency "$GLM_MAX_CONCURRENCY" --min-model-len "$GLM_MIN_MODEL_LEN"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
