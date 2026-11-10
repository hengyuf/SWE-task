#!/usr/bin/env bash
# train-acc-v2 (2026-10-24): one 4 h inference worker = one node, one TP8 vLLM replica of STUDENT_MODEL_PATH for the
# fleet at STUDENT_ROOT (serve_model.sh: the student fleet's serve_student.sh with its own registry and the
# lenient-think parser). Submitted only through the schedule helper (babysit pair, one candidate per account); the
# losing candidate exits at once. Env from the submitter (--export=ALL via the helper): STUDENT_ROOT,
# STUDENT_MODEL_PATH, RESULTS_DIR, EXP_NAME; the helper adds BABYSIT_ENABLED/LEG/INDEX.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # nothing secret-like or session-bound from the submitting shell reaches the worker
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${STUDENT_ROOT:?}" "${STUDENT_MODEL_PATH:?}"
export SLURM_EXPORT_ENV=ALL  # the srun step below inherits this job's environment
MIG=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration
end="${SLURM_JOB_END_TIME:-}"
[[ "$end" =~ ^[0-9]+$ ]] || end="$(SLURM_TIME_FORMAT=%s squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)"
export STUDENT_END_EPOCH="$end" STUDENT_JOB_START="${SLURM_JOB_START_TIME:-$(date +%s)}"
# The eval's parity check (run by serve_model.sh, recorded only) expects the lenient parser's arguments.
export DEEPSWE_REASONING_PARSER=nano_v3_lenient_think
echo "[$(date -Is)] job $SLURM_JOB_ID (${SLURM_JOB_NAME:-?}) on ${SLURMD_NODENAME:-$(hostname)}: fleet $STUDENT_ROOT," \
  "model $STUDENT_MODEL_PATH, end ${STUDENT_END_EPOCH:-unknown}"
# Same container and mounts as the student fleet (student/worker.sbatch). exec keeps the helper's payload lock (fd 8).
exec srun --ntasks=1 --mpi=none --export=ALL \
  --container-image="$MIG/runtime/images/nemo-rl-cuda13.2-base-20260903-x86_64.sqsh" \
  --container-mounts="/lustre/fsw:/lustre/fsw,$MIG/source/RL:/opt/nemo-rl:ro" \
  --no-container-mount-home --container-workdir=/tmp \
  /bin/bash /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/serve_model.sh
