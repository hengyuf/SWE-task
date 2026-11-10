#!/usr/bin/env bash
# One persistent student inference worker = one 4h job = one node with one TP8 vLLM replica (serve_student.sh).
# Submitted by fleet.py with a scrubbed environment and --export=NONE; the job name (swe-task-student-wNN), the
# idle-reaper exemption (--comment) and --exclude come from its sbatch command line.
# This file is frozen at submit time, so it only resolves the job's start/end and enters the container; all worker
# logic lives in serve_student.sh, read when the job starts (fixes reach jobs that are already queued).
#SBATCH --account=nemotron_agents_dev
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
export SLURM_EXPORT_ENV=ALL  # sbatch --export=NONE must not carry over to the srun step
STUDENT=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
MIG=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration

end="${SLURM_JOB_END_TIME:-}"
[[ "$end" =~ ^[0-9]+$ ]] || end="$(SLURM_TIME_FORMAT=%s squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)"
export STUDENT_END_EPOCH="$end" STUDENT_JOB_START="${SLURM_JOB_START_TIME:-$(date +%s)}"
echo "[$(date -Is)] job $SLURM_JOB_ID (${SLURM_JOB_NAME:-?}, partition ${SLURM_JOB_PARTITION:-?}) on" \
  "${SLURMD_NODENAME:-$(hostname)}: start $STUDENT_JOB_START end ${STUDENT_END_EPOCH:-unknown}" \
  "restart ${SLURM_RESTART_COUNT:-0}"

# Same container and mounts as the eval's vLLM (EVAL/scripts/eval.sbatch), minus the Gym-only provider overlay.
exec srun --ntasks=1 --mpi=none --export=ALL \
  --container-image="$MIG/runtime/images/nemo-rl-cuda13.2-base-20260903-x86_64.sqsh" \
  --container-mounts="/lustre/fsw:/lustre/fsw,$MIG/source/RL:/opt/nemo-rl:ro" \
  --no-container-mount-home --container-workdir=/tmp \
  /bin/bash "$STUDENT/serve_student.sh"
