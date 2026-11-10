#!/bin/bash
# The official4 queue runner (run_queue.py) as a small CPU job, submitted through hengyuf's dual-account helper by
# hand (2026-10-22; this queue4 copy 2026-11-02): two held candidates of this script, on nemotron_agents_dev and
# nemotron_rl_algo, then released;
# the first to start wins in babysit_enter and cancels its sibling, a loser exits 0 at once. Submit it with
# --export=NONE, so the job gets the user's login environment (the environment run.sbatch builds always had; the
# runner passes it on to the builds it submits, without secret-like names and Slurm settings). Because --export=NONE
# carries no variables, this script finds its leg and index in the helper's state by its own job id.
# State: RESULTS_DIR=synth/queue4/runner (helper state under runner/babysit/); log: synth/queue4/logs/%x-%j.out.
#SBATCH --job-name=swe-task-queue4-runner
#SBATCH --partition=cpu_datamover
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue4/logs/%x-%j.out
set -uo pipefail
export PATH=${PATH:+$PATH:}/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin  # login PATH first
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
export RESULTS_DIR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue4/runner
state=$RESULTS_DIR/babysit
leg=$(cat "$state/current" 2>/dev/null || echo 0)
index=""
for i in 0 1; do
  if [[ "$(cat "$state/legs/$leg/job.$i" 2>/dev/null)" == "${SLURM_JOB_ID:-}" ]]; then index=$i; fi
done
if [[ -z "$index" ]]; then
  echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} is not a recorded candidate of leg $leg in $state: exiting"
  exit 1
fi
export BABYSIT_ENABLED=1 BABYSIT_LEG=$leg BABYSIT_INDEX=$index
# shellcheck source=/dev/null
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_enter
rc=$?
if (( rc == 10 )); then
  echo "[$(date -Is)] babysit: the sibling candidate won; job $SLURM_JOB_ID exits"
  exit 0
fi
(( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed with status $rc"; exit 1; }
echo "[$(date -Is)] babysit: runner job $SLURM_JOB_ID won leg $leg (account ${SLURM_JOB_ACCOUNT:-?}) on $(hostname)"
for name in $(compgen -e); do  # nothing secret-like reaches the runner or, through it, the builds
  case "${name^^}" in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;;
  esac
done
export PYTHONDONTWRITEBYTECODE=1
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue4
# exec keeps the helper's payload lock (fd 8) for the job's lifetime: a second runner leg can never overlap
exec /usr/bin/python3 -I -B run_queue.py run
