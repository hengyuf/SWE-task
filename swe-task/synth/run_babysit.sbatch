#!/bin/bash
# run.sbatch submitted through hengyuf's dual-account helper (hengyuf, 2026-10-22: "尝试给之后所有的job 都配置上这个
# helper"). /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh babysit_submit sends two held candidates of
# this script, on accounts nemotron_agents_dev and nemotron_rl_algo, then releases them. The first one to start wins
# in babysit_enter and cancels its pending sibling; a loser exits 0 at once.
# The helper passes no script arguments, so the command is read from $RESULTS_DIR/babysit/argv (NUL-separated,
# written by synth/queue2/run_queue.py before each submission). RESULTS_DIR is the run dir, synth/runs/<run>.
# Everything else is as in run.sbatch. The helper sets --account and --export=ALL; the submitting environment is
# the runner's scrubbed one.
#SBATCH --job-name=synth
#SBATCH --partition=cpu_datamover
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs/slurm/%x-%j.out
set -uo pipefail
export PATH=${PATH:+$PATH:}/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin  # babysit_enter: scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
# shellcheck source=/dev/null
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_enter
rc=$?
if (( rc == 10 )); then
  echo "[$(date -Is)] babysit: the sibling candidate won; job $SLURM_JOB_ID exits"
  exit 0
fi
(( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed with status $rc"; exit 1; }
echo "[$(date -Is)] babysit: job $SLURM_JOB_ID won leg ${BABYSIT_LEG:-?} (account ${SLURM_JOB_ACCOUNT:-?})"
set -e
mapfile -d '' -t ARGS < "$RESULTS_DIR/babysit/argv"
(( ${#ARGS[@]} > 0 )) || { echo "empty $RESULTS_DIR/babysit/argv"; exit 1; }
# The code reads every credential from a file when it needs it, so drop anything secret-like or session-bound.
for name in $(compgen -e); do
  case "${name^^}" in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;;
  esac
done
unset SYNTH_STUDENT_BASE_URL  # every job starts (or finds) its own student router (student/client.py)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
# exec keeps the helper's payload lock (fd 8) open for the whole job, so an overlapping leg can never start.
exec /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B "${ARGS[@]}"
