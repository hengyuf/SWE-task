#!/bin/bash
# One step of the Kimi-SFT corpus hand-off to HSG (2026-10-23), as a small CPU job: the Claude sandbox cannot resolve
# or reach HSG, and Data Mover copies are Slurm jobs, so both run from here. Usage (standalone sbatch, two held
# candidates through hengyuf's dual-account helper, by hand):
#   sbatch --parsable --hold --export=NONE --account=<acct> --job-name=sft-hsg-<action> --comment=cc-<token>-L<leg>-<i>
#          hsg_step.sbatch <action>
# <action> names actions/<action>.sh (a fixed, reviewed script); helper state lives in state/<action>/babysit/.
# The job finds its leg and index in that state by its own job id (--export=NONE carries no variables).
#SBATCH --partition=cpu_datamover
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer/logs/%x-%j.out
set -uo pipefail
umask 077
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin:/home/svc-datamover/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
X=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer
action="${1:-}"
[[ "$action" =~ ^[a-z0-9][a-z0-9-]*$ && -f "$X/actions/$action.sh" ]] || { echo "unknown action '$action'"; exit 2; }
export RESULTS_DIR=$X/state/$action
state=$RESULTS_DIR/babysit
leg=$(cat "$state/current" 2>/dev/null || echo 0)
index=""
for i in 0 1; do
  if [[ "$(cat "$state/legs/$leg/job.$i" 2>/dev/null)" == "${SLURM_JOB_ID:-}" ]]; then index=$i; fi
done
[[ -n "$index" ]] || { echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} is not a candidate of $action leg $leg"; exit 1; }
export BABYSIT_ENABLED=1 BABYSIT_LEG=$leg BABYSIT_INDEX=$index
# shellcheck source=/dev/null
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_enter
rc=$?
if (( rc == 10 )); then echo "[$(date -Is)] babysit: the sibling won; job $SLURM_JOB_ID exits"; exit 0; fi
(( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed ($rc)"; exit 1; }
echo "[$(date -Is)] $action: job $SLURM_JOB_ID won leg $leg (account ${SLURM_JOB_ACCOUNT:-?}) on $(hostname)"
for name in $(compgen -e); do  # no secret-like variable reaches the action (the HSG helpers read .env themselves)
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|*_PW) unset "$name" 2>/dev/null || true ;; esac
done
out=$X/out/$action
mkdir -p "$out"
cd "$out" || exit 1
bash "$X/actions/$action.sh" "$out"
rc=$?
echo "[$(date -Is)] $action finished rc=$rc"
printf '%s\n' "$rc" > "$out/RC"
exit "$rc"
