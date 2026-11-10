#!/bin/bash
# The swe-task dashboard as a small CPU job: server.py (token-protected HTTP on 0.0.0.0 of the node, reached by port
# forwarding) with its collector thread (collect.py every 120 s): it idles between cycles, which are mostly file reads.
# Submitted by launch.sh with a scrubbed environment and --partition (tools/pick_partition.py, among cpu_datamover /
# cpu_dataprocessing / cpu_interactive; the 7-day walltime excludes cpu_interactive, which caps at 1 day), so there is
# no partition line here. Arguments after the script go to server.py and win over the defaults below (launch.sh passes
# --port and, on request, --reuse-token); DASHBOARD_PORT, if set, changes the default port.
# The URL with the token goes to state/server.json (mode 600) only, never to this job's log.
#SBATCH --account=nemotron_agents_dev
#SBATCH --job-name=swe-task-dashboard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --export=NONE
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard/logs/%x-%j.out
set -euo pipefail
# The collector reads nothing secret from the environment (the hub budget comes from tools/hub_budget.py, which reads
# its key from a file), and --export=NONE still loads the login environment: drop anything secret-like it holds.
for name in $(compgen -e); do
  case "${name^^}" in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;;
  esac
done
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
# The collector's squeue/sacct (it also passes this path itself).
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf}"
D=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard
cd "$D"
echo "[$(date -Is)] dashboard job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname)} (partition" \
  "${SLURM_JOB_PARTITION:-?}), $(/usr/bin/python3 --version 2>&1)"
# Dual-account helper (hengyuf 2026-10-22: every job through schdule_helper.sh): when babysit_submit started this job
# (BABYSIT_ENABLED=1, BABYSIT_LEG/INDEX in its environment; helper state in $D/babysit/), only the first candidate to
# start runs the dashboard and cancels its pending sibling; the other exits 0 at once. A launch.sh start (no BABYSIT_*)
# runs as before. DASHBOARD_REUSE_AUTH=1 adds --reuse-token (a name without "TOKEN": the loop above unsets those).
if [[ "${BABYSIT_ENABLED:-}" == 1 ]]; then
  export RESULTS_DIR=$D
  # shellcheck source=/dev/null
  source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
  set +e; babysit_enter; rc=$?; set -e
  if (( rc == 10 )); then echo "[$(date -Is)] babysit: the sibling candidate won; job ${SLURM_JOB_ID:-?} exits"; exit 0; fi
  (( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed with status $rc"; exit 1; }
  echo "[$(date -Is)] babysit: job ${SLURM_JOB_ID:-?} won leg ${BABYSIT_LEG:-?} (account ${SLURM_JOB_ACCOUNT:-?})"
fi
exec /usr/bin/python3 -I -B "$D/server.py" --host 0.0.0.0 --port "${DASHBOARD_PORT:-8765}" --interval 120 \
  ${DASHBOARD_REUSE_AUTH:+--reuse-token} "$@"
