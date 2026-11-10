#!/usr/bin/env bash
# fleetd.sbatch (the student fleet daemon) submitted through hengyuf's dual-account helper (2026-10-22). It runs the
# same daemon loop for up to 7 days, but does NOT queue a successor itself: synth/queue4/run_queue.py (earlier queue2, queue3) submits a new
# leg if the daemon ends while builds still need serving.
# Workers: fleet.py itself submits them on the fleet.json worker_accounts (hengyuf 2026-11-02: "yes you can use
# rl_algo account"; 2026-10-22 to 11-02 single-account): a new slot's job on the first account, its sibling on the
# other one after pair_delay_min (15) in the queue, and it settles the pairs; not through the helper, whose --comment
# tag would void the workers' idle-reaper exemption. The loop below restarts the daemon after any non-zero exit, e.g.
# status 3 after `touch student/state/RELOAD` (loads the fleet.py on disk; needs a daemon from 2026-11-02 or later),
# or 137 after a SIGKILL of the daemon python; a SIGTERM (status 0) ends the leg instead.
# The helper sets --account, --job-name and --export=ALL. RESULTS_DIR=student/state/fleetd-babysit holds its state.
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
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
STUDENT=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=/cm/local/apps/python3/bin/python3

end="${SLURM_JOB_END_TIME:-}"
[[ "$end" =~ ^[0-9]+$ ]] || end="$(SLURM_TIME_FORMAT=%s squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)"
[[ "$end" =~ ^[0-9]+$ ]] || end=$(( $(date +%s) + 7 * 86400 ))
until_epoch=$(( end - 300 ))
echo "[$(date -Is)] fleetd job $SLURM_JOB_ID (babysit leg ${BABYSIT_LEG:-?}, account ${SLURM_JOB_ACCOUNT:-?}) on" \
  "$(hostname) ($("$PY" --version 2>&1)); daemon until $(date -d "@$until_epoch" -Is)"

# Lease (2026-10-22; queue3 since 2026-10-26, queue4 since 2026-11-02): the queue runner (synth/queue4/run_queue.py) owns fleet.json "target". If its
# heartbeat is more than an hour old (the runner job died and nobody restarted it) and it did not stop cleanly, set
# the target to 0, so the daemon stops replacing 4 h workers for builds that are no longer driven.
lease_watch() {
  while sleep 600; do
    "$PY" -I -B - <<'PYEOF' || true
import json, os, time
from pathlib import Path
beat_path = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue4/heartbeat.json")
fleet_path = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/fleet.json")
try:
    beat = json.loads(beat_path.read_text())
    fleet = json.loads(fleet_path.read_text())
except (OSError, ValueError):
    raise SystemExit(0)
age = time.time() - float(beat.get("at") or 0)
if beat.get("note") == "stopped" or age < 3600 or not fleet.get("target"):
    raise SystemExit(0)
fleet["target"] = 0
tmp = fleet_path.with_name(".fleet.json.lease.%d.tmp" % os.getpid())
tmp.write_text(json.dumps(fleet, indent=2) + "\n")
tmp.replace(fleet_path)
print("[%s] lease: queue4 runner heartbeat %.0f s old: fleet target -> 0" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), age),
      flush=True)
PYEOF
  done
}
lease_watch &
while (( $(date +%s) < until_epoch )); do
  "$PY" -B "$STUDENT/fleet.py" daemon --interval 60 --until "$until_epoch" --wait-lock
  rc=$?
  echo "[$(date -Is)] daemon exited with status $rc"
  [[ -e "$STUDENT/state/STOP" ]] && { echo "STOP flag set: not restarting"; exit 0; }
  (( rc == 0 )) && break
  sleep 60
done
