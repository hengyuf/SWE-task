#!/usr/bin/env bash
# Fleet maintainer daemon for the persistent student workers: `fleet.py daemon` on a few CPUs for up to 7 days.
# Submitted by `fleet.py start-daemon` (scrubbed environment, --export=NONE) with --partition/--time from fleet.json
# daemon_partition / daemon_time, i.e. cpu_datamover for 7 days; the cpu_long directive below is only the default of
# a bare `sbatch fleetd.sbatch`. Continuity: the job queues its successor right away (same name,
# --dependency=singleton, so it starts only once this job has ended, also if this one dies early; `start-daemon
# --successor` ignores this job and daemon jobs no longer pending or running, e.g. COMPLETING); the daemon stops
# 5 min before the walltime. A daemon that exits abnormally (watchdog, crash) is restarted here after 60 s.
# `fleet.py stop` sets the STOP flag and cancels this job and its successor.
#SBATCH --account=nemotron_agents_dev
#SBATCH --partition=cpu_long
#SBATCH --job-name=swe-task-student-fleetd
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
STUDENT=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=/cm/local/apps/python3/bin/python3

end="${SLURM_JOB_END_TIME:-}"
[[ "$end" =~ ^[0-9]+$ ]] || end="$(SLURM_TIME_FORMAT=%s squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)"
[[ "$end" =~ ^[0-9]+$ ]] || end=$(( $(date +%s) + 7 * 86400 ))
until_epoch=$(( end - 300 ))
echo "[$(date -Is)] fleetd job $SLURM_JOB_ID on $(hostname) ($("$PY" --version 2>&1)); daemon until" \
  "$(date -d "@$until_epoch" -Is)"

successor() {  # queue the next daemon job unless one is queued already or the fleet is stopped
  local attempt
  for attempt in 1 2 3; do
    "$PY" -B "$STUDENT/fleet.py" start-daemon --successor && return 0
    echo "[$(date -Is)] successor submission failed (attempt $attempt)"
    sleep 60
  done
  return 1
}

successor
while (( $(date +%s) < until_epoch )); do
  "$PY" -B "$STUDENT/fleet.py" daemon --interval 60 --until "$until_epoch" --wait-lock
  rc=$?
  echo "[$(date -Is)] daemon exited with status $rc"
  [[ -e "$STUDENT/state/STOP" ]] && { echo "STOP flag set: not restarting"; exit 0; }
  (( rc == 0 )) && break
  sleep 60
done
successor  # in case the queued successor was cancelled meanwhile
