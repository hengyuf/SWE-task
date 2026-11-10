#!/usr/bin/env bash
# Submit (or resume) one task_builder build as a CPU job of synth/run.sbatch. Every stage is idempotent, so
# resubmitting the same --run/--task continues where the previous job stopped. Defaults are the official setting
# (task_builder: 10 planned candidates, top 5 built, 5 branches, 3 rounds; repo = top eligible top100.jsonl entry).
# Partition: unless --partition is given, tools/pick_partition.py test-submits (sbatch --test-only, nothing is
# submitted) this exact job to cpu_datamover, cpu_dataprocessing and cpu_interactive and takes the earliest
# estimated start (ties: more idle nodes, then that order). The probes are recorded in runs/<run>/launch.jsonl.
# Jobs are submitted with a scrubbed environment (nothing from the submitting shell reaches them).
# Resume = the same command. Without "-- build args", the build args of the run's last launch are reused
# (e.g. "--rounds 1"), so a resume cannot silently change the build; --no-reuse-args uses task_builder's defaults.
# Usage: tools/launch_build.sh --run RUN --task TASK [--job-name NAME] [--time 24:00:00] [--partition P |
#          --partitions a,b,c] [--no-reuse-args] [--dry-run] [-- build args]
#   e.g. tools/launch_build.sh --run official1-bandit --task bandit-incremental-cache-control
set -euo pipefail
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
RUN="" TASK="" NAME="" WALLTIME=24:00:00 PARTITION="" PARTITIONS=cpu_datamover,cpu_dataprocessing,cpu_interactive
DRY=0 EXTRA=() REUSE=1
while (($#)); do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --job-name) NAME=$2; shift 2 ;;
    --time) WALLTIME=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --partitions) PARTITIONS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --no-reuse-args) REUSE=0; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && -n "$TASK" ]] || { echo "usage: $0 --run RUN --task TASK ..." >&2; exit 2; }
[[ -d "$S/synth/cache/deep-swe/tasks/$TASK" ]] || { echo "unknown DeepSWE task $TASK" >&2; exit 2; }
NAME=${NAME:-synth-build-$RUN}
if ((REUSE)) && ((${#EXTRA[@]} == 0)) && [[ -s "$S/synth/runs/$RUN/launch.jsonl" ]]; then
  last=$(tail -n 1 "$S/synth/runs/$RUN/launch.jsonl" | /usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("extra") or "")')
  if [[ -n "$last" ]]; then
    read -r -a EXTRA <<< "$last"
    echo "reusing the build args of the last launch: ${EXTRA[*]}" >&2
  fi
fi
job_args=(--job-name="$NAME" --time="$WALLTIME" "$S/synth/run.sbatch"
          task_builder.py build --run "$RUN" --task "$TASK" "${EXTRA[@]}")
# SLURM_CONF: login nodes run no slurmd, so the client needs the config path (fleet.py does the same).
clean() {
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" LANG=C.UTF-8 \
    SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf}" \
    PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin "$@"
}
if [[ -n "$PARTITION" ]]; then
  pick="{\"partition\": \"$PARTITION\", \"probes\": [], \"chosen_by\": \"--partition\"}"
else
  pick=$(cd "$S/synth" && clean /usr/bin/python3 -I -B "$S/tools/pick_partition.py" --partitions "$PARTITIONS" \
         -- "${job_args[@]}")
  PARTITION=$(printf '%s' "$pick" | /usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin)["partition"])')
fi
argv=(sbatch --parsable --partition="$PARTITION" "${job_args[@]}")
if ((DRY)); then echo "probe: $pick"; echo "DRY: ${argv[*]}"; exit 0; fi
job=$(cd "$S/synth" && clean "${argv[@]}")
mkdir -p "$S/synth/runs/$RUN"
printf '{"submitted_at": "%s", "job_id": "%s", "run": "%s", "task": "%s", "time": "%s", "partition": "%s", "extra": "%s", "log": "%s", "partition_probe": %s}\n' \
  "$(date -Is)" "$job" "$RUN" "$TASK" "$WALLTIME" "$PARTITION" "${EXTRA[*]}" "$S/synth/runs/slurm/$NAME-$job.out" "$pick" \
  | tee -a "$S/synth/runs/$RUN/launch.jsonl"
