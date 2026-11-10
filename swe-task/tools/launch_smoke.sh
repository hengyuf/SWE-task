#!/usr/bin/env bash
# Submit Baihe's smoke sequence (synth/README.md) as one Slurm dependency chain, with the local student (sc14):
#   tests -> smoke s1 s3 s4 -> smoke s2                (run smoke1)
#   tests -> failure_analyzer (3 val tasks) -> build   (run val1)
# Each step is one CPU job of synth/run.sbatch. Jobs are submitted with a scrubbed environment so nothing from the
# submitting shell (session tokens, keys) reaches them; the pipeline reads its credentials from files.
# Partition: unless --partition is given, tools/pick_partition.py test-submits (nothing is submitted) a 12 h job of
# this shape to cpu_datamover, cpu_dataprocessing and cpu_interactive once, and the whole chain goes to the one
# with the earliest estimated start (ties: more idle nodes, then that order).
# Usage: tools/launch_smoke.sh [--smoke-run smoke1] [--val-run val1] [--partition P | --partitions a,b,c] [--dry-run]
set -euo pipefail
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
SMOKE_RUN=smoke1 VAL_RUN=val1 DRY=0 PARTITION="" PARTITIONS=cpu_datamover,cpu_dataprocessing,cpu_interactive
while (($#)); do
  case "$1" in
    --smoke-run) SMOKE_RUN=$2; shift 2 ;;
    --val-run) VAL_RUN=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --partitions) PARTITIONS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
TASKS=abs-stepped-slices,fastapi-implicit-head-options,valibot-recursive-schema-composition

clean() {  # scrubbed environment; SLURM_CONF because login nodes run no slurmd (fleet.py does the same)
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" LANG=C.UTF-8 \
    SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf}" \
    PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin "$@"
}
if [[ -z "$PARTITION" ]]; then
  PICK=$(cd "$S/synth" && clean /usr/bin/python3 -I -B "$S/tools/pick_partition.py" --partitions "$PARTITIONS" \
         -- --time=12:00:00 --job-name=synth-smoke "$S/synth/run.sbatch" smoke.py --run "$SMOKE_RUN" s1)
  PARTITION=$(printf '%s' "$PICK" | /usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin)["partition"])')
  echo "partition probe: $PICK" >&2
fi

submit() {  # submit <name> <time> <dependency or -> <run.sbatch args...>; prints the job id
  local name=$1 walltime=$2 dep=$3; shift 3
  local argv=(sbatch --parsable --partition="$PARTITION" --job-name="$name" --time="$walltime")
  [[ "$dep" != - ]] && argv+=(--dependency="$dep" --kill-on-invalid-dep=yes)
  argv+=("$S/synth/run.sbatch" "$@")
  if ((DRY)); then echo "DRY: ${argv[*]}" >&2; echo "DRY-$name"; return; fi
  (cd "$S/synth" && clean "${argv[@]}")
}

tests=$(submit synth-tests 01:00:00 - -m unittest discover -s tests -v)
smoke=$(submit synth-smoke 03:00:00 "afterok:$tests" smoke.py --run "$SMOKE_RUN" s1 s3 s4)
s2=$(submit synth-s2 12:00:00 "afterany:$smoke" smoke.py --run "$SMOKE_RUN" s2)
fa=$(submit synth-fa 12:00:00 "afterok:$tests" failure_analyzer.py --run "$VAL_RUN" --tasks "$TASKS")
build=$(submit synth-build 12:00:00 "afterany:$fa" task_builder.py build --run "$VAL_RUN" --task abs-stepped-slices \
  --repo deepnoodle-ai/risor --candidates 2 --branches 2 --rounds 1)

record="$S/synth/runs/launch-$(date +%Y%m%dT%H%M%S).json"
printf '{"submitted_at": "%s", "partition": "'"$PARTITION"'", "smoke_run": "%s", "val_run": "%s", "jobs": {"tests": "%s", "smoke_s1_s3_s4": "%s", "smoke_s2": "%s", "failure_analyzer": "%s", "build": "%s"}}\n' \
  "$(date -Is)" "$SMOKE_RUN" "$VAL_RUN" "$tests" "$smoke" "$s2" "$fa" "$build" | tee "$record"
((DRY)) && rm -f "$record"
exit 0
