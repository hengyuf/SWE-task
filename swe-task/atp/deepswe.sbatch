#!/bin/bash
# Supply --account=nemotron_agents_dev or --account=nemotron_rl_algo to sbatch.
#SBATCH --job-name=atp-deepswe
#SBATCH --partition=cpu_interactive
#SBATCH --qos=normal
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue
#SBATCH --output=runs/slurm/deepswe-%j.out
#SBATCH --error=runs/slurm/deepswe-%j.err

set -euo pipefail
umask 077
ATP_APP=${ATP_APP:-${SLURM_SUBMIT_DIR:?}}
ATP_CAMPAIGN=
ATP_SHARD_OFFSET=
ATP_SOURCE=
ATP_RECOVERY_ALLOWANCE=
while (($#)); do
    case "$1" in
        --app) ATP_APP=$2; shift 2 ;;
        --campaign) ATP_CAMPAIGN=$2; shift 2 ;;
        --source) ATP_SOURCE=$2; shift 2 ;;
        --recovery-allowance) ATP_RECOVERY_ALLOWANCE=$2; shift 2 ;;
        --shard-offset) ATP_SHARD_OFFSET=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z $ATP_CAMPAIGN || ! $ATP_SHARD_OFFSET =~ ^(0|5)$ || ${SLURM_NNODES:-0} != 5 ]]; then
    echo 'Usage: sbatch --account=ACCOUNT deepswe.sbatch --campaign PATH --shard-offset 0|5 [--source PATH]' >&2
    exit 2
fi
ATP_SOURCE=${ATP_SOURCE:-$ATP_CAMPAIGN/source}
test -r "$ATP_SOURCE/scripts/run_deepswe.py"
test -r "$ATP_SOURCE/scripts/deepswe_stage.py"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
ATP_STEP_PID=
forward_stop() {
    if [[ -n $ATP_STEP_PID ]]; then
        kill -TERM "$ATP_STEP_PID" 2>/dev/null || true
    fi
}
trap forward_stop TERM INT

srun --ntasks=5 --ntasks-per-node=1 --cpus-per-task=24 --kill-on-bad-exit=0 \
    bash -c '
        set -euo pipefail
        ATP_WORKER_APP=$1
        ATP_WORKER_CAMPAIGN=$2
        ATP_WORKER_OFFSET=$3
        ATP_WORKER_SOURCE=$4
        ATP_WORKER_ALLOWANCE=$5
        ATP_SCRATCH_BASE=/raid/scratch
        if [[ ! -d $ATP_SCRATCH_BASE || ! -w $ATP_SCRATCH_BASE ]]; then
            ATP_SCRATCH_BASE=/tmp
        fi
        ATP_WORKER_SCRATCH=$ATP_SCRATCH_BASE/atp-deepswe-${SLURM_JOB_ID}
        mkdir -p "$ATP_WORKER_SCRATCH/scripts"
        cp "$ATP_WORKER_SOURCE/scripts/deepswe_stage.py" "$ATP_WORKER_SOURCE/scripts/deepswe_source.py" \
            "$ATP_WORKER_SOURCE/scripts/deepswe_credentials.py" \
            "$ATP_WORKER_SCRATCH/scripts/"
        export ATP_DEEPSWE_SOURCE=$ATP_WORKER_SOURCE
        cd "$ATP_WORKER_SCRATCH"
        /usr/bin/python3 scripts/deepswe_stage.py --app "$ATP_WORKER_APP" --scratch "$ATP_WORKER_SCRATCH"
        export ATP_DEEPSWE_SCRATCH=$ATP_WORKER_SCRATCH
        ATP_RECOVERY_ARGS=()
        if [[ -n $ATP_WORKER_ALLOWANCE ]]; then
            ATP_RECOVERY_ARGS=(--recovery-allowance "$ATP_WORKER_ALLOWANCE")
        fi
        exec "$ATP_WORKER_SCRATCH/controller/bin/python" "$ATP_WORKER_SCRATCH/scripts/run_deepswe.py" \
            worker --app "$ATP_WORKER_APP" --campaign "$ATP_WORKER_CAMPAIGN" \
            --source "$ATP_WORKER_SOURCE" --shard-id "$((ATP_WORKER_OFFSET + SLURM_PROCID))" "${ATP_RECOVERY_ARGS[@]}"
    ' bash "$ATP_APP" "$ATP_CAMPAIGN" "$ATP_SHARD_OFFSET" "$ATP_SOURCE" "$ATP_RECOVERY_ALLOWANCE" &
ATP_STEP_PID=$!
ATP_STEP_STATUS=0
wait "$ATP_STEP_PID" || ATP_STEP_STATUS=$?
# A trapped advance warning interrupts wait before worker cleanup finishes.
if kill -0 "$ATP_STEP_PID" 2>/dev/null; then
    wait "$ATP_STEP_PID" || ATP_STEP_STATUS=$?
fi
exit "$ATP_STEP_STATUS"
