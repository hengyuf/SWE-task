#!/usr/bin/env bash
# kimi-rollouts-v3 (2026-10-26): submit Kimi-K3 probe campaign legs through the schedule helper (babysit pairs: one
# candidate per account), as teacher-rollouts-v2's submit.sh; every campaign has its own helper state dir under babysit/.
#   bash submit.sh pilot                         # 1 rollout of driver.PILOT_TASK, 8 h job
#   bash submit.sh baihe-pilot                   # 1 rollout of driver.BAIHE_PILOT_TASK (Baihe sandbox credential), 8 h
#   bash submit.sh TASKSET GROUP [GROUP ...]     # TASKSET sel50|rest47 (groups 0-4) or baihe (groups 0-7), 12 h jobs
#   bash submit.sh ctrl-old                      # control arm: driver.CTRL_TASKS x 5 with the previous Kimi settings, 8 h
# A rerun of the same pair after its job ended resumes the campaign (next leg). Env knobs:
#   KR_MAX_CONCURRENCY_OVERRIDE  rollouts in flight per campaign, fixed when the campaign is created (defaults below)
#   KR_PARTITION   CPU partition(s) (default cpu_interactive,cpu_dataprocessing,cpu_datamover: Slurm starts the job in
#                  whichever frees first; probe with sbatch --test-only first)
#   KR_CPUS, KR_MEM, KR_TIME   per-job request (default 2 CPUs, 16G; a running driver: ~0.25 CPU, 3-6 GB RSS)
# Prints "<exp name> <id0>,<id1>".
set -euo pipefail
KR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
PARTITION="${KR_PARTITION:-cpu_interactive,cpu_dataprocessing,cpu_datamover}"
RESOURCES=(--cpus-per-task="${KR_CPUS:-2}" --mem="${KR_MEM:-16G}")
declare -A DEFAULT_CONCURRENCY=([sel50]=10 [rest47]=10 [baihe]=4)
declare -A GROUPS_OF=([sel50]=5 [rest47]=5 [baihe]=8)
mkdir -p "$KR/logs" "$KR/babysit"
submit_pair() {  # $1 exp name, $2 state dir, $3 time limit
  export RESULTS_DIR="$2" BATCH_SCRIPT="$KR/driver.sbatch" EXP_NAME="$1"
  mkdir -p "$RESULTS_DIR"
  ids=$(source "$HELPER" && babysit_submit --partition="$PARTITION" "${RESOURCES[@]}" --time="${KR_TIME:-$3}")
  echo "$EXP_NAME $ids"
}
case "${1:-}" in
  pilot|baihe-pilot)
    export KR_TASKSET="$1" KR_GROUP=0 KR_MAX_CONCURRENCY=1
    submit_pair "kr3-$1" "$KR/babysit/$1" 08:00:00 ;;
  ctrl-old)
    export KR_TASKSET="$1" KR_GROUP=0 KR_MAX_CONCURRENCY="${KR_MAX_CONCURRENCY_OVERRIDE:-10}"
    submit_pair "kr3-$1" "$KR/babysit/$1" 08:00:00 ;;
  sel50|rest47|baihe)
    taskset="$1"; shift
    [[ $# -gt 0 ]] || { echo "usage: $0 $taskset GROUP [GROUP ...]" >&2; exit 2; }
    for group in "$@"; do
      [[ "$group" =~ ^[0-9]+$ && "$group" -lt "${GROUPS_OF[$taskset]}" ]] || { echo "bad group $group" >&2; exit 2; }
      export KR_TASKSET="$taskset" KR_GROUP="$group"
      export KR_MAX_CONCURRENCY="${KR_MAX_CONCURRENCY_OVERRIDE:-${DEFAULT_CONCURRENCY[$taskset]}}"
      submit_pair "kr3-$taskset-g$group" "$KR/babysit/$taskset-g$group" 12:00:00
    done ;;
  *) echo "usage: $0 pilot | baihe-pilot | ctrl-old | sel50|rest47|baihe GROUP [GROUP ...]" >&2; exit 2 ;;
esac
