#!/usr/bin/env bash
# kimi-rollouts-v5 (2026-10-28): submit Kimi-K3 probe campaign legs through the schedule helper (babysit pairs: one
# candidate per account), as kimi-rollouts-v4's submit.sh; every campaign has its own helper state dir under babysit/.
#   bash submit.sh pilot                         # 1 rollout of driver.PILOT_TASK (campaign "pilot"), 8 h job
#   bash submit.sh official2 GROUP [GROUP ...]   # GROUP 0-3 (official2-tasks.json tasks[g::4]), 23 h jobs
# A rerun of the same pair after its job ended resumes the campaign (next leg). Env knobs:
#   KR_MAX_CONCURRENCY_OVERRIDE  rollouts in flight per campaign, fixed when the campaign is created (default 8:
#                  4 campaigns x 8 = 32 in flight; README "Concurrency")
#   KR_PARTITION   CPU partition(s) (default cpu_interactive,cpu_dataprocessing,cpu_datamover: Slurm starts the job in
#                  whichever frees first; probe with sbatch --test-only first)
#   KR_CPUS, KR_MEM, KR_TIME   per-job request (default 2 CPUs, 16G; a running driver: ~0.25 CPU, 3-6 GB RSS)
# Ordering: the user first asked for this batch AFTER kimi-rollouts-v4 ("这组rollout 跑完之后", 2026-10-28) and this
# script refused until v4 had finished. On 2026-10-29 ~02:00 the user lifted that (verbatim: "v5 可以直接开始跑了"), so
# v5 runs while v4 still runs and the after-v4 check (and KR5_SKIP_V4_CHECK) is gone.
# Prints "<exp name> <id0>,<id1>". A STOP file in babysit/<campaign>/ refuses further legs of that campaign.
set -euo pipefail
KR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
PARTITION="${KR_PARTITION:-cpu_interactive,cpu_dataprocessing,cpu_datamover}"
RESOURCES=(--cpus-per-task="${KR_CPUS:-2}" --mem="${KR_MEM:-16G}")
declare -A DEFAULT_CONCURRENCY=([official2]=8)
declare -A GROUPS_OF=([official2]=4)
submit_pair() {  # $1 exp name, $2 state dir, $3 time limit
  export RESULTS_DIR="$2" BATCH_SCRIPT="$KR/driver.sbatch" EXP_NAME="$1"
  mkdir -p "$RESULTS_DIR"
  ids=$(source "$HELPER" && babysit_submit --partition="$PARTITION" "${RESOURCES[@]}" --time="${KR_TIME:-$3}")
  echo "$EXP_NAME $ids"
}
case "${1:-}" in
  pilot)
    mkdir -p "$KR/logs" "$KR/babysit"
    export KR_TASKSET=pilot KR_GROUP=0 KR_MAX_CONCURRENCY=1
    submit_pair "kr5-pilot" "$KR/babysit/pilot" 08:00:00 ;;
  official2)
    taskset="$1"; shift
    [[ $# -gt 0 ]] || { echo "usage: $0 $taskset GROUP [GROUP ...]" >&2; exit 2; }
    for group in "$@"; do
      [[ "$group" =~ ^[0-9]+$ && "$group" -lt "${GROUPS_OF[$taskset]}" ]] || { echo "bad group $group" >&2; exit 2; }
    done
    mkdir -p "$KR/logs" "$KR/babysit"
    for group in "$@"; do
      export KR_TASKSET="$taskset" KR_GROUP="$group"
      export KR_MAX_CONCURRENCY="${KR_MAX_CONCURRENCY_OVERRIDE:-${DEFAULT_CONCURRENCY[$taskset]}}"
      submit_pair "kr5-$taskset-g$group" "$KR/babysit/$taskset-g$group" 23:00:00
    done ;;
  *) echo "usage: $0 pilot | official2 GROUP [GROUP ...]" >&2; exit 2 ;;
esac
