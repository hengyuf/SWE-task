#!/usr/bin/env bash
# glm-rollouts-sftv3 (2026-11-07): submit campaign legs through the schedule helper (babysit pair: one candidate per
# account), as kimi-rollouts-v4's submit.sh; every campaign has its own helper state dir under babysit/.
#   bash submit.sh pilot | baihe-pilot              # 1 rollout (v4's pilot tasks), max concurrency 1
#   bash submit.sh hengyuf|baihe GROUP [GROUP ...]   # groups 0-3 of the 124 hengyuf / 96 Baihe tasks, 5 rollouts each
# A rerun of the same pair after its job ended resumes the campaign (next leg). Env knobs:
#   GR_MAX_CONCURRENCY_OVERRIDE  rollouts in flight per campaign, fixed when the campaign is created
#                                (defaults: hengyuf 24 x 4 = 96 on hengyuf's sandbox account, baihe 12 x 4 = 48 on Baihe's)
#   GR_PARTITION  CPU partitions (default cpu_datamover,cpu_interactive,cpu; Slurm starts the job where it fits first)
#   GR_CPUS, GR_MEM, GR_TIME   per-job request (default 4 CPUs, 32G, 23 h)
# Prints "<exp name> <id0>,<id1>". A STOP file in babysit/<campaign>/ refuses further legs of that campaign.
set -euo pipefail
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107
PARTITION="${GR_PARTITION:-cpu_datamover,cpu_interactive,cpu}"
RESOURCES=(--cpus-per-task="${GR_CPUS:-4}" --mem="${GR_MEM:-32G}" --time="${GR_TIME:-23:00:00}")
declare -A DEFAULT_CONCURRENCY=([hengyuf]=24 [baihe]=12)
declare -A GROUPS_OF=([hengyuf]=4 [baihe]=4)
mkdir -p "$GR/logs" "$GR/babysit"
submit_pair() {  # $1 exp name, $2 state dir
  export RESULTS_DIR="$2" BATCH_SCRIPT="$GR/driver.sbatch" EXP_NAME="$1"
  mkdir -p "$RESULTS_DIR"
  [[ ! -e "$RESULTS_DIR/STOP" ]] || { echo "$RESULTS_DIR/STOP exists; not submitting $1" >&2; return 1; }
  ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
        babysit_submit --partition="$PARTITION" "${RESOURCES[@]}")
  echo "$EXP_NAME $ids"
}
case "${1:-}" in
  pilot|baihe-pilot)
    export GR_TASKSET="$1" GR_GROUP=0 GR_MAX_CONCURRENCY=1
    submit_pair "gr-$1" "$GR/babysit/$1" ;;
  hengyuf|baihe)
    taskset="$1"; shift
    [[ $# -gt 0 ]] || { echo "usage: $0 $taskset GROUP [GROUP ...]" >&2; exit 2; }
    for group in "$@"; do
      [[ "$group" =~ ^[0-9]+$ && "$group" -lt "${GROUPS_OF[$taskset]}" ]] || { echo "bad group $group" >&2; exit 2; }
      export GR_TASKSET="$taskset" GR_GROUP="$group"
      export GR_MAX_CONCURRENCY="${GR_MAX_CONCURRENCY_OVERRIDE:-${DEFAULT_CONCURRENCY[$taskset]}}"
      submit_pair "gr-$taskset-g$group" "$GR/babysit/$taskset-g$group"
    done ;;
  *) echo "usage: $0 pilot | baihe-pilot | hengyuf|baihe GROUP [GROUP ...]" >&2; exit 2 ;;
esac
