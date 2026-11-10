#!/usr/bin/env bash
# teacher-rollouts-v2 (2026-10-25): submit hub-teacher probe campaigns through the schedule helper (babysit pairs: one
# candidate per account), on cpu_dataprocessing; every pair has its own helper state dir (RESULTS_DIR) under babysit/.
#   bash submit.sh pilot [glm|deepseek ...]             # 2 tasks x 1 rollout per model (driver.py PILOT_TASKS), 8 h jobs
#   bash submit.sh full MODEL GROUP [GROUP ...]          # full batch: tasks[GROUP::8] x 5 rollouts, 12 h jobs
# A rerun of the same pair after its job ended resumes the campaign (next leg). Env knobs:
#   TR_HUB_KEY_NAMES   ordered hub key NAMES for the leg (default NVAPI_KEY_NEW_2,NVAPI_KEY_NEW_3; user, 2026-10-26 ~00:45:
#                      NEW_2 first, NEW_3 once NEW_2 is used up); driver.py uses the first with >= TR_MIN_REMAINING
#                      left and records the leg's key in legs.jsonl and the results file. NVAPI_KEY_NEW_1 (every leg
#                      submitted before 2026-10-26 01:00; it also funds the r5 RL judge) is refused for new legs.
#   TR_MIN_REMAINING   the driver refuses to start/resume a campaign on a key with less left (default 330 = ~10% of
#                      NVAPI_KEY_NEW_1's 3,267 USD at the start of this work)
#   TR_MAX_CONCURRENCY_OVERRIDE  rollouts in flight per campaign of a full-batch group, fixed when the campaign is
#                      created (default: glm 6, deepseek 2, i.e. 48 GLM and 16 DeepSeek rollouts over 8 groups each;
#                      keeps our share of each model's hub-wide token limit low; GLM-5.3 is also the r4 RL judge)
#   TR_PARTITION       CPU partition(s) (default cpu_dataprocessing; a comma list lets Slurm start the job in whichever
#                      partition frees first; cpu_datamover/cpu_interactive share another pool)
#   TR_CPUS, TR_MEM    per-job request (default 8 CPUs, 48G as driver.sbatch; measured: ~0.25 CPU, 3.4-5.5 GB RSS)
# Prints "<exp name> <id0>,<id1>".
set -euo pipefail
TR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
export TR_MIN_REMAINING="${TR_MIN_REMAINING:-330}"
export TR_HUB_KEY_NAMES="${TR_HUB_KEY_NAMES:-NVAPI_KEY_NEW_2,NVAPI_KEY_NEW_3}"  # NAMES only; driver.sbatch passes them on
IFS=, read -ra key_names <<< "$TR_HUB_KEY_NAMES"
[[ ${#key_names[@]} -gt 0 ]] || { echo "TR_HUB_KEY_NAMES is empty" >&2; exit 2; }
for key_name in "${key_names[@]}"; do
  [[ "$key_name" == NVAPI_KEY_NEW_2 || "$key_name" == NVAPI_KEY_NEW_3 ]] || {
    echo "TR_HUB_KEY_NAMES: $key_name is not allowed for new legs (NVAPI_KEY_NEW_2 / NVAPI_KEY_NEW_3 only)" >&2; exit 2; }
done
grep -q 'key-names "$hub_key_names"' "$TR/driver.sbatch" || { echo "driver.sbatch predates TR_HUB_KEY_NAMES" >&2; exit 2; }
PARTITION="${TR_PARTITION:-cpu_dataprocessing}"  # may be a list, e.g. cpu_dataprocessing,cpu_datamover
RESOURCES=(--cpus-per-task="${TR_CPUS:-8}" --mem="${TR_MEM:-48G}")  # a running driver: ~0.25 CPU, 3.4-5.5 GB RSS
declare -A DEFAULT_CONCURRENCY=([glm]=6 [deepseek]=2)
mkdir -p "$TR/logs" "$TR/babysit"
submit_pair() {  # $1 exp name, $2 state dir, $3 time limit
  export RESULTS_DIR="$2" BATCH_SCRIPT="$TR/driver.sbatch" EXP_NAME="$1"
  mkdir -p "$RESULTS_DIR"
  ids=$(source "$HELPER" && babysit_submit --partition="$PARTITION" "${RESOURCES[@]}" --time="$3")
  echo "$EXP_NAME $ids"
}
case "${1:-}" in
  pilot)
    shift
    models=("$@"); [[ ${#models[@]} -gt 0 ]] || models=(glm deepseek)
    for model in "${models[@]}"; do
      [[ "$model" == glm || "$model" == deepseek ]] || { echo "bad model $model" >&2; exit 2; }
      export TR_MODEL="$model" TR_TASKSET=pilot TR_GROUP=0 TR_MAX_CONCURRENCY="${TR_MAX_CONCURRENCY:-32}"
      submit_pair "tr-pilot-$model" "$TR/babysit/pilot-$model" 08:00:00
    done ;;
  full)
    model="${2:?model}"; shift 2
    [[ "$model" == glm || "$model" == deepseek ]] || { echo "bad model $model" >&2; exit 2; }
    [[ $# -gt 0 ]] || { echo "usage: $0 full MODEL GROUP [GROUP ...]" >&2; exit 2; }
    for group in "$@"; do
      [[ "$group" =~ ^[0-7]$ ]] || { echo "bad group $group" >&2; exit 2; }
      export TR_MODEL="$model" TR_TASKSET=full TR_GROUP="$group"
      export TR_MAX_CONCURRENCY="${TR_MAX_CONCURRENCY_OVERRIDE:-${DEFAULT_CONCURRENCY[$model]}}"
      submit_pair "tr-$model-g$group" "$TR/babysit/full-$model-g$group" 12:00:00
    done ;;
  *) echo "usage: $0 pilot [MODEL ...] | full MODEL GROUP [GROUP ...]" >&2; exit 2 ;;
esac
