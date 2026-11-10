#!/usr/bin/env bash
# teacher-deepswe (2026-10-30): submit hub-teacher probe campaign legs through the schedule helper (babysit pairs: one
# candidate per account), as kimi-rollouts-v4's submit.sh; every campaign (model x wave x group) has its own helper
# state dir babysit/<model>-<campaign>/ (a STOP file there refuses further legs of that campaign).
#   bash submit.sh pilot MODEL [MODEL ...]          # 1 rollout of driver.PILOT_TASK per model, 8 h job, 1 in flight
#   bash submit.sh MODEL w1 GROUP [GROUP ...]        # wave 1 (79 tasks): groups 0-2, 23 h jobs
#   bash submit.sh MODEL w2 GROUP [GROUP ...]        # wave 2 (34 tasks): groups 0-1, 23 h jobs
# Rollouts in flight per campaign (fixed when the campaign is created; the cap is 8): kimi 8, glm 8, deepseek 5.
#   DeepSeek-V4-Pro's hub-wide limit is 16M tokens/min (x-ratelimit priority_model) and one running rollout uses ~0.72M
#   tokens/min (TR2 median), so 3 x 8 = 24 in flight would need ~17.4M/min (109%); TR2 already drove it below zero
#   remaining at 16 in flight. 3 x 5 = 15 keeps it at TR2's level (analysis/hub-ratelimits.json).
# MODEL = kimi | glm | deepseek. A rerun of the same pair after its job ended resumes the campaign (next leg).
# Refuses (exit 3) when: the campaign already has results; another TDW_MAX_CAMPAIGNS (default 9) tdw- campaigns are
#   queued or running; a NEW wave-2 campaign would start before one more of the model's wave-1 campaigns finished
#   (results/<model>-w1-g*.json; override TDW_FORCE_W2=1); the driver's --dry-run preflight fails.
# Env knobs:
#   TDW_MAX_CONCURRENCY_OVERRIDE  rollouts in flight per campaign (1-8; default per model above), fixed when the
#                   campaign is created (a resumed campaign keeps its own; watch.py passes it)
#   TDW_PARTITION   CPU partition(s), a comma list of cpu_interactive / cpu_dataprocessing / cpu_datamover (default
#                   all three: Slurm starts the job in whichever frees first; probe with sbatch --test-only first)
#   TDW_CPUS, TDW_MEM, TDW_TIME   per-job request (default 2 CPUs, 16G; 23:00:00, pilot 08:00:00)
# Prints "<exp name> <id0>,<id1>".
set -euo pipefail
TD=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
HELPER_SHA256=62056f42a6368949aa3ad22a7b12079b565666c471a7b6fcf317239f2deb6978
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3
PARTITION="${TDW_PARTITION:-cpu_interactive,cpu_dataprocessing,cpu_datamover}"
RESOURCES=(--cpus-per-task="${TDW_CPUS:-2}" --mem="${TDW_MEM:-16G}")
MAX_CAMPAIGNS="${TDW_MAX_CAMPAIGNS:-9}"
declare -A GROUPS_OF=([w1]=3 [w2]=2)
declare -A DEFAULT_CONCURRENCY=([kimi]=8 [glm]=8 [deepseek]=5)
declare -A TASKSET_OF=([w1]=deepswe-w1 [w2]=deepswe-w2)
[[ "$(sha256sum "$HELPER" | cut -d' ' -f1)" == "$HELPER_SHA256" ]] || { echo "schedule helper changed (sha256)" >&2; exit 2; }
IFS=, read -ra parts <<< "$PARTITION"
for part in "${parts[@]}"; do
  [[ "$part" == cpu_interactive || "$part" == cpu_dataprocessing || "$part" == cpu_datamover ]] || {
    echo "partition $part refused: CPU partitions only (cpu_interactive, cpu_dataprocessing, cpu_datamover)" >&2; exit 2; }
done
mkdir -p "$TD/logs" "$TD/babysit"

active_campaigns() {  # distinct tdw- job names queued or running (a babysit pair shares one name)
  squeue --noheader -u "${USER:-hengyuf}" --format='%j|%T' | awk -F'|' '$1 ~ /^tdw-/ && $2 != "COMPLETED" {print $1}' | sort -u
}

submit_pair() {  # $1 model, $2 campaign, $3 taskset, $4 group, $5 concurrency, $6 time limit
  local model="$1" campaign="$2" exp="tdw-$1-$2" active n_active finished started
  if [[ -e "$TD/results/$model-$campaign.json" ]]; then echo "$exp: results exist (campaign complete); not submitting" >&2; return 3; fi
  [[ -e "$TD/babysit/$model-$campaign/STOP" ]] && { echo "$exp: babysit/$model-$campaign/STOP exists; not submitting" >&2; return 3; }
  active="$(active_campaigns)" || { echo "squeue failed; not submitting $exp" >&2; return 3; }
  if ! grep -qx "$exp" <<< "$active"; then
    n_active=$(grep -c . <<< "$active" || true)
    if (( n_active >= MAX_CAMPAIGNS )); then echo "$exp: $n_active tdw- campaigns already queued/running (max $MAX_CAMPAIGNS)" >&2; return 3; fi
  fi
  if [[ "$campaign" == w2-* && ! -d "$TD/probes/$model/$campaign" && "${TDW_FORCE_W2:-0}" != 1 ]] && ! grep -qx "$exp" <<< "$active"; then
    finished=$(compgen -G "$TD/results/$model-w1-g*.json" | wc -l || true)
    # distinct wave-2 campaigns started: a probes/<model>/w2-g<K> dir OR a queued/running job (counted once each)
    started=$( { for w2 in "$TD"/probes/"$model"/w2-g[0-9]; do [[ -d "$w2" ]] && basename "$w2"; done
                 grep -o "^tdw-$model-w2-g[0-9]$" <<< "$active" | sed "s/^tdw-$model-//"; } | sort -u | grep -c . || true)
    if (( started >= finished )); then
      echo "$exp: $model has $finished finished wave-1 campaign(s) and $started wave-2 campaign(s) started; wave 2 starts only as wave 1 finishes (TDW_FORCE_W2=1 overrides)" >&2
      return 3
    fi
  fi
  (cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth && \
     "$PY" -B "$TD/driver.py" --model "$model" --taskset "$3" --group "$4" --max-concurrency "$5" --dry-run > /dev/null) || {
     echo "$exp: driver.py --dry-run preflight failed; not submitting" >&2; return 3; }
  export TDW_MODEL="$model" TDW_TASKSET="$3" TDW_GROUP="$4" TDW_MAX_CONCURRENCY="$5"
  export RESULTS_DIR="$TD/babysit/$model-$campaign" BATCH_SCRIPT="$TD/driver.sbatch" EXP_NAME="$exp"
  mkdir -p "$RESULTS_DIR" || return 1
  ids=$(source "$HELPER" && babysit_submit --partition="$PARTITION" "${RESOURCES[@]}" --time="${TDW_TIME:-$6}") || {
    echo "$exp: babysit_submit failed (see the helper's message above)" >&2; return 1; }
  echo "$EXP_NAME $ids"
}
# submit_pair runs in a || context (set -e is off inside it): every step above checks its own status
rc=0

valid_model() { [[ "$1" == kimi || "$1" == glm || "$1" == deepseek ]]; }
case "${1:-}" in
  pilot)
    shift
    [[ $# -gt 0 ]] || { echo "usage: $0 pilot MODEL [MODEL ...]" >&2; exit 2; }
    for model in "$@"; do
      valid_model "$model" || { echo "bad model $model" >&2; exit 2; }
      submit_pair "$model" pilot pilot 0 1 08:00:00 || rc=$?
    done ;;
  kimi|glm|deepseek)
    model="$1"; wave="${2:-}"; shift 2 || true
    [[ "$wave" == w1 || "$wave" == w2 ]] || { echo "usage: $0 MODEL w1|w2 GROUP [GROUP ...]" >&2; exit 2; }
    [[ $# -gt 0 ]] || { echo "usage: $0 $model $wave GROUP [GROUP ...]" >&2; exit 2; }
    concurrency="${TDW_MAX_CONCURRENCY_OVERRIDE:-${DEFAULT_CONCURRENCY[$model]}}"
    [[ "$concurrency" =~ ^[1-8]$ ]] || { echo "TDW_MAX_CONCURRENCY_OVERRIDE must be 1-8" >&2; exit 2; }
    for group in "$@"; do
      [[ "$group" =~ ^[0-9]+$ && "$group" -lt "${GROUPS_OF[$wave]}" ]] || { echo "bad group $group for $wave" >&2; exit 2; }
      submit_pair "$model" "$wave-g$group" "${TASKSET_OF[$wave]}" "$group" "$concurrency" 23:00:00 || rc=$?
    done ;;
  *) echo "usage: $0 pilot MODEL [MODEL ...] | MODEL w1|w2 GROUP [GROUP ...]   (MODEL = kimi|glm|deepseek)" >&2; exit 2 ;;
esac
exit $rc
