#!/bin/bash
# ---- r3 babysit vendored copy (2026-10-22) ---------------------------------------------------------------
# Source: /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
#   sha256 1a1809e04e2f0b12aaaa479ae3073370f5259e655f245124bfcd01ceb3fdafc8 (192 lines, the user's helper).
#   2026-10-22 (user-approved) the source itself was fixed: _babysit_state read squeue %T|%R, which prints a
#   pending reason in parentheses ("PENDING|(JobHeldUser)"), so its JobHeldUser/JobHeldAdmin comparisons never
#   matched (recovery after a lost sbatch response always failed closed; a release retry wrote "released" for a
#   pair that was still held). It now reads %T|%r. Previous source: schdule_helper.sh.bak-20261022 (sha
#   6cc144bc152f5c02c4aff87d869319adcf389cf5fd36e0bcda4f922c6a89a6a8).
# This copy is pinned by the r3-network-v1 layer (verify_override.SCHEDULE_HELPER_SHA256 + input-manifest), so
# later edits of the user's file never reach a running chain. The ONLY change below this header (every other
# line is verbatim) is reaper safety: the candidate tag " cc-<token>-L<leg>-<i>" goes INSIDE a JSON-object
# --comment, just before its closing '"}}' (i.e. at the end of the OccupiedIdleGPUsJobReaper "description"
# value), so each candidate's Slurm comment stays one valid JSON object with all three reaper fields. A
# non-JSON comment keeps the original form. request.<i> still stores the exact candidate comment, and
# _babysit_recover_id still compares it verbatim with squeue %k, so recovery matching is unchanged.
#   -    candidate_comment="${comment:+$comment }cc-$token-L$leg-$index"
#   +    if [[ "$comment" == '{'* ]]; then
#   +      [[ "$comment" == *'"}}' ]] || { _babysit_error 'a JSON --comment must end with a string value and "}}'; return 1; }
#   +      candidate_comment="${comment%'"}}'} cc-$token-L$leg-$index\"}}"
#   +    else
#   +      candidate_comment="${comment:+$comment }cc-$token-L$leg-$index"
#   +    fi
# ------------------------------------------------------------------------------------------------------------
# Sourced scheduling helper. The launcher/leg retain training, progress and stop policy.
# Jobs share one Slurm controller; these are two accounts, not two clusters.
_babysit_error() { printf 'babysit: %s\n' "$*" >&2; return 1; }
_babysit_write() { printf '%s\n' "$2" > "$1.tmp" && mv -- "$1.tmp" "$1"; }
_babysit_read() { if [[ -f "$1" ]]; then cat -- "$1"; else printf '%s\n' "${2:-}"; fi; }
_babysit_lock() { exec 9>"$1/lock" && flock 9; }
_babysit_unlock() { flock -u 9; exec 9>&-; }

_babysit_state() {
  local output
  [[ "$1" =~ ^[0-9]+$ ]] || { _babysit_error "invalid candidate ID: $1"; return 1; }
  if output="$(LC_ALL=C squeue --noheader --jobs="$1" --format='%T|%r' 2>&1)"; then
    printf '%s\n' "$output"
  else
    case "$output" in
      'slurm_load_jobs error: Invalid job id specified'|'squeue: error: Invalid job id specified') ;;
      *) _babysit_error "cannot query candidate $1: $output"; return 1 ;;
    esac
  fi
}

_babysit_active() {
  case "${1%%|*}" in
    ''|COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED) return 1 ;;
    *) return 0 ;;
  esac
}

# Called only while holding the run lock. Unknown submission outcomes fail closed.
_babysit_recover_id() {
  local dir="$1" index="$2" comment="$3" account="$4" output id actual state actual_account actual_comment found=''
  output="$(squeue --noheader --user="${USER}" --name="${EXP_NAME:?}" --format='%i|%j|%T|%a|%k')" || return 1
  while IFS='|' read -r id actual state actual_account actual_comment; do
    [[ "$actual" == "$EXP_NAME" && "$actual_account" == "$account" && "$actual_comment" == "$comment" ]] || continue
    [[ "$id" =~ ^[0-9]+$ && "$state" == PENDING && -z "$found" ]] || {
      _babysit_error "ambiguous candidate for $comment; inspect held jobs"; return 1;
    }
    found="$id"
  done <<< "$output"
  [[ -n "$found" ]] || { _babysit_error "submission outcome unknown for $comment; refusing duplicate"; return 1; }
  state="$(_babysit_state "$found")" || return 1
  [[ "$state" == 'PENDING|JobHeldUser' ]] || { _babysit_error "candidate $found is not user-held; inspect before recovery"; return 1; }
  _babysit_write "$dir/job.$index" "$found"
}

babysit_submit() (
  # A subshell closes its run-lock descriptor on every error; no environment is saved.
  # Successor submission must not retain the running parent's payload lock.
  exec 8>&-
  local state="${RESULTS_DIR:?}/babysit" leg="${BABYSIT_LEG:-}" current token dir index id output name query previous comment='' candidate_comment
  local -a args=() accounts=(nemotron_agents_dev nemotron_rl_algo)
  [[ -f "${BATCH_SCRIPT:?}" ]] || { _babysit_error 'batch script missing'; return 1; }
  : "${EXP_NAME:?}"
  while (( $# )); do
    output="$1"; shift
    case "$output" in
      --account=*|--job-name=*|--parsable) ;;
      --comment=*) comment="${output#*=}" ;;
      --comment)
        (( $# )) || { _babysit_error '--comment requires a value'; return 1; }
        comment="$1"; shift ;;
      --hold|--export*|--array*|--wrap*) _babysit_error "helper owns option $output"; return 1 ;;
      *) args+=("$output") ;;
    esac
  done
  mkdir -p "$state/legs" || return 1
  _babysit_lock "$state" || return 1
  for name in STOP DONE STALLED; do
    [[ ! -e "${RESULTS_DIR}/$name" ]] || { _babysit_error "$name prevents submission"; return 1; }
  done
  if [[ -n "${NRL_MAX_STEPS:-}" && -d "${CHECKPOINT_DIR:-${RESULTS_DIR}}/step_${NRL_MAX_STEPS}" ]]; then
    _babysit_error "step_${NRL_MAX_STEPS} present: training complete"; return 1
  fi
  token="$(_babysit_read "$state/token")" || return 1
  if [[ -z "$token" ]]; then
    token="$(date +%s)-$$"; _babysit_write "$state/token" "$token" || return 1
  fi
  current="$(_babysit_read "$state/current" 0)" || return 1
  [[ "$current" =~ ^(0|[1-9][0-9]*)$ ]] || { _babysit_error 'invalid current leg'; return 1; }
  if [[ -z "$leg" ]]; then
    leg="$current"
    if (( leg == 0 )); then
      leg=1
    elif [[ -f "$state/legs/$leg/released" ]]; then
      local active=0
      for index in 0 1; do
        id="$(_babysit_read "$state/legs/$leg/job.$index")" || return 1
        query="$(_babysit_state "$id")" || return 1
        if _babysit_active "$query"; then active=1; fi
      done
      if (( active == 0 )); then
        # A cancelled successor must not permit a fresh chain over its still-running parent.
        previous="$(_babysit_read "$state/legs/$((leg - 1))/winner")" || return 1
        if [[ -n "$previous" ]]; then
          query="$(_babysit_state "$previous")" || return 1
          if _babysit_active "$query"; then _babysit_error "predecessor $previous still active"; return 1; fi
        fi
        leg=$((leg + 1))
      fi
    fi
  fi
  [[ "$leg" =~ ^[1-9][0-9]*$ ]] || { _babysit_error 'BABYSIT_LEG must be positive'; return 1; }
  (( leg >= current && leg <= current + 1 )) || { _babysit_error 'BABYSIT_LEG must be the current or next leg'; return 1; }
  dir="$state/legs/$leg"
  mkdir -p "$dir" || return 1
  if (( leg > current )); then _babysit_write "$state/current" "$leg" || return 1; fi
  for index in 0 1; do
    [[ ! -f "$dir/job.$index" ]] || continue
    # Keep the experiment job name visible to the existing validation watcher.
    if [[ "$comment" == '{'* ]]; then
      [[ "$comment" == *'"}}' ]] || { _babysit_error 'a JSON --comment must end with a string value and "}}'; return 1; }
      candidate_comment="${comment%'"}}'} cc-$token-L$leg-$index\"}}"
    else
      candidate_comment="${comment:+$comment }cc-$token-L$leg-$index"
    fi
    if [[ -f "$dir/request.$index" ]]; then
      candidate_comment="$(_babysit_read "$dir/request.$index")" || return 1
      _babysit_recover_id "$dir" "$index" "$candidate_comment" "${accounts[$index]}" || return 1
    else
      _babysit_write "$dir/request.$index" "$candidate_comment" || return 1
      if ! output="$(BABYSIT_ENABLED=1 BABYSIT_LEG="$leg" BABYSIT_INDEX="$index" \
          sbatch "${args[@]}" --parsable --hold --export=ALL --account="${accounts[$index]}" --job-name="$EXP_NAME" --comment="$candidate_comment" "$BATCH_SCRIPT")"; then
        _babysit_error "submission failed or response lost for $candidate_comment; retry reconciles held IDs"; return 1
      fi
      id="${output%%;*}"
      [[ "$id" =~ ^[0-9]+$ ]] || { _babysit_error "invalid response for $candidate_comment"; return 1; }
      _babysit_write "$dir/job.$index" "$id" || return 1
    fi
  done
  id="$(cat "$dir/job.0"),$(cat "$dir/job.1")"
  # Cutover only: keep the recorded pair held until the replaced pending job exits.
  # A subsequent ordinary submit releases this same pair; successors do not inherit it.
  if [[ "${BABYSIT_DEFER_RELEASE:-0}" != 1 && ! -f "$dir/released" ]]; then
    if [[ -f "$dir/release.requested" ]]; then
      for index in 0 1; do
        output="$(cat "$dir/job.$index")" || return 1
        query="$(_babysit_state "$output")" || return 1
        case "$query" in
          'PENDING|JobHeldUser') scontrol release "$output" >&2 || return 1 ;;
          'PENDING|JobHeldAdmin') _babysit_error "candidate $output held by administrator"; return 1 ;;
          PENDING\|*|RUNNING\|*|CONFIGURING\|*|COMPLETING\|*) ;;
          *) [[ -f "$dir/winner" ]] || { _babysit_error "candidate $output disappeared during release"; return 1; } ;;
        esac
      done
    else
      _babysit_write "$dir/release.requested" "$id" || return 1
      scontrol release "$id" >&2 || return 1
    fi
    _babysit_write "$dir/released" "$id" || return 1
  fi
  printf 'babysit: leg %s candidates %s\n' "$leg" "$id" >&2
  printf '%s\n' "$id"
)

babysit_enter() {
  local state="${RESULTS_DIR:?}/babysit" leg="${BABYSIT_LEG:-}" index="${BABYSIT_INDEX:-}" dir id winner sibling
  [[ "$leg" =~ ^[1-9][0-9]*$ && "$index" =~ ^[01]$ ]] || { _babysit_error 'invalid candidate identity'; return 1; }
  id="${SLURM_JOB_ID:-}"; dir="$state/legs/$leg"
  [[ -d "$dir" ]] || { _babysit_error 'candidate manifest missing'; return 1; }
  _babysit_lock "$state" || return 1
  if [[ ! "$id" =~ ^[0-9]+$ || "$id" != "$(_babysit_read "$dir/job.$index")" ]]; then
    _babysit_unlock; _babysit_error 'job ID does not match candidate'; return 1
  fi
  winner="$(_babysit_read "$dir/winner")" || { _babysit_unlock; return 1; }
  if [[ -n "$winner" && "$winner" != "$id" ]]; then _babysit_unlock; return 10; fi
  # One run-wide payload lock also fails closed on accidental overlapping leg entry.
  if ! { exec 8>"$state/payload.lock" && flock -n 8; }; then
    exec 8>&-; _babysit_unlock; return 10
  fi
  if [[ -z "$winner" ]]; then
    if ! _babysit_write "$dir/winner" "$id"; then exec 8>&-; _babysit_unlock; return 1; fi
  fi
  sibling="$(_babysit_read "$dir/job.$((1 - index))")" || { exec 8>&-; _babysit_unlock; return 1; }
  if [[ "$sibling" =~ ^[0-9]+$ ]]; then
    scancel --state=PENDING "$sibling" >&2 || printf 'babysit: pending sibling cancellation failed; gate remains closed\n' >&2
  fi
  _babysit_unlock
  return 0
}

babysit_cancel_pending() (
  local state="${RESULTS_DIR:?}/babysit" leg="${BABYSIT_LEG:-}" next index file id rc=0
  [[ -d "$state" ]] || return 0
  _babysit_lock "$state" || return 1
  [[ -n "$leg" ]] || leg="$(_babysit_read "$state/current" 0)"
  [[ "$leg" =~ ^[1-9][0-9]*$ ]] || { _babysit_error 'invalid cancellation leg'; return 1; }
  for next in "$leg" "$((leg + 1))"; do
    for index in 0 1; do
      file="$state/legs/$next/job.$index"
      [[ -f "$file" ]] || continue
      id="$(cat "$file")" || return 1
      [[ "$id" =~ ^[0-9]+$ ]] || { _babysit_error 'invalid recorded cancellation ID'; return 1; }
      scancel --state=PENDING "$id" >&2 || rc=1
    done
  done
  return "$rc"
)