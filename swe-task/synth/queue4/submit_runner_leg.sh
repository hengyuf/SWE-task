#!/bin/bash
# Submit the next leg of the official4 queue runner (runner.sbatch) the way the dual-account helper's babysit_submit
# does -- two held candidates (nemotron_agents_dev, nemotron_rl_algo), recorded under runner/babysit/legs/<leg>/, then
# released; the first to start wins in babysit_enter -- except with --export=NONE (the helper refuses any --export),
# so the runner job gets the user's login environment (runner.sbatch finds its leg and index by its own job id).
# This is the "manual helper protocol" of queue3/queue4 legs 1-4, written down 2026-11-05 for leg 5.
#   bash submit_runner_leg.sh            # refuses while the previous leg's candidates are still active
set -euo pipefail
export PATH=${PATH:+$PATH:}/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
Q=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue4
STATE=$Q/runner/babysit
NAME=swe-task-queue4-runner
ACCOUNTS=(nemotron_agents_dev nemotron_rl_algo)
exec 9>"$STATE/lock"
flock 9
current=$(cat "$STATE/current")
token=$(cat "$STATE/token")
for i in 0 1; do  # the previous leg must be over (a running runner holds the queue)
  id=$(cat "$STATE/legs/$current/job.$i" 2>/dev/null || true)
  [[ -z "$id" ]] && continue
  st=$(squeue --noheader --jobs="$id" --format='%T' 2>/dev/null || true)
  [[ -z "$st" ]] || { echo "leg $current candidate $id is still $st: not submitting" >&2; exit 1; }
done
leg=$((current + 1))
dir=$STATE/legs/$leg
mkdir -p "$dir"
printf '%s\n' "$leg" > "$STATE/current.tmp" && mv "$STATE/current.tmp" "$STATE/current"
for i in 0 1; do
  comment="cc-$token-L$leg-$i"
  printf '%s\n' "$comment" > "$dir/request.$i"
  out=$(sbatch --parsable --hold --export=NONE --account="${ACCOUNTS[$i]}" --job-name="$NAME" --comment="$comment" \
        "$Q/runner.sbatch")
  id=${out%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "bad sbatch answer for $comment: $out" >&2; exit 1; }
  printf '%s\n' "$id" > "$dir/job.$i"
done
ids="$(cat "$dir/job.0"),$(cat "$dir/job.1")"
printf '%s\n' "$ids" > "$dir/release.requested"
scontrol release "$ids"
printf '%s\n' "$ids" > "$dir/released"
echo "runner leg $leg candidates $ids"
