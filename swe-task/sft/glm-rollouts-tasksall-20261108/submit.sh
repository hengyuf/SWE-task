#!/usr/bin/env bash
# glm-rollouts-tasksall (2026-11-08): submit one leg of a campaign through the schedule helper (babysit pair: one
# candidate per account). keeper.py calls this for released campaigns; a rerun after a leg ended resumes the campaign.
#   bash submit.sh CAMPAIGN [TAG]     # campaign from tasks.json (e.g. t1-h00); TAG names this leg's babysit state dir
#                                     # babysit/<campaign>/<TAG> (keeper: k01, k02, ...; default m<epoch>)
# Every leg gets a FRESH babysit state dir: a lost sbatch response then cannot wedge the campaign (the helper refuses
# to resubmit an unknown outcome in the same state dir). Only one leg runs at a time: the keeper submits a leg only
# when no gta-<campaign> job exists. babysit/<campaign>/STOP refuses (exit 1). Env knobs:
#   GR_PARTITION  CPU partitions (default cpu_datamover,cpu_interactive,cpu; Slurm starts the job where it fits first)
#   GR_CPUS, GR_MEM, GR_TIME   per-job request (default 4 CPUs, 32G, 23 h)
# Prints "<exp name> <id0>,<id1>".
set -euo pipefail
# never pass a caller's babysit identity or resource-shaping SLURM_* (e.g. the keeper job's) to the helper / sbatch
unset BABYSIT_LEG BABYSIT_INDEX BABYSIT_ENABLED BABYSIT_DEFER_RELEASE SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU \
  SLURM_MEM_PER_GPU SLURM_CPUS_PER_TASK SLURM_CPUS_PER_GPU SLURM_TRES_PER_TASK
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108
PARTITION="${GR_PARTITION:-cpu_datamover,cpu_interactive,cpu}"
RESOURCES=(--cpus-per-task="${GR_CPUS:-4}" --mem="${GR_MEM:-32G}" --time="${GR_TIME:-23:00:00}")
campaign="${1:-}"; tag="${2:-m$(date +%s)}"
[[ "$campaign" =~ ^t[1-4]-[hb][0-9]{2}$ ]] || { echo "usage: $0 CAMPAIGN [TAG] (bad campaign name '$campaign')" >&2; exit 2; }
[[ "$tag" =~ ^[a-z][0-9]+$ ]] || { echo "bad tag $tag" >&2; exit 2; }
/usr/bin/python3 -I -c 'import json,sys; sys.exit(0 if sys.argv[2] in {c["name"] for c in json.load(open(sys.argv[1]))["campaigns"]} else 1)' \
  "$GR/tasks.json" "$campaign" || { echo "$campaign is not in tasks.json" >&2; exit 2; }
[[ ! -e "$GR/babysit/$campaign/STOP" ]] || { echo "$GR/babysit/$campaign/STOP exists; not submitting" >&2; exit 1; }
[[ ! -e "$GR/babysit/$campaign/$tag" ]] || { echo "$GR/babysit/$campaign/$tag exists; use a new tag" >&2; exit 2; }
export GR_CAMPAIGN="$campaign" RESULTS_DIR="$GR/babysit/$campaign/$tag" BATCH_SCRIPT="$GR/driver.sbatch" EXP_NAME="gta-$campaign"
mkdir -p "$GR/logs" "$RESULTS_DIR"
ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
      babysit_submit --partition="$PARTITION" "${RESOURCES[@]}")
echo "$EXP_NAME $ids"
