#!/bin/bash
# Delete leftover node-local GLM-5.3 weight copies (/raid/scratch/<user>-glm53, /raid/scratch/root-glm53, 704 GB each)
# on ONE node. They were left by runs before 2026-11-07, when serve_glm53_rank.sh started deleting its copy at exit.
# Submitted only by cleanup_staged.sh (schedule-helper babysit pair, pinned with --nodelist). 1 GPU because the GPU
# partitions reject jobs without a GPU request.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/logs/cleanup/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
found=0
for d in "/raid/scratch/${SLURM_JOB_USER:-$USER}-glm53" /raid/scratch/root-glm53; do
  [[ -d "$d" ]] || continue
  found=1
  echo "[$(date -Is)] $(hostname): $(du -sh "$d" 2>/dev/null | cut -f1) in $d"
  if rm -rf "$d"; then echo "[$(date -Is)] $(hostname): removed $d"; else echo "[$(date -Is)] $(hostname): FAILED to remove $d"; fi
done
(( found )) || echo "[$(date -Is)] $(hostname): no GLM-5.3 copy on /raid/scratch"
