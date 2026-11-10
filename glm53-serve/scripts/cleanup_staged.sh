#!/usr/bin/env bash
# Queue one cleanup_staged.sbatch per node (schedule-helper babysit pair, batch_short, pinned with --nodelist).
#   bash scripts/cleanup_staged.sh pool0-00786 pool0-00791
# Results: logs/cleanup/glm53-cleanup-<node>-<job>.out
# batch_short allows 20 queued jobs per user and every node costs 2 (one per account) until its node frees up, so at
# most 3 nodes per call (MAX_NODES overrides); otherwise this blocks your other batch_short submissions for hours.
# Nodes that still held a copy on 2026-11-07 (from runs before the rank script deleted its copy at exit):
#   pool0-00786 pool0-00791 pool0-00835 pool0-00837 pool0-01560 pool0-01654 pool0-01664 pool0-01731 pool0-01758
#   pool0-01891 pool0-01912 pool0-01916 (/raid/scratch/hengyuf-glm53) and pool0-00082 pool0-01472 (root-glm53)
set -euo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
(( $# <= ${MAX_NODES:-3} )) || { echo "refusing $# nodes (> MAX_NODES=${MAX_NODES:-3}); see the header" >&2; exit 2; }
mkdir -p "$G/logs/cleanup"
for node in "$@"; do
  export RESULTS_DIR="$G/runs/cleanup/$node" BATCH_SCRIPT="$G/scripts/cleanup_staged.sbatch" EXP_NAME="glm53-cleanup-$node"
  mkdir -p "$RESULTS_DIR"
  ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
        babysit_submit --partition="${PARTITION:-batch_short}" --nodelist="$node")
  echo "$node $ids"
done
