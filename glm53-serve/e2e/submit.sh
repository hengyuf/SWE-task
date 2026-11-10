#!/usr/bin/env bash
# Submit one GLM-5.3 local-endpoint E2E campaign as a babysit pair (agents_dev + rl_algo) on CPU partitions.
#   bash submit.sh <taskset pilot|safe14|group> <base-url> [repeats] [max-concurrency] [group]
# Campaign root: e2e/root, or $E2E_ROOT (one campaign dir per taskset/repeats under <root>/probes/glmlocal/);
# E2E_TAG prefixes the job/babysit name (use one per root). Rerunning the same
# command after the job ended resumes the campaign. Prints "<exp name> <id0>,<id1>".
set -euo pipefail
E=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/e2e
taskset="${1:?taskset}"; base_url="${2:?base url}"; repeats="${3:-1}"; conc="${4:-8}"; group="${5:-0}"
export GLM_ROOT="${E2E_ROOT:-$E/root}" GLM_BASE_URL="$base_url" GLM_TASKSET="$taskset" GLM_GROUP="$group" GLM_REPEATS="$repeats"
export GLM_MAX_CONCURRENCY="$conc" GLM_MIN_MODEL_LEN=262144
name="glmlocal-$taskset-x$repeats"; [[ "$taskset" == group ]] && name="glmlocal-g$group-x$repeats"
name="${E2E_TAG:+$E2E_TAG-}$name"
export RESULTS_DIR="$E/babysit/$name" BATCH_SCRIPT="$E/driver.sbatch" EXP_NAME="$name"
mkdir -p "$RESULTS_DIR" "$GLM_ROOT"
ids=$(source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh && \
      babysit_submit --partition="${E2E_PARTITION:-cpu_interactive,cpu_datamover}" --cpus-per-task=4 --mem=32G --time=12:00:00)
echo "$EXP_NAME $ids"
