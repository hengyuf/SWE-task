#!/bin/bash
# glm-rollouts-tasksall (2026-11-08): one campaign leg in a CPU job = router.py (127.0.0.1, spreads the leg's GLM-5.3
# calls over the RUNNING glm53-serve-ta-* GPU jobs) + driver.py (ATP campaign; OpenCode runs in OpenSandbox, the model on
# our vLLM servers). Submitted only through submit.sh (schedule helper babysit pair); the losing candidate exits at once.
# Env from submit.sh: GR_CAMPAIGN (a campaign of tasks.json), RESULTS_DIR, EXP_NAME.
# Install changes with write + mv, never edit in place (running jobs read this file).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/logs/%x-%j.out
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
for name in $(compgen -e); do  # the code reads every credential from its file when it needs it
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*) unset "$name" 2>/dev/null || true ;; esac
done
unset SYNTH_STUDENT_BASE_URL GLM53_BASE_URL
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_rc=0
babysit_enter || babysit_rc=$?
[[ $babysit_rc == 10 ]] && { echo "babysit: sibling candidate won; exiting"; exit 0; }
[[ $babysit_rc == 0 ]] || { echo "babysit_enter failed rc=$babysit_rc"; exit 1; }
: "${GR_CAMPAIGN:?}" "${GR_JOB_PREFIX:=glm53-serve-ta}"
GR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?} account ${SLURM_JOB_ACCOUNT:-?}: campaign $GR_CAMPAIGN, backends $GR_JOB_PREFIX-*"

router_pid=""
for try in 1 2 3; do
  port=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  static=()
  for url in ${GR_BACKENDS:-}; do static+=(--backend "$url"); done  # optional fixed backends
  $PY -B "$GR/router.py" --port "$port" --job-prefix "$GR_JOB_PREFIX" --log "$GR/logs/router-$EXP_NAME-$SLURM_JOB_ID.jsonl" \
    ${static[@]+"${static[@]}"} > "$GR/logs/router-$EXP_NAME-$SLURM_JOB_ID.out" 2>&1 &
  router_pid=$!
  for _ in $(seq 30); do
    curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 1
  done
  if curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
  kill "$router_pid" 2>/dev/null; router_pid=""
done
[[ -n "$router_pid" ]] || { echo "router did not start"; exit 1; }
trap 'kill "$router_pid" 2>/dev/null' EXIT
echo "[$(date -Is)] router pid $router_pid on http://127.0.0.1:$port/v1 ($(curl -s -m 5 http://127.0.0.1:$port/health))"

cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
$PY -B "$GR/driver.py" --campaign "$GR_CAMPAIGN" --base-url "http://127.0.0.1:$port/v1"
rc=$?
echo "[$(date -Is)] driver exit $rc"
exit $rc
