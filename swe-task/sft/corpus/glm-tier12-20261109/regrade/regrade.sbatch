#!/bin/bash
# GLM-5.3 tier-1/2 corpus, old-grader re-grade (2026-11-09; DESIGN.md "Old-grader rule"): regrade_glm.py
#   plan (only when plan.json is missing) -> stage -> validate -> regrade -> status
# in ONE CPU job (8 CPUs, 16G, 8 h, --no-requeue). Resumable: a new job skips final results (results.json).
# Submitted ONLY through hengyuf's schedule helper (/lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh): a
# babysit pair, one held candidate per account (nemotron_agents_dev, nemotron_rl_algo); the first to start wins in
# babysit_enter, the other exits at once. Logs: <R>/jobs/logs/%x-%j.out; helper state: <R>/jobs/state.
#
# Sandboxes: regrade_glm.py reads the sandbox credentials from files by NAME, nothing prints them. hengyuf tasks use
# S/secrets/opensandbox.env (Config.load(ATP_ENV); must equal S/atp/.env's CELL3_SANDBOX_API_KEY setup), Baihe's
# tasks use glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env (BAIHE_SANDBOX_API_KEY). Every credential-like
# environment variable is unset after babysit_enter, as in sft_v3.sbatch (*KEY* / *TOKEN* / *SECRET* / *PASSWORD* /
# *CREDENTIAL* / *_PW, plus OPENSANDBOX_*, which would override those files); nothing here reads a credential from the
# environment or calls a model. TMPDIR = the job's private temp dir (removed at exit with this job's
# common.scratch() dir). Concurrency: GLM_REGRADE_CONCURRENCY (default 16) sandboxes at a time.
#
# R=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/regrade
# Submission A, one command (when sbatch may run inside a script, i.e. the Claude Bash sandbox is off):
#   mkdir -p $R/jobs/logs $R/jobs/state && RESULTS_DIR=$R/jobs/state EXP_NAME=glm-regrade BATCH_SCRIPT=$R/regrade.sbatch \
#     bash -c 'source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh \
#     && babysit_submit --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover'
# Submission B, by hand (standalone Slurm calls; the job finds its leg / index by its own id, as sft_v3.sbatch):
#   W=$R/jobs; B=$W/state/babysit; T=<token, e.g. $(date +%s)-glmrg>; mkdir -p $B/legs/1 $W/logs
#   echo $T > $B/token; echo 1 > $B/current; echo cc-$T-L1-0 > $B/legs/1/request.0; echo cc-$T-L1-1 > $B/legs/1/request.1
#   sbatch --parsable --hold --export=NONE --account=nemotron_agents_dev --job-name=glm-regrade --comment=cc-$T-L1-0 \
#     --partition=<p> --output=$W/logs/%x-%j.out $R/regrade.sbatch            # id -> $B/legs/1/job.0
#   sbatch ... --account=nemotron_rl_algo ... --comment=cc-$T-L1-1 ...         # id -> $B/legs/1/job.1
#   echo <id0>,<id1> > $B/legs/1/release.requested; scontrol release <id0>,<id1>; echo <id0>,<id1> > $B/legs/1/released
# A later leg (resume after a timeout or infra results): the same with legs/2 (echo 2 > $B/current), or Submission A
# again (babysit_submit moves to the next leg once both candidates of the current one have ended).
#SBATCH --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/regrade/jobs/logs/%x-%j.out
set -uo pipefail
umask 027
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
R=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/regrade
export RESULTS_DIR=$R/jobs/state
state=$RESULTS_DIR/babysit
if [[ -z "${BABYSIT_LEG:-}" || -z "${BABYSIT_INDEX:-}" ]]; then  # manual pair (--export=NONE): our slot by job id
  leg=$(cat "$state/current" 2>/dev/null || echo 0)
  index=""
  for i in 0 1; do
    if [[ "$(cat "$state/legs/$leg/job.$i" 2>/dev/null)" == "${SLURM_JOB_ID:-}" ]]; then index=$i; fi
  done
  [[ -n "$index" ]] || { echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} is not a candidate of leg $leg"; exit 1; }
  export BABYSIT_ENABLED=1 BABYSIT_LEG=$leg BABYSIT_INDEX=$index
fi
# shellcheck source=/dev/null
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_enter
rc=$?
if (( rc == 10 )); then echo "[$(date -Is)] babysit: the sibling won; job $SLURM_JOB_ID exits"; exit 0; fi
(( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed ($rc)"; exit 1; }
for name in $(compgen -e); do  # wholesale, as sft_v3.sbatch: the sandbox credentials are read from FILES (see above)
  case "${name^^}" in
    OPENSANDBOX_*|*KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|*_PW) unset "$name" 2>/dev/null || true ;;
  esac
done
unset PYTHONPATH PYTHONHOME  # Gym's loader check (task_builder.validate_task) runs a non-isolated python subprocess
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "not inside a Slurm job"; exit 1; }
JOBTMP=$(mktemp -d "/tmp/glm-regrade-${SLURM_JOB_ID}-XXXXXX") || exit 1
SYNTH_SCRATCH="synth-${SLURM_JOB_ID}-$(id -u)"  # synth/common.scratch() of this job: sandbox.py's transfer files
trap 'rm -rf "$JOBTMP" "/tmp/$SYNTH_SCRATCH" "/raid/scratch/$SYNTH_SCRATCH"' EXIT
export TMPDIR=$JOBTMP  # as sft_v3.sbatch: Python / subprocess temporaries stay in this job's private dir
cd "$JOBTMP" || exit 1  # python -I: nothing is imported from the cwd; Gym's loader check runs from here too
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3
TOOL=$R/regrade_glm.py
CONC=${GLM_REGRADE_CONCURRENCY:-16}
echo "[$(date -Is)] glm-regrade: job ${SLURM_JOB_ID:-?} won leg $BABYSIT_LEG (account ${SLURM_JOB_ACCOUNT:-?}) on" \
     "$(hostname), partition ${SLURM_JOB_PARTITION:-?}; concurrency $CONC"
step() { echo "[$(date -Is)] run: $*"; "$@"; local r=$?; echo "[$(date -Is)] exit $r"; return $r; }
if [[ ! -f $R/plan.json ]]; then
  step "$PY" -I -B -X utf8 "$TOOL" plan || exit 3
fi
step "$PY" -I -B -X utf8 "$TOOL" stage || exit 4  # per-task stage failures are recorded (exit 0); nonzero = fatal
# validate: exit 1 = an unexpected error in some task (recorded ones are not errors); its passes then stay "infra"
# in regrade, every other task goes on. 143 = cancelled (SIGTERM, e.g. the time limit): stop.
step "$PY" -I -B -X utf8 "$TOOL" validate --concurrency "$CONC"
v=$?
(( v == 0 || v == 1 )) || exit 5
step "$PY" -I -B -X utf8 "$TOOL" regrade --concurrency "$CONC"
g=$?
step "$PY" -I -B -X utf8 "$TOOL" status --verbose
echo "[$(date -Is)] glm-regrade done (validate exit $v, regrade exit $g): $R/results.json"
(( v == 0 && g == 0 )) || exit 6
exit 0
