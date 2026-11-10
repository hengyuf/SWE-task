#!/bin/bash
# GLM-5.3 tier-1/2 SFT corpus (2026-11-09): tools/sources_glm.py -> tools/build_glm.py -> tools/verify_glm.py
# (-> SHA256SUMS) in ONE CPU job. Submitted ONLY through hengyuf's schedule helper (/lustre/fsw/portfolios/nemotron/
# users/hengyuf/schdule_helper.sh): a babysit pair, one held candidate per account (nemotron_agents_dev,
# nemotron_rl_algo); the first to start wins in babysit_enter, the other exits at once. No credential is needed or read.
# Design: /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/DESIGN.md (rev 2).
#
# C = /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-glm-tier12-20261109 (the clean corpus dir:
#     tools/ + what build writes; the transfer refuses anything outside SHA256SUMS, so NO dry run or state goes there)
# W = /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109 (work dir)
# Logs + helper state: W/scratch/jobs/ (logs/, state-<mode>/). DATA_ROOT (AGA, baked into the blend / receipt /
# verify) = GLMSFT_DATA_ROOT, default /scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/
# sft-glm-tier12-20261109 (DESIGN "Paths and names").
#
# Mode = $1, or GLMSFT_MODE when the helper's babysit_submit submits it (it passes no script arguments):
#   dryrun  output W/scratch/dryrun-job/build (moved to W/scratch/dryrun-job/build-<date> first if a finished one
#           exists): sources_glm.py --limit GLMSFT_LIMIT (default 12 records per batch) or --only-file GLMSFT_ONLY_FILE,
#           (whatever W/regrade/results.json holds at that time; not the corpus dir, so no completeness guard);
#           build_glm.py; verify_glm.py --readme; SHA256SUMS.
#           Writes W/scratch/dryrun-job/RC (exit code).
#   build   output C itself: sources_glm.py (every tier-1/2 final record; refuses while some class B pass has no
#           re-grade result in W/regrade/results.json, i.e. the re-grade is missing or still running, unless
#           GLMSFT_ALLOW_MISSING_REGRADE=1: those passes are then excluded as old_grader_not_regraded), build_glm.py,
#           verify_glm.py --readme, SHA256SUMS. Refuses if C/raw_sft exists.
#   rebuild output C from the EXISTING C/sources (no new selection): build_glm.py + verify --readme + SHA256SUMS.
#           Refuses if C/raw_sft exists or C/sources/glm/selection_report.json is missing.
#   verify  re-run ONLY verify_glm.py --readme + SHA256SUMS on the existing build in C (e.g. after a transient verify
#           failure: verify --readme replaces the earlier result line). Refuses unless C/raw_sft and
#           C/DATA_PREP_COMPLETE.json exist and C/SHA256SUMS does not (a sealed corpus is not re-verified in place).
# DATA_ROOT must be DESIGN's AGA DATA_ROOT for build / rebuild / verify (build_glm.py and verify_glm.py refuse another
# one) unless GLMSFT_ALLOW_OTHER_DATA_ROOT=1. The effective options are printed in the job log header.
#
# Submission A, one command (when sbatch may run inside a script, i.e. the Claude Bash sandbox is off):
#   mkdir -p W/scratch/jobs/logs && GLMSFT_MODE=build RESULTS_DIR=W/scratch/jobs/state-build EXP_NAME=glmsft-build \
#     BATCH_SCRIPT=C/tools/glm_sft.sbatch bash -c 'source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh \
#     && babysit_submit --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover'
#   (dryrun: GLMSFT_MODE=dryrun RESULTS_DIR=W/scratch/jobs/state-dryrun EXP_NAME=glmsft-dryrun; optional GLMSFT_LIMIT,
#    GLMSFT_ONLY_FILE, GLMSFT_DATA_ROOT, GLMSFT_ALLOW_MISSING_REGRADE, GLMSFT_ALLOW_OTHER_DATA_ROOT are passed through
#    the environment: babysit_submit uses --export=ALL, so unset any stale GLMSFT_* in the submitting shell first)
# Submission B, by hand (standalone Slurm calls; the job finds its leg / index by its own id, as xfer/hsg_step.sbatch):
#   J=W/scratch/jobs; B=$J/state-<mode>/babysit; T=<token, e.g. $(date +%s)-glmsft>; mkdir -p $B/legs/1 $J/logs
#   echo $T > $B/token; echo 1 > $B/current; echo cc-$T-L1-0 > $B/legs/1/request.0; echo cc-$T-L1-1 > $B/legs/1/request.1
#   sbatch --parsable --hold --export=NONE --account=nemotron_agents_dev --job-name=glmsft-<mode> --comment=cc-$T-L1-0 \
#     --partition=<p> --output=$J/logs/%x-%j.out C/tools/glm_sft.sbatch <mode>        # id -> $B/legs/1/job.0
#   sbatch ... --account=nemotron_rl_algo ... --comment=cc-$T-L1-1 ...                   # id -> $B/legs/1/job.1
#   echo <id0>,<id1> > $B/legs/1/release.requested; scontrol release <id0>,<id1>; echo <id0>,<id1> > $B/legs/1/released
#   (with --export=NONE, set options as arguments instead of the environment: `dryrun`, `build`, `rebuild`, `verify`;
#    the defaults apply to LIMIT / DATA_ROOT)
#SBATCH --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/scratch/jobs/logs/%x-%j.out
set -uo pipefail
umask 027
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
C=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-glm-tier12-20261109
W=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109
J=$W/scratch/jobs
D=$W/scratch/dryrun-job
MODE="${1:-${GLMSFT_MODE:-}}"
DESIGN_ROOT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/sft-glm-tier12-20261109
DATA_ROOT="${GLMSFT_DATA_ROOT:-$DESIGN_ROOT}"
LIMIT="${GLMSFT_LIMIT:-12}"
ONLY_FILE="${GLMSFT_ONLY_FILE:-}"
ALLOW=()
if [[ "${GLMSFT_ALLOW_MISSING_REGRADE:-0}" == 1 ]]; then ALLOW=(--allow-missing-regrade); fi
ROOTOPT=()
if [[ "${GLMSFT_ALLOW_OTHER_DATA_ROOT:-0}" == 1 ]]; then ROOTOPT=(--allow-other-data-root); fi
case "$MODE" in
  dryrun|build|rebuild|verify) export RESULTS_DIR=$J/state-$MODE ;;
  *) echo "usage: glm_sft.sbatch dryrun|build|rebuild|verify (or GLMSFT_MODE)"; exit 2 ;;
esac
state=$RESULTS_DIR/babysit
if [[ -z "${BABYSIT_LEG:-}" || -z "${BABYSIT_INDEX:-}" ]]; then  # manual pair (--export=NONE): our slot by job id
  leg=$(cat "$state/current" 2>/dev/null || echo 0)
  index=""
  for i in 0 1; do
    if [[ "$(cat "$state/legs/$leg/job.$i" 2>/dev/null)" == "${SLURM_JOB_ID:-}" ]]; then index=$i; fi
  done
  [[ -n "$index" ]] || { echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} is not a candidate of $MODE leg $leg"; exit 1; }
  export BABYSIT_ENABLED=1 BABYSIT_LEG=$leg BABYSIT_INDEX=$index
fi
# shellcheck source=/dev/null
source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh
babysit_enter
rc=$?
if (( rc == 10 )); then echo "[$(date -Is)] babysit: the sibling won; job $SLURM_JOB_ID exits"; exit 0; fi
(( rc == 0 )) || { echo "[$(date -Is)] babysit_enter failed ($rc)"; exit 1; }
for name in $(compgen -e); do  # nothing here needs a credential
  case "${name^^}" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|*_PW) unset "$name" 2>/dev/null || true ;; esac
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
JOBTMP=$(mktemp -d "/tmp/glmsft-${SLURM_JOB_ID:-x}-XXXXXX") || exit 1
trap 'rm -rf "$JOBTMP"' EXIT
export TMPDIR=$JOBTMP  # verify's datasets cache (Arrow copies of the shards) and sim outputs
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
T=$C/tools
NCPU=${SLURM_CPUS_PER_TASK:-16}
NW=$(( NCPU > 8 ? NCPU - 4 : NCPU ))
SCAN=$(( NCPU > 16 ? 16 : NCPU ))  # concurrent http.jsonl readers (Lustre)
SIMW=$(( NCPU > 16 ? 6 : 2 ))      # sim_sft_getitem.py subprocesses (each tokenizes a 524,288-token row)
echo "[$(date -Is)] glm-sft $MODE: job ${SLURM_JOB_ID:-?} won leg $BABYSIT_LEG (account ${SLURM_JOB_ACCOUNT:-?}) on" \
     "$(hostname), partition ${SLURM_JOB_PARTITION:-?}, $NCPU CPUs"
echo "[$(date -Is)] options: DATA_ROOT=$DATA_ROOT ($([[ "$DATA_ROOT" == "$DESIGN_ROOT" ]] && echo "DESIGN's AGA DATA_ROOT" \
     || echo "NOT DESIGN's AGA DATA_ROOT")) allow-other-data-root=${GLMSFT_ALLOW_OTHER_DATA_ROOT:-0}" \
     "allow-missing-regrade=${GLMSFT_ALLOW_MISSING_REGRADE:-0} limit=$LIMIT only-file=${ONLY_FILE:-none}"
step() { echo "[$(date -Is)] run: $*"; "$@"; local r=$?; echo "[$(date -Is)] exit $r"; return $r; }
fail() { [[ "$MODE" == dryrun ]] && echo "$1" > "$D/RC"; exit "$1"; }
cd "$JOBTMP" || exit 1  # python -I: nothing is imported from the cwd
if [[ "$MODE" == dryrun ]]; then
  mkdir -p "$D"
  if [[ -e $D/build ]]; then
    [[ -f $D/RC ]] || { echo "[$(date -Is)] $D/build has no RC (dry run still running?): refusing"; exit 2; }
    mv "$D/build" "$D/build-$(date +%Y%m%d-%H%M%S)" && rm -f "$D/RC" || exit 2
  fi
  OUT=$D/build
  mkdir -p "$OUT"
  if [[ -n "$ONLY_FILE" ]]; then SEL=(--only-file "$ONLY_FILE"); else SEL=(--limit "$LIMIT"); fi
  step "$PY" -I -B "$T/sources_glm.py" --out "$OUT" --workers "$SCAN" "${SEL[@]}" || fail 3
  step "$PY" -I -B "$T/build_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" || fail 4
  step "$PY" -I -B "$T/verify_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    --sim-workers "$SIMW" --readme || fail 5
elif [[ "$MODE" == rebuild ]]; then
  [[ ! -e $C/raw_sft ]] || { echo "[$(date -Is)] $C/raw_sft exists: refusing to rebuild over it"; exit 2; }
  [[ -f $C/sources/glm/selection_report.json ]] || { echo "[$(date -Is)] no $C/sources/glm/selection_report.json"; exit 2; }
  OUT=$C
  step "$PY" -I -B "$T/build_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    ${ROOTOPT[@]+"${ROOTOPT[@]}"} || fail 4
  step "$PY" -I -B "$T/verify_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    --sim-workers "$SIMW" --readme ${ROOTOPT[@]+"${ROOTOPT[@]}"} || fail 5
elif [[ "$MODE" == verify ]]; then
  [[ -d $C/raw_sft && -f $C/DATA_PREP_COMPLETE.json ]] || { echo "[$(date -Is)] no built corpus in $C: nothing to verify"; exit 2; }
  [[ ! -e $C/SHA256SUMS ]] || { echo "[$(date -Is)] $C/SHA256SUMS exists (a sealed corpus): refusing to re-verify in place"; exit 2; }
  OUT=$C
  step "$PY" -I -B "$T/verify_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    --sim-workers "$SIMW" --readme ${ROOTOPT[@]+"${ROOTOPT[@]}"} || fail 5
else
  [[ ! -e $C/raw_sft ]] || { echo "[$(date -Is)] $C/raw_sft exists: refusing to rebuild over it"; exit 2; }
  OUT=$C
  step "$PY" -I -B "$T/sources_glm.py" --out "$OUT" --workers "$SCAN" ${ALLOW[@]+"${ALLOW[@]}"} || fail 3
  step "$PY" -I -B "$T/build_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    ${ROOTOPT[@]+"${ROOTOPT[@]}"} || fail 4
  step "$PY" -I -B "$T/verify_glm.py" --out "$OUT" --data-root "$DATA_ROOT" --workers "$NW" \
    --sim-workers "$SIMW" --readme ${ROOTOPT[@]+"${ROOTOPT[@]}"} || fail 5
fi
# SHA256SUMS of every file of the output, "./path" entries (hidden temporaries excluded)
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.*' ! -path '*/__pycache__/*' -print0 \
   | sort -z | xargs -0 sha256sum > .SHA256SUMS.tmp && mv .SHA256SUMS.tmp SHA256SUMS) || fail 6
[[ "$MODE" == dryrun ]] && echo 0 > "$D/RC"
echo "[$(date -Is)] glm-sft $MODE done: $OUT ($(wc -l < "$OUT/SHA256SUMS") files in SHA256SUMS)"
exit 0
