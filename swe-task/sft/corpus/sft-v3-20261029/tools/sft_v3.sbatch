#!/bin/bash
# SFT-v3 (2026-10-29): tools/sources_v4v5.py -> tools/build_sft_v3.py -> tools/verify_sft_v3.py (-> SHA256SUMS) in ONE
# CPU job. Submitted ONLY through hengyuf's schedule helper (/lustre/fsw/portfolios/nemotron/users/hengyuf/
# schdule_helper.sh): a babysit pair, one held candidate per account (nemotron_agents_dev, nemotron_rl_algo); the first
# to start wins in babysit_enter, the other exits at once. No credential is needed or read.
#
# Mode = $1, or SFTV3_MODE when the helper's babysit_submit submits it (it passes no script arguments):
#   dryrun  output <C>/dryrun/build: sources_v4v5.py --allow-partial (v5 + every v4 campaign whose results file exists),
#           build_sft_v3.py --v2-limit 10 (20 v2 conversations + all of those v4 / v5 passes), verify_sft_v3.py.
#           Helper state <C>/dryrun/state, log <C>/dryrun/logs (pass --output).
#           `dryrun [pilots] snapshot` uses --snapshot-incomplete instead of --allow-partial.
#   rebuild output <C>, from the EXISTING <C>/sources (no new selection / snapshot): build_sft_v3.py + verify --readme +
#           SHA256SUMS. Refuses if <C>/raw_sft exists or a selection_report.json is missing. Used 2026-10-29 for the
#           literal stop-token rule (same 17:03 v4 snapshot as the first build). Helper state <C>.work/state-rebuild.
#   build   output <C> itself (the real corpus): sources_v4v5.py refuses unless all 18 v4 + 4 v5 results files exist
#           (`build pilots` also takes the 3 pilot rollouts; `build [pilots] snapshot` takes the campaigns still running
#           from a snapshot of their terminal slots, the rest listed as pending: sources_v4v5.py --snapshot-incomplete);
#           refuses if <C>/raw_sft exists; moves a FINISHED dry run (<C>/dryrun/RC present) to <C>.work/ first;
#           verify --readme; then SHA256SUMS. Helper state <C>.work/state, log <C>.work/logs.
#
# Submission A, one command (when sbatch may run inside a script, i.e. the Claude Bash sandbox is off):
#   mkdir -p <C>.work/logs && SFTV3_MODE=build RESULTS_DIR=<C>.work/state EXP_NAME=sftv3-build \
#     BATCH_SCRIPT=<C>/tools/sft_v3.sbatch bash -c 'source /lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh \
#     && babysit_submit --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover'
# Submission B, by hand (standalone Slurm calls; the job finds its leg / index by its own id, as xfer/hsg_step.sbatch):
#   W=<C>.work; B=$W/state/babysit; T=<token, e.g. $(date +%s)-sftv3b>; mkdir -p $B/legs/1 $W/logs
#   echo $T > $B/token; echo 1 > $B/current; echo cc-$T-L1-0 > $B/legs/1/request.0; echo cc-$T-L1-1 > $B/legs/1/request.1
#   sbatch --parsable --hold --export=NONE --account=nemotron_agents_dev --job-name=sftv3-build --comment=cc-$T-L1-0 \
#     --partition=<p> --output=$W/logs/%x-%j.out <C>/tools/sft_v3.sbatch build        # id -> $B/legs/1/job.0
#   sbatch ... --account=nemotron_rl_algo ... --comment=cc-$T-L1-1 ...                    # id -> $B/legs/1/job.1
#   echo <id0>,<id1> > $B/legs/1/release.requested; scontrol release <id0>,<id1>; echo <id0>,<id1> > $B/legs/1/released
#SBATCH --partition=cpu_dataprocessing,cpu_interactive,cpu_datamover
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-v3-20261029.work/logs/%x-%j.out
set -uo pipefail
umask 027
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf
C=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-v3-20261029
WORK=$C.work
MODE="${1:-${SFTV3_MODE:-}}"
# options after the mode (or SFTV3_PILOTS=1 / SFTV3_SNAPSHOT=1):
#   pilots    also take the v4 / v5 pilot rollouts (default: left out)
#   snapshot  sources_v4v5.py --snapshot-incomplete: campaigns without a results file (driver still running) contribute
#             their TERMINAL slots, read now with the driver's record logic; their other slots are listed as pending
#             (user 2026-10-29 ~16:25: "不等这些长尾rollout了,直接开始构建SFT-v3的 sft data 并传到HSG")
PILOTS=()
SNAP=()
for opt in "${@:2}"; do
  case "$opt" in
    pilots) PILOTS=(--include-pilots) ;;
    snapshot) SNAP=(--snapshot-incomplete) ;;
    *) echo "unknown option $opt (pilots | snapshot)"; exit 2 ;;
  esac
done
if [[ "${SFTV3_PILOTS:-0}" == 1 ]]; then PILOTS=(--include-pilots); fi
if [[ "${SFTV3_SNAPSHOT:-0}" == 1 ]]; then SNAP=(--snapshot-incomplete); fi
PARTIAL=(--allow-partial)  # dry run without a snapshot: skip the campaigns without a results file
if (( ${#SNAP[@]} )); then PARTIAL=("${SNAP[@]}"); fi
case "$MODE" in
  dryrun) export RESULTS_DIR=$C/dryrun/state ;;
  build) export RESULTS_DIR=$WORK/state ;;
  rebuild) export RESULTS_DIR=$WORK/state-rebuild ;;
  *) echo "usage: sft_v3.sbatch dryrun|build|rebuild (or SFTV3_MODE)"; exit 2 ;;
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
JOBTMP=$(mktemp -d "/tmp/sftv3-${SLURM_JOB_ID:-x}-XXXXXX") || exit 1
trap 'rm -rf "$JOBTMP"' EXIT
export TMPDIR=$JOBTMP  # verify's datasets cache (Arrow copies of the shards)
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
T=$C/tools
NCPU=${SLURM_CPUS_PER_TASK:-16}
NW=$(( NCPU > 8 ? NCPU - 4 : NCPU ))
SCAN=$(( NCPU > 16 ? 16 : NCPU ))  # concurrent http.jsonl readers (Lustre)
echo "[$(date -Is)] sft-v3 $MODE: job ${SLURM_JOB_ID:-?} won leg $BABYSIT_LEG (account ${SLURM_JOB_ACCOUNT:-?}) on" \
     "$(hostname), partition ${SLURM_JOB_PARTITION:-?}, $NCPU CPUs; options:" \
     "${PILOTS[@]+"${PILOTS[@]}"} ${SNAP[@]+"${SNAP[@]}"}"
step() { echo "[$(date -Is)] run: $*"; "$@"; local r=$?; echo "[$(date -Is)] exit $r"; return $r; }
fail() { [[ "$MODE" == dryrun ]] && echo "$1" > "$C/dryrun/RC"; exit "$1"; }
cd "$JOBTMP" || exit 1  # python -I: nothing is imported from the cwd
if [[ "$MODE" == dryrun ]]; then
  OUT=$C/dryrun/build
  mkdir -p "$OUT"
  step "$PY" -I -B "$T/sources_v4v5.py" --out "$OUT" "${PARTIAL[@]}" --workers "$SCAN" \
    ${PILOTS[@]+"${PILOTS[@]}"} || fail 3
  step "$PY" -I -B "$T/build_sft_v3.py" --out "$OUT" --v2-limit 10 --workers "$NW" || fail 4
  step "$PY" -I -B "$T/verify_sft_v3.py" --out "$OUT" --workers "$NW" --readme || fail 5
elif [[ "$MODE" == rebuild ]]; then
  [[ ! -e $C/raw_sft ]] || { echo "[$(date -Is)] $C/raw_sft exists: refusing to rebuild over it"; exit 2; }
  for b in v4 v5; do
    [[ -f $C/sources/$b/selection_report.json ]] || { echo "[$(date -Is)] no $C/sources/$b/selection_report.json"; exit 2; }
  done
  OUT=$C
  step "$PY" -I -B "$T/build_sft_v3.py" --out "$OUT" --workers "$NW" || fail 4
  step "$PY" -I -B "$T/verify_sft_v3.py" --out "$OUT" --workers "$NW" --readme || fail 5
else
  [[ ! -e $C/raw_sft ]] || { echo "[$(date -Is)] $C/raw_sft exists: refusing to rebuild over it"; exit 2; }
  if [[ -e $C/dryrun ]]; then  # keep the corpus dir to the corpus files (xfer-v3 copies the whole dir)
    [[ -f $C/dryrun/RC ]] || { echo "[$(date -Is)] $C/dryrun has no RC (dry run still running?): refusing"; exit 2; }
    mkdir -p "$WORK" && mv "$C/dryrun" "$WORK/dryrun-$(date +%Y%m%d-%H%M%S)" || exit 2
  fi
  OUT=$C
  step "$PY" -I -B "$T/sources_v4v5.py" --out "$OUT" --workers "$SCAN" ${PILOTS[@]+"${PILOTS[@]}"} \
    ${SNAP[@]+"${SNAP[@]}"} || fail 3
  step "$PY" -I -B "$T/build_sft_v3.py" --out "$OUT" --workers "$NW" || fail 4
  step "$PY" -I -B "$T/verify_sft_v3.py" --out "$OUT" --workers "$NW" --readme || fail 5
fi
# SHA256SUMS of every file of the output (hidden temporaries, the dry-run dir and helper state excluded)
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.*' ! -path './dryrun/*' ! -path './state/*' -print0 \
   | sort -z | xargs -0 sha256sum > .SHA256SUMS.tmp && mv .SHA256SUMS.tmp SHA256SUMS) || fail 6
[[ "$MODE" == dryrun ]] && echo 0 > "$C/dryrun/RC"
echo "[$(date -Is)] sft-v3 $MODE done: $OUT ($(wc -l < "$OUT/SHA256SUMS") files in SHA256SUMS)"
exit 0
