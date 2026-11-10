# xfer-v2: move the Kimi-pass SFT corpus v2 from dfw to HSG with Data Mover (user's recipe, 2026-10-23):
# dfw Lustre -> pdx-staging:hengyuf-dm-staging/<name> (dm job copy on dfw), then on HSG
# pdx-staging:hengyuf-dm-staging/<name> -> HSG Lustre (dm job copy over hsg-ssh), then sha256 check on HSG.
# Run by hsg_step.sbatch; refuses to overwrite an existing HSG destination.
set -uo pipefail
out="$1"
NAME=sft-kimi-pass-v2-20261023
SRC=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023
STAGE=pdx-staging:hengyuf-dm-staging/$NAME
DEST=/lustre/fsw/portfolios/nemotron/users/hengyuf/$NAME
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
DM=/home/svc-datamover/bin/dm
log() { echo "[$(date -Is)] $*"; }

job_id() { grep -Eo '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$1" | head -1; }
wait_dm() {  # poll a dm job until it ends; rc 0 only for "finished"
  local id="$1" state
  for _ in $(seq 1 720); do
    state=$("$DM" job status "$id" --format json 2>/dev/null | /usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null)
    case "$state" in
      finished) log "dm $id finished"; return 0 ;;
      failed|cancelled|canceled|error|errored) log "dm $id ended in state $state"; "$DM" job status "$id" --errors 2>&1 | tail -30; return 1 ;;
    esac
    sleep 30
  done
  log "dm $id still not finished after 6 h"; return 1
}

test -s "$SRC/DATA_PREP_COMPLETE.json" && test -s "$SRC/SHA256SUMS" || { log "corpus not ready at $SRC"; exit 2; }
(cd "$SRC" && sha256sum --quiet -c SHA256SUMS) || { log "local SHA256SUMS check failed"; exit 2; }
log "local corpus verified: $(du -sh "$SRC" | cut -f1), $(find "$SRC" -type f | wc -l) files"

if timeout 120 "$H/hsg-ssh" "test -e '$DEST'"; then log "HSG destination $DEST already exists: refusing to overwrite"; exit 3; fi

# The nested dm sbatch must not inherit this job's Slurm settings.
for v in $(compgen -e); do case "$v" in SLURM_CONF) ;; SLURM_*|SBATCH_*|SRUN_*|SALLOC_) unset "$v" ;; esac; done

log "leg 1 (dfw): $SRC -> $STAGE"
"$DM" job copy "$SRC" "$STAGE" --yes --format json --slurm-account nemotron_agents_dev --slurm-partition cpu_dataprocessing \
  --slurm-nodes 1 --slurm-cpus-per-task 2 --slurm-time 03:00:00 > "$out/leg1.json" 2> "$out/leg1.err"
id1=$(job_id "$out/leg1.json"); [[ -n "$id1" ]] || id1=$(job_id "$out/leg1.err")
[[ -n "$id1" ]] || { log "leg 1: no dm job id"; tail -20 "$out/leg1.err"; exit 4; }
log "leg 1 dm job $id1"; echo "$id1" > "$out/leg1.id"
wait_dm "$id1" || exit 5

log "leg 2 (HSG): $STAGE -> $DEST"
timeout 600 "$H/hsg-ssh" "export PATH=\$PATH:/home/svc-datamover/bin; dm job copy '$STAGE' '$DEST' --yes --format json --slurm-account nemotron_evals_dev --slurm-partition cpu_datamover --slurm-nodes 1 --slurm-cpus-per-task 4 --slurm-time 03:00:00" > "$out/leg2.json" 2> "$out/leg2.err"
id2=$(job_id "$out/leg2.json"); [[ -n "$id2" ]] || id2=$(job_id "$out/leg2.err")
[[ -n "$id2" ]] || { log "leg 2: no dm job id"; tail -20 "$out/leg2.err"; exit 6; }
log "leg 2 dm job $id2"; echo "$id2" > "$out/leg2.id"
wait_dm "$id2" || exit 7

log "verify on HSG"
for attempt in 1 2 3 4 5 6; do  # the HSG login node sometimes refuses ssh (kex closed): retry
  timeout 1800 "$H/hsg-ssh" "cd '$DEST' && sha256sum --quiet -c SHA256SUMS && echo HSG_SHA256_OK && ls -la '$DEST' '$DEST/raw_sft' '$DEST/blends'" > "$out/verify.txt" 2>&1
  grep -q HSG_SHA256_OK "$out/verify.txt" && break
  log "verify attempt $attempt failed: $(tail -1 "$out/verify.txt")"; sleep $((60 * attempt))
done
cat "$out/verify.txt"
grep -q HSG_SHA256_OK "$out/verify.txt" || { log "HSG verification failed"; exit 8; }
log "transfer complete: $DEST"
