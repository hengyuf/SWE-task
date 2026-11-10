# xfer-v3-leg2: FALLBACK for xfer-v3 when its leg 1 finished but its single leg-2 ssh was refused (HSG kex closed).
# Leg 2 only (pdx-staging -> HSG with dm over hsg-ssh, the same command as xfer-v3) + the HSG sha256 check. Refuses
# unless xfer-v3's leg-1 dm job is "finished" and xfer-v3 has no leg-2 dm job id; retries the ssh only while it fails
# at connection setup (kex / connection closed, no dm job id), so no second copy can be started by a retry.
set -uo pipefail
out="$1"
NAME=sft-v3-20261029
STAGE=pdx-staging:hengyuf-dm-staging/$NAME
DEST=/lustre/fsw/portfolios/nemotron/users/hengyuf/$NAME
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
DM=/home/svc-datamover/bin/dm
V3OUT=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer/out/xfer-v3
log() { echo "[$(date -Is)] $*"; }
job_id() { grep -Eo '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$1" | head -1; }
dm_state() { "$DM" job status "$1" --format json 2>/dev/null | /usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null; }
wait_dm() {
  local id="$1" state
  for _ in $(seq 1 720); do
    state=$(dm_state "$id")
    case "$state" in
      finished) log "dm $id finished"; return 0 ;;
      failed|cancelled|canceled|error|errored) log "dm $id ended in state $state"; "$DM" job status "$id" --errors 2>&1 | tail -30; return 1 ;;
    esac
    sleep 30
  done
  log "dm $id still not finished after 6 h"; return 1
}
id1=$(cat "$V3OUT/leg1.id" 2>/dev/null)
[[ -n "$id1" && "$(dm_state "$id1")" == finished ]] || { log "xfer-v3 leg 1 ($id1) is not finished: refusing"; exit 2; }
[[ -s "$V3OUT/leg2.id" ]] && { log "xfer-v3 already has a leg-2 dm job $(cat "$V3OUT/leg2.id"): refusing"; exit 2; }
for v in $(compgen -e); do case "$v" in SLURM_CONF) ;; SLURM_*|SBATCH_*|SRUN_*|SALLOC_) unset "$v" ;; esac; done
id2=""
for attempt in 1 2 3 4 5 6; do
  log "leg 2 (HSG) attempt $attempt: $STAGE -> $DEST"
  timeout 600 "$H/hsg-ssh" "export PATH=\$PATH:/home/svc-datamover/bin; dm job copy '$STAGE' '$DEST' --yes --format json --slurm-account nemotron_evals_dev --slurm-partition cpu_datamover --slurm-nodes 1 --slurm-cpus-per-task 4 --slurm-time 03:00:00" > "$out/leg2.json" 2> "$out/leg2.err"
  id2=$(job_id "$out/leg2.json"); [[ -n "$id2" ]] || id2=$(job_id "$out/leg2.err")
  [[ -n "$id2" ]] && break
  if grep -q 'kex_exchange_identification\|Connection closed by\|Connection reset by\|Connection refused' "$out/leg2.err"; then
    log "ssh refused at connection setup: $(tail -1 "$out/leg2.err")"; sleep $((60 * attempt)); continue
  fi
  log "leg 2: no dm job id and not a connection-setup failure: stopping"; tail -20 "$out/leg2.err"; exit 6
done
[[ -n "$id2" ]] || { log "leg 2: ssh refused 6 times"; exit 6; }
log "leg 2 dm job $id2"; echo "$id2" > "$out/leg2.id"
wait_dm "$id2" || exit 7
log "verify on HSG"
for attempt in 1 2 3 4 5 6; do
  timeout 1800 "$H/hsg-ssh" "cd '$DEST' && sha256sum --quiet -c SHA256SUMS && echo HSG_SHA256_OK && ls -la '$DEST' '$DEST/raw_sft' '$DEST/blends'" > "$out/verify.txt" 2>&1
  grep -q HSG_SHA256_OK "$out/verify.txt" && break
  log "verify attempt $attempt failed: $(tail -1 "$out/verify.txt")"; sleep $((60 * attempt))
done
cat "$out/verify.txt"
grep -q HSG_SHA256_OK "$out/verify.txt" || { log "HSG verification failed"; exit 8; }
log "transfer complete: $DEST"
