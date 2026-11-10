# verify-aga-glm12: the AGA check of xfer-aga-glm12 on its own (sha256 of every SHA256SUMS entry, no file outside
# SHA256SUMS, same file count as dfw), run by hsg_step.sbatch; only needed when xfer-aga-glm12 exited 8. Keep
# NAME/SRC/AGA_ROOT identical to xfer-aga-glm12.sh. Retries ssh only before the remote shell starts, never after a
# rejected password.
set -uo pipefail
out="$1"
NAME=sft-glm-tier12-20261109
SRC=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-glm-tier12-20261109
AGA_ROOT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf
DEST=$AGA_ROOT/$NAME
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
A=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer/actions
log() { echo "[$(date -Is)] $*"; }
for v in NAME SRC AGA_ROOT; do [[ -n "${!v}" && "${!v}" != FILL_ME ]] || { log "$v is not filled in"; exit 2; }; done
[[ "$AGA_ROOT/$NAME" =~ ^/[A-Za-z0-9/._-]*$ ]] || { log "unsafe AGA path"; exit 2; }
nfiles=$(cd "$SRC" && find . -type f | wc -l)
for attempt in 1 2 3 4 5 6; do
  timeout 1800 "$H/aga-ssh" "echo REMOTE_BEGIN; bash -s -- '$DEST' '$nfiles'" < "$A/aga_verify_remote.sh" \
    > "$out/verify.txt" 2> "$out/verify.err"; rc=$?
  grep -q '^REMOTE_BEGIN$' "$out/verify.txt" && break
  if (( rc == 5 )) || grep -q 'Permission denied' "$out/verify.err"; then log "password rejected: stop (AD lockout)"; exit 5; fi
  log "attempt $attempt did not reach AGA: $(tail -1 "$out/verify.err")"; sleep $((60 * attempt))
done
cat "$out/verify.txt"
grep -q '^AGA_SHA256_OK' "$out/verify.txt" || { log "AGA verification failed"; exit 1; }
log "verified: $DEST"
