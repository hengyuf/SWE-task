# AGA-side preflight for xfer-aga-*.sh, sent as `aga-ssh "bash -s -- <DEST> <ACCOUNT> <PARTITION>" < this file`.
# Read-only; prints one marker per check (the caller requires all *_OK / *_PRESENT / DEST_ABSENT / PARENT_WRITABLE).
# No secret is printed: storage_locations and starfleet_tokens are only grepped for names and the expiry line.
# Every command that could read stdin gets </dev/null (the script itself arrives on stdin).
set -u
export LC_ALL=C
DEST="$1" ACCT="$2" PART="$3"
export PATH=$PATH:/home/svc-datamover/bin
me=$(id -un)
if [[ -e "$DEST" || -L "$DEST" ]]; then echo DEST_EXISTS; else echo DEST_ABSENT; fi
p=$(dirname "$DEST"); if [[ -d "$p" && -w "$p" ]]; then echo PARENT_WRITABLE; else echo PARENT_NOT_WRITABLE; fi
command -v dm >/dev/null && echo DM_PRESENT || echo DM_MISSING
command -v sbatch >/dev/null && echo SBATCH_PRESENT || echo SBATCH_MISSING
# dm hard-codes its Slurm worker binary (seen in dfw sacct: srun ... /home/svc-datamover/image/data-mover)
[[ -x /home/svc-datamover/image/data-mover ]] && echo AGENT_PRESENT || echo AGENT_MISSING
grep -oE '^[A-Za-z0-9_.-]+:' "$HOME/.config/datamover/storage_locations" 2>/dev/null | grep -qx 'pdx-staging:' \
  && echo STAGE_LOC_PRESENT || echo STAGE_LOC_MISSING
exp=$(awk '/^client_token:/{f=1;next} /^[A-Za-z_]+:/{f=0} f && $1=="expiry:"{gsub(/"/,"",$2); print $2; exit}' \
      "$HOME/.config/datamover/starfleet_tokens" 2>/dev/null)
if [[ -n "$exp" ]] && (( $(date -d "$exp" +%s 2>/dev/null || echo 0) > $(date +%s) + 6 * 3600 )); then
  echo "AGA_DM_TOKEN_OK $exp"
  # identity only (same Starfleet user as on dfw lets dfw poll the leg-2 job); no token is printed
  timeout 60 dm user who --format json </dev/null 2>/dev/null | grep -oE '"(preferred_username|name)": *"[^"]*"' | head -2 | sed 's/^/DM_USER /'
else
  echo "AGA_DM_LOGIN_NEEDED ${exp:-none}"
fi
timeout 60 sacctmgr -nP show assoc user="$me" format=account </dev/null 2>/dev/null | grep -qx "$ACCT" \
  && echo ACCOUNT_OK || echo ACCOUNT_NOT_ASSOCIATED
timeout 30 scontrol show partition "$PART" -o </dev/null 2>/dev/null | grep -q "PartitionName=$PART " \
  && echo PARTITION_OK || echo PARTITION_MISSING
timeout 30 scontrol show partition "$PART" -o </dev/null 2>/dev/null | tr ' ' '\n' | \
  grep -E '^(AllowAccounts|AllowQos|MaxTime|DefMemPerNode|DefMemPerCPU|State)=' | tr '\n' ' '; echo
mnt=$(df --output=target "$p" 2>/dev/null | tail -1)
[[ -n "$mnt" ]] && timeout 60 lfs quota -h -u "$me" "$mnt" </dev/null 2>&1 | head -4
echo PREFLIGHT_END
