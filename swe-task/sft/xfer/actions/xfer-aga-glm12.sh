# xfer-aga-glm12: move the GLM-5.3 tier-1/2 SFT corpus from dfw to AGA (oci-aga-slurm-1) with Data Mover. Installed
# 2026-11-09 from the aga-transfer research draft (corpus/glm-tier12-20261109/scratch/aga-transfer) with the AGA probe's
# values; next to aga_preflight_remote.sh and aga_verify_remote.sh; run by hsg_step.sbatch like xfer-v3.
# Same two legs as xfer-v3 (dfw -> HSG, 2026-10-29):
#   leg 1, on dfw: dm job copy <dfw corpus dir> pdx-staging:hengyuf-dm-staging/<NAME>   (worker = dfw Slurm job)
#   leg 2, ON AGA: dm job copy pdx-staging:hengyuf-dm-staging/<NAME> <AGA dir>   run over aga-ssh on the AGA login
#                  node (its worker is an AGA Slurm job; dfw workers cannot see AGA Lustre)
#   then on AGA: sha256 of every SHA256SUMS entry + no file outside SHA256SUMS + same file count as dfw.
# Fill the FILL_ME values from the aga-probe1 report; the script refuses while any is unset.
# Changes vs xfer-v3, all fail closed:
#  - dfw dm login preflight (offline token expiry + `dm user who`): leg 1 cannot hang on a browser device login;
#  - one AGA preflight call with explicit markers. xfer-v3's `hsg-ssh "test -e DEST"` guard counts a refused ssh as
#    "absent" (it happened at 18:28:03 in job 21026694). Also checks: parent writable, dm/sbatch/worker present,
#    pdx-staging defined, a valid dm login on AGA, account/partition exist;
#  - ssh is retried only while it fails before the remote shell starts (no REMOTE_BEGIN line), and never after a
#    rejected password (AD lockout);
#  - --runtime slurm is explicit on both legs. dm otherwise infers it from the environment; its default is dm-cluster;
#  - --mem goes through --slurm-sbatch-extra-args. dm sends no --mem, and dfw cpu_* partitions have
#    DefMemPerNode=UNLIMITED (a whole-node wait);
#  - a non-empty staging prefix stops leg 1, and leg 2 is polled from dfw (same Starfleet user), else over aga-ssh.
set -uo pipefail
out="$1"
NAME=sft-glm-tier12-20261109    # = basename of SRC = staging prefix = AGA dir name
SRC=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-glm-tier12-20261109  # clean corpus dir
AGA_ROOT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf  # probe 2026-11-09 22:09 (readlink -f)
AGA_ACCOUNT=nemotron_n4_post    # the only AGA association of hengyuf (probe)
AGA_PARTITION=cpu_datamover     # AllowAccounts=ALL, MaxTime UNLIMITED, DefMemPerNode UNLIMITED (probe)
AGA_QOS=cpu-datamover           # dm's AGA global config adds --qos=cpu-datamover; passed explicitly (--slurm-qos)
AGA_MEM=16G                     # "" to omit (check DefMemPerNode of AGA_PARTITION in the probe)
DFW_ACCOUNT=nemotron_agents_dev
DFW_PARTITION=cpu_interactive   # sbatch --test-only cpu_interactive / cpu_datamover / cpu_dataprocessing first
DFW_MEM=16G                     # 2026-10-29: without it the leg-1 worker sat in Resources until MinMemoryNode was set
AGA_LOG_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/dm-logs  # global one NOT-writable
RESUME_AT_LEG2=0                # 1 = leg 1 of a previous run of this action finished (out/leg1.id) and no leg 2 exists:
                                #     skip stage check + leg 1 and start at leg 2 (the xfer-v3-leg2.sh case, built in)
DEST=$AGA_ROOT/$NAME
STAGE=pdx-staging:hengyuf-dm-staging/$NAME
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
A=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer/actions
DM=/home/svc-datamover/bin/dm
log() { echo "[$(date -Is)] $*"; }

for v in NAME SRC AGA_ROOT AGA_ACCOUNT AGA_PARTITION; do
  [[ -n "${!v}" && "${!v}" != FILL_ME ]] || { log "$v is not filled in"; exit 2; }
done
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$(basename "$SRC")" == "$NAME" ]] || { log "NAME must equal basename of SRC"; exit 2; }
[[ "$AGA_ROOT" == /* && "$AGA_ROOT$AGA_ACCOUNT$AGA_PARTITION$AGA_MEM$AGA_QOS" =~ ^[A-Za-z0-9/._-]*$ ]] || { log "unsafe AGA value"; exit 2; }
[[ -x "$H/aga-ssh" && -s "$A/aga_preflight_remote.sh" && -s "$A/aga_verify_remote.sh" ]] || { log "helpers missing"; exit 2; }
[[ -z "$AGA_LOG_DIR" || "$AGA_LOG_DIR" =~ ^/[A-Za-z0-9/._-]*$ ]] || { log "unsafe AGA_LOG_DIR"; exit 2; }
params="NAME=$NAME SRC=$SRC DEST=$DEST STAGE=$STAGE"
[[ ! -s "$out/leg2.id" ]] || { log "a leg-2 dm job already exists ($(cat "$out/leg2.id")): inspect / verify-aga-glm12, never re-copy"; exit 2; }
if [[ "$RESUME_AT_LEG2" == 1 ]]; then
  [[ -s "$out/leg1.id" && "$(cat "$out/params.txt" 2>/dev/null)" == "$params" ]] || { log "resume: no matching leg 1 in $out"; exit 2; }
elif [[ -e "$out/leg1.id" ]]; then
  log "a previous run exists in $out: move it aside for a fresh start, or set RESUME_AT_LEG2=1 to continue at leg 2"; exit 2
fi
printf '%s\n' "$params" > "$out/params.txt"

job_id() {  # the dm job UUID: "id" of the first JSON object in the file, else the first UUID in it
  /usr/bin/python3 -I -c 'import json,sys
s=open(sys.argv[1]).read(); print(json.JSONDecoder().raw_decode(s[s.index("{"):])[0]["id"])' "$1" 2>/dev/null \
    || grep -Eo '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$1" 2>/dev/null | head -1
}
json_state() { /usr/bin/python3 -I -c 'import json,sys
s=sys.stdin.read(); print(json.JSONDecoder().raw_decode(s[s.index("{"):])[0].get("state",""))' 2>/dev/null; }
aga_once() {  # aga_once <timeout> <outfile> <remote command> [stdin file]: one ssh; rc 0 iff the remote shell ran
  local t="$1" f="$2" cmd="$3" in="${4:-/dev/null}" rc
  timeout "$t" "$H/aga-ssh" "echo REMOTE_BEGIN; $cmd" < "$in" > "$f" 2> "$f.err"; rc=$?
  grep -q '^REMOTE_BEGIN$' "$f" && return 0
  if (( rc == 5 )) || grep -q 'Permission denied' "$f.err"; then
    log "AGA rejected the password (rc $rc): stopping, no retry (AD lockout)"; exit 5
  fi
  return 1
}
aga() {  # like aga_once, retried only while ssh fails before the remote shell starts (kex refusals seen on HSG)
  local attempt
  for attempt in 1 2 3 4 5 6; do
    aga_once "$@" && return 0
    log "aga-ssh attempt $attempt did not reach the remote shell: $(tail -1 "$2.err" 2>/dev/null)"; sleep $((60 * attempt))
  done
  return 1
}
wait_dm() {  # wait_dm <id> [local|remote]: 0 finished; 1 failed/cancelled/timeout; 3 not visible from dfw
  local id="$1" where="${2:-local}" state empty=0
  for _ in $(seq 1 720); do
    if [[ "$where" == remote ]]; then
      state=""
      aga_once 300 "$out/poll.txt" "export PATH=\$PATH:/home/svc-datamover/bin; dm job status '$id' --format json </dev/null" \
        && state=$(json_state < "$out/poll.txt")
    else
      state=$(timeout 120 "$DM" job status "$id" --format json </dev/null 2>/dev/null | json_state)
    fi
    case "$state" in
      finished) log "dm $id finished ($where poll)"; return 0 ;;
      failed|cancelled|canceled|error|errored)
        log "dm $id ended in state $state"
        [[ "$where" == local ]] && timeout 120 "$DM" job status "$id" --errors --format json </dev/null 2>&1 | tail -30
        return 1 ;;
      "") empty=$((empty + 1)); [[ "$where" == local ]] && (( empty >= 10 )) && return 3 ;;
      *) empty=0 ;;
    esac
    if [[ "$where" == remote ]]; then sleep 120; else sleep 30; fi
  done
  log "dm $id not finished within the polling window"; return 1
}
token_ok() {  # offline: the 30-day Starfleet client_token in starfleet_tokens must outlive the transfer (6 h margin)
  local exp
  exp=$(awk '/^client_token:/{f=1;next} /^[A-Za-z_]+:/{f=0} f && $1=="expiry:"{gsub(/"/,"",$2); print $2; exit}' \
        "$HOME/.config/datamover/starfleet_tokens" 2>/dev/null)
  log "dfw dm client_token expiry: ${exp:-none}"
  [[ -n "$exp" ]] && (( $(date -d "$exp" +%s 2>/dev/null || echo 0) > $(date +%s) + 6 * 3600 ))
}

# --- 1. local corpus (as xfer-v3) --------------------------------------------------------------------------------
test -s "$SRC/DATA_PREP_COMPLETE.json" && test -s "$SRC/SHA256SUMS" || { log "corpus not ready at $SRC"; exit 2; }
/usr/bin/python3 -I -c 'import json,sys; s=json.load(open(sys.argv[1]))["state"]; sys.exit(0 if s == "complete" else 1)' \
  "$SRC/DATA_PREP_COMPLETE.json" || { log "DATA_PREP_COMPLETE.json state is not complete (dry run?)"; exit 2; }
grep -q 'VERIFY_RESULT_PLACEHOLDER' "$SRC/README.md" 2>/dev/null && { log "README has no verification result"; exit 2; }
# verify_glm.py --readme writes exactly one 'Result (<time>): OK.' line on success (FAILED otherwise)
[[ "$(grep -cE '^Result \([^)]*\): OK\.' "$SRC/README.md" 2>/dev/null)" == 1 ]] \
  && ! grep -qE '^Result \([^)]*\): FAILED' "$SRC/README.md" || { log "README has no single verify 'Result (...): OK.' line"; exit 2; }
# the AGA destination must be the DATA_ROOT the corpus was built for (blend / receipt paths are absolute)
/usr/bin/python3 -I -c 'import json,sys
src, dest = sys.argv[1], sys.argv[2]
r = json.load(open(src + "/DATA_PREP_COMPLETE.json"))
b = json.load(open(src + "/blends/raw_sft.json"))
paths = [x for x in b["train"] if "/" in x]
ok = r.get("data_root") == dest and paths and all(p.startswith(dest + "/raw_sft/") for p in paths) \
     and all(s["path"].startswith(dest + "/raw_sft/") for s in r["shards"])
sys.exit(0 if ok else 1)' "$SRC" "$DEST" || { log "DATA_ROOT / blend / shard paths of the corpus are not under $DEST"; exit 2; }
(cd "$SRC" && sha256sum --quiet -c SHA256SUMS) || { log "local SHA256SUMS check failed"; exit 2; }
# file listing / sizes via os.walk, not find / du: on 2026-11-10 statfs() on the dfw Lustre mount hung (an
# unresponsive target) and gnulib fts (GNU find, du) calls fstatfs, so find blocked in ptlrpc_set_wait for 45+ min
# while ls, os.walk and file reads were instant
listfiles() { /usr/bin/python3 -I -c 'import os, sys
root = sys.argv[1]
for d, ds, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        if os.path.isfile(p) and not os.path.islink(p):
            print("./" + os.path.relpath(p, root))' "$1"; }
treesize() { /usr/bin/python3 -I -c 'import os, sys
print(sum(os.path.getsize(os.path.join(d, f)) for d, ds, fs in os.walk(sys.argv[1]) for f in fs))' "$1"; }
extra=$(cd "$SRC" && LC_ALL=C comm -23 <(listfiles "$SRC" | grep -vx './SHA256SUMS' | LC_ALL=C sort) \
                              <(sed -E 's/^[0-9a-f]{64}  //' SHA256SUMS | LC_ALL=C sort))
[[ -z "$extra" ]] || { log "files not in SHA256SUMS (move them out first): $(echo "$extra" | head -5 | tr '\n' ' ')"; exit 2; }
nfiles=$(listfiles "$SRC" | wc -l)
log "local corpus verified: $(treesize "$SRC") bytes, $nfiles files"

# --- 2. dfw dm login ---------------------------------------------------------------------------------------------
token_ok || { log "DFW_DM_LOGIN_NEEDED: the user must run $DM user who in a dfw terminal (browser device login)"; exit 9; }
timeout 120 "$DM" user who --format json </dev/null > "$out/dm-who.json" 2> "$out/dm-who.err"
if grep -q 'web browser is required' "$out/dm-who.json" "$out/dm-who.err"; then log "dfw dm wants a browser login"; exit 9; fi
log "dfw dm user: $(grep -oE '"preferred_username": *"[^"]*"' "$out/dm-who.json" | head -1)"

# --- 3. AGA preflight (one remote call, markers) -------------------------------------------------------------------
aga 300 "$out/aga-preflight.txt" "bash -s -- '$DEST' '$AGA_ACCOUNT' '$AGA_PARTITION'" "$A/aga_preflight_remote.sh" \
  || { log "AGA unreachable"; exit 6; }
cat "$out/aga-preflight.txt"
for m in DEST_ABSENT PARENT_WRITABLE DM_PRESENT SBATCH_PRESENT AGENT_PRESENT STAGE_LOC_PRESENT AGA_DM_TOKEN_OK \
         ACCOUNT_OK PARTITION_OK PREFLIGHT_END; do
  grep -q "^$m" "$out/aga-preflight.txt" || { log "AGA preflight: $m missing: refusing"; exit 3; }
done

# The nested dm sbatch must not inherit this job's Slurm settings (memory: feedback_slurm_nested_submit_env_leak).
for v in $(compgen -e); do case "$v" in SLURM_CONF) ;; SLURM_*|SBATCH_*|SRUN_*|SALLOC_*) unset "$v" ;; esac; done

if [[ "$RESUME_AT_LEG2" == 1 ]]; then
  id1=$(cat "$out/leg1.id")
  st=$(timeout 120 "$DM" job status "$id1" --format json </dev/null 2>/dev/null | json_state)
  [[ "$st" == finished ]] || { log "resume: leg-1 dm job $id1 is '$st', not finished: refusing"; exit 2; }
  log "resume: leg 1 ($id1) finished earlier; starting at leg 2"
else
# --- 4. staging prefix must be empty (a fresh NAME); stale objects would also be caught by the AGA check -------------
timeout 300 "$DM" location list-objects "$STAGE" --format json </dev/null > "$out/stage-before.json" 2> "$out/stage-before.err"
n_staged=$(/usr/bin/python3 -I -c 'import json,sys
s=open(sys.argv[1]).read().strip()
if not s: print(0); raise SystemExit
try: d=json.loads(s)
except Exception: print("unparsed"); raise SystemExit
if isinstance(d, dict): d=next((v for v in d.values() if isinstance(v, list)), [])
print(len(d) if isinstance(d, list) else "unparsed")' "$out/stage-before.json")
case "$n_staged" in
  0) log "staging prefix $STAGE is empty" ;;
  unparsed) log "WARNING: could not parse list-objects output (see stage-before.*); the AGA extra-files check still guards" ;;
  *) log "staging prefix $STAGE already holds $n_staged entries: refusing (pick a new NAME)"; exit 3 ;;
esac

# --- 5. leg 1 (dfw) ----------------------------------------------------------------------------------------------
log "leg 1 (dfw): $SRC -> $STAGE"
timeout 1800 "$DM" job copy "$SRC" "$STAGE" --yes --format json --runtime slurm --slurm-account "$DFW_ACCOUNT" \
  --slurm-partition "$DFW_PARTITION" --slurm-nodes 1 --slurm-cpus-per-task 2 --slurm-time 03:00:00 \
  ${DFW_MEM:+"--slurm-sbatch-extra-args=--mem=$DFW_MEM"} </dev/null > "$out/leg1.json" 2> "$out/leg1.err"
id1=$(job_id "$out/leg1.json"); [[ -n "$id1" ]] || id1=$(job_id "$out/leg1.err")
[[ -n "$id1" ]] || { log "leg 1: no dm job id"; tail -20 "$out/leg1.err"; exit 4; }
log "leg 1 dm job $id1"; echo "$id1" > "$out/leg1.id"
wait_dm "$id1" local || exit 5
fi

# --- 6. leg 2 (AGA, over aga-ssh; never re-run once the remote shell started) --------------------------------------
log "leg 2 (AGA): $STAGE -> $DEST"
leg2="export PATH=\$PATH:/home/svc-datamover/bin; dm job copy '$STAGE' '$DEST' --yes --format json --runtime slurm"
leg2+=" --slurm-account '$AGA_ACCOUNT' --slurm-partition '$AGA_PARTITION' --slurm-qos '$AGA_QOS' --slurm-nodes 1 --slurm-cpus-per-task 4"
leg2+=" --slurm-time 03:00:00${AGA_MEM:+ --slurm-sbatch-extra-args=--mem=$AGA_MEM}${AGA_LOG_DIR:+ --slurm-log-dir '$AGA_LOG_DIR'} </dev/null"
aga 900 "$out/leg2.json" "$leg2" || { log "leg 2: AGA unreachable; leg 1 is done: set RESUME_AT_LEG2=1 and re-run"; exit 6; }
id2=$(job_id "$out/leg2.json"); [[ -n "$id2" ]] || id2=$(job_id "$out/leg2.json.err")
[[ -n "$id2" ]] || { log "leg 2: remote ran but no dm job id: inspect (dm job list), do NOT blindly re-run"; tail -20 "$out/leg2.json.err"; exit 6; }
log "leg 2 dm job $id2"; echo "$id2" > "$out/leg2.id"
wait_dm "$id2" local; rc=$?
if (( rc == 3 )); then log "leg-2 status not visible from dfw: polling over aga-ssh"; wait_dm "$id2" remote; rc=$?; fi
(( rc == 0 )) || exit 7

# --- 7. verify on AGA ----------------------------------------------------------------------------------------------
log "verify on AGA"
aga 1800 "$out/verify.txt" "bash -s -- '$DEST' '$nfiles'" "$A/aga_verify_remote.sh" \
  || { log "verify: AGA unreachable (re-check later with verify-aga-glm12)"; exit 8; }
cat "$out/verify.txt"
grep -q '^AGA_SHA256_OK' "$out/verify.txt" || { log "AGA verification failed"; exit 8; }
log "transfer complete: $DEST"
