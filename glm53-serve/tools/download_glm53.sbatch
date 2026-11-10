#!/bin/bash
# Download zai-org/GLM-5.3 (FP8, ~756 GB, 156 files) from Hugging Face to dfw Lustre at a pinned revision, then
# verify every LFS file's sha256 against the Hub's LFS oid. Resumable: rerun the same sbatch.
#SBATCH --job-name=glm53-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/logs/%x-%j.out
set -uo pipefail
REPO=zai-org/GLM-5.3
REV=ba32f8c19c8738e2c57c71798c601d477a7bf844
DEST=/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3
export HF_HOME=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN="$(/cm/local/apps/python39/bin/python3 -B -c "
import shlex
for w in shlex.split(open('/lustre/fsw/portfolios/nemotron/users/hengyuf/.env').read(), comments=True):
    k, s, v = w.partition('=')
    if s and k == 'HF_TOKEN':
        print(v); break
")"
mkdir -p "$DEST" "$HF_HOME"
echo "[$(date -Is)] host $(hostname) start download $REPO@$REV -> $DEST"
for attempt in 1 2 3 4 5 6; do
  /home/hengyuf/.local/bin/hf download "$REPO" --revision "$REV" --local-dir "$DEST" --max-workers 16 --quiet > /dev/null && break
  echo "[$(date -Is)] download attempt $attempt failed; retrying in $((60*attempt)) s"
  sleep $((60*attempt))
done
echo "[$(date -Is)] download finished; du: $(du -sh "$DEST" | cut -f1)"
echo "$REV" > "$DEST/HF_REVISION"
echo "[$(date -Is)] verifying sha256 of LFS files against the Hub tree"
/cm/local/apps/python39/bin/python3 -B - "$REPO" "$REV" "$DEST" <<'PY'
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
repo, rev, dest = sys.argv[1:4]
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
tree = json.load(urllib.request.urlopen(req, timeout=120))
files = [f for f in tree if f.get("type") == "file"]
def check(f):
    p = os.path.join(dest, f["path"])
    if not os.path.isfile(p):
        return f["path"], "MISSING"
    if os.path.getsize(p) != f["size"]:
        return f["path"], f"SIZE {os.path.getsize(p)} != {f['size']}"
    lfs = f.get("lfs")
    if not lfs:
        return f["path"], "ok-size"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return f["path"], "ok-sha" if h.hexdigest() == lfs["oid"] else f"SHA MISMATCH {h.hexdigest()} != {lfs['oid']}"
with ThreadPoolExecutor(14) as ex:
    res = list(ex.map(check, files))
bad = [r for r in res if not r[1].startswith("ok")]
json.dump({"repo": repo, "revision": rev, "files": len(res), "bad": bad, "results": res},
          open(os.path.join(dest, "VERIFY.json"), "w"), indent=1)
print(f"verified {len(res)} files, bad {len(bad)}")
for b in bad: print("BAD", b)
if not bad:
    open(os.path.join(dest, "DOWNLOAD_OK"), "w").write(rev + "\n")
sys.exit(1 if bad else 0)
PY
rc=$?
echo "[$(date -Is)] verify rc=$rc"
exit $rc
