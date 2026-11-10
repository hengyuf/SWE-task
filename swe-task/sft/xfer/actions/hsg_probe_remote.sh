# Read-only report, run on the HSG login node through `hsg-ssh bash -s` (no writes outside $TMPDIR-free stdout).
set -u
P=/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/zhenghaox/swe-rubric-pipeline/super-filtered-glm53-v5-20260922
echo "## host $(hostname) user $(id -un) date $(date -Is)"
echo "## ls P"; ls -la "$P" 2>&1 | head -200
echo "## ls parent"; ls -la "$(dirname "$P")" 2>&1 | head -100
echo "## find P (depth 4; iter_*, checkpoints, .git, __pycache__ pruned)"
timeout 300 find "$P" -maxdepth 4 \( -name 'iter_*' -o -name checkpoints -o -name .git -o -name __pycache__ \) -prune -o -printf '%y %s %TY-%Tm-%Td %p\n' 2>&1 | head -6000
echo "## sc14 candidates on hsg"
for d in /lustre/fsw/portfolios/nemotron/users/hengyuf /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf \
         /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/training_actual_0603/super_n4_post/conv_wrappers/swonlysc-step0014 \
         /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/training_actual_0603/super_n4_post/conv_wrappers/swonlysc-step0014/evals \
         /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/training_actual_0603/super_n4_post/conv_wrappers/swonlysc-step0014/evals/hf; do
  echo "# $d"; ls -la "$d" 2>&1 | head -60
done
echo "## lustre mounts"; df -h 2>/dev/null | grep -i lustre | head; ls -la /lustre 2>&1 | head
echo "## data mover"; ls -la /home/svc-datamover/bin/dm 2>&1; ls -la ~/.config/datamover/ 2>&1
grep -o '^[A-Za-z0-9_-]*:' ~/.config/datamover/storage_locations 2>&1 | head
echo "## slurm accounts"; sacctmgr -nP show assoc user=hengyuf format=account,partition,qos 2>&1 | head -30
sinfo -s 2>&1 | head -30
echo "## quota"; lfs quota -h -u hengyuf /lustre/fsw 2>&1 | head -8; lfs quota -h -u hengyuf /lustre/fs1 2>&1 | head -8
echo "## data files in P"
python3 - "$P" <<'PY'
import collections, json, os, sys
P = sys.argv[1]
skip = {"checkpoints", "__pycache__", ".git", "runtime", "megatron-lm", "Megatron-LM"}
files = []
for root, dirs, names in os.walk(P):
    dirs[:] = [d for d in dirs if not d.startswith("iter_") and d not in skip]
    if root[len(P):].count(os.sep) >= 6:
        dirs[:] = []
    for n in names:
        if ".jsonl" in n or n.endswith((".parquet", ".idx", ".bin", ".npy")):
            p = os.path.join(root, n)
            try:
                files.append((p, os.path.getsize(p)))
            except OSError:
                pass
print("n_data_files", len(files))
for p, s in sorted(files)[:400]:
    print("F", s, p)

def describe(path):
    print("### describe", path)
    try:
        with open(path, "rb") as f:
            line = f.readline(400_000_000)
    except Exception as e:
        print("unreadable", e); return
    try:
        d = json.loads(line)
    except Exception as e:
        print("not json", e, line[:300]); return
    print("top keys", list(d)[:40] if isinstance(d, dict) else type(d).__name__)
    msgs = d.get("messages") if isinstance(d, dict) else None
    if not isinstance(msgs, list):
        print({k: (str(v)[:300]) for k, v in d.items()} if isinstance(d, dict) else str(d)[:600]); return
    print("n_messages", len(msgs), "line_bytes", len(line))
    print("key sets", collections.Counter(tuple(sorted(m)) for m in msgs))
    print("roles", collections.Counter(m.get("role") for m in msgs))
    sysm = [m for m in msgs if m.get("role") == "system"]
    print("n_system", len(sysm))
    for m in sysm[:3]:
        print("system extra fields", {k: (v if not isinstance(v, (str, list)) else (v[:200] if isinstance(v, str) else f"list[{len(v)}]")) for k, v in m.items() if k != "content"})
    seen = set()
    for i, m in enumerate(msgs[:80]):
        r = m.get("role")
        if (r, i > 3) in seen:
            continue
        seen.add((r, i > 3))
        c = m.get("content")
        c = c if isinstance(c, str) else json.dumps(c)[:2000]
        print(f"--- [{i}] role={r} len={len(c)} extra={[k for k in m if k not in ('role', 'content')]}")
        print("HEAD:", repr(c[:700]))
        print("TAIL:", repr(c[-300:]))

kinds = {}
for p, s in sorted(files):
    if ".jsonl" not in p:
        continue
    kind = os.path.basename(p).split(".", 1)[-1]
    key = (os.path.dirname(p).split(os.sep)[-1], kind)
    if key not in kinds and len(kinds) < 6:
        kinds[key] = p
for p in kinds.values():
    describe(p)
PY
