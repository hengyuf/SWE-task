#!/usr/bin/env python3
"""verify-1 misc checks: hmm-loop rollouts by task, loop-call outcomes, unreasoned action calls in http, vLLM serving
load from worker logs, infra sensitivity (v3etp10 model-side infra as fail), parity files. Stdlib only.
    python3 -I -B misc.py -> out/misc.txt"""
import collections, glob, gzip, json, math, random, re, statistics
from fractions import Fraction
from pathlib import Path
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
lines = []
for m in ("v3etp5", "v3etp10", "v3etp20"):
    rows = [json.loads(l) for l in gzip.open(OUT / f"http-{m}.jsonl.gz", "rt")]
    hm = collections.Counter()
    loops = runaway = 0
    unre = collections.Counter()
    for s in rows:
        calls = [c for c in s["calls"] if "non200" not in c]
        if any(any(r["hmm_share"] >= 0.3 for r in c.get("r_loops") or []) for c in calls):
            hm[s["task_id"].split("-cand")[0][4:]] += 1
        for c in calls:
            if c.get("r_loops") and not c.get("comp"):
                loops += 1
                runaway += c["finish"] == "length"
            if c["tools"] and not c.get("comp"):
                if c.get("r_len") is None:
                    unre["reasoning None"] += 1
                elif (c.get("r_strip_len") or 0) == 0:
                    unre["reasoning empty/blank"] += 1
    lines.append(f"{m}: hmm-loop rollouts by task {dict(hm)} total {sum(hm.values())}; loop calls (non-summary) {loops}, "
                 f"finish=length {runaway} ({100 * runaway / max(1, loops):.0f}%); unreasoned action calls (http, all calls incl. subagents) {dict(unre)}")
STATS = re.compile(rb"Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs")
for m, fleet in (("sc14", "base"), ("etp20", "etp20"), ("v3etp5", "v3etp5"), ("v3etp10", "v3etp10"), ("v3etp20", "v3etp20")):
    gen, run = [], []
    for log in glob.glob(str(TA / "fleets" / fleet / "logs" / "vllm-*.log")):
        with open(log, "rb") as h:
            for line in h:
                if b"Running:" in line:
                    mm = STATS.search(line)
                    if mm and int(mm.group(2)) > 0:
                        gen.append(float(mm.group(1)))
                        run.append(int(mm.group(2)))
    lines.append(f"{m}: vLLM samples {len(run)} mean running {statistics.fmean(run):.2f} tok/s per running seq {sum(gen) / sum(run):.1f}")
# infra sensitivity: v3etp10 with its 3 model-side infra slots as fail, vs v3etp20
def load(m):
    d = {}
    for f in sorted(TA.glob(f"results/{m}-group-*.json")):
        for t, rs in json.loads(f.read_text())["records"].items():
            d[t] = [(r["slot"], r["status"]) for r in rs]
    return d
a, b = load("v3etp10"), load("v3etp20")
ms = {("syn-wasmtime-cand-c04-r1b1-bbb510ca", "v3etp10-r0"), ("syn-wasmtime-cand-c04-r1b1-bbb510ca", "v3etp10-r1"),
      ("syn-gomacro-cand-c02-r1b0-e8804dd9", "v3etp10-r4")}
rows = []
for t in sorted(a):
    ga = [(s, st) for s, st in a[t] if st in ("pass", "fail") or (t, s) in ms]
    pa = sum(st == "pass" for s, st in ga)
    gb = [st for s, st in b[t] if st in ("pass", "fail")]
    rows.append((pa, len(ga), sum(x == "pass" for x in gb), len(gb)))
diff = lambda s: 100 * (sum(r[2] for r in s) / sum(r[3] for r in s) - sum(r[0] for r in s) / sum(r[1] for r in s))
rng = random.Random(99)
boots = sorted(diff([rows[rng.randrange(50)] for _ in range(50)]) for _ in range(20000))
diffs = [Fraction(r[2], r[3]) - Fraction(r[0], r[1]) for r in rows]
den = 1
for x in diffs:
    den = den * x.denominator // math.gcd(den, x.denominator)
ints = [int(x * den) for x in diffs]
dist = {0: 1}
for v in ints:
    nd = collections.defaultdict(int)
    for s_, c in dist.items():
        nd[s_ + v] += c; nd[s_ - v] += c
    dist = nd
p = sum(c for s_, c in dist.items() if abs(s_) >= abs(sum(ints))) / 2 ** 50
lines.append(f"infra sensitivity v3etp20 vs v3etp10 (3 model-side infra as fail): {sum(r[0] for r in rows)}/{sum(r[1] for r in rows)} -> "
             f"{sum(r[2] for r in rows)}/{sum(r[3] for r in rows)} diff {diff(rows):+.2f} CI [{boots[500]:.1f}, {boots[19499]:.1f}] exact sign-flip p {p:.3f}")
# parity files
par = collections.Counter()
mism = collections.Counter()
for f in glob.glob(str(TA / "fleets/v3etp20/logs/parity-*.json")):
    d = json.loads(Path(f).read_text())
    par[str(d.get("status") or d.get("result") or d.get("parity"))] += 1
    for k in ("mismatches", "diffs", "failures", "differences"):
        if d.get(k):
            mism[json.dumps(d[k], sort_keys=True)[:300]] += 1
lines.append(f"parity files {sum(par.values())}: {dict(par)}; distinct mismatch payloads {len(mism)}: {list(mism.items())[:2]}")
(OUT / "misc.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
