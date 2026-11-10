#!/usr/bin/env python3
"""train-acc-v2 (2026-10-31): did any rollout see a serving gap? Per model, from each group's student router request
log (TA/routers/<model>-group-<K>/student-router-requests.jsonl: one line per routed call with status, waited_s = time
the router held the call for lack of a routable worker, attempts, aborts, moved_from) and the router logs
(student-router.log: "no routable worker" / worker state changes).

    python3 -I -B serving_gaps.py [MODEL ...] -> serving_gaps.json (stdout: summary)
Read-only on the routers.
2026-11-04: copy in analysis/v3etp20/compare (HERE = this directory), + v3etp20.
"""
import collections
import json
from pathlib import Path
import re
import sys
import time

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
MODELS = sys.argv[1:] or ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def scan(model):
    out = {"requests": 0, "status": collections.Counter(), "waited_gt_0": 0, "waited_gt_10s": 0, "waited_gt_60s": 0,
           "waited_max_s": 0.0, "waited_sum_s": 0.0, "attempts_gt_1": 0, "aborts": 0, "moved": 0,
           "moved_why": collections.Counter(), "errors": collections.Counter(), "non200": [], "waits": []}
    for path in sorted(TA.glob(f"routers/{model}-group-*/student-router-requests.jsonl")):
        group = path.parent.name.rsplit("-", 1)[1]
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out["requests"] += 1
            out["status"][str(r.get("status"))] += 1
            w = r.get("waited_s") or 0.0
            out["waited_sum_s"] += w
            out["waited_max_s"] = max(out["waited_max_s"], w)
            out["waited_gt_0"] += w > 0
            out["waited_gt_10s"] += w > 10
            out["waited_gt_60s"] += w > 60
            if w > 10:
                out["waits"].append({"group": group, "ts": time.strftime("%H:%M:%S", time.localtime(r["ts"])),
                                     "waited_s": round(w, 1), "status": r.get("status"), "affinity": r.get("affinity")})
            out["attempts_gt_1"] += (r.get("attempts") or 1) > 1
            out["aborts"] += r.get("aborts") or 0
            if r.get("moved_from"):
                out["moved"] += 1
                out["moved_why"][str(r.get("moved_why"))] += 1
            if r.get("status") != 200:
                out["errors"][str(r.get("error"))[:120]] += 1
                out["non200"].append({"group": group, "ts": time.strftime("%m-%d %H:%M:%S", time.localtime(r["ts"])),
                                      "status": r.get("status"), "worker": r.get("worker"), "workers": r.get("workers"),
                                      "attempts": r.get("attempts"), "waited_s": r.get("waited_s"),
                                      "error": str(r.get("error"))[:200], "affinity": r.get("affinity"),
                                      "duration_s": r.get("duration_s")})
    no_worker = collections.Counter()
    for log in sorted(TA.glob(f"routers/{model}-group-*/student-router.log")):
        with open(log, errors="replace") as handle:
            for line in handle:
                low = line.lower()
                if "no routable" in low or "no worker" in low or "429" in line:
                    no_worker[re.sub(r"\d", "0", line[24:120])] += 1
    out["router_log_no_worker_lines"] = dict(no_worker.most_common(10))
    out["status"] = dict(out["status"])
    out["moved_why"] = dict(out["moved_why"])
    out["errors"] = dict(out["errors"])
    out["waited_sum_s"] = round(out["waited_sum_s"], 1)
    out["waits"] = out["waits"][:50]
    return out


def main():
    result = {m: scan(m) for m in MODELS}
    tmp = HERE / ".serving_gaps.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    tmp.replace(HERE / "serving_gaps.json")
    for m, r in result.items():
        print(m, {k: v for k, v in r.items() if k not in ("non200", "waits")})
        for x in r["non200"][:12]:
            print("   non200", x)
        for x in r["waits"][:12]:
            print("   wait", x)


if __name__ == "__main__":
    main()
