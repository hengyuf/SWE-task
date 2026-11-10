#!/usr/bin/env python3
"""verify-2: router request-log stats for one model (routers/<model>-group-*/student-router-requests.jsonl)."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")


def hm(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


def main(model):
    n = 0
    st = Counter()
    waits = []
    non200 = []
    workers = Counter()
    late = Counter()
    by_worker_last = {}
    clamps = 0
    for f in sorted((TA / "routers").glob(f"{model}-group-*/student-router-requests.jsonl")):
        g = f.parent.name
        for line in open(f):
            d = json.loads(line)
            n += 1
            st[d.get("status")] += 1
            if (d.get("waited_s") or 0) > 0:
                waits.append((g, d["ts"], d["waited_s"]))
            if d.get("status") != 200:
                non200.append((g, hm(d["ts"]), d["ts"], d.get("status"), d.get("error"), d.get("attempts"), d.get("duration_s"),
                               d.get("worker")))
            workers[d.get("worker")] += 1
            if d.get("clamps"):
                clamps += 1
            by_worker_last[d.get("worker")] = max(by_worker_last.get(d.get("worker"), 0), d["ts"])
    print(model, "routed", n, "status", dict(st), "waits>0", len(waits), "with clamps", clamps)
    for w in waits[:20]:
        print("  wait", w)
    for x in non200:
        print("  non200", x)
    for w, c in sorted(workers.items(), key=lambda kv: str(kv[0])):
        print("  worker", w, c, "last", hm(by_worker_last[w]))


if __name__ == "__main__":
    main(sys.argv[1])
