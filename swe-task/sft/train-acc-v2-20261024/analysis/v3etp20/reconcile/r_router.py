#!/usr/bin/env python3
"""reconcile (2026-11-04): router accounting for v3etp20 from the raw router request logs (read-only).
Counts per worker set (leg 1 / tail / leg 2 by Slurm job id from fleets/v3etp20 vllm logs), status, served-by ('worker')
vs routed-to ('workers' list), and calls after 05:50 PST.

    /usr/bin/python3 -I -B r_router.py -> out/router.txt
"""
import collections
import json
import time
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
TAIL = {"21280428", "21280430", "21280432", "21280434"}


def hm(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def main():
    recs = []
    for p in sorted(TA.glob("routers/v3etp20-group-*/student-router-requests.jsonl")):
        g = int(p.parent.name.rsplit("-", 1)[1])
        for line in open(p):
            r = json.loads(line)
            r["_g"] = g
            recs.append(r)
    workers_all = sorted({w for r in recs for w in (r.get("workers") or [])} | {r.get("worker") for r in recs if r.get("worker")})
    L = [f"router records {len(recs)}; status {dict(collections.Counter(r.get('status') for r in recs))}",
         f"waited_s > 0: {sum(1 for r in recs if (r.get('waited_s') or 0) > 0)}; attempts > 1: {sum(1 for r in recs if (r.get('attempts') or 1) > 1)}",
         f"workers seen: {workers_all}"]
    served = collections.Counter(r.get("worker") for r in recs if r.get("status") == 200)
    routed_tail = [r for r in recs if TAIL & set(r.get("workers") or [])]
    served_tail = [r for r in recs if r.get("worker") in TAIL and r.get("status") == 200]
    L.append(f"served (status 200) by a tail worker: {len(served_tail)}; records whose 'workers' list has a tail worker: {len(routed_tail)} "
             f"(status {dict(collections.Counter(r.get('status') for r in routed_tail))}); "
             f"routed to tail but served by another worker: {sum(1 for r in routed_tail if r.get('status') == 200 and r.get('worker') not in TAIL)}")
    L.append("served per tail worker: " + str({w: served[w] for w in sorted(TAIL)}))
    # calls after 05:50 local (PST) on 2026-11-04
    cut = time.mktime(time.strptime("2026-11-04 05:50:00", "%Y-%m-%d %H:%M:%S"))
    after = [r for r in recs if r["ts"] >= cut]
    L.append(f"records with ts >= 05:50: {len(after)}; status {dict(collections.Counter(r.get('status') for r in after))}; "
             f"served by {dict(collections.Counter(r.get('worker') for r in after))}; groups {dict(collections.Counter(r['_g'] for r in after))}")
    for r in recs:
        if r.get("status") != 200:
            L.append(f"  non-200: group {r['_g']} {hm(r['ts'])} +{r.get('duration_s')} s status {r.get('status')} worker {r.get('worker')} workers {r.get('workers')}")
    (OUT / "router.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
