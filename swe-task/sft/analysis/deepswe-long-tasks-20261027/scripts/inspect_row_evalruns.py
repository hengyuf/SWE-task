#!/usr/bin/env python3
"""Print top-level keys/types/small values of the first row of a rollouts.jsonl (read-only)."""
import json, sys
p = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
with open(p) as f:
    for i, line in enumerate(f):
        if i < n:
            continue
        r = json.loads(line)
        break
def show(d, pre="", depth=0):
    for k, v in d.items():
        if isinstance(v, dict) and depth < 2:
            print(f"{pre}{k}: dict({len(v)})")
            show(v, pre + "  ", depth + 1)
        elif isinstance(v, (list, tuple)):
            print(f"{pre}{k}: list({len(v)}) first={json.dumps(v[0])[:200] if v else None}")
        elif isinstance(v, str):
            print(f"{pre}{k}: str({len(v)}) {v[:200]!r}")
        else:
            print(f"{pre}{k}: {v!r}"[:300])
show(r)
