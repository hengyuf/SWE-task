#!/usr/bin/env python3
"""Show Responses-API output item types/keys of row N in a rollouts.jsonl (read-only)."""
import json, sys
from collections import Counter
p, n = sys.argv[1], int(sys.argv[2])
with open(p) as f:
    for i, line in enumerate(f):
        if i == n:
            r = json.loads(line)
            break
out = r["response"]["output"]
c = Counter(it.get("type") for it in out)
print(c)
seen = set()
for it in out:
    t = it.get("type")
    if t in seen:
        continue
    seen.add(t)
    print(t, {k: (json.dumps(v)[:200]) for k, v in it.items()})
inp = r["responses_create_params"]["input"]
print("input items", len(inp), [ (x.get("role"), x.get("type"), len(json.dumps(x))) for x in inp])
print("tools", len(r["responses_create_params"].get("tools") or []))
