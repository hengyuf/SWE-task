"""verify-2 adhoc: print the reasoning around tool-call markup / special tokens for v3etp20 turns flagged by extract."""
import gzip
import json
import re
import sys

src = sys.argv[1]
targets = []
for l in gzip.open(src, "rt"):
    s = json.loads(l)
    for c in s["calls"]:
        r = c.get("resp")
        if r and (r["rmark"] or r["rspec"]):
            targets.append((s["attempt_dir"], r["rid"], r["ntc"], r["fin"], r["rmark"], r["rspec"], s["task"], s["slot"]))
print(len(targets))
for ad, rid, ntc, fin, rm, rs, task, slot in targets:
    with open(ad + "/http.jsonl", "rb") as f:
        for raw in f:
            if rid.encode() in raw and (b'"event": "response"' in raw[:2000] or b'"event":"response"' in raw[:2000]):
                d = json.loads(raw)
                b = json.loads(d["body"])
                m = b["choices"][0]["message"]
                rea = m.get("reasoning") or ""
                mm = re.search(r"<tool_call>|<function=|<\|[A-Za-z_]+\|>", rea)
                print("==", task[4:34], slot, rid, "ntc", ntc, "fin", fin, "mark", rm, "spec", rs,
                      "has <tool_call>", "<tool_call>" in rea, "has <function=", "<function=" in rea, "rlen", len(rea))
                print("   ", repr(rea[max(0, mm.start() - 150):mm.start() + 250]) if mm else None)
                break
