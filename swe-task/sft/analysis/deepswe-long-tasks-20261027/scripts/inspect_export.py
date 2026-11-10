#!/usr/bin/env python3
"""Show the structure of one OpenCode export.json (read-only)."""
import json, sys
p = sys.argv[1]
with open(p) as f:
    ex = json.load(f)
print("top keys:", list(ex.keys()))
for k, v in ex.items():
    if k != "messages":
        print(k, json.dumps(v)[:600])
msgs = ex.get("messages") or []
print("n messages", len(msgs))
for i, m in enumerate(msgs[:3]):
    info = m.get("info") or {}
    print(i, "info keys", list(info.keys()))
    print("  info", json.dumps({k: v for k, v in info.items() if k not in ("system",)})[:900])
    print("  parts", [ (p.get("type"), p.get("tool")) for p in (m.get("parts") or [])][:10])
sess = {}
for i, m in enumerate(msgs):
    info = m.get("info") or {}
    sess.setdefault(info.get("sessionID"), []).append(i)
print("sessions:", {k: len(v) for k, v in sess.items()})
# print assistant message token rows
for i, m in enumerate(msgs):
    info = m.get("info") or {}
    if info.get("role") == "assistant":
        t = info.get("tokens") or {}
        c = t.get("cache") or {}
        tot = (t.get("input") or 0) + (c.get("read") or 0) + (c.get("write") or 0) + (t.get("output") or 0)
        print(i, info.get("mode"), info.get("agent"), info.get("summary"), info.get("finish"), "in", t.get("input"), "out", t.get("output"), "reas", t.get("reasoning"), "cr", c.get("read"), "cw", c.get("write"), "tot", tot, "err", json.dumps(info.get("error"))[:150] if info.get("error") else "")
    elif info.get("role") == "user" and i > 0:
        print(i, "user", info.get("summary") if not isinstance(info.get("summary"), dict) else "summary-dict", [ (p.get("type")) for p in (m.get("parts") or [])][:5])
