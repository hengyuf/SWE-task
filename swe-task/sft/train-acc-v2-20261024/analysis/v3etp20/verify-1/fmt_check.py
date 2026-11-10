#!/usr/bin/env python3
"""verify-1: format checks from each final-attempt export.json (my own implementation of the format lens's stated
definitions): ARG_SPILL (a string tool argument holding '</think>', <tool_call>, </tool_call>, <function=, </function>,
<parameter=, </parameter>) and its completed edit/write subset; reasoning ending in call markup with no tool part (B01);
call markup in the visible text (B05); 'invalid' tool parts; serialized tool input >= 65,536 chars.
    python3 -I -B fmt_check.py MODEL... -> out/fmt_check.txt"""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
SPILL = re.compile(r"</think>|</?tool_call>|<function=|</function>|<parameter=|</parameter>")
TAIL = re.compile(r"(</tool_call>|</function>|</parameter>)\s*$")
MARK = re.compile(r"<tool_call>|</tool_call>|<function=|</function>|<parameter=")
def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
def scan(a):
    try:
        exp = json.loads((Path(a) / "export.json").read_text())
    except (OSError, ValueError):
        return None
    o = {"spill": 0, "spill_file": 0, "b01": 0, "b05": 0, "invalid": 0, "big": 0}
    for m in exp.get("messages") or []:
        info = m.get("info") or {}
        if info.get("role") != "assistant" or info.get("mode") == "compaction" or info.get("summary") is True:
            continue
        parts = m.get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        s = text.lstrip()
        if s.startswith("<think>"):
            if "</think>" in s:
                r, v = s[7:].split("</think>", 1)
            else:
                r, v = s[7:], ""
        else:
            r, v = "", text
        r = "".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + r
        tools = [p for p in parts if p.get("type") == "tool"]
        if TAIL.search(r.rstrip()) and not tools:
            o["b01"] += 1
        if MARK.search(v):
            o["b05"] += 1
        for t in tools:
            st = t.get("state") or {}
            inp = st.get("input")
            if t.get("tool") == "invalid":
                o["invalid"] += 1
            if inp is not None and len(json.dumps(inp)) >= 65536:
                o["big"] += 1
            if any(SPILL.search(x) for x in strings(inp)):
                o["spill"] += 1
                if t.get("tool") in ("edit", "write") and st.get("status") == "completed":
                    o["spill_file"] += 1
    return o
lines = []
for model in sys.argv[1:]:
    ads = [r["attempt_dir"] for f in sorted(TA.glob(f"results/{model}-group-*.json")) for rs in json.loads(f.read_text())["records"].values() for r in rs]
    with Pool(12) as pool:
        rows = [x for x in pool.map(scan, ads, chunksize=2) if x]
    lines.append(f"{model}: rollouts {len(rows)} | run-on (ARG_SPILL) calls {sum(r['spill'] for r in rows)} rollouts {sum(r['spill'] > 0 for r in rows)}; "
                 f"written into a file: calls {sum(r['spill_file'] for r in rows)} rollouts {sum(r['spill_file'] > 0 for r in rows)} | "
                 f"B01 reasoning ends in call markup, no tool part: rollouts {sum(r['b01'] > 0 for r in rows)} | B05 visible markup: rollouts "
                 f"{sum(r['b05'] > 0 for r in rows)} | invalid tool rollouts {sum(r['invalid'] > 0 for r in rows)} calls {sum(r['invalid'] for r in rows)} | "
                 f">=65,536-char input: calls {sum(r['big'] for r in rows)}")
(HERE / "out/fmt_check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
