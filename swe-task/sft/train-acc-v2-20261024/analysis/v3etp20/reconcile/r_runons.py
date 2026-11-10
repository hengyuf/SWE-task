#!/usr/bin/env python3
"""reconcile (2026-11-04): own count of argument run-ons (a string argument of a tool call that holds next-turn markup:
'</think>', <tool_call>, </tool_call>, <function=, </function>, <parameter=, </parameter>), excluding OpenCode's
'invalid' tool parts; 'written into a file' = an edit / write part that completed. From each slot's final export.json
(non-summary assistant messages). Also the largest serialized tool input. Read-only; stdlib only.

    /usr/bin/python3 -I -B r_runons.py -> out/runons.txt
"""
import json
import re
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
SPILL = re.compile(r"</think>|<tool_call>|</tool_call>|<function=|</function>|<parameter=|</parameter>")
MODELS = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def scan(ad):
    p = Path(ad) / "export.json" if ad else None
    if not p or not p.exists():
        return None
    calls = files = invalid = 0
    biggest = 0
    for m in json.loads(p.read_text()).get("messages") or []:
        info = m.get("info") or {}
        if info.get("role") != "assistant" or info.get("summary") is True or info.get("mode") == "compaction":
            continue
        for part in m.get("parts") or []:
            if part.get("type") != "tool":
                continue
            st = part.get("state") or {}
            inp = st.get("input")
            biggest = max(biggest, len(json.dumps(inp)) if inp is not None else 0)
            if part.get("tool") == "invalid":
                invalid += 1
                continue
            if any(SPILL.search(s) for s in strings(inp)):
                calls += 1
                if part.get("tool") in ("edit", "write") and st.get("status") == "completed":
                    files += 1
    return calls, files, invalid, biggest


def main():
    L = []
    for m in MODELS:
        ads = []
        for path in sorted(TA.glob(f"results/{m}-group-*.json")):
            for recs in json.loads(path.read_text())["records"].values():
                ads += [r.get("attempt_dir") for r in recs]
        with Pool(12) as pool:
            res = [r for r in pool.map(scan, ads, chunksize=4) if r is not None]
        L.append(f"{m:8s} rollouts {len(res)}: run-on calls {sum(r[0] for r in res)} in {sum(1 for r in res if r[0])} rollouts; "
                 f"written into a file {sum(r[1] for r in res)} in {sum(1 for r in res if r[1])} rollouts; "
                 f"'invalid' tool parts {sum(r[2] for r in res)} in {sum(1 for r in res if r[2])} rollouts; largest tool input {max(r[3] for r in res):,} chars")
    (OUT / "runons.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
