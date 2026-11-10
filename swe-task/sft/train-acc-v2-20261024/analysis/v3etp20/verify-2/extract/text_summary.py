#!/usr/bin/env python3
"""verify-2: summarize out/text_<model>.jsonl.gz (textscan.py)."""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

for model in sys.argv[1:]:
    rows = [json.loads(x) for x in gzip.open(HERE / "out" / f"text_{model}.jsonl.gz", "rt")]
    n = len(rows)
    f_any = f_main = f_cjk = 0
    scripts = Counter()
    ex = []
    loop_calls = Counter()
    hmm_rolls = 0
    loop_rolls = 0
    units = Counter()
    for r in rows:
        fr = [c for c in r["calls"] if c.get("foreign")]
        if fr:
            f_any += 1
            if any(not c.get("comp") for c in fr):
                f_main += 1
            if any(any(k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL") for k in c["foreign"]) for c in fr):
                f_cjk += 1
            for c in fr:
                scripts.update(c["foreign"].keys())
            if len(ex) < 6:
                ex.append((r["task"][4:30], r["slot"], fr[0]["foreign_ex"][:120]))
        lc = [c for c in r["calls"] if c.get("loops")]
        if lc:
            loop_rolls += 1
        if any(any(g["hmm"] for g in c["loops"]) for c in lc):
            hmm_rolls += 1
        for c in lc:
            if c["fin"] == "length":
                k = "length"
            elif any(g["after"] >= 2000 for g in c["loops"]):
                k = "escaped"
            else:
                k = "self_ended"
            loop_calls[k] += 1
            for g in c["loops"]:
                units["hmm" if g["hmm"] else g["top"]] += 1
    tot = sum(loop_calls.values())
    print(f"{model}: slots {n}; rollouts with spontaneous non-Latin script in reasoning (any call) {f_any}, excluding compaction "
          f"calls {f_main}, CJK/kana/hangul {f_cjk}; scripts {dict(scripts.most_common(8))}")
    print(f"  loop calls (any unit) {tot}: {dict(loop_calls)}; share length {loop_calls['length'] / max(1, tot):.2f}; rollouts with a loop "
          f"{loop_rolls}; with an hmm loop region {hmm_rolls}; top region units {units.most_common(8)}")
    for e in ex:
        print("   ex", e)
