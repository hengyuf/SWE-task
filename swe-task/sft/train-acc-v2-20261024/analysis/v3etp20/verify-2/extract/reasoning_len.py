#!/usr/bin/env python3
"""verify-2: reasoning length per reasoning turn (main-session non-compaction calls with non-empty stripped reasoning),
in CHARACTERS (no tokenizer in /usr/bin/python3; a proxy for the analyst's token counts), the two-sample KS distance to
base sc14, the share of main-session action turns with visible text, and per-call completion tokens."""
import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(model):
    rs, act, vis = [], 0, 0
    with gzip.open(HERE / "out" / f"{model}.jsonl.gz", "rt") as fh:
        for line in fh:
            s = json.loads(line)
            calls = s["calls"]
            main_sys = calls[0].get("sys") if calls else None
            for c in calls:
                if c.get("comp") or not c.get("resp") or c.get("sys") != main_sys:
                    continue
                r = c["resp"]
                if r["rs"] > 0:
                    rs.append(r["rs"])
                if r["ntc"] > 0:
                    act += 1
                    vis += r["cs"] > 0
    return np.array(rs), act, vis


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


models = sys.argv[1:]
data = {m: load(m) for m in models}
base = data.get("sc14", (None,))[0]
for m in models:
    rs, act, vis = data[m]
    line = (f"{m}: reasoning turns {len(rs)}; chars median {statistics.median(rs):.0f} mean {rs.mean():.0f} p90 "
            f"{np.percentile(rs, 90):.0f}; % < 150 chars {100 * (rs < 150).mean():.1f}; action turns with visible text "
            f"{100 * vis / max(1, act):.1f}% ({vis}/{act})")
    if base is not None and m != "sc14":
        line += f"; KS distance to sc14 (chars) {ks(rs, base):.3f}"
    print(line)
