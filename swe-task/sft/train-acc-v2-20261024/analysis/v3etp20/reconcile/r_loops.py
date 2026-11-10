#!/usr/bin/env python3
"""reconcile (2026-11-04): how the text lens's reasoning-loop calls split by outcome under two definitions, from the
lens's own flagged-call records (../degenerate/text/out/models_calls_flagged.jsonl; non-summary calls with >= 1 loop
region = 2,000-char windows with zlib < 0.10). Read-only; stdlib only.

    /usr/bin/python3 -I -B r_loops.py -> out/loops.txt

  lens rule (aggregate.py): terminal if the last region ends within 2,000 chars of the end of the reasoning OR the last
      20,000 chars compress below 0.15; terminal + finish length = runaway, terminal otherwise = self-ended, else escaped
  region-only rule (textlib.py docstring): escaped = >= 2,000 chars of text after the last region; otherwise runaway
      (finish length) / self-ended
  also: calls with finish 'length' among the loop calls, and loop calls per rollout
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "degenerate/text/out/models_calls_flagged.jsonl"


def main():
    out = collections.defaultdict(lambda: collections.Counter())
    roll = collections.defaultdict(set)
    hmm_roll = collections.defaultdict(set)
    for line in open(SRC):
        r = json.loads(line)
        if r["dataset"] != "trainacc":
            continue
        R = r.get("R") or {}
        regs = R.get("regions")
        if not regs or r["comp"]:
            continue
        m = r["model"]
        last = regs[-1]
        tail = R.get("tail")
        big = max(regs, key=lambda g: g["end"] - g["start"])
        lens_term = last["after"] <= 2000 or (tail is not None and tail < 0.15)
        reg_term = last["after"] < 2000
        out[m]["calls"] += 1
        out[m]["finish_length"] += r["finish"] == "length"
        out[m]["lens:" + ("runaway" if lens_term and r["finish"] == "length" else "self-ended" if lens_term else "escaped")] += 1
        out[m]["region:" + ("runaway" if reg_term and r["finish"] == "length" else "self-ended" if reg_term else "escaped")] += 1
        roll[m].add(r["id"])
        if big["cls"] == "hmm":
            hmm_roll[m].add(r["id"])
    L = []
    for m in ("sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"):
        c = out[m]
        L.append(f"{m:8s} loop calls {c['calls']:3d} in {len(roll[m]):2d} rollouts (hmm-dominant {len(hmm_roll[m])}); finish length {c['finish_length']} "
                 f"({100 * c['finish_length'] / max(1, c['calls']):.0f}%) | lens rule runaway/self-ended/escaped "
                 f"{c['lens:runaway']}/{c['lens:self-ended']}/{c['lens:escaped']} | region-only rule {c['region:runaway']}/{c['region:self-ended']}/{c['region:escaped']}")
    (HERE / "out/loops.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
