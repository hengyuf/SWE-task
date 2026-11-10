#!/usr/bin/env python3
"""verify-1: reasoning-text measures per rollout from each slot's final-attempt export.json (my own turn split as in
scan_export.py). Stdlib only, read-only.
  * reasoning quality over action turns with reasoning: exact repeat of an earlier reasoning of the rollout, near repeat
    (first 60 chars, lowercased, digits -> 0), short (< 100 chars) generic (no path/identifier/number), and loop-like
    long reasoning (>= 4,000 chars with whole-text zlib ratio < 0.15);
  * floods in every non-summary turn's reasoning, outside loop regions (2k windows stride 1k, zlib < 0.10; regions cut
    out with 1,001-space gaps): max interjections (hm+/wait/hold on/actually/oh+/ah+/ugh+/whoa/oops/argh+/huh/whatever/
    hmph) in any 1,000-char window >= 22 with the window's zlib ratio >= 0.25; max em dashes in any 1,000-char window
    >= 26 with ratio >= 0.25 (thresholds = the text lens's corpus-derived ones, used as given).
    python3 -I -B text_scan.py MODEL... -> out/text_scan.txt
"""
import collections
import gzip
import json
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERJ = re.compile(r"(?i)(?<![\w'])(hm+|wait|hold on|actually|oh+|ah+|ugh+|whoa|oops|argh+|huh|whatever|hmph)(?![\w'])")
SPECIFIC = re.compile(r"[/_`()\[\]{}.:=<>#]|\d|[a-z][A-Z]")


def zr(s):
    b = s.encode("utf-8", "surrogatepass")
    return len(zlib.compress(b, 6)) / max(1, len(b))


def regions(text):
    n = len(text)
    if n < 500:
        return []
    if n <= 2000:
        wins = [(0, n)]
    else:
        st = list(range(0, n - 2000 + 1, 1000))
        if st[-1] != n - 2000:
            st.append(n - 2000)
        wins = [(s, s + 2000) for s in st]
    regs, cur = [], None
    for s, e in wins:
        if zr(text[s:e]) < 0.10:
            if cur and s <= cur[1]:
                cur[1] = e
            else:
                if cur:
                    regs.append(cur)
                cur = [s, e]
        elif cur:
            regs.append(cur)
            cur = None
    if cur:
        regs.append(cur)
    return regs


def maxwin(pos, width=1000):
    best, at, i = 0, 0, 0
    for j in range(len(pos)):
        while pos[j] - pos[i] > width:
            i += 1
        if j - i + 1 > best:
            best, at = j - i + 1, pos[i]
    return best, at


def split_think(text):
    s = text.lstrip()
    if not s.startswith("<think>"):
        return None, text
    if "</think>" not in s:
        return s[len("<think>"):], ""
    a, b = s[len("<think>"):].split("</think>", 1)
    return a, b


def scan(args):
    model, task, slot, adir = args
    out = {"model": model, "task": task, "slot": slot, "export": False}
    try:
        exp = json.loads((Path(adir) / "export.json").read_text())
    except (OSError, ValueError):
        return out
    out["export"] = True
    c = collections.Counter()
    seen, seen_head = set(), set()
    flood_i = flood_e = 0
    for m in exp.get("messages") or []:
        info = m.get("info") or {}
        if info.get("role") != "assistant" or info.get("mode") == "compaction" or info.get("summary") is True:
            continue
        parts = m.get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        inner, vis = split_think(text)
        r = ("".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or "")).strip()
        tools = [p for p in parts if p.get("type") == "tool"]
        if r:
            regs = regions(r)
            if regs:
                pieces, last = [], 0
                for s, e in regs:
                    pieces.append(r[last:s])
                    last = e
                pieces.append(r[last:])
                base = (" " * 1001).join(p for p in pieces if p)
            else:
                base = r
            pos = [mm.start() for mm in INTERJ.finditer(base)]
            if len(pos) >= 22:
                mw, at = maxwin(pos)
                if mw >= 22 and zr(base[at:at + 1000]) >= 0.25:
                    flood_i += 1
            pos = [i for i, ch in enumerate(base) if ch == "—"]
            if len(pos) >= 26:
                mw, at = maxwin(pos)
                if mw >= 26 and zr(base[at:at + 1000]) >= 0.25:
                    flood_e += 1
        if not tools or not r:
            continue
        c["reasoned_actions"] += 1
        if r in seen:
            c["exact"] += 1
        seen.add(r)
        head = re.sub(r"\d+", "0", " ".join(r.lower().split()))[:60]
        if head in seen_head:
            c["near"] += 1
        seen_head.add(head)
        if len(r) < 100 and not SPECIFIC.search(r):
            c["generic"] += 1
        if len(r) >= 4000 and zr(r) < 0.15:
            c["looplike"] += 1
    out.update(c)
    out["flood_interj_calls"] = flood_i
    out["flood_emdash_calls"] = flood_e
    return out


def main():
    TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
    lines = []
    for model in sys.argv[1:]:
        jobs = []
        for f in sorted(TA.glob(f"results/{model}-group-*.json")):
            d = json.loads(f.read_text())
            for t, rs in d["records"].items():
                for r in rs:
                    jobs.append((model, t, r["slot"], r["attempt_dir"]))
        with Pool(12) as pool:
            rows = pool.map(scan, jobs, chunksize=2)
        ex = [r for r in rows if r["export"]]
        ra = sum(r.get("reasoned_actions", 0) for r in ex)
        pct = lambda k: 100 * sum(r.get(k, 0) for r in ex) / max(1, ra)
        lines.append(f"{model}: rollouts {len(ex)} reasoned actions {ra} | exact {pct('exact'):.2f}% near {pct('near'):.2f}% "
                     f"generic {pct('generic'):.2f}% | loop-like (whole zlib) turns {sum(r.get('looplike', 0) for r in ex)} rollouts "
                     f"{sum(r.get('looplike', 0) > 0 for r in ex)} | interjection-flood rollouts {sum(r['flood_interj_calls'] > 0 for r in ex)} "
                     f"em-dash-flood rollouts {sum(r['flood_emdash_calls'] > 0 for r in ex)}")
    (HERE / "out/text_scan.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
