#!/usr/bin/env python3
"""verify-2: own text scan of every 200 response (final attempt of each slot) for
  * spontaneous non-Latin script runs in the reasoning (letters of a non-Latin script whose run does not occur in the
    call's context = system/user/tool message text of its request; Greek runs < 3 chars ignored),
  * repetition-loop regions in the reasoning (2,000-char windows, stride 1,000, zlib ratio < 0.10), with how the call
    ended (length cut / region at the end of a self-ended call / escaped = >= 2,000 chars after the region) and whether
    the region's most common word token is an "hmm" (h+m+).
Writes out/text_<model>.jsonl.gz (one row per slot).

    python3 -I -B textscan.py <model> [--procs N]
"""
import argparse
import gzip
import json
import re
import sys
import unicodedata
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
LATIN_OK = {"LATIN"}
RE_WORD = re.compile(r"\w+")


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join((x.get("text") or "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def foreign_runs(text):
    """maximal runs of letters whose Unicode name starts with a non-Latin script word (CJK, HIRAGANA, CYRILLIC, ...)."""
    runs = []
    cur, cur_sc, start = [], None, 0
    for i, ch in enumerate(text):
        sc = None
        if ord(ch) > 0x24F and unicodedata.category(ch).startswith("L"):
            try:
                sc = unicodedata.name(ch).split()[0]
            except ValueError:
                sc = "UNNAMED"
            if sc in ("LATIN", "MODIFIER", "FULLWIDTH", "HALFWIDTH", "MATHEMATICAL", "SMALL"):
                sc = None
            if sc in ("KATAKANA-HIRAGANA",):
                sc = "KATAKANA"
        if sc:
            if cur and sc != cur_sc and not (sc in ("CJK", "HIRAGANA", "KATAKANA") and cur_sc in ("CJK", "HIRAGANA", "KATAKANA")):
                runs.append((cur_sc, "".join(cur), start))
                cur = []
            if not cur:
                start = i
                cur_sc = sc
            cur.append(ch)
        else:
            if cur:
                runs.append((cur_sc, "".join(cur), start))
                cur = []
    if cur:
        runs.append((cur_sc, "".join(cur), start))
    return [r for r in runs if not (r[0] == "GREEK" and len(r[1]) < 3)]


def zr(s):
    b = s.encode("utf-8", "surrogatepass")
    return len(zlib.compress(b, 6)) / max(1, len(b))


def loop_regions(text):
    n = len(text)
    if n < 2000:
        return []
    starts = list(range(0, n - 2000 + 1, 1000))
    if starts[-1] != n - 2000:
        starts.append(n - 2000)
    regs, cur = [], None
    for s in starts:
        r = zr(text[s:s + 2000])
        if r < 0.10:
            if cur and s <= cur[1]:
                cur[1] = s + 2000
            else:
                if cur:
                    regs.append(cur)
                cur = [s, s + 2000]
        else:
            if cur:
                regs.append(cur)
                cur = None
    if cur:
        regs.append(cur)
    out = []
    for s, e in regs:
        words = Counter(w.lower() for w in RE_WORD.findall(text[s:e][:30000]))
        top = words.most_common(1)[0][0] if words else ""
        out.append({"s": s, "e": e, "after": n - e, "top": top[:20], "hmm": bool(re.fullmatch(r"h+m+", top)),
                    "sample": text[s + (e - s) // 2: s + (e - s) // 2 + 40]})
    return out


def work(row):
    ad = Path(row["attempt_dir"])
    res = {k: row[k] for k in ("model", "task", "slot", "status")}
    calls = []
    last_req = {}
    hp = ad / "http.jsonl"
    if not hp.exists():
        res["calls"] = calls
        return res
    with open(hp, "rb") as fh:
        for raw in fh:
            if b'"event": "request"' in raw[:2000] or b'"event":"request"' in raw[:2000]:
                d = json.loads(raw)
                if d.get("event") == "request":
                    last_req[d.get("model_call_id")] = raw
                continue
            if b'"event": "response"' not in raw[:3000] and b'"event":"response"' not in raw[:3000]:
                continue
            d = json.loads(raw)
            if d.get("event") != "response" or d.get("status") != 200 or not d.get("body"):
                continue
            try:
                b = json.loads(d["body"]) if isinstance(d["body"], str) else d["body"]
            except ValueError:
                continue
            ch = (b.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            fin = ch.get("finish_reason")
            rec = {}
            fr = foreign_runs(reasoning)
            if fr:
                rq = last_req.get(d.get("model_call_id"))
                ctx = ""
                comp = False
                if rq is not None:
                    p = json.loads(rq).get("payload") or {}
                    msgs = p.get("messages") or []
                    if msgs and msgs[0].get("role") == "system" and text_of(msgs[0].get("content")).startswith(
                            "You are an anchored context summarization assistant"):
                        comp = True
                    ctx = "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "user", "tool"))
                spont = [(sc, run) for sc, run, _ in fr if run not in ctx]
                if spont:
                    first = next(x for x in fr if x[1] not in ctx)
                    rec["foreign"] = Counter(sc for sc, _ in spont)
                    rec["foreign_ex"] = reasoning[max(0, first[2] - 40): first[2] + len(first[1]) + 40]
                    rec["comp"] = comp
            regs = loop_regions(reasoning)
            if regs:
                rec["loops"] = regs
                rec["fin"] = fin
                rec["rlen"] = len(reasoning)
            if rec:
                rec["rid"] = b.get("id")
                calls.append(rec)
            last_req.pop(d.get("model_call_id"), None)
    res["calls"] = calls
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    rows = []
    for f in sorted((TA / "results").glob(f"{a.model}-group-*.json")):
        d = json.loads(f.read_text())
        for t, recs in d["records"].items():
            for r in recs:
                rows.append({"model": a.model, "task": t, "slot": r["slot"], "status": r["status"], "attempt_dir": r["attempt_dir"]})
    OUT.mkdir(exist_ok=True)
    with Pool(a.procs) as pool, gzip.open(OUT / f"text_{a.model}.jsonl.gz", "wt") as fo:
        for i, res in enumerate(pool.imap_unordered(work, rows, chunksize=1)):
            fo.write(json.dumps(res) + "\n")
            if i % 25 == 0:
                print(a.model, i, file=sys.stderr, flush=True)
    print("done", a.model, len(rows))


if __name__ == "__main__":
    main()
