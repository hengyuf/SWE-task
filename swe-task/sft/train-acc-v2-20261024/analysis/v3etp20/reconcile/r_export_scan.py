#!/usr/bin/env python3
"""reconcile (2026-11-04): own per-slot scan of the OpenCode session export (export.json) of each slot's FINAL attempt
(attempt_dir from results/<model>-group-*.json). Read-only on the probes; stdlib only.

    /usr/bin/python3 -I -B r_export_scan.py MODEL [MODEL ...] [--procs N] -> out/export-<MODEL>.jsonl.gz

Per slot:
  summaries   assistant messages with info.summary == True or info.mode == 'compaction': tool-call markup anywhere in the
              text (<tool_call>, </tool_call>, <function=, </function>, <parameter=), tool parts, finish, '## Goal' in
              the body (text after </think>), commit instruction in the body (the endings lens's RE_COMMIT_INSTR, copied
              verbatim), body head
  actions     non-summary assistant messages with >= 1 tool part; unreasoned = reasoning (inside <think>..</think>, or
              the whole unclosed think) empty after strip
  markup      non-summary reasoning containing tool-call markup: opener '<tool_call>' present?, tool parts of the message,
              finish, the markup strings, a snippet
  length      messages with finish 'length': output tokens, reasoning length, zlib ratio of the last 20,000 chars,
              dominant token of the last 4,000 chars (tokens = \\w+ or one non-space symbol), last 120 chars
  foreign     maximal non-Latin letter runs in reasoning (the text lens's character class; Greek runs < 3 chars skipped);
              copied = the run occurs in user text / tool output / tool error seen before the message, else spontaneous;
              40 chars either side; the rest of the line on either side (up to 400 chars)
"""
import argparse
import collections
import gzip
import json
import re
import sys
import unicodedata
import zlib
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
RE_TOOLCALL = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")
RE_COMMIT_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work))"
    r"|commit\b[^.\n]{0,40}\b(?:when|once|after)\b[^.\n]{0,30}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and|,)\s+commit\b"
    r"|commit(?:ting)? (?:it |them |everything |all )?at (?:the )?end"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`)"
    r"|new branch from main and commit|\bcommit\s*[.;]?\s*(?:\n|$)|^\s*[-*\d.)]+\s*commit\b)", re.I | re.M)
RE_FOREIGN = re.compile(
    "[Ͱ-ϿЀ-ԯ԰-֏֐-׿؀-ۿ܀-ࣿऀ-෿"
    "฀-๿຀-࿿က-႟Ⴀ-ჿᄀ-ᇿሀ-᎟぀-ヿ"
    "㄀-ㄯ㄰-㆏ㆠ-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯"
    "豈-﫿ｦ-ﾟ]+")
RE_TOK = re.compile(r"\w+|[^\w\s]")


def split_think(text):
    """(reasoning, visible, closed) of an assistant text that starts with <think>."""
    s = text.lstrip()
    if not s.startswith("<think>"):
        return "", text, None
    body = s[len("<think>"):]
    if "</think>" in body:
        r, v = body.split("</think>", 1)
        return r, v, True
    return body, "", False


def zr(s):
    b = s.encode("utf-8", "surrogatepass")
    return len(zlib.compress(b, 6)) / max(1, len(b))


def script(ch):
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNNAMED"


def scan(slot):
    ad = Path(slot["attempt_dir"]) if slot.get("attempt_dir") else None
    res = dict(slot)
    path = ad / "export.json" if ad else None
    if not path or not path.exists():
        res["export"] = False
        return res
    res["export"] = True
    d = json.loads(path.read_text())
    msgs = d.get("messages") or []
    ctx = []          # list of strings seen so far (user text, tool output/error)
    ctx_join = ""
    ctx_dirty = False
    summaries, markup, length, foreign = [], [], [], []
    n_action = n_unreasoned = n_assistant = 0
    unreasoned = []
    for i, m in enumerate(msgs):
        info = m.get("info") or {}
        parts = m.get("parts") or []
        role = info.get("role")
        if role == "user":
            for p in parts:
                if p.get("type") == "text" and p.get("text"):
                    ctx.append(p["text"])
                    ctx_dirty = True
            continue
        if role != "assistant":
            continue
        n_assistant += 1
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        tools = [p for p in parts if p.get("type") == "tool"]
        reasoning, visible, closed = split_think(text)
        is_summary = info.get("summary") is True or info.get("mode") == "compaction"
        fin = info.get("finish")
        out_tok = (info.get("tokens") or {}).get("output")
        if is_summary:
            body = visible if closed else ("" if closed is False else text)
            summaries.append({"i": i, "markup": bool(RE_TOOLCALL.search(text)), "markup_in_body": bool(RE_TOOLCALL.search(body)),
                              "tools": len(tools), "finish": fin, "goal": "## Goal" in body, "goal_anywhere": "## Goal" in text,
                              "instr": bool(RE_COMMIT_INSTR.search(body)), "body_len": len(body), "think_len": len(reasoning),
                              "closed": closed, "head": body[:160], "out": out_tok})
        else:
            if tools:
                n_action += 1
                if not reasoning.strip():
                    n_unreasoned += 1
                    unreasoned.append({"i": i, "closed": closed, "text_head": text[:80], "tools": [p.get("tool") for p in tools]})
            mk = RE_TOOLCALL.findall(reasoning)
            if mk:
                j = RE_TOOLCALL.search(reasoning).start()
                markup.append({"i": i, "opener": "<tool_call>" in reasoning, "kinds": sorted(set(mk)), "tools": len(tools),
                               "finish": fin, "closed": closed, "snippet": reasoning[max(0, j - 120): j + 120]})
        if fin == "length":
            tail = reasoning[-4000:]
            toks = RE_TOK.findall(tail)
            c = collections.Counter(toks)
            top = c.most_common(3)
            length.append({"i": i, "summary": is_summary, "out": out_tok, "r_len": len(reasoning), "closed": closed,
                           "tail_zr": round(zr(reasoning[-20000:]), 4) if reasoning else None, "top": top,
                           "spaces_tail": tail.count(" "), "last": reasoning[-120:]})
        # foreign runs in reasoning (all assistant messages, incl. summaries)
        if reasoning and not reasoning.isascii():
            if ctx_dirty:
                ctx_join = "\n".join(ctx)
                ctx_dirty = False
            for mm in RE_FOREIGN.finditer(reasoning):
                run = mm.group(0)
                sc = script(run[0])
                if sc == "GREEK" and len(run) < 3:
                    continue
                ls = reasoning.rfind("\n", 0, mm.start()) + 1
                le = reasoning.find("\n", mm.end())
                le = len(reasoning) if le < 0 else le
                foreign.append({"i": i, "summary": is_summary, "run": run[:24], "script": sc, "copied": run in ctx_join,
                                "left": reasoning[max(0, mm.start() - 40):mm.start()], "right": reasoning[mm.end():mm.end() + 40],
                                "line_left": reasoning[max(ls, mm.start() - 400):mm.start()], "line_right": reasoning[mm.end():min(le, mm.end() + 400)],
                                "line_left_full": mm.start() - ls <= 400})
        for p in tools:
            st = p.get("state") or {}
            o = st.get("output")
            if o is not None:
                ctx.append(o if isinstance(o, str) else json.dumps(o))
                ctx_dirty = True
            e = st.get("error")
            if isinstance(e, str) and e:
                ctx.append(e)
                ctx_dirty = True
    res.update({"n_assistant": n_assistant, "n_action": n_action, "n_unreasoned": n_unreasoned, "unreasoned": unreasoned,
                "summaries": summaries, "markup": markup, "length": length, "foreign": foreign,
                "n_compaction_req": sum(1 for m in msgs if (m.get("info") or {}).get("role") == "user"
                                        and any(p.get("type") == "compaction" for p in (m.get("parts") or [])))})
    return res


def slots_of(model):
    out = []
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, recs in data["records"].items():
            for r in recs:
                out.append({"model": model, "group": data["group"], "task_id": task_id, "rep": int(r["slot"].rsplit("-r", 1)[1]),
                            "status": r["status"], "patch_chars": r.get("patch_chars") or 0, "agent_error": r.get("agent_error"),
                            "attempt_dir": r.get("attempt_dir")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for model in a.models:
        slots = slots_of(model)
        with Pool(a.procs) as pool:
            rows = pool.map(scan, slots, chunksize=2)
        with gzip.open(OUT / f"export-{model}.jsonl.gz", "wt") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(model, len(rows), "slots;", sum(r["export"] for r in rows), "with export", file=sys.stderr)


if __name__ == "__main__":
    main()
