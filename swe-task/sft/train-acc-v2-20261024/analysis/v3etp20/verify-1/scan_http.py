#!/usr/bin/env python3
"""verify-1 (independent): per-call features from the http.jsonl journal of each slot's FINAL attempt
(results/<model>-group-*.json attempt_dir). Read-only on the probes. Stdlib only.

    python3 -I -B scan_http.py MODEL [--procs N]  -> out/http-<MODEL>.jsonl.gz (one line per slot, calls inside)

Per slot: request count, response status counter, retry events; per 200 response (= model call):
  id, call_id, model_call_id, http_attempt, req_ts (request journal timestamp of the same call_id + http_attempt),
  resp_ts, finish, prompt/completion tokens, compaction (request system prompt = OpenCode's anchored-summary agent),
  reasoning: None/len, tool-call markup, '<tool_call>' opener, special tokens <|...|>, tail zlib ratio (last 20,000
  chars, for len >= 2,000), 2k-window loop regions (zlib < 0.10, stride 1k) with dominant word, tail 160 chars,
  foreign-script runs (non-Latin letters) and whether each occurs in the non-assistant context of the request;
  content: None/len, tool markup, '## Goal', head 200; tool_calls: names, max argument length.
"""
import argparse
import collections
import gzip
import json
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|</function>|<parameter=")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")
# non-Latin letters: Greek, Cyrillic, Armenian, Hebrew, Arabic, Syriac..Indic, Thai, Lao, Tibetan, Myanmar, Georgian,
# Hangul Jamo, Ethiopic.., Hiragana/Katakana, Bopomofo, CJK ext A, CJK unified, Yi, Hangul syllables, CJK compat, halfwidth kana
FOREIGN = re.compile("[Ͱ-ϿЀ-ԯ԰-֏֐-׿؀-ۿ܀-ࣿऀ-෿"
                     "฀-๿຀-࿿က-႟Ⴀ-ჿᄀ-ᇿሀ-᎟぀-ヿ"
                     "㄀-ㄯ㄰-㆏ㆠ-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯"
                     "豈-﫿ｦ-ﾟ]+")
WORD = re.compile(r"\w+")
COMPACTION_SYS = b"You are an anchored context summarization assistant"


def zr(s):
    b = s.encode("utf-8", "surrogatepass")
    return len(zlib.compress(b, 6)) / max(1, len(b))


def loop_regions(text):
    n = len(text)
    if n < 500:
        return []
    if n <= 2000:
        wins = [(0, n)]
    else:
        starts = list(range(0, n - 2000 + 1, 1000))
        if starts[-1] != n - 2000:
            starts.append(n - 2000)
        wins = [(s, s + 2000) for s in starts]
    regs, cur = [], None
    for s, e in wins:
        if zr(text[s:e]) < 0.10:
            if cur and s <= cur[1]:
                cur[1] = e
            else:
                if cur:
                    regs.append(cur)
                cur = [s, e]
        else:
            if cur:
                regs.append(cur)
                cur = None
    if cur:
        regs.append(cur)
    out = []
    for s, e in regs:
        seg = text[s:e]
        words = [w.lower() for w in WORD.findall(seg)]
        c = collections.Counter(words)
        top, topn = (c.most_common(1)[0] if c else ("", 0))
        hmm = sum(v for k, v in c.items() if re.fullmatch(r"h+m+", k))
        out.append({"s": s, "e": e, "top": top[:30], "top_share": round(topn / max(1, len(words)), 3),
                    "hmm_share": round(hmm / max(1, len(words)), 3), "nwords": len(words),
                    "terminal": e >= n - 1000, "escaped": n - e >= 2000, "sample": seg[-80:]})
    return out


def text_of_messages(payload):
    """concatenated non-assistant message text of a request (system, user, tool)."""
    parts = []
    for m in payload.get("messages") or []:
        if m.get("role") == "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts += [x.get("text") or "" for x in c if isinstance(x, dict)]
    return "\n".join(parts)


def scan(slot):
    path = Path(slot["attempt_dir"]) / "http.jsonl"
    res = {k: slot[k] for k in ("model", "group", "task_id", "slot", "status", "patch_chars", "agent_error", "attempt_dir")}
    res.update({"http_missing": not path.exists(), "requests": 0, "resp_status": {}, "events": {}, "calls": []})
    if not path.exists():
        return res
    req_ts, req_raw, req_comp = {}, {}, {}
    status = collections.Counter()
    events = collections.Counter()
    first_req = last_resp = None
    with open(path, "rb") as handle:
        for raw in handle:
            head = raw[:1500]
            m = re.search(rb'"event": ?"([a-z_]+)"', head)
            ev = m.group(1).decode() if m else "?"
            events[ev] += 1
            if ev == "request":
                d = json.loads(raw)
                res["requests"] += 1
                key = (d.get("call_id"), d.get("http_attempt"))
                req_ts[key] = d.get("timestamp")
                if first_req is None or (d.get("timestamp") or 0) < first_req:
                    first_req = d.get("timestamp")
                msgs = (d.get("payload") or {}).get("messages") or []
                sysc = msgs[0].get("content") if msgs else ""
                if isinstance(sysc, list):
                    sysc = "".join(x.get("text") or "" for x in sysc if isinstance(x, dict))
                req_comp[d.get("call_id")] = isinstance(sysc, str) and sysc.startswith(COMPACTION_SYS.decode())
                req_raw[d.get("call_id")] = d.get("payload")
                continue
            if ev != "response":
                continue
            d = json.loads(raw)
            st = d.get("status")
            status[str(st)] += 1
            ts = d.get("timestamp")
            if last_resp is None or (ts or 0) > last_resp:
                last_resp = ts
            body = d.get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = None
            if st != 200 or not isinstance(body, dict) or not body.get("choices"):
                res["calls"].append({"non200": st, "call_id": d.get("call_id"), "http_attempt": d.get("http_attempt"),
                                     "ts": ts, "body_head": (d.get("body") or "")[:200] if isinstance(d.get("body"), str) else None})
                continue
            ch = body["choices"][0]
            msg = ch.get("message") or {}
            reasoning = msg.get("reasoning")
            if reasoning is None:
                reasoning = msg.get("reasoning_content")
            content = msg.get("content")
            tcs = msg.get("tool_calls") or []
            usage = body.get("usage") or {}
            cid = d.get("call_id")
            rec = {"id": body.get("id"), "call_id": cid, "model_call_id": d.get("model_call_id"),
                   "http_attempt": d.get("http_attempt"), "req_ts": req_ts.get((cid, d.get("http_attempt"))), "ts": ts,
                   "finish": ch.get("finish_reason"), "pt": usage.get("prompt_tokens"), "ct": usage.get("completion_tokens"),
                   "comp": req_comp.get(cid),
                   "tools": [((t.get("function") or {}).get("name")) for t in tcs],
                   "argmax": max([len((t.get("function") or {}).get("arguments") or "") for t in tcs], default=0)}
            if isinstance(reasoning, str):
                rec["r_len"] = len(reasoning)
                rec["r_strip_len"] = len(reasoning.strip())
                if MARKUP.search(reasoning):
                    rec["r_markup"] = True
                    rec["r_opener"] = "<tool_call>" in reasoning
                sp = SPECIAL.findall(reasoning)
                if sp:
                    rec["r_special"] = sorted(set(sp))
                if len(reasoning) >= 2000:
                    rec["r_tail_zr"] = round(zr(reasoning[-20000:]), 4)
                if len(reasoning) >= 500:
                    regs = loop_regions(reasoning)
                    if regs:
                        rec["r_loops"] = regs
                rec["r_tail"] = reasoning[-160:]
                runs = FOREIGN.findall(reasoning)
                if runs:
                    ctx = text_of_messages(req_raw.get(cid) or {})
                    rec["r_foreign"] = [[r[:20], r in ctx] for r in runs[:50]]
                    rec["r_foreign_n"] = len(runs)
                    rec["r_foreign_spont"] = sum(1 for r in runs if r not in ctx)
                    ex = [reasoning[max(0, reasoning.find(r) - 30): reasoning.find(r) + len(r) + 30] for r in runs if r not in ctx][:2]
                    if ex:
                        rec["r_foreign_ex"] = ex
            else:
                rec["r_len"] = None
            if isinstance(content, str):
                rec["c_len"] = len(content)
                if MARKUP.search(content):
                    rec["c_markup"] = True
                rec["c_goal"] = "## Goal" in content
                if rec["comp"]:
                    rec["c_text"] = content[:20000]
                rec["c_head"] = content[:200]
                rec["c_tail"] = content[-200:]
            else:
                rec["c_len"] = None
            if rec["comp"] and isinstance(reasoning, str):
                rec["r_head"] = reasoning[:200]
            res["calls"].append(rec)
            req_raw.pop(cid, None)
    res["resp_status"] = dict(status)
    res["events"] = dict(events)
    res["first_req"] = first_req
    res["last_resp"] = last_resp
    return res


def slots(model):
    out = []
    for f in sorted(TA.glob(f"results/{model}-group-*.json")):
        d = json.loads(f.read_text())
        for t, rs in d["records"].items():
            for r in rs:
                out.append({"model": model, "group": d["group"], "task_id": t, "slot": r["slot"], "status": r["status"],
                            "patch_chars": r.get("patch_chars"), "agent_error": r.get("agent_error"),
                            "attempt_dir": r.get("attempt_dir")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ss = slots(args.model)
    # largest files first for balance
    ss.sort(key=lambda s: -(Path(s["attempt_dir"]) / "http.jsonl").stat().st_size
            if (Path(s["attempt_dir"]) / "http.jsonl").exists() else 0)
    tmp = OUT / f".http-{args.model}.jsonl.gz.tmp"
    n = 0
    with Pool(args.procs) as pool, gzip.open(tmp, "wt") as out:
        for res in pool.imap_unordered(scan, ss, chunksize=1):
            out.write(json.dumps(res) + "\n")
            n += 1
            if n % 25 == 0:
                print(args.model, n, "slots", flush=True)
    tmp.replace(OUT / f"http-{args.model}.jsonl.gz")
    print(args.model, "done", n, file=sys.stderr)


if __name__ == "__main__":
    main()
