#!/usr/bin/env python3
"""verify-2 (2026-11-04): independent per-slot / per-call extraction from RAW attempt dirs (final attempt of each slot,
per results/<model>-group-*.json attempt_dir). Reads http.jsonl (requests + responses), started.json, terminal.json,
generation-validity-v2-r7.json. Writes out/<model>.jsonl.gz (one JSON line per slot with its calls).

    python3 -I -B extract_calls.py <model> [--procs N]
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"

RE_TOOLMARK = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")
RE_SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")
RE_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
# a short CJK run glued into ASCII prose: ASCII letter or "word " on the left, and space/ASCII letter/punct on the right
RE_CJK_GLITCH = re.compile(r"(?:[A-Za-z]|[A-Za-z][ ])[぀-ヿ㐀-䶿一-鿿가-힯]{1,8}(?=[ A-Za-z,.;:])")
RE_HMM_RUN = re.compile(r"(?:\bhmm+\b[\s\W]{0,6}){10,}", re.I)
# my own commit-instruction matcher (task: "work on this in a new branch from main and commit everything when you are done")
RE_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work)|all changes)"
    r"|commit\b[^.\n]{0,60}\b(?:when|once|after)\b[^.\n]{0,40}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and)\s+commit\b"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`))", re.I)
COMP_SYS = "You are an anchored context summarization assistant"
CONT = "Continue if you have next steps"


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join((x.get("text") or "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def zr(s):
    b = s[-20000:].encode("utf-8", "surrogatepass")
    return round(len(zlib.compress(b)) / max(1, len(b)), 4)


def max_hmm_run(s):
    best = 0
    for m in RE_HMM_RUN.finditer(s):
        best = max(best, len(re.findall(r"hmm+", m.group(0), re.I)))
    return best


def slot_rows(model):
    rows = []
    for f in sorted((TA / "results").glob(f"{model}-group-*.json")):
        d = json.loads(f.read_text())
        for t, recs in d["records"].items():
            for r in recs:
                rows.append({"model": model, "group": d["group"], "task": t, "slot": r["slot"], "status": r["status"],
                             "patch_chars": r.get("patch_chars"), "agent_error": r.get("agent_error"),
                             "verifier_error": r.get("verifier_error"), "attempt_dir": r["attempt_dir"]})
    return rows


def work(row):
    ad = Path(row["attempt_dir"])
    out = dict(row)
    out["attempt"] = ad.name
    out["n_attempt_dirs"] = len(list(ad.parent.glob("attempt-*")))
    try:
        st = json.loads((ad / "started.json").read_text())
        out["started_at"], out["deadline"] = st.get("started_at"), st.get("deadline")
    except Exception as e:  # noqa: BLE001
        out["started_err"] = repr(e)[:100]
    try:
        gv = json.loads((ad / "generation-validity-v2-r7.json").read_text())
        out["budget_timeout"] = gv.get("budget_timeout")
        out["gv_failure"] = gv.get("failure")
    except Exception as e:  # noqa: BLE001
        out["gv_err"] = repr(e)[:100]
    try:
        term = json.loads((ad / "terminal.json").read_text())
        resp = term.get("response") or {}
        out["term"] = {"opencode_finished": term.get("opencode_finished"), "resp_created_at": resp.get("created_at"),
                       "opencode_exit_code": term.get("opencode_exit_code"),
                       "opencode_run_error_type": term.get("opencode_run_error_type"),
                       "failure_reason": (term.get("failure_reason") or "")[:200] or None,
                       "stderr_tail": (term.get("opencode_run_stderr") or "")[-120:],
                       "model_patch_bytes": term.get("model_patch_bytes"), "reward": term.get("reward"),
                       "verifier_error": (term.get("verifier_error") or "")[:200] or None}
        del term
    except Exception as e:  # noqa: BLE001
        out["term_err"] = repr(e)[:100]
    calls = {}
    order = []
    events = {}
    tools = None
    hp = ad / "http.jsonl"
    if hp.exists():
        with open(hp, "rb") as fh:
            for raw in fh:
                try:
                    d = json.loads(raw)
                except ValueError:
                    events["bad_json"] = events.get("bad_json", 0) + 1
                    continue
                ev = d.get("event")
                events[ev] = events.get(ev, 0) + 1
                mcid = d.get("model_call_id")
                if mcid is None:
                    continue
                c = calls.get(mcid)
                if c is None:
                    c = calls[mcid] = {"mcid": mcid, "n_req": 0, "statuses": []}
                    order.append(mcid)
                if ev == "request":
                    c["n_req"] += 1
                    ts = d.get("timestamp")
                    c.setdefault("t0", ts)
                    c["t_last_req"] = ts
                    p = d.get("payload") or {}
                    if c["n_req"] == 1:
                        msgs = p.get("messages") or []
                        c["nmsg"] = len(msgs)
                        c["max_tokens"] = p.get("max_tokens")
                        sys0 = text_of(msgs[0].get("content")) if msgs and msgs[0].get("role") == "system" else ""
                        c["sys"] = hashlib.md5(sys0.encode("utf-8", "surrogatepass")).hexdigest()[:8]
                        fu = next((text_of(m.get("content")) for m in msgs if m.get("role") == "user"), "")
                        c["fu"] = fu[:60]
                        if sys0.startswith(COMP_SYS):
                            last = text_of(msgs[-1].get("content")) if msgs else ""
                            c["comp"] = "create" if last.startswith("Create a new anchored summary") else (
                                "update" if last.startswith("Update the anchored summary") else "other:" + last[:40])
                        else:
                            c["comp"] = None
                            if tools is None and p.get("tools"):
                                tools = sorted((t.get("function") or {}).get("name") for t in p["tools"])
                        if msgs and msgs[-1].get("role") == "user" and text_of(msgs[-1].get("content")).startswith(CONT):
                            c["after_continue"] = True
                        # commit instruction anywhere in user messages / assistant text (not tool calls / outputs)
                        instr = False
                        orig = False
                        for m in msgs:
                            if m.get("role") in ("user", "assistant"):
                                tx = text_of(m.get("content"))
                                if "commit everything when you are done" in tx:
                                    orig = True
                                if RE_INSTR.search(tx):
                                    instr = True
                        c["ctx_instr"] = instr
                        c["ctx_orig_instr"] = orig
                        c["first_user_wdwdsf"] = bool(len(msgs) > 1 and msgs[1].get("role") == "user"
                                                      and text_of(msgs[1].get("content")).startswith("What did we do so far"))
                elif ev == "response":
                    status = d.get("status")
                    c["statuses"].append(status)
                    body = d.get("body")
                    if status == 200 and body:
                        try:
                            b = json.loads(body) if isinstance(body, str) else body
                        except ValueError:
                            c["body_err"] = True
                            continue
                        ch = (b.get("choices") or [{}])[0]
                        msg = ch.get("message") or {}
                        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                        content = msg.get("content") or ""
                        tcs = msg.get("tool_calls") or []
                        usage = b.get("usage") or {}
                        r = {"rid": b.get("id"), "fin": ch.get("finish_reason"), "ct": usage.get("completion_tokens"),
                             "pt": usage.get("prompt_tokens"), "rlen": len(reasoning), "rs": len(reasoning.strip()),
                             "cs": len(content.strip()), "ntc": len(tcs),
                             "tn": [((t.get("function") or {}).get("name")) for t in tcs],
                             "targ": max([len((t.get("function") or {}).get("arguments") or "") for t in tcs] or [0]),
                             "rmark": bool(RE_TOOLMARK.search(reasoning)), "cmark": bool(RE_TOOLMARK.search(content)),
                             "rspec": bool(RE_SPECIAL.search(reasoning)), "cspec": bool(RE_SPECIAL.search(content)),
                             "cjk": len(RE_CJK.findall(reasoning)), "cjkg": len(RE_CJK_GLITCH.findall(reasoning)),
                             "hmm": max_hmm_run(reasoning) if "hmm" in reasoning.lower() else 0}
                        if r["cjkg"]:
                            m = RE_CJK_GLITCH.search(reasoning)
                            r["cjkg_ex"] = reasoning[max(0, m.start() - 30): m.end() + 30]
                        if (r["ct"] or 0) >= 4096 or r["fin"] == "length":
                            r["z"] = zr(reasoning + content)
                            r["tail"] = (reasoning + content)[-240:]
                        if c.get("comp"):
                            r["c_head"] = content[:240]
                            r["c_goal"] = "## Goal" in content
                            r["c_instr"] = bool(RE_INSTR.search(content))
                            r["c_commit_word"] = "commit" in content.lower()
                            r["c_len"] = len(content)
                        c["resp"] = r
                elif ev == "model_call_completed":
                    c["t_done"] = d.get("timestamp")
                    c["result_type"] = d.get("result_type")
                elif not (ev or "").startswith("api_gate"):
                    c.setdefault("other_ev", []).append(ev)
    out["events"] = events
    out["tools"] = tools
    out["calls"] = [calls[k] for k in order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    rows = slot_rows(a.model)
    OUT.mkdir(exist_ok=True)
    with Pool(a.procs) as pool, gzip.open(OUT / f"{a.model}.jsonl.gz", "wt") as fo:
        for i, res in enumerate(pool.imap_unordered(work, rows, chunksize=1)):
            fo.write(json.dumps(res) + "\n")
            if i % 25 == 0:
                print(a.model, i, file=sys.stderr, flush=True)
    print("done", a.model, len(rows))


if __name__ == "__main__":
    main()
