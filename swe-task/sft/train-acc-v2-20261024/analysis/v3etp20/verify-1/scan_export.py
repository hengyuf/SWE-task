#!/usr/bin/env python3
"""verify-1 (independent): per-rollout session features from each slot's FINAL attempt (results attempt_dir):
export.json (OpenCode root session), terminal.json, started.json, generation-validity*.json. Stdlib only, read-only.

    python3 -I -B scan_export.py MODEL [MODEL ...] [--procs N] -> out/export-<MODEL>.jsonl.gz

Turn = assistant message that is not a compaction summary (mode 'compaction' or summary True). Reasoning of a turn =
'reasoning' parts + the inner of a leading <think> block of the text parts (non-synthetic); visible = text after
</think>. Compaction = a user message holding a 'compaction' part; its summary = the next assistant message with
mode 'compaction' / summary True.
"""
import argparse
import collections
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|</function>|<parameter=")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")
GITCOMMIT = re.compile(r"\bgit\b[^;&|\n]*\bcommit\b")
COMMIT_OK = re.compile(r"^\[([^\]\n]+?) (?:\(root-commit\) )?([0-9a-f]{7,40})\]", re.M)
EDIT_TOOLS = {"edit", "write", "multiedit", "patch", "apply_patch"}
# my own definition of "the summary still carries the task's commit instruction"
MY_INSTR = re.compile(
    r"(\bcommit(?:ting)?\s+(?:everything|all\b|all (?:the )?(?:changes|work)|the (?:changes|work)|(?:my|our|your) (?:changes|work)|changes\b|work\b)"
    r"|\bcommit\b[^\n]{0,60}\b(?:when|once|after)\b[^\n]{0,40}\b(?:done|finished|complete|completed)\b"
    r"|\b(?:and|then|must|should|need to|needs to|to)\s+commit\b)", re.I)
MENTIONS_COMMIT = re.compile(r"\bcommit", re.I)
# the analyst's RE_COMMIT_INSTR (endlib.py), copied verbatim as a definition only, to separate pipeline from definition
THEIR_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work))"
    r"|commit\b[^.\n]{0,40}\b(?:when|once|after)\b[^.\n]{0,30}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and|,)\s+commit\b"
    r"|commit(?:ting)? (?:it |them |everything |all )?at (?:the )?end"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`)"
    r"|new branch from main and commit|\bcommit\s*[.;]?\s*(?:\n|$)|^\s*[-*\d.)]+\s*commit\b)", re.I | re.M)


def split_think(text):
    s = text.lstrip()
    if not s.startswith("<think>"):
        return None, text, False
    if "</think>" not in s:
        return s[len("<think>"):], "", True
    inner, after = s[len("<think>"):].split("</think>", 1)
    return inner, after, False


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def scan(slot):
    a = Path(slot["attempt_dir"])
    res = dict(slot)
    term = read_json(a / "terminal.json", {}) or {}
    started = read_json(a / "started.json", {}) or {}
    gvs = sorted(a.glob("generation-validity*.json"))
    gv = read_json(gvs[-1], {}) if gvs else {}
    res.update({"finished": term.get("opencode_finished"), "export_found": term.get("opencode_export_found"),
                "stdout_len": len(term.get("opencode_run_stdout") or ""), "stderr_tail": (term.get("opencode_run_stderr") or "")[-120:],
                "run_error_type": term.get("opencode_run_error_type"), "failure_reason": (term.get("failure_reason") or "")[:200],
                "model_patch_bytes": term.get("model_patch_bytes"), "resp_created_at": (term.get("response") or {}).get("created_at"),
                "verifier_error": (term.get("verifier_error") or "")[:200] if term.get("verifier_error") else None,
                "started_at": started.get("started_at"), "deadline": term.get("_atp_deadline"),
                "gv_file": gvs[-1].name if gvs else None, "budget_timeout": gv.get("budget_timeout"),
                "gv_failure": gv.get("failure"), "attempt_index": started.get("attempt_index")})
    exp = read_json(a / "export.json")
    if not exp:
        res["export"] = False
        return res
    res["export"] = True
    msgs = exp.get("messages") or []
    times = []
    turns, comps = [], []
    pending = None
    commits_ok = 0
    commit_attempts = 0
    edits_ok = 0
    invalid = 0
    tool_names = collections.Counter()
    after_cont = False
    for idx, m in enumerate(msgs):
        info = m.get("info") or {}
        parts = m.get("parts") or []
        tm = info.get("time") or {}
        for k in ("created", "completed"):
            if isinstance(tm.get(k), (int, float)):
                times.append(tm[k])
        role = info.get("role")
        if role == "user":
            if idx == 0:
                continue
            if any(p.get("type") == "compaction" for p in parts):
                pending = {"ord": len(comps) + 1, "turn_before": len(turns), "summary": None,
                           "overflow": any(p.get("type") == "compaction" and p.get("overflow") for p in parts)}
                comps.append(pending)
            elif any(p.get("synthetic") for p in parts):
                after_cont = True
            continue
        if role != "assistant":
            continue
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        rparts = "".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning")
        tools = [p for p in parts if p.get("type") == "tool"]
        tok = info.get("tokens") or {}
        if info.get("mode") == "compaction" or info.get("summary") is True:
            inner, after, unclosed = split_think(text)
            body = after if inner is not None else text
            s = {"tools": len(tools), "markup": bool(MARKUP.search(text)), "finish": info.get("finish"),
                 "out": tok.get("output"), "goal": "## Goal" in body, "body_len": len(body.strip()),
                 "my_instr": bool(MY_INSTR.search(body)), "their_instr": bool(THEIR_INSTR.search(body)),
                 "mentions_commit": bool(MENTIONS_COMMIT.search(body)), "unclosed": unclosed,
                 "head": body.strip()[:160]}
            if pending is not None and pending["summary"] is None:
                pending["summary"] = s
            else:
                comps.append({"ord": len(comps) + 1, "orphan": True, "summary": s, "turn_before": len(turns)})
            continue
        inner, visible, unclosed = split_think(text)
        reasoning = (rparts + (inner or "")).strip()
        names = [t.get("tool") for t in tools]
        tool_names.update(names)
        invalid += names.count("invalid")
        for t in tools:
            st = t.get("state") or {}
            inp = st.get("input") if isinstance(st.get("input"), dict) else {}
            if t.get("tool") in EDIT_TOOLS and st.get("status") == "completed":
                edits_ok += 1
            if t.get("tool") == "bash":
                cmd = inp.get("command") if isinstance(inp.get("command"), str) else ""
                outp = st.get("output") if isinstance(st.get("output"), str) else ((st.get("metadata") or {}).get("output") or "")
                if GITCOMMIT.search(cmd):
                    commit_attempts += 1
                    if COMMIT_OK.search(outp or ""):
                        commits_ok += 1
        err = info.get("error")
        turns.append({"n_tools": len(tools), "reasoned": bool(reasoning), "r_len": len(reasoning),
                      "r_markup": bool(MARKUP.search(reasoning)), "r_opener": "<tool_call>" in reasoning,
                      "r_special": sorted(set(SPECIAL.findall(reasoning))), "unclosed": unclosed,
                      "v_markup": bool(MARKUP.search(visible or "")), "v_special": bool(SPECIAL.search(visible or "")),
                      "visible_len": len((visible or "").strip()), "finish": info.get("finish"), "out": tok.get("output"),
                      "err": err.get("name") if isinstance(err, dict) else None, "after_cont": after_cont,
                      "think_only": inner is not None and not (visible or "").strip() and not tools})
        after_cont = False
    res["first_t"] = min(times) if times else None
    res["last_t"] = max(times) if times else None
    res["n_turns"] = len(turns)
    res["action_turns"] = sum(t["n_tools"] > 0 for t in turns)
    res["unreasoned_actions"] = sum(t["n_tools"] > 0 and not t["reasoned"] for t in turns)
    res["unreasoned_detail"] = [{"i": i, "r_len": t["r_len"], "unclosed": t["unclosed"], "out": t["out"]}
                                for i, t in enumerate(turns) if t["n_tools"] > 0 and not t["reasoned"]]
    res["markup_turns"] = sum(t["r_markup"] for t in turns)
    res["markup_headless_turns"] = sum(t["r_markup"] and not t["r_opener"] for t in turns)
    res["special_turns"] = sum(bool(t["r_special"]) for t in turns)
    res["special_tokens"] = sorted({s for t in turns for s in t["r_special"]})
    res["unclosed_turns"] = sum(t["unclosed"] for t in turns)
    res["commit_attempts"] = commit_attempts
    res["commits_ok"] = commits_ok
    res["edits_ok"] = edits_ok
    res["invalid_tool_calls"] = invalid
    res["tool_names"] = dict(tool_names)
    res["comps"] = comps
    res["n_comp"] = len(comps)
    last = turns[-1] if turns else None
    res["last_turn"] = last
    res["reasoning_lens"] = [t["r_len"] for t in turns if t["reasoned"]]
    res["visible_action_turns"] = sum(t["n_tools"] > 0 and t["visible_len"] > 0 for t in turns)
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
    ap.add_argument("models", nargs="+")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        ss = slots(model)
        with Pool(args.procs) as pool:
            rows = pool.map(scan, ss, chunksize=2)
        tmp = OUT / f".export-{model}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        tmp.replace(OUT / f"export-{model}.jsonl.gz")
        print(model, len(rows), "rows", sum(r.get("export", False) for r in rows), "with export", flush=True)


if __name__ == "__main__":
    main()
