#!/usr/bin/env python3
"""Session-ending lens, step 1: compact per-rollout session records (read-only on runs / probes / classifier outputs).

usage: python3 -I -B extract_endings.py [--procs N] [--only NAME[,NAME...]]
  -> out/sessions/<dataset>__<model>.jsonl.gz, one line per rollout

Rollouts = the empty-patch classifier's all.jsonl rows (one per (task, repeat); its code / subcode / wall_s are kept):
  deepswe  113 tasks x 5 per run (classifier outputs listed in CLASSIFIER below; same files as sftv3-etp REPORT.md)
  ta       train-acc-v2 50 tasks x 5 per model (analysis/empty-patch/<model>/all.jsonl, final attempt of each slot)
Each rollout's OpenCode export.json (root session) is parsed as think_watch.py / thinkx.py do: every assistant message
that is not a compaction summary (info.summary True or mode 'compaction') is one model call ("turn").

Per turn: message index, created / first-part / completed times (ms; the eval server answers in one piece, so
created -> first part = model time, first part -> completed = tools + overhead, as analyze.py), output and prompt
tokens, finish, error name, reasoning / visible lengths, unclosed think, zlib ratio of the last 20,000 chars of the
text (thinkx.py loop measure; only for turns with >= 4,096 output tokens or finish 'length'), whether it is the first
turn after OpenCode's synthetic "Continue if you have next steps..." prompt or after a compaction, and its tool calls:
tool, status, bash command (first 600 chars; flags computed on the full command), exit code, OpenCode's bash timeout
marker, output length, pass / fail marker counts on the output (first 200k chars), file path, error text head.
For the last 3 turns: visible text head / tail and reasoning tail. Per compaction: position, summary output tokens,
finish, '## Goal' present, tool-call markup in the summary, overflow flag. For unfinished sessions: the in-flight
message (created, parts) and any tool part still 'running'.
No secrets are copied: commands / texts are truncated and pass through a redaction regex.

2026-11-04: copy in train-acc analysis/v3etp20/degenerate/endings-ds-crosscheck: DeepSWE v3etp10 and v3etp20 only.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import endlib as L  # noqa: E402
OUT = HERE / "out/sessions"
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis")
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
CLASSIFIER = {
    ("deepswe", "base"): A / "empty-patch-20261018/base-step0-rep5-out131k/all.jsonl",
    ("deepswe", "base-rerun"): A / "sftv3-etp-20261030/classifier/base-rerun/all.jsonl",
    ("deepswe", "v2-5ep"): A / "lenient-think-20261023/empty-patch/v2-lenient/all.jsonl",
    ("deepswe", "v2-10ep"): A / "lenient-think-20261023/empty-patch/v2-10ep-lenient/all.jsonl",
    ("deepswe", "v2-20ep"): A / "lenient-think-20261023/empty-patch/v2-20ep-lenient/all.jsonl",
    ("deepswe", "etp5"): A / "etp-sft-20261027/classifier/etp5/all.jsonl",
    ("deepswe", "etp10"): A / "etp-sft-20261027/classifier/etp10/all.jsonl",
    ("deepswe", "etp20"): A / "etp-sft-20261027/classifier/etp20/all.jsonl",
    ("deepswe", "v3etp5"): A / "sftv3-etp-20261030/classifier/v3etp5/all.jsonl",
    ("deepswe", "v3etp10"): A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
}
# 2026-11-04 copy (train-acc v3etp20 analysis, DeepSWE cross-check of the compaction breakage): DeepSWE v3etp10
# (published rows, reproduction check) and v3etp20 (the DeepSWE v3etp20 analysis's classifier rows, read-only) only.
CLASSIFIER = {("deepswe", "v3etp10"): A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
              ("deepswe", "v3etp20"): A / "sftv3-etp20-20261104/classifier/v3etp20/all.jsonl"}

RE_TOOLCALL = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")
RE_SPECIAL_TOK = re.compile(r"<\|[A-Za-z_]+\|>")
RE_TIMEOUT = re.compile(r"terminated command after exceeding timeout|Command timed out|timed out after \d+", re.I)
RE_FAILM = re.compile(r"(\bFAIL(?:ED|URES?)?\b|\b[1-9]\d* (?:failed|failing|errors?)\b|^\s*--- FAIL|panic:|error\[E\d+|"
                      r"\bTraceback \(most recent call last\)|AssertionError|✗|✕|× |^E\s{3}|test result: FAILED|"
                      r"Tests?:\s+\d+ failed|\bnot ok \d)", re.M)
RE_PASSM = re.compile(r"(\b\d+ passed\b|^ok\s+\S|^PASS\b|test result: ok|Tests?:\s+\d+ passed|✓|✔|"
                      r"\ball tests pass|^OK(?: \(|$)|\b0 failed\b|# pass \d+)", re.M | re.I)
SECRETISH = re.compile(r"(?i)(nvapi-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                       r"bearer\s+[A-Za-z0-9._\-]{16,}|xox[abpr]-[A-Za-z0-9\-]{10,}|(api[_-]?key|token|secret|password)"
                       r"\s*[=:]\s*['\"]?[A-Za-z0-9._\-]{12,})")
RE_KILL = re.compile(r"(\bkill\b|\bpkill\b|\bkillall\b)")
RE_SLEEP = re.compile(r"\bsleep\s+\d")
RE_BG = re.compile(r"(?:^|[^&|>])&\s*(?:$|;|\n|disown|sleep|echo|\()|\bnohup\b|\bsetsid\b|\bdisown\b")
RE_COMMIT = re.compile(r"\bgit\b[^;&|\n]*\bcommit\b")
RE_COMMIT_OK_LINE = re.compile(r"^\[([^\]\n]+?) (?:\(root-commit\) )?([0-9a-f]{7,40})\]", re.M)
RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def red(s):
    return SECRETISH.sub("[REDACTED]", s or "")


def split_think(text):
    s = text.strip()
    if not s.startswith("<think>"):
        return None, s
    if "</think>" not in s:
        return s[len("<think>"):], None
    inner, after = s[len("<think>"):].split("</think>", 1)
    return inner, after.strip()


def zr(s):
    b = s[-20000:].encode("utf-8", "surrogatepass")
    return round(len(zlib.compress(b)) / max(1, len(b)), 4)


def call_rec(part):
    st = part.get("state") or {}
    tool = part.get("tool") or "?"
    inp = st.get("input") if isinstance(st.get("input"), dict) else {}
    out = st.get("output") if isinstance(st.get("output"), str) else ""
    md = st.get("metadata") if isinstance(st.get("metadata"), dict) else {}
    if not out and isinstance(md.get("output"), str):
        out = md["output"]
    out = RE_ANSI.sub("", out)
    err = st.get("error") if isinstance(st.get("error"), str) else ""
    tm = st.get("time") or {}
    rec = {"tool": tool, "st": st.get("status"), "olen": len(out), "t0": tm.get("start")}
    if err:
        rec["err"] = red(err[:160])
    if tool == "bash":
        cmd = inp.get("command") if isinstance(inp.get("command"), str) else ""
        rec.update({"cmd": red(cmd[:600]), "clen": len(cmd), "exit": md.get("exit"),
                    "tmo": bool(RE_TIMEOUT.search(out[-2000:])) or bool(RE_TIMEOUT.search(err)),
                    "tmo_arg": inp.get("timeout"), "wd": (inp.get("workdir") or "")[:80] if isinstance(inp.get("workdir"), str) else "",
                    "kill": bool(RE_KILL.search(cmd)), "sleep": bool(RE_SLEEP.search(cmd)), "bg": bool(RE_BG.search(cmd)),
                    "commit": bool(RE_COMMIT.search(cmd)), "commit_ok": bool(RE_COMMIT.search(cmd) and RE_COMMIT_OK_LINE.search(out)),
                    "nfail": len(RE_FAILM.findall(out[:200000])), "npass": len(RE_PASSM.findall(out[:200000])),
                    "otail": red(out[-160:]) if out else ""})
    elif tool in ("read", "edit", "write", "multiedit", "patch", "apply_patch"):
        fp = inp.get("filePath") or inp.get("path") or ""
        rec["fp"] = str(fp)[:200]
        if tool == "edit":
            old = inp.get("oldString") if isinstance(inp.get("oldString"), str) else ""
            new = inp.get("newString") if isinstance(inp.get("newString"), str) else ""
            rec["eh"] = (zlib.crc32(old.encode("utf-8", "replace")), zlib.crc32(new.encode("utf-8", "replace")))
            rec["dl"] = len(new) - len(old)
        if tool == "write":
            c = inp.get("content") if isinstance(inp.get("content"), str) else ""
            rec["eh"] = (0, zlib.crc32(c.encode("utf-8", "replace")))
            rec["dl"] = len(c)
    elif tool in ("grep", "glob", "list"):
        rec["fp"] = str(inp.get("path") or "")[:200]
        rec["pat"] = str(inp.get("pattern") or "")[:120]
    elif tool == "todowrite":
        todos = inp.get("todos") if isinstance(inp.get("todos"), list) else []
        rec["todo_n"] = len(todos)
        rec["todo_done"] = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
    elif tool == "task":
        rec["desc"] = red(str(inp.get("description") or "")[:120])
    return rec


def session(path):
    data = json.loads(Path(path).read_text())
    msgs = data.get("messages") or []
    turns, comps = [], []
    first_t = None
    after_continue = after_comp = False
    pending_comp = None
    for idx, m in enumerate(msgs):
        info = m.get("info") or {}
        parts = m.get("parts") or []
        tm = info.get("time") or {}
        if first_t is None and tm.get("created"):
            first_t = tm["created"]
        if info.get("role") == "user":
            if idx == 0:
                continue
            if any(p.get("type") == "compaction" for p in parts):
                pending_comp = {"i": idx, "nturn": len(turns), "t": tm.get("created"),
                                "overflow": any(p.get("type") == "compaction" and p.get("overflow") for p in parts)}
                comps.append(pending_comp)
                continue
            synth = any(p.get("synthetic") for p in parts)
            txt = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
            if synth:
                after_continue = True
                if pending_comp is not None:
                    pending_comp["cont_text"] = txt[:120]
            else:
                # a non-synthetic user message after the task (should not happen in these harnesses)
                comps.append({"i": idx, "user_msg": red(txt[:200]), "nturn": len(turns)})
            continue
        if info.get("role") != "assistant":
            continue
        tok = info.get("tokens") or {}
        cache = tok.get("cache") or {}
        text = "".join(p.get("text") or "" for p in parts
                       if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        if info.get("summary") is True or info.get("mode") == "compaction":
            inner, after = split_think(text)
            body = after if after is not None else ("" if inner is not None else text)
            srec = {"si": idx, "s_out": tok.get("output"), "s_fin": info.get("finish"), "s_len": len(body),
                    "s_goal": "## Goal" in body or "Goal" in body[:400],
                    "s_toolmarkup": bool(RE_TOOLCALL.search(text)), "s_tools": sum(1 for p in parts if p.get("type") == "tool"),
                    "s_think_only": inner is not None and not (after or "").strip(),
                    "s_head": red(body[:160]), "s_commit": bool(L.RE_MENTIONS_COMMIT.search(body)),
                    "s_commit_instr": bool(L.RE_COMMIT_INSTR.search(body)), "s_err": ((info.get("error") or {}).get("name") if isinstance(info.get("error"), dict) else None),
                    "s_tc": tm.get("created"), "s_td": tm.get("completed")}
            if pending_comp is not None:
                pending_comp.update(srec)
            else:
                comps.append(dict(srec, orphan=True))
            after_comp = True
            continue
        inner, visible = split_think(text)
        reasoning = ("".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or ""))
        ptimes = [((p.get("time") or {}).get("start") or ((p.get("state") or {}).get("time") or {}).get("start"))
                  for p in parts if p.get("type") in ("text", "tool", "reasoning")]
        ptimes = [x for x in ptimes if x]
        calls = [call_rec(p) for p in parts if p.get("type") == "tool"]
        out = tok.get("output")
        err = info.get("error")
        t = {"i": idx, "tc": tm.get("created"), "tp": min(ptimes) if ptimes else None, "td": tm.get("completed"),
             "out": out, "pr": (tok.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0),
             "fin": info.get("finish"), "err": (err.get("name") if isinstance(err, dict) else (str(err)[:60] if err else None)),
             "rl": len(reasoning.strip()), "vl": len((visible or "").strip()), "unc": inner is not None and visible is None,
             "markup": bool(RE_TOOLCALL.search(reasoning) or RE_SPECIAL_TOK.search(reasoning)),
             "vmarkup": bool(RE_TOOLCALL.search(visible or "") or RE_SPECIAL_TOK.search(visible or "")),
             "ac": after_continue, "acomp": after_comp, "calls": calls}
        if (isinstance(out, (int, float)) and out >= 4096) or info.get("finish") == "length":
            t["z"] = zr(text if text else reasoning)
        if isinstance(err, dict):
            t["err_msg"] = red(json.dumps(err)[:240])
        turns.append(t)
        after_continue = after_comp = False
    # texts of the last 3 turns
    asst = [m for m in msgs if (m.get("info") or {}).get("role") == "assistant"
            and not ((m.get("info") or {}).get("summary") is True or (m.get("info") or {}).get("mode") == "compaction")]
    for t, m in zip(turns[-3:], asst[-3:]):
        parts = m.get("parts") or []
        text = "".join(p.get("text") or "" for p in parts
                       if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        inner, visible = split_think(text)
        reasoning = ("".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or "")).strip()
        visible = (visible or "").strip()
        t["vhead"] = red(visible[:400])
        t["vtail"] = red(visible[-400:]) if len(visible) > 400 else ""
        t["rtail"] = red(reasoning[-600:])
        t["flags"] = L.final_flags(visible, reasoning)
        t["rhead"] = red(reasoning[:200])
    # in-flight state
    running = []
    for m in msgs[-2:]:
        for p in m.get("parts") or []:
            if p.get("type") == "tool" and (p.get("state") or {}).get("status") in ("running", "pending"):
                st = p.get("state") or {}
                inp = st.get("input") if isinstance(st.get("input"), dict) else {}
                running.append({"tool": p.get("tool"), "cmd": red(str(inp.get("command") or inp.get("filePath") or "")[:300]),
                                "t0": (st.get("time") or {}).get("start")})
    last = msgs[-1] if msgs else {}
    li = last.get("info") or {}
    last_t = None
    for m in msgs:
        tm = (m.get("info") or {}).get("time") or {}
        for k in ("created", "completed"):
            if isinstance(tm.get(k), (int, float)):
                last_t = tm[k] if last_t is None else max(last_t, tm[k])
        for p in m.get("parts") or []:
            for tt in ((p.get("time") or {}), ((p.get("state") or {}).get("time") or {})):
                for k in ("start", "end"):
                    if isinstance(tt.get(k), (int, float)):
                        last_t = max(last_t or tt[k], tt[k])
    return {"turns": turns, "comps": comps, "running": running, "first_t": first_t, "last_t": last_t,
            "last_role": li.get("role"), "last_completed": bool((li.get("time") or {}).get("completed")),
            "last_is_summary": li.get("summary") is True or li.get("mode") == "compaction",
            "n_msgs": len(msgs)}


KEEP = ("task_id", "ti", "ri", "reward", "model_patch_bytes", "opencode_finished", "opencode_export_found", "code",
        "subcode", "wall_s", "empty_patch", "not_finished", "verifier_error", "stderr", "evaluation_completed", "shard")
KEEP_F = ("n_compactions", "n_commits", "commit_attempts", "edited", "edit_ok_app", "first_edit_rel_s", "last_edit_rel_s",
          "last_commit_rel_s", "n_calls_at_cap", "last_running_tool", "last_running_cmd", "last_running_start_rel_s",
          "last_error", "last_error_msg", "visible_mentions_commit", "visible_defers_commit", "visible_asks_user",
          "think_mentions_wdwdsf", "head_events", "after_commit_events", "net_vs_patch", "n_net_files", "broken_summaries",
          "last_summary_has_goal", "n_synthetic_continue", "last_user_kind", "steps_after_last_user", "model_wait_s",
          "tools_overhead_s", "msg_span_s", "n_assistant")


def work(args):
    dataset, model, line = args
    r = json.loads(line)
    rec = {"dataset": dataset, "model": model}
    for k in KEEP:
        v = r.get(k)
        if k == "stderr" and isinstance(v, str):
            v = red(v[-200:])
        if k == "verifier_error" and isinstance(v, str):
            v = red(v[:200])
        rec[k] = v
    f = r.get("f") or {}
    rec["f"] = {k: f.get(k) for k in KEEP_F}
    if isinstance(rec["f"].get("last_running_cmd"), str):
        rec["f"]["last_running_cmd"] = red(rec["f"]["last_running_cmd"][:300])
    if isinstance(rec["f"].get("last_error_msg"), str):
        rec["f"]["last_error_msg"] = red(rec["f"]["last_error_msg"][:300])
    path = r.get("opencode_results_fpath") if r.get("opencode_export_found") else None
    if path and os.path.exists(path):
        try:
            rec["s"] = session(path)
        except Exception as exc:  # noqa: BLE001
            rec["s_error"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        rec["s"] = None
    return json.dumps(rec, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(x for x in args.only.split(",") if x)
    for (dataset, model), cpath in CLASSIFIER.items():
        name = f"{dataset}__{model}"
        if only and name not in only and model not in only:
            continue
        t0 = time.time()
        lines = [l for l in open(cpath) if l.strip()]
        out = OUT / f"{name}.jsonl.gz"
        tmp = out.with_name(f".{name}.tmp{os.getpid()}.gz")
        with Pool(args.procs) as pool, gzip.open(tmp, "wt") as handle:
            n = 0
            for rec in pool.imap(work, [(dataset, model, l) for l in lines], chunksize=4):
                handle.write(rec + "\n")
                n += 1
        os.replace(tmp, out)
        print(f"{name}: {n} rollouts from {cpath} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
