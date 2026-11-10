#!/usr/bin/env python3
"""FORMAT-level degeneration of model calls in OpenCode exports (DeepSWE eval runs and train-acc probes). v2.

Read-only on runs / probes. Writes only OUT_DIR/<NAME>.rollouts.jsonl (one record per rollout).

usage: fmt_detect.py --out OUT_DIR [--procs N] NAME=<classifier all.jsonl> [NAME=...]

Rollout list = the empty-patch classifier's all.jsonl of the run (DeepSWE: 565 rows = 113 tasks x 5; train-acc:
analysis/empty-patch/<model>/all.jsonl = 250 slots, final attempt each). opencode_results_fpath = the ROOT OpenCode session
export (subagent sessions are not in it).

Unit "call" = one assistant message of the export that is not a compaction summary (= think_watch.py / thinkx.py turn =
one model call). reasoning = reasoning parts + the leading <think>...</think> block of the text (the eval server writes
the parsed reasoning that way; analyze.py split_think); visible = the text after the first </think>.
Unit "tool call" = one part of type "tool" (OpenCode records an unknown tool name as tool "invalid"; a zod schema failure
as status "error" with "The <tool> tool was called with invalid arguments: SchemaError(...)").

Per-call flags
  R_CALL_TAIL      the reasoning ENDS with tool-call closing markup (</tool_call>, </function> or </parameter>): a call
                   written inside the think block. sub: opener (<tool_call> in the reasoning) / headless; and whether
                   the call made no parsed tool call ('lost': the session-ending E3b case) or still made one
  R_CALL_MENTION   tool-call markup inside the reasoning that is not a trailing call (quoted / cited markup, e.g. the
                   model re-reading its own earlier call, or markup it found inside a file)
  R_SPECIAL        a <|name|> string in the reasoning (sc14's only real ones are <|im_start|>/<|im_end|>); sub = names
  FOREIGN          another chat format's tool / turn syntax in reasoning, visible text or a tool argument: <|open|>,
                   <|close|>, <|sep|> (sc14 base's own habit), harmony <|start|>/<|channel|>/<|constrain|>/to=functions.,
                   DeepSeek DSML (｜DSML｜), <invoke name=, <function_calls>, [TOOL_CALLS], Kimi <|tool_call_begin|>;
                   sub = where:token
  V_CALL           tool-call markup in the visible text that the parser did not turn into a call (malformed visible
                   call); sub = shape: no_function_tag (<tool_call> then <parameter=), param_as_function
                   (<parameter=bash> as the function tag), bracket ([function=...]), dsml, bare_body (<tool_call> then
                   raw text), headless (closing tags without an opener), well_formed (looks complete), other
  THINK_TAG        doubled / misplaced think tags: </think> or <think> inside the visible text, <think> inside the
                   reasoning; sub = which
  UNCLOSED         text starts with <think> and never closes (in exports these are only the visible ones; calls the
                   lenient parser rescued look closed - see the train-acc counting-parser logs)
  EMPTY            call that ended by itself (finish stop / tool-calls) with no reasoning, no visible text, no tool call
  LE1              call that ended by itself with <= 1 output token (EOS right after the <think> generation prefix)
  TC_NO_TOOLS      finish == tool-calls but no tool part
  CUT_IN_CALL      finish == length while a <tool_call>/<function= block is open (cut inside a call by the 131k cap)
  DUP_PARALLEL     >= 2 tool parts of the call with identical (tool, input)  [also counted: extra copies]
  MANY_PARALLEL    >= 10 tool parts in one call
  TOOL_RESP        <tool_response> / </tool_response> written by the model (reasoning or visible): a self-written result
  ROLE             <|im_start|> / <|im_end|> written in reasoning / visible text
Per-tool-call flags
  T_INVALID        tool "invalid". sub: shell_cmd (sed, cat, python3, git, test, ...: a command used as the tool name),
                   markup_in_name (the name holds a newline / <parameter / </ ), placeholder (command,
                   *_placeholder, function_name, ...), near_miss (edit distance <= 2 to a real tool), other
  T_SCHEMA         schema error. sub: tool:missing:key / tool:type:key (+ ':with_foreign_keys' when the input also has
                   another tool's or a malformed key, i.e. merged / boundary-broken input)
  T_UNKNOWN_KEY    an input key outside the tool's parameter set. sub: base_habit (bash.description: sc14's habit, not in
                   this OpenCode's schema), cross_tool (a key of another tool, e.g. write.command, edit.workdir,
                   bash.filePath), case_variant (filepath, newstring), malformed (key holds whitespace / markup / a value:
                   '<parameter=KEY' without '>'), placeholder (parameter_name, function_name, ...), novel (env, label, ...)
  ARG_SPILL        a string argument that holds next-turn markup: '</think>' (the model's next reasoning + visible text +
                   a new <tool_call> absorbed into the parameter) / call markup (<tool_call>, <function=, </function>,
                   </tool_call>) / parameter tags (<parameter=, </parameter>). sub = think / call / param; plus
                   ARG_SPILL_FILE when the tool is edit / write and it completed (the markup was written into a file)
  T_PATH_BAD       filePath / path / workdir with leading/trailing whitespace, a newline, markup or wrapping quotes
  T_BIG_ARG        serialized input >= 65,536 chars (T_HUGE_ARG >= 200,000)
  T_ARG_LOOP       a string argument with a degenerate repetition: a block of 1-60 lines repeated with no change over
                   >= max(30, 3 x block) consecutive lines, or a <= 40-char unit repeated >= 100 times in a row
  T_EMPTY_REQ      a must-be-non-empty argument (command, filePath, pattern, url, ...) is blank
  T_NOOP_EDIT      edit with oldString == newString
  T_LINENO         edit oldString / write content copied with the read tool's "N: " line-number prefixes

2026-11-04: copied unchanged (detector logic) into train-acc-v2-20261024/analysis/v3etp20/degenerate/format; run on
ta-sc14, ta-etp20, ta-v3etp5, ta-v3etp10 (reproduction check against the published out/ta-*.rollouts.jsonl) and ta-v3etp20.
"""
import argparse
import json
import os
import re
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import orjson

RE_TOOLCALL = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")  # analyze.py RE_TOOLCALL
RE_CALL_TAIL = re.compile(r"(</tool_call>|</function>|</parameter>)\s*$")
RE_SPECIAL_TOK = re.compile(r"<\|[A-Za-z_]+\|>")
FOREIGN = ["<|open|>", "<|close|>", "<|sep|>", "<|tool_call_begin|>", "<|tool_calls_section_begin|>",
           "<|tool_call_argument_begin|>", "<|tool_call_end|>", "[TOOL_CALLS]", "<function_calls>", "<invoke name=",
           "to=functions.", "<|channel|>", "<|message|>", "<|call|>", "<|start|>", "<|constrain|>", "<|python_tag|>",
           "<|eom_id|>", "<|eot_id|>", "<|start_header_id|>", "｜DSML｜", "<｜tool▁call▁begin｜>", "<start_function_call>",
           "<|plugin|>", "<|action_start|>", "<|interpreter|>"]
RE_FOREIGN = re.compile("|".join(re.escape(s) for s in FOREIGN))
RE_TOOL_RESP = re.compile(r"</?tool_response>")
RE_ROLE = re.compile(r"<\|im_start\|>|<\|im_end\|>")
RE_SPILL_THINK = re.compile(r"</think>")
RE_SPILL_CALL = re.compile(r"<tool_call>|</tool_call>|<function=[^>\n]*>|</function>")
RE_SPILL_PARAM = re.compile(r"<parameter=[^>\n]*>|</parameter>")
RE_LINENO = re.compile(r"^\s*\d+: ")
RE_SCHEMA = re.compile(r"The (\w+) tool was called with invalid arguments: SchemaError\((.*?)\n\s*at (\[.*?\])\)", re.S)
RE_UNIT = re.compile(r"(.{1,40}?)\1{99,}", re.S)
PLACEHOLDERS = {"parameter_name", "parameter", "function_name", "tool_name", "name_of_parameter", "param", "arg",
                "argument", "example_parameter", "param_name", "key", "value", "placeholder"}
SHELL_CMDS = {"sed", "cat", "python", "python3", "git", "grep", "rg", "ls", "find", "head", "tail", "awk", "test",
              "build", "go", "npm", "npx", "node", "cargo", "make", "pytest", "echo", "cd", "diff", "tac", "wc", "w",
              "timeout", "sh", "view", "bazel", "pip", "uv", "tsc", "jest", "nl", "cp", "mv", "rm", "mkdir", "sort"}
SCHEMA = {  # OpenCode 1.17.11 tool parameters = the eval request payload's tools = the SFT system prompt's <tools>
    "bash": ({"command", "timeout", "workdir"}, {"command"}),
    "edit": ({"filePath", "oldString", "newString", "replaceAll"}, {"filePath", "oldString", "newString"}),
    "glob": ({"pattern", "path"}, {"pattern"}),
    "grep": ({"pattern", "path", "include"}, {"pattern"}),
    "read": ({"filePath", "offset", "limit"}, {"filePath"}),
    "skill": ({"name"}, {"name"}),
    "task": ({"description", "prompt", "subagent_type", "task_id", "command"}, {"description", "prompt", "subagent_type"}),
    "todowrite": ({"todos"}, {"todos"}),
    "webfetch": ({"url", "format", "timeout"}, {"url"}),
    "write": ({"content", "filePath"}, {"content", "filePath"}),
}
ALL_KEYS = set().union(*(a for a, _ in SCHEMA.values()))
NONEMPTY = {"command", "filePath", "pattern", "url", "prompt", "description", "subagent_type", "name", "todos"}
BIG = 65536
HUGE = 200000
MAX_EX = 3


def split_think(text):
    s = text.strip()
    if not s.startswith("<think>"):
        return None, s
    if "</think>" not in s:
        return s[len("<think>"):], None
    inner, after = s[len("<think>"):].split("</think>", 1)
    return inner, after.strip()


def excerpt(s, m=None, width=160):
    if s is None:
        return None
    if m is None:
        return s[:2 * width]
    a, b = max(0, m.start() - width), min(len(s), m.end() + width)
    return ("..." if a else "") + s[a:b] + ("..." if b < len(s) else "")


def strings_of(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings_of(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings_of(v, f"{path}[{i}]")


def canon(obj):
    try:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode()
    except TypeError:
        return json.dumps(obj, sort_keys=True, default=str)


def open_call_at_end(s):
    last_open = max(s.rfind("<tool_call>"), s.rfind("<function="))
    if last_open < 0:
        return False
    return max(s.rfind("</tool_call>"), s.rfind("</function>")) < last_open


def edit_distance_le2(a, b):
    if abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] <= 2


def v_call_shape(vis):
    i = vis.find("<tool_call>")
    if "｜DSML｜" in vis:
        return "dsml"
    if i < 0:
        return "headless" if RE_TOOLCALL.search(vis) else "other"
    after = vis[i + len("<tool_call>"):].lstrip()
    if after.startswith("<function="):
        body = after
        if "</function>" in body and "</tool_call>" in body and ("<parameter=" not in body or "</parameter>" in body):
            return "well_formed"
        return "function_unclosed"
    if after.startswith("[function="):
        return "bracket"
    if re.match(r"<parameter=(bash|read|edit|write|grep|glob|todowrite|webfetch|task)>", after):
        return "param_as_function"
    if after.startswith("<parameter="):
        return "no_function_tag"
    if after.startswith("<|") or after.startswith("{"):
        return "foreign_or_json"
    return "bare_body"


def arg_loop(s):
    """(kind, detail) if the string holds a degenerate repetition, else None."""
    if len(s) < 2000:
        return None
    tail = s[-20000:]
    raw = tail.encode("utf-8", "surrogatepass")
    if len(zlib.compress(raw)) / max(1, len(raw)) >= 0.15:  # prefilter: natural code / text compresses to ~0.2-0.45
        return None
    lines = s.split("\n")[-600:]
    n = len(lines)
    best = None
    for p in range(1, 61):
        run = 0
        for i in range(n - p):
            if lines[i] == lines[i + p] and lines[i].strip():
                run += 1
                if run >= max(30, 3 * p) and (best is None or run > best[1]):
                    best = (p, run)
            else:
                run = 0
    if best:
        return "line_block", f"block of {best[0]} line(s) repeated over {best[1] + best[0]} lines"
    m = RE_UNIT.search(tail)
    if m:
        return "unit", f"{m.group(1)!r} x {len(m.group(0)) // len(m.group(1))}"
    return None


def unknown_key_kind(tool, k):
    if tool == "bash" and k == "description":
        return "base_habit"
    if k in PLACEHOLDERS:
        return "placeholder"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
        return "malformed"
    allowed = SCHEMA[tool][0]
    if any(k.lower() == a.lower() for a in allowed):
        return "case_variant"
    if k in ALL_KEYS:
        return "cross_tool"
    return "novel"


class Rec:
    def __init__(self, meta):
        self.meta = meta
        self.calls = Counter()
        self.tools = Counter()
        self.sub = {}
        self.ex = {}
        self.n_calls = 0
        self.n_tool_parts = 0
        self.n_finished_calls = 0
        self.tool_names = Counter()
        self.arg_sizes = []
        self.max_arg = 0
        self.max_parallel = 0
        self.par_hist = Counter()
        self.last_flags = []
        self.flag_calls = {}     # flag -> list of call indices (first 50)

    def add(self, unit, flag, example=None, sub=None, n=1, call=None):
        (self.calls if unit == "call" else self.tools)[flag] += n
        if sub is not None:
            self.sub.setdefault(flag, Counter())[sub] += 1
        if example is not None:
            lst = self.ex.setdefault(flag, [])
            if len(lst) < MAX_EX:
                lst.append(example)
        if call is not None:
            lst = self.flag_calls.setdefault(flag, [])
            if len(lst) < 50:
                lst.append(call)

    def out(self):
        return {**self.meta, "n_calls": self.n_calls, "n_finished_calls": self.n_finished_calls,
                "n_tool_parts": self.n_tool_parts, "calls": dict(self.calls), "tools": dict(self.tools),
                "sub": {k: dict(v) for k, v in self.sub.items()}, "ex": self.ex, "tool_names": dict(self.tool_names),
                "arg_sizes_ge16k": self.arg_sizes, "max_arg": self.max_arg, "max_parallel": self.max_parallel,
                "par_hist": {str(k): v for k, v in self.par_hist.items()}, "last_flags": self.last_flags,
                "flag_calls": self.flag_calls}


def analyze(row):
    meta = {"task_id": row.get("task_id"), "ri": row.get("ri"), "shard": row.get("shard"), "code": row.get("code"),
            "subcode": row.get("subcode"), "reward": row.get("reward"), "empty_patch": row.get("empty_patch"),
            "export": row.get("opencode_results_fpath")}
    rec = Rec(meta)
    path = row.get("opencode_results_fpath")
    if not path or row.get("opencode_export_found") is False:
        meta["no_export"] = True
        return rec.out()
    try:
        data = orjson.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        meta["no_export"] = True
        return rec.out()
    idx = -1
    for message in data.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") != "assistant" or info.get("summary") is True or info.get("mode") == "compaction":
            continue
        idx += 1
        rec.n_calls += 1
        parts = message.get("parts") or []
        text = "".join(p.get("text") or "" for p in parts
                       if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        inner, visible = split_think(text)
        reasoning = "".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or "")
        vis = visible or ""
        tparts = [p for p in parts if p.get("type") == "tool"]
        finish = info.get("finish")
        out_tok = (info.get("tokens") or {}).get("output")
        self_ended = finish in ("stop", "tool-calls") and not info.get("error")
        rec.n_finished_calls += self_ended
        rec.par_hist[min(len(tparts), 10)] += 1
        where = {"task_id": meta["task_id"], "ri": meta["ri"], "call": idx, "finish": finish, "out_tok": out_tok,
                 "n_tools": len(tparts)}
        flags = []

        def f(flag, m=None, s=None, sub=None, ex=None):
            flags.append(flag)
            rec.add("call", flag, {**where, "excerpt": ex if ex is not None else excerpt(s, m)}, sub, call=idx)

        # reasoning: calls written inside the think block vs quoted markup
        rs = reasoning.rstrip()
        if RE_TOOLCALL.search(reasoning):
            if RE_CALL_TAIL.search(rs):
                seg = rs[max(rs.rfind("</think>"), 0):]
                opener = "<tool_call>" in rs[-4000:] if len(rs) > 4000 else "<tool_call>" in rs
                f("R_CALL_TAIL", None, None, sub=("opener" if opener else "headless") + (":lost" if not tparts else ":with_call"),
                  ex=rs[-400:])
            else:
                f("R_CALL_MENTION", RE_TOOLCALL.search(reasoning), reasoning)
        sp = RE_SPECIAL_TOK.findall(reasoning)
        if sp:
            f("R_SPECIAL", RE_SPECIAL_TOK.search(reasoning), reasoning, sub="+".join(sorted(set(sp))))
        for where_name, s in (("reasoning", reasoning), ("visible", vis)):
            m = RE_FOREIGN.search(s)
            if m:
                f("FOREIGN", m, s, sub=f"{where_name}:{m.group(0)}")
            m = RE_TOOL_RESP.search(s)
            if m:
                f("TOOL_RESP", m, s, sub=where_name)
            m = RE_ROLE.search(s)
            if m:
                f("ROLE", m, s, sub=f"{where_name}:{m.group(0)}")
        if "<think>" in reasoning:
            f("THINK_TAG", re.search("<think>", reasoning), reasoning, sub="reasoning:<think>")
        if vis:
            m = RE_TOOLCALL.search(vis)
            if m or "｜DSML｜" in vis:
                f("V_CALL", m, vis, sub=v_call_shape(vis))
            m = re.search("</think>", vis)
            if m:
                f("THINK_TAG", m, vis, sub="visible:</think>")
            m = re.search("<think>", vis)
            if m:
                f("THINK_TAG", m, vis, sub="visible:<think>")
        if inner is not None and "</think>" not in text:
            f("UNCLOSED", None, None, ex=reasoning[-300:])
        if self_ended and not tparts and not reasoning.strip() and not vis.strip():
            f("EMPTY", None, None, ex=text[:200])
        if self_ended and isinstance(out_tok, (int, float)) and out_tok <= 1:
            f("LE1", None, None, ex=text[:200])
        if finish == "tool-calls" and not tparts:
            f("TC_NO_TOOLS", None, None, ex=reasoning[-200:] + " || " + vis[-200:])
        if finish == "length" and (open_call_at_end(vis) or (visible is None and open_call_at_end(reasoning))):
            f("CUT_IN_CALL", None, None, ex=(vis or reasoning)[-300:])
        rec.max_parallel = max(rec.max_parallel, len(tparts))
        if len(tparts) >= 10:
            f("MANY_PARALLEL", None, None, sub=len(tparts), ex=", ".join(p.get("tool") or "?" for p in tparts)[:300])
        seen = Counter(((p.get("tool"), canon((p.get("state") or {}).get("input"))) for p in tparts))
        dups = sum(c - 1 for c in seen.values() if c > 1)
        if dups:
            (tool, inp), _ = max(seen.items(), key=lambda kv: kv[1])
            f("DUP_PARALLEL", None, None, sub=tool,
              ex=f"{len(tparts)} calls, {dups} extra identical; e.g. {tool} {inp[:240]}")
            rec.calls["DUP_PARALLEL_extra_copies"] += dups
        # tool calls
        for p in tparts:
            rec.n_tool_parts += 1
            tool = p.get("tool")
            rec.tool_names[tool] += 1
            st = p.get("state") or {}
            inp = st.get("input")
            status = st.get("status")
            error = st.get("error") if isinstance(st.get("error"), str) else (str(st.get("error")) if st.get("error") else "")
            twhere = {**where, "tool": tool, "status": status}

            def t(flag, ex=None, sub=None):
                flags.append(flag)
                rec.add("tool", flag, {**twhere, "excerpt": ex}, sub, call=idx)

            ser = canon(inp)
            size = len(ser)
            rec.max_arg = max(rec.max_arg, size)
            if size >= 16384:
                rec.arg_sizes.append([tool, size])
            if tool == "invalid":
                name = (inp or {}).get("tool") if isinstance(inp, dict) else None
                msg = str((inp or {}).get("error") if isinstance(inp, dict) else "")
                nm = str(name)
                if "unavailable tool" not in msg:
                    kind = "other_error"
                elif "\n" in nm or "<" in nm or "}" in nm or " " in nm.strip():
                    kind = "markup_or_text_in_name"
                elif nm.split("-")[0] in SHELL_CMDS or nm.split("-")[0].rstrip("s") in SHELL_CMDS or nm in SHELL_CMDS:
                    kind = "shell_cmd"
                elif nm in PLACEHOLDERS or "placeholder" in nm or nm in ("command", "endofmessage"):
                    kind = "placeholder"
                elif any(edit_distance_le2(nm.strip("`"), real) for real in SCHEMA):
                    kind = "near_miss"
                else:
                    kind = "other"
                t("T_INVALID", f"tool={name!r} | {msg[:160]}", sub=kind)
                rec.sub.setdefault("invalid_names", Counter())[nm[:80]] += 1
                continue
            foreign_keys = []
            spill = None
            for kp, s in strings_of(inp):
                m = RE_SPILL_THINK.search(s)
                kind = "think" if m else None
                if not m:
                    m = RE_SPILL_CALL.search(s)
                    kind = "call" if m else None
                if not m:
                    m = RE_SPILL_PARAM.search(s)
                    kind = "param" if m else None
                if m and (spill is None or ["param", "call", "think"].index(kind) > ["param", "call", "think"].index(spill[0])):
                    spill = (kind, kp, excerpt(s, m, 140))
                m = RE_FOREIGN.search(s)
                if m:
                    t("FOREIGN_ARG", f"{kp}: {excerpt(s, m, 100)}", sub=f"arg:{m.group(0)}")
                lp = arg_loop(s) if len(s) >= 2000 else None
                if lp:
                    t("T_ARG_LOOP", f"{kp}: {len(s)} chars, {lp[1]}: ...{s[-160:]!r}", sub=f"{tool}:{lp[0]}")
            if isinstance(inp, dict) and tool in SCHEMA:
                allowed, required = SCHEMA[tool]
                for k in inp:
                    if k not in allowed:
                        kind = unknown_key_kind(tool, k)
                        if kind != "base_habit":
                            foreign_keys.append(k)
                        t("T_UNKNOWN_KEY", f"{k[:120]!r}={canon(inp[k])[:160]}",
                          sub=f"{kind}:{tool}.{k[:40]}" + (":with_spill" if spill else ""))
                for k in required:
                    v = inp.get(k)
                    if k in NONEMPTY and isinstance(v, str) and not v.strip():
                        t("T_EMPTY_REQ", f"{k}={v!r}", sub=f"{tool}.{k}")
                for k in ("filePath", "path", "workdir"):
                    v = inp.get(k)
                    if isinstance(v, str) and v and (v != v.strip() or "\n" in v or "<" in v or
                                                      (len(v) > 1 and v[0] == v[-1] and v[0] in "'\"`")):
                        t("T_PATH_BAD", f"{k}={v[:160]!r}", sub=f"{tool}.{k}")
                if tool == "edit":
                    old, new = inp.get("oldString"), inp.get("newString")
                    if isinstance(old, str) and old == new:
                        t("T_NOOP_EDIT", f"oldString==newString ({len(old)} chars): {old[:160]!r}")
                if tool in ("edit", "write"):
                    s = inp.get("oldString") if tool == "edit" else inp.get("content")
                    if isinstance(s, str):
                        lines = [ln for ln in s.split("\n") if ln.strip()]
                        if len(lines) >= 2 and sum(bool(RE_LINENO.match(ln)) for ln in lines) >= 0.8 * len(lines):
                            t("T_LINENO", s[:200], sub=tool)
            elif tool not in SCHEMA and tool is not None:
                rec.sub.setdefault("other_tool_names", Counter())[str(tool)] += 1
            if status == "error" and "called with invalid arguments" in error:
                m = RE_SCHEMA.search(error)
                if m:
                    what, keypath = m.group(2), m.group(3)
                    kind = "missing" if what.startswith("Missing key") else "type" if what.startswith("Expected") else "other"
                    sub = (f"{tool}:{kind}:{keypath}" + (":with_foreign_keys" if foreign_keys else "")
                           + (":with_spill" if spill else ""))
                else:
                    sub = f"{tool}:unparsed"
                t("T_SCHEMA", f"{error[:180]} | input={ser[:240]}", sub=sub)
            if spill:
                t("ARG_SPILL", f"{spill[1]}: {spill[2]}", sub=f"{spill[0]}:{tool}")
                if tool in ("edit", "write") and status == "completed" and spill[1] in ("newString", "content"):
                    t("ARG_SPILL_FILE", f"{spill[1]}: {spill[2]}", sub=f"{spill[0]}:{tool}")
            if size >= BIG:
                t("T_BIG_ARG", f"{size} chars; {ser[:160]}", sub=tool)
            if size >= HUGE:
                t("T_HUGE_ARG", f"{size} chars", sub=tool)
        rec.last_flags = sorted(set(flags))
    return rec.out()


def run(name, path, out_dir, procs):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out = Path(out_dir) / f"{name}.rollouts.jsonl"
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    with ProcessPoolExecutor(procs) as pool, open(tmp, "w") as handle:
        for res in pool.map(analyze, rows, chunksize=4):
            res["model"] = name
            handle.write(json.dumps(res, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(name, len(rows), "->", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.runs:
        name, path = item.split("=", 1)
        run(name, path, args.out, args.procs)


if __name__ == "__main__":
    main()
