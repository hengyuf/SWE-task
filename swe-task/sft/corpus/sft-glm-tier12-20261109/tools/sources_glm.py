#!/usr/bin/env python3
"""GLM-5.3 tier-1/2 SFT corpus (2026-11-09), sources: select GLM-5.3's successful OpenCode rollouts on the tier-1/2
tasks and extract them, with their OpenCode context windows, into the v1 batch line layout that build_glm.py renders.

User request (hengyuf, GitHub issue hengyuf/SWE-task#1, comment 6758873905, verbatim): "现在收集tier1/2 的 successful
traces （包括有compaction的）， 将其加工为类似SFT v3 一样的dataset， 这里对于一个rollout 的多个segment（有compaction的），
将其视为多条独立的traces，task ID 可以用一个用于标记，但训练时保留各自边界。标准 SFT 可以把 segment 当独立样本，做
packing时，用现有 packer 的边界/attention 语义，别让 loss 或上下文跨样本串起来。pack 好data 把data 用datamover 移动到 AGA
（oci-aga-slurm-1-login-01.nvidia.com）上。（AGA 访问密码与HSG 相同）"
Design: /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/DESIGN.md (rev 2).

Inputs (read-only): the GLM final records (every re-run applied) of glm-rollouts-tasksall-20261108
(analysis/rollouts.jsonl, tier 1 and 2 records) and glm-rollouts-sftv3-20261107 (analysis/final-rollouts.jsonl, the 220
SFT-v3 tasks, all tier 1 by inventory), the attempt dirs they name, each probe campaign's tasks/<task_id>/tests/, the
task inventory tasks-all-20261106/all_tasks.jsonl (tier rule of tier12_accept_compare.py: 1 = accepted_or_strong,
2 = audit partial) and that script's tier12-accept-compare.json (GLM-teacher verdict, informational), and the old-grader
classes WORK/regrade/classes.json + re-grade results WORK/regrade/results.json (may not exist yet).

Selection, per considered final record (every tier-1/2 record); the FIRST failing rule is the exclusion reason (the
counts are also reported independently, among final passes):
  1. final_status_<s>         the final record's status is not "pass" (fail, or infra)
  2. not_clean                v3's rule (sources_v4v5.examine / decide): terminal.json evaluation_completed and reward 1,
                              export.json + http.jsonl + model.patch present, no cached generation-validity failure
  3. old_grader_*             DESIGN "Old-grader rule": classes.json class of the pass ("n/a" = current grader, "A",
                              "B"); class B needs a re-grade result (results.json) with reward 1:
                              old_grader_not_regraded (no result) / old_grader_regrade_fail (reward != 1); a class A /
                              n/a pass whose (check) re-grade returned reward 0 is old_grader_regrade_fail as well, an
                              inconclusive check (reward null) keeps it; old_grader_unclassified = the pass is missing
                              from classes.json or its attempt_dir differs there
  4. degenerate_reply         v3's rule: degenerate.py classify() (the v4 dir's, exec'd from its source text by
                              sources_v4v5.py, block sha256 asserted) labels some status-200 reply of http.jsonl pad /
                              special / empty, task subagents and summary-writing (compaction) calls included. classify()
                              reads message.reasoning_content OR message.reasoning (asserted below), so GLM's
                              reasoning-only replies are not "empty" and no normalization is needed
  5. harness_unexpected       v3's rule, per window: the first request of every window has v1's system base and v1's
                              tool set (and the first request of http.jsonl is window 1's first request)
  6. two_source_mismatch      the two-source gate (port of scratch/glm-compaction/segments.py, below) found a difference
                              or an unexpected structure
Compaction is not an exclusion: a compacted rollout is cut into its context windows.

Two-source gate (hard; run on every clean final pass), ported from scratch/glm-compaction/segments.py (load_calls /
walk / export_segments / compare / analyze) and analyze_attempt.py (transition check):
  http.jsonl  a call = a request event joined with its response on (call_id, http_attempt); kept if status 200 with a
              parsable body. A request with no tools whose last user message starts with OpenCode's anchored-summary
              prompt is a summary-writing (compaction) call. pred(B) = the latest earlier 200 call A (not compaction)
              whose messages[1:] are a strict prefix of B's (per-message sha, the system message excluded); every such
              transition must add exactly [assistant == A's response (content, reasoning, tool calls: id, name, parsed
              arguments), A's tool results in order] (analyze_attempt.check_assistant). Window 1 = the chain from the
              head [system, user == export task]; the main compaction k = the first compaction call after window k's
              last call sharing assistant messages with the main windows; window k+1 = the chain from the first head
              after it whose messages[1] is "What did we do so far?" (and messages[2] the summary). A branch point (two
              200 successors), a 200 call of the main session outside every window, a transition that is not
              [response, results] or a change of the tool set inside a window fails the gate.
  export.json every window rebuilt with OpenCode 1.17.11 + Gym semantics (window 1 = task + the window's assistant /
              tool messages; window k>1 = "What did we do so far?", the summary, the tail from the compaction part's
              tail_start_id, the synthetic "Continue ..." message, the window's own messages); an assistant message
              with no text and no tool part is not sent (skipped); the tool results of the window's LAST sent message
              are not part of the window (fix of the prototype, which cut the results of the last export message even
              when that message was skipped, e.g. the errored empty final message of tasksall t1-h00 11-2).
  compare     the window's last request + its response vs the rebuild, message by message: role, content, reasoning,
              tool calls (id, name, parsed arguments), tool_call_id; system messages not compared (they change inside
              some windows: date line, AGENTS.md); #windows == #main compaction calls + 1 == #compaction parts + 1.
Turns that are never targets (window info "dropped"): export assistant messages with no text and no tool part
("empty_no_parts", or "errored_no_call" when info.error is set: no 200 call exists for them), and a reply with
finish_reason "length" ("finish_length": only allowed as the last call of a window; the window then ends at the previous
call). Every other call of a window is a target; all but the last target of a window must have tool calls.
Window k>=2 context = its first request's messages[1:] (user "What did we do so far?", the summary with the compaction
call's reasoning, the tail, the synthetic "Continue ..." user message) normalized for the chat template (content as text,
reasoning_content, tool calls {type, function {name, arguments: the parsed JSON object}}, tool {role, content}; ids
dropped) and redacted with build_batch.py's patterns (as the export). System prompt + tools of every window = its FIRST
request (v2's harness rule generalized; flag system_changed_in_window). How a window's system prompt changes (fix
2026-11-10): system_change_kinds = the parts of build_batch.split_system that differ from the first request's in some
later request of the window ("date" = the env block's "Today's date:" line, "env" = another env line, "instructions" =
the text after the env block: AGENTS.md / CLAUDE.md content and the skills list, "base" = the base instructions, "other"
= a difference outside the message text), system_first_change_call, system_changed_targets (target calls whose own
request had another system prompt than the sample's) and system_content_changed_targets (the same, ignoring a different
date line). end_tool_results of a window whose last target made tool calls: "next_window_context" when OpenCode's tail
of the next window holds that turn (its results are then in the next window's context piece), else "in_no_sample" (the
next window's tail does not hold it, or there is no next window: the results are in no sample); null when the window
ends in a final answer.
Task language (fix 2026-11-10) = synth/verifier/test_paths.py repo_language(top100.jsonl, target, repo) of the task's
inventory row for every task (the rule of regrade/touch_scan.py, i.e. the language classes.json carries for classes A /
B; a mismatch is listed in the report). Re-run provenance (fix 2026-11-10): a final record whose attempt dir lies in
another probe campaign is a re-run; rerun_campaign / rerun_slot / rerun_reason / original_attempt_dir / original_status
come from the re-run mapping of the rollout campaigns' results/ (tasksall glmlocal-dip*.json + glmlocal-dip*-parts/,
reason = the file's rule; sftv3 glmlocal-infra-rerun1.json, reason = its request; the drivers' sandbox-failure
re-runs, results/glmlocal-<campaign>.json sandbox_failure_reruns), matched on (campaign, task_id, slot, re-run attempt).

Output (<out>/sources/):
  glm/trajectories.jsonl.gz   one line per kept rollout {meta, trajectory (export.json, redacted), final_patch} written
                              by v3's sources_v4v5.write_batch (meta_of replaced by the GLM meta below); meta["glm"]
                              holds the window info: route ("v2_convert" = a non-compacted pass ending in a final answer
                              with no dropped turn: v2's convert() unchanged; else "segment"), and per window: first /
                              last target call index, chain span, target export indices, dropped turns with reason, the
                              system / tools / params sha256 of the window's first request, system_changed_in_window and
                              the system change fields above, end_tool_results, post_final_continue, the compaction call
                              that follows it, and for k>=2 the context
  glm/harness.json            v3's layout (write_batch: system variants / tool sets / params of each rollout's first
                              request, keyed by build_batch.sha256_json of the redacted value) plus every window's first
                              request ("window_heads" counts)
  glm/selection_report.json   every considered record: kept or the first failing rule (+ detail); counts per rule,
                              independent counts among final passes, old-grader / re-grade summary, window counts, inputs
                              with sha256 (incl. the v1 / v2 harness.json references of rule 5, top100.jsonl,
                              test_paths.py and the re-run mapping files)
  task_tests.json             per task (inventory key): test.patch / grader.py sha256, the test files test.patch creates
  regrade/classes.json, regrade/results.json   byte copies of the old-grader inputs read (sha256 in the report)
Ids: rollout = glm53-<batch>/<campaign>/<task_id>/<slot>; sample (build_glm.py) = rollout + "#seg<k>of<K>".

  PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
  $PY -I -B tools/sources_glm.py [--out DIR] [--workers 16] [--only KEY ... | --only-file F | --limit N]
      [--allow-missing-regrade] [--regrade-results PATH]
KEY = "<batch>/<campaign>/<task_id>/<slot>" (classes.json's key) or the rollout id. --only / --only-file / --limit and
--regrade-results PATH (another results file, e.g. a test fixture) are dry runs (report "partial": true); the real corpus
dir refuses them, and refuses a run while some class B pass has no re-grade result (regrade/results.json missing or
still being written) unless --allow-missing-regrade. A re-grade result whose model_patch_sha256 is not the sha256 of the
pass's model.patch is a re-grade of another patch: ignored (the pass counts as not re-graded).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
WORK = S / "sft/corpus/glm-tier12-20261109"
CORPUS = S / "sft/corpus/sft-glm-tier12-20261109"
DESIGN = WORK / "DESIGN.md"
V3_SOURCES = S / "sft/corpus/sft-v3-20261029/tools/sources_v4v5.py"
V3_SOURCES_SHA256 = "f06c89aab868a68daec96995734fda6d4954f09485c8992040649b0be67c5aab"  # sft-v3-20261029 SHA256SUMS
TASKSALL = S / "sft/glm-rollouts-tasksall-20261108"
SFTV3 = S / "sft/glm-rollouts-sftv3-20261107"
FINAL_RECORDS = {"tasksall": TASKSALL / "analysis/rollouts.jsonl", "sftv3": SFTV3 / "analysis/final-rollouts.jsonl"}
BATCHES = ("tasksall", "sftv3")
INVENTORY = S / "sft/tasks-all-20261106/all_tasks.jsonl"
SFTV3_TASKS = SFTV3 / "tasks.json"
ACCEPT_COMPARE = TASKSALL / "analysis/tier12-accept-compare.json"
CLASSES = WORK / "regrade/classes.json"
REGRADE_RESULTS = WORK / "regrade/results.json"
TEST_PATHS = S / "synth/verifier/test_paths.py"  # repo_language(): the language rule of regrade/touch_scan.py
TOP100 = S / "top100.jsonl"
RERUN_RESULTS = {"tasksall": TASKSALL / "results", "sftv3": SFTV3 / "results"}  # re-run mapping files (provenance)
PREFIX = "glm53"
SOURCE = "glm"                 # the source label (sources/glm/); rollout_batch = tasksall | sftv3
PATH_ALIASES = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/",)
PATH_CANON = "/lustre/fsw/portfolios/nemotron/users/hengyuf/"
REQUEST = ("hengyuf, GitHub issue hengyuf/SWE-task#1, comment 6758873905 (verbatim): 现在收集tier1/2 的 successful "
           "traces （包括有compaction的）， 将其加工为类似SFT v3 一样的dataset， 这里对于一个rollout 的多个segment"
           "（有compaction的）， 将其视为多条独立的traces，task ID 可以用一个用于标记，但训练时保留各自边界。标准 SFT 可以把 "
           "segment 当独立样本，做packing时，用现有 packer 的边界/attention 语义，别让 loss 或上下文跨样本串起来。pack 好"
           "data 把data 用datamover 移动到 AGA（oci-aga-slurm-1-login-01.nvidia.com）上。（AGA 访问密码与HSG 相同）")
RULES = ("final pass (re-runs applied) -> clean completion (v3) -> old-grader rule (class B needs a current-verifier "
         "re-grade with reward 1; a class A / n/a check re-grade with reward 0 excludes) -> no degenerate reply anywhere "
         "(degenerate.py classify over every 200 reply, subagents and summary-writing calls included) -> v1 harness base "
         "and tool set for every window's first request -> two-source gate (export rebuild == http.jsonl prefix chain, "
         "every window)")


# ---------------------------------------------------------------- v3's sources_v4v5.py, imported from its path
def _load_v3_sources():
    data = V3_SOURCES.read_bytes()
    assert hashlib.sha256(data).hexdigest() == V3_SOURCES_SHA256, "sft-v3 sources_v4v5.py changed"
    spec = importlib.util.spec_from_file_location("sft_v3_sources_v4v5", V3_SOURCES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its own asserts: build_batch.py helper block, degenerate.py block (sha256)
    return module


sv = _load_v3_sources()
BB, redact_obj, sha256_json, sha256_bytes, sha256_file = sv.BB, sv.redact_obj, sv.sha256_json, sv.sha256_bytes, \
    sv.sha256_file
# journal_of's length cut is finish_reason == "length" for GLM: its completion-cap clause used Kimi's 131,072 output cap
# (GLM: max_tokens 262,144 = the window), so the cap is disabled; COMPACT_AT (262,144 - 20,000) is GLM's too.
sv.OUTPUT_CAP = math.inf
assert sv.COMPACT_AT == 262144 - 20000
# degenerate.py's classify() must read GLM's message.reasoning (a reasoning-only reply is not empty): checked on probes;
# if it did not, the classify() that journal_of calls would get a normalized COPY (reasoning -> reasoning_content)
_PROBE_REASONING_ONLY = {"choices": [{"message": {"content": None, "reasoning": "r", "tool_calls": None}}]}
_PROBE_EMPTY = {"choices": [{"message": {"content": None, "reasoning": None, "tool_calls": None}}]}
CLASSIFY_READS_REASONING = sv.classify(_PROBE_REASONING_ONLY)[0] is None
assert sv.classify(_PROBE_EMPTY)[0] == "empty"
if not CLASSIFY_READS_REASONING:  # (not the case for the pinned block; kept for robustness)
    _v3_classify = sv.classify

    def _classify_normalized(body):
        body = json.loads(json.dumps(body))
        for choice in body.get("choices") or []:
            message = (choice or {}).get("message") or {}
            if message.get("reasoning") and not message.get("reasoning_content"):
                message["reasoning_content"] = message["reasoning"]
        return _v3_classify(body)
    sv.classify = _classify_normalized
V1H = json.loads(sv.V1_HARNESS.read_text())
V2H = json.loads(sv.V2_BAIHE_HARNESS.read_text())
V1_BASES = {v["base_sha256"] for v in V1H["system_prompt_variants"].values()}
V1_TOOLS = set(V1H["tool_sets"])


def _load_test_paths():
    """synth/verifier/test_paths.py (stdlib only), imported from its path: repo_language() = the top100.jsonl language
    of a build's repository, the rule regrade/touch_scan.py used for classes.json's language (sha256 in the report)."""
    spec = importlib.util.spec_from_file_location("swe_task_verifier_test_paths", TEST_PATHS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TP = _load_test_paths()
_LANGUAGES = {}


def language_of(row):
    """The task language of an inventory row: repo_language(top100.jsonl, target, repo), None when unknown."""
    key = (row.get("target"), row.get("repo"))
    if key not in _LANGUAGES:
        try:
            _LANGUAGES[key] = TP.repo_language(TOP100, row["target"], row["repo"])
        except (TP.UnknownLanguage, KeyError, TypeError, AttributeError):
            _LANGUAGES[key] = None
    return _LANGUAGES[key]


def norm(path):
    return sv.norm(path)


def read_json(path, default=None):
    return sv.read_json(path, default)


# ---------------------------------------------------------------- two-source gate (port of glm-compaction/segments.py)
COMPACT_PROMPTS = ("Create a new anchored summary from the conversation history.",
                   "Update the anchored summary below using the conversation history above.")
WDWDSF = "What did we do so far?"
CONTINUE = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
GYM_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)  # Gym's THINK_TAG_PATTERN (first match = reasoning)


def _mh(message) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8", "surrogatepass")).hexdigest()[:16]


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


DATE_LINE_RE = re.compile(r"(?m)^[ \t]*Today's date:.*$")  # OpenCode's env block line ("  Today's date: Sat Nov 07 2026")
SYSTEM_PARTS = ("base", "env", "date", "instructions")


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def system_parts(message):
    """(base, env block without its date line, the date line, the rest) hashes of a request's system message text, split
    as build_batch.split_system does (base instructions / <env> block / instruction files and skills), or None."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return None
    parts = BB["split_system"](text_of(message.get("content")))
    env = parts["env"]
    return (_h(parts["base"]), _h(DATE_LINE_RE.sub("", env)), _h("\n".join(DATE_LINE_RE.findall(env))),
            _h(parts["rest"]))


def system_change_kinds(head, other) -> set:
    """Which parts of the system message differ between two calls (empty: the same system message)."""
    if other["sys"] == head["sys"]:
        return set()
    if head["sys_parts"] is None or other["sys_parts"] is None:
        return {"other"}
    kinds = {name for name, a, b in zip(SYSTEM_PARTS, head["sys_parts"], other["sys_parts"]) if a != b}
    return kinds or {"other"}


def _event_kind(raw: bytes):
    head = raw[:3000]
    if b'"event":"request"' in head or b'"event": "request"' in head:
        return "request"
    if b'"event":"response"' in head or b'"event": "response"' in head:
        return "response"
    return None


def resp_as_message(r: dict) -> dict:
    m = {"role": "assistant", "content": r.get("content") or ""}
    reasoning = r.get("reasoning") or r.get("reasoning_content") or ""
    if reasoning:
        m["reasoning_content"] = reasoning
    if r.get("tool_calls"):
        m["tool_calls"] = [{"id": t.get("id"), "type": "function",
                            "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                           for t in r["tool_calls"]]
    return m


def norm_msg(m: dict) -> dict:
    tc = []
    for t in m.get("tool_calls") or []:
        args = t["function"].get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            pass
        tc.append((t.get("id"), t["function"]["name"], json.dumps(args, sort_keys=True, ensure_ascii=False)))
    return {"role": m.get("role"), "content": text_of(m.get("content")),
            "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": tc, "tool_call_id": m.get("tool_call_id")}


def compare(a_msgs, b_msgs):
    probs = []
    if len(a_msgs) != len(b_msgs):
        probs.append(f"length {len(a_msgs)} != {len(b_msgs)}")
    for i, (a, b) in enumerate(zip(a_msgs, b_msgs)):
        na, nb = norm_msg(a), norm_msg(b)
        if na != nb:
            probs.append(f"msg {i} ({na['role']}) differs in {[k for k in na if na[k] != nb[k]]}")
            if len(probs) > 6:
                break
    return probs


def _transition_problems(delta, pred_resp):
    """analyze_attempt.py's check: the messages a request adds after its predecessor are exactly [the predecessor's
    response as assistant, one tool message per tool call in order]."""
    if not pred_resp or pred_resp.get("message") is None:
        return ["predecessor has no parsable response"]
    if not delta or delta[0].get("role") != "assistant":
        return ["delta does not start with an assistant message"]
    probs = []
    expect = resp_as_message(pred_resp["message"])
    if norm_msg(delta[0]) != norm_msg(expect):
        a, b = norm_msg(delta[0]), norm_msg(expect)
        probs.append(f"re-sent assistant != response ({[k for k in a if a[k] != b[k]]})")
    ids = [t.get("id") for t in (pred_resp["message"].get("tool_calls") or [])]
    tools = delta[1:]
    if [m.get("role") for m in tools] != ["tool"] * len(tools):
        probs.append("after the assistant: roles " + "".join((m.get("role") or "?")[0] for m in tools))
    elif [m.get("tool_call_id") for m in tools] != ids:
        probs.append("tool_call_ids differ from the response's tool call ids")
    return probs


def load_calls(http_path):
    """Pass 1 over http.jsonl: every request (per-message hashes of messages[1:], system / tools hashes, predecessor,
    transition check) joined with its response (status, message, finish_reason, usage)."""
    calls, resp = [], {}
    with open(http_path, "rb") as stream:
        for line_no, raw in enumerate(stream):
            kind = _event_kind(raw)
            if kind is None:
                continue
            event = json.loads(raw)
            if event.get("event") != kind:
                continue
            if kind == "request":
                payload = event.get("payload") or {}
                msgs = payload.get("messages") or []
                last = msgs[-1] if msgs else {}
                ckind = "call"
                if (not payload.get("tools") and last.get("role") == "user"
                        and text_of(last.get("content")).startswith(COMPACT_PROMPTS)):
                    ckind = "compaction"
                hs = [_mh(m) for m in msgs[1:]]
                pred = None
                if ckind == "call":
                    for d in reversed(calls):
                        if d["kind"] != "call" or (resp.get(d["key"]) or {}).get("status") != 200:
                            continue
                        if d["n"] < len(msgs) and hs[:d["n"] - 1] == d["hs"]:
                            pred = d["i"]
                            break
                transition = None
                if pred is not None:
                    transition = _transition_problems(msgs[calls[pred]["n"]:], resp.get(calls[pred]["key"]))
                calls.append({"i": len(calls), "line": line_no, "key": (event.get("call_id"), event.get("http_attempt")),
                              "kind": ckind, "n": len(msgs), "hs": hs, "pred": pred, "transition": transition,
                              "ahs": {h for h, m in zip(hs, msgs[1:]) if m.get("role") == "assistant"},
                              "sys": _mh(msgs[0]) if msgs else None,
                              "sys_parts": system_parts(msgs[0]) if msgs and ckind == "call" else None,
                              "tools_h": _mh(payload.get("tools") or []),
                              "n_tools": len(payload.get("tools") or []),
                              "m1": text_of(msgs[1].get("content")) if len(msgs) > 1 else None,
                              "m2_content": text_of(msgs[2].get("content")) if len(msgs) > 2 else None})
            else:
                r = {"status": event.get("status"), "message": None, "finish": None, "usage": {}}
                if r["status"] == 200:
                    try:
                        body = json.loads(event.get("body") or "")
                        choice = body["choices"][0]
                        message = choice["message"]
                        r["message"] = {k: message.get(k) for k in ("content", "reasoning", "reasoning_content",
                                                                    "tool_calls")}
                        r["finish"] = choice.get("finish_reason")
                        usage = body.get("usage") or {}
                        r["usage"] = {"prompt_tokens": usage.get("prompt_tokens"),
                                      "completion_tokens": usage.get("completion_tokens")}
                    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                        r["message"] = None
                resp[(event.get("call_id"), event.get("http_attempt"))] = r
    for c in calls:
        r = resp.get(c["key"]) or {}
        c["status"] = r.get("status")
        c["msg"] = r.get("message") if r.get("status") == 200 else None
        c["finish"], c["usage"] = r.get("finish"), r.get("usage") or {}
    return calls


def fetch_requests(http_path, lines):
    """Pass 2: the payloads of the request events at the given line numbers."""
    want, out = set(lines), {}
    with open(http_path, "rb") as stream:
        for line_no, raw in enumerate(stream):
            if line_no in want:
                out[line_no] = json.loads(raw)["payload"]
                if len(out) == len(want):
                    break
    return out


def walk(calls, head):
    children = {}
    for c in calls:
        if c["pred"] is not None and c["msg"] is not None:
            children.setdefault(c["pred"], []).append(c["i"])

    def depth(i):
        best, stack = 0, [(i, 0)]
        while stack:
            j, d = stack.pop()
            best = max(best, d)
            stack += [(k, d + 1) for k in children.get(j, [])]
        return best

    members, cur, branch_points = [head], head, []
    while children.get(cur):
        nxt = children[cur]
        if len(nxt) > 1:
            branch_points.append({"at": cur, "children": nxt})
        cur = max(nxt, key=lambda k: (depth(k), k))
        members.append(cur)
    return members, branch_points


def split_think(text):
    """Gym request-side conversion of an assistant text part: first <think> block -> reasoning, all blocks removed."""
    found = GYM_THINK_RE.findall(text)
    return (found[0] if found else ""), GYM_THINK_RE.sub("", text)


def export_assistant(m):
    """Export assistant message -> (OpenAI assistant message or None when OpenCode does not send it, [tool msgs])."""
    texts, calls, results = [], [], []
    for p in m["parts"]:
        t = p["type"]
        if t == "text":
            if p.get("ignored"):
                continue
            texts.append(p["text"])
        elif t == "reasoning":
            texts.append("<think>" + p.get("text", "") + "</think>")
        elif t == "tool":
            s = p["state"]
            calls.append({"id": p.get("callID"), "type": "function",
                          "function": {"name": p["tool"], "arguments": json.dumps(s.get("input"), ensure_ascii=False)}})
            if s["status"] == "completed":
                out = "[Old tool result content cleared]" if (s.get("time") or {}).get("compacted") else s["output"]
            elif s["status"] == "error":
                out = s["error"]
            else:
                out = "[Tool execution was interrupted]"
            results.append({"role": "tool", "tool_call_id": p.get("callID"), "content": out})
        elif t in ("step-start", "step-finish", "patch", "snapshot"):
            pass
        else:
            raise AssertionError(("unexpected assistant part", t))
    assert len(texts) <= 1, "more than one text part"
    if not texts and not calls:
        return None, []
    reasoning, visible = split_think(texts[0]) if texts else ("", "")
    a = {"role": "assistant", "content": visible}
    if reasoning:
        a["reasoning_content"] = reasoning
    if calls:
        a["tool_calls"] = calls
    return a, results


def export_user_text(m):
    texts = []
    for p in m["parts"]:
        if p["type"] == "text" and not p.get("ignored"):
            texts.append(p["text"])
        elif p["type"] == "compaction":
            texts.append(WDWDSF)  # OpenCode renders the compaction marker as this user text
        elif p["type"] not in ("step-start", "step-finish"):
            raise AssertionError(("unexpected user part", p["type"]))
    return "\n".join(texts)


def export_windows(export):
    """Every window rebuilt from export.json (segments.export_segments, with the last-SENT-message fix)."""
    em = export["messages"]
    comp_users = [i for i, m in enumerate(em) if m["info"]["role"] == "user"
                  and any(p["type"] == "compaction" for p in m["parts"])]
    by_id = {m["info"]["id"]: i for i, m in enumerate(em)}
    assert em[0]["info"]["role"] == "user", "export does not start with the user task"
    bounds = [(1, comp_users[0] if comp_users else len(em), [{"role": "user", "content": export_user_text(em[0])}], 0,
               None)]
    for k, cu in enumerate(comp_users):
        part = next(p for p in em[cu]["parts"] if p["type"] == "compaction")
        assert part.get("auto") and not part.get("overflow"), ("unsupported compaction", part)
        summ = em[cu + 1]
        assert summ["info"]["role"] == "assistant" and summ["info"].get("summary") and \
            summ["info"].get("mode") == "compaction", "compaction user msg not followed by its summary"
        cont = em[cu + 2]
        assert cont["info"]["role"] == "user" and export_user_text(cont) == CONTINUE, "no synthetic continue msg"
        ctx = [{"role": "user", "content": WDWDSF}]
        a, _ = export_assistant(summ)
        n_ctx = 0
        if a is not None:
            ctx.append(a)
            n_ctx = 1
        tail_id = part.get("tail_start_id")
        tail = []
        if tail_id:
            ts = by_id[tail_id]
            prev_cu = comp_users[k - 1] if k else -1
            assert prev_cu < ts < cu, "tail_start_id outside the previous window"
            for j in range(ts, cu):
                mj = em[j]
                assert mj["info"]["role"] == "assistant" and mj["info"].get("mode") != "compaction", "tail not assistant"
                a, res = export_assistant(mj)
                if a is not None:
                    ctx.append(a)
                    ctx += res
                    n_ctx += 1
                    tail.append(j)
        ctx.append({"role": "user", "content": CONTINUE})
        end = comp_users[k + 1] if k + 1 < len(comp_users) else len(em)
        bounds.append((cu + 3, end, ctx, n_ctx, {"compaction_user": cu, "summary": cu + 1, "continue": cu + 2,
                                                  "tail_start_id": tail_id, "tail": tail}))
    out = []
    for j, (lo, hi, ctx, n_ctx, where) in enumerate(bounds):
        own = list(range(lo, hi))
        assert all(em[i]["info"]["role"] == "assistant" for i in own), "unexpected user message inside a window"
        built = [(i,) + export_assistant(em[i]) for i in own]
        sent = [(i, a, res) for i, a, res in built if a is not None]
        skipped = [i for i, a, _ in built if a is None]
        msgs = list(ctx)
        for q, (i, a, res) in enumerate(sent):
            msgs.append(a)
            if q < len(sent) - 1:
                msgs += res  # the last sent message of the window: its results are not part of this window
        out.append({"messages": msgs, "sent": [i for i, _, _ in sent], "skipped": skipped,
                    "n_context_assistants": n_ctx, "export_range": [lo, hi], "compaction": where})
    return out


def context_messages(msgs):
    """A post-compaction request's messages[1:] -> chat-template messages (content text, reasoning_content, tool calls
    with parsed arguments; ids dropped). Raises on an unexpected role or argument."""
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": text_of(m.get("content"))})
        elif role == "assistant":
            a = {"role": "assistant", "content": text_of(m.get("content"))}
            reasoning = m.get("reasoning_content")
            if reasoning is None:
                reasoning = m.get("reasoning")
            if reasoning:
                a["reasoning_content"] = reasoning
            if m.get("tool_calls"):
                calls = []
                for t in m["tool_calls"]:
                    f = t.get("function") or {}
                    args = f.get("arguments")
                    args = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(args, dict):
                        raise ValueError("context tool call arguments are not a JSON object")
                    calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
                a["tool_calls"] = calls
            out.append(a)
        elif role == "tool":
            out.append({"role": "tool", "content": text_of(m.get("content"))})
        else:
            raise ValueError(f"unexpected role {role!r} in a post-compaction context")
    return out


def window_harness(payload):
    """(system_messages, tools, params, their sha256s, base sha256) of a request body, redacted as v3 (write_batch)."""
    h = BB["harness_of"](payload)
    system_messages = redact_obj(h["system_messages"], Counter())
    tools = redact_obj(h["tools"], Counter())
    text = "\n".join(BB["_text"](m) for m in system_messages)
    return {"system_messages": system_messages, "tools": tools, "params": h["params"],
            "system_sha256": sha256_json(system_messages), "tools_sha256": sha256_json(tools),
            "params_sha256": sha256_json(h["params"]),
            "base_sha256": sha256_bytes(BB["split_system"](text)["base"].encode("utf-8", "surrogatepass"))}


def gate(gen_dir: Path, export: dict, first_request: dict):
    """The two-source gate and the window info of one rollout (see the module docstring)."""
    problems = []
    http_path = gen_dir / "http.jsonl"
    calls = load_calls(http_path)
    ok = [c for c in calls if c["msg"] is not None]
    em = export["messages"]
    task = export_user_text(em[0])
    n_comp_export = sum(1 for m in em if m["info"]["role"] == "user" and any(p["type"] == "compaction"
                                                                               for p in m["parts"]))
    res = {"ok": False, "requests": len(calls), "ok_calls": len(ok),
           "non200": [[c["i"], c["status"]] for c in calls if c["msg"] is None],
           "compaction_calls": [c["i"] for c in ok if c["kind"] == "compaction"],
           "export_compactions": n_comp_export, "problems": problems, "windows": []}
    heads_all = [c for c in ok if c["kind"] == "call" and c["pred"] is None]
    start = [c["i"] for c in heads_all if c["n"] == 2 and c["m1"] == task]
    if len(start) != 1:
        problems.append(f"{len(start)} main-session start heads")
        return res
    chains, main_comps = [], []
    main_ahs = set()
    head = start[0]
    while head is not None:
        members, branch_points = walk(calls, head)
        if branch_points:
            problems.append(f"window {len(chains) + 1}: branch points {branch_points[:3]}")
        chains.append(members)
        main_ahs |= calls[members[-1]]["ahs"]
        last = members[-1]
        comp = next((c["i"] for c in ok if c["kind"] == "compaction" and c["i"] > last
                     and c["ahs"] and (c["ahs"] & main_ahs)), None)
        if comp is None:
            break
        main_comps.append(comp)
        summ = calls[comp]["msg"].get("content") or ""
        nxt = [c["i"] for c in heads_all if c["i"] > comp and c["m1"] == WDWDSF
               and (c["m2_content"] == summ or not summ)]
        head = nxt[0] if nxt else None
        if head is None:
            problems.append(f"no post-compaction head after compaction call {comp}")
    res["main_compaction_calls"] = main_comps
    in_windows = {i for ch in chains for i in ch}
    orphans = [c["i"] for c in ok if c["kind"] == "call" and c["i"] not in in_windows and c["m1"] in (task, WDWDSF)]
    if orphans:
        problems.append(f"main-session 200 calls outside every window: {orphans[:5]}")
    res["subagent_heads"] = sum(1 for c in heads_all if c["i"] not in {ch[0] for ch in chains}
                                and c["m1"] not in (task, WDWDSF))
    if len(main_comps) != n_comp_export:
        problems.append(f"{len(main_comps)} main compaction calls in http vs {n_comp_export} in export")
    if len(chains) != n_comp_export + 1:
        problems.append(f"{len(chains)} windows vs {n_comp_export} compaction parts + 1")
    try:
        ewins = export_windows(export)
    except (AssertionError, KeyError, StopIteration, TypeError) as error:
        problems.append("export reconstruction failed: " + repr(error)[:300])
        return res
    if len(ewins) != len(chains):
        problems.append(f"export windows {len(ewins)} != http windows {len(chains)}")
        return res
    payloads = fetch_requests(http_path, [calls[ch[0]]["line"] for ch in chains] + [calls[ch[-1]]["line"]
                                                                                       for ch in chains])
    harness_store = {"system": {}, "tools": {}, "params": {}}
    windows = []
    K = len(chains)
    for j, (members, ew) in enumerate(zip(chains, ewins)):
        k = j + 1
        head, last = calls[members[0]], calls[members[-1]]
        p_last, p_head = payloads[last["line"]], payloads[head["line"]]
        http_msgs = p_last["messages"] + [resp_as_message(last["msg"])]
        diff = compare([http_msgs[0]] + ew["messages"], http_msgs)
        if diff:
            problems.append(f"window {k}: export != http: {diff[:3]}")
        user_idx = [i for i, m in enumerate(http_msgs) if m.get("role") == "user"]
        n_own_http = sum(1 for m in http_msgs[user_idx[-1]:] if m.get("role") == "assistant") if user_idx else -1
        if n_own_http != len(members) or len(ew["sent"]) != len(members):
            problems.append(f"window {k}: {n_own_http} http / {len(ew['sent'])} export target assistants != "
                            f"{len(members)} calls")
        for c in members[1:]:
            if calls[c]["transition"]:
                problems.append(f"window {k}: call {c} transition: {calls[c]['transition'][:2]}")
        if any(calls[c]["tools_h"] != head["tools_h"] for c in members):
            problems.append(f"window {k}: the tool set changes inside the window")
        system_variants = len({calls[c]["sys"] for c in members})
        # targets: every call of the chain except a finish "length" reply (only allowed last)
        dropped, targets = [], []
        for q, c in enumerate(members):
            e = ew["sent"][q] if q < len(ew["sent"]) else None
            if calls[c]["finish"] == "length":
                if q != len(members) - 1:
                    problems.append(f"window {k}: finish length reply at call {c} is not the window's last call")
                dropped.append({"export_index": e, "call_index": c, "reason": "finish_length",
                                "completion_tokens": calls[c]["usage"].get("completion_tokens")})
            else:
                targets.append((c, e))
        for e in ew["skipped"]:
            info = em[e]["info"]
            dropped.append({"export_index": e, "call_index": None,
                            "reason": "errored_no_call" if info.get("error") else "empty_no_parts",
                            "error": (info.get("error") or {}).get("name")})
        dropped.sort(key=lambda d: (d["export_index"] is None, d["export_index"] or 0))
        if not targets:
            problems.append(f"window {k}: no target turn")
        for c, _ in targets[:-1]:
            if not (calls[c]["msg"].get("tool_calls")):
                problems.append(f"window {k}: call {c} is a reply without tool calls before the window's last target")
        # how the system prompt changes inside the window (the sample uses the window's first request's)
        change_kinds = set()
        for c in members:
            change_kinds |= system_change_kinds(head, calls[c])
        changed_targets = [c for c, _ in targets if calls[c]["sys"] != head["sys"]]
        h = window_harness(p_head)
        for name, field in (("system", "system_messages"), ("tools", "tools"), ("params", "params")):
            harness_store[name][h[f"{name}_sha256"]] = h[field]
        if k == 1 and first_request is not None:
            if (h["system_sha256"], h["tools_sha256"]) != (first_request.get("system_sha256"),
                                                           first_request.get("tools_sha256")):
                problems.append("window 1's first request differs from http.jsonl's first request (system / tools)")
        w = {"segment": k, "n_segments": K, "head_call": members[0], "last_chain_call": members[-1],
             "chain_calls": len(members), "target_calls": [c for c, _ in targets],
             "target_export_indices": [e for _, e in targets], "n_targets": len(targets),
             "first_call_index": targets[0][0] if targets else None,
             "last_call_index": targets[-1][0] if targets else None,
             "export_range": ew["export_range"], "dropped": dropped,
             "system_sha256": h["system_sha256"], "tools_sha256": h["tools_sha256"],
             "params_sha256": h["params_sha256"], "base_sha256": h["base_sha256"],
             "system_changed_in_window": system_variants > 1, "system_variants_in_window": system_variants,
             "system_change_kinds": sorted(change_kinds),
             "system_first_change_call": next((c for c in members if calls[c]["sys"] != head["sys"]), None),
             "system_changed_targets": len(changed_targets),
             "system_content_changed_targets": sum(1 for c in changed_targets
                                                   if system_change_kinds(head, calls[c]) - {"date"}),
             "ends_with": ("final_answer" if targets and not calls[targets[-1][0]]["msg"].get("tool_calls")
                           else "tool_calls"),
             "end_tool_results": None,
             "last_finish": last["finish"], "glm_prompt_tokens": last["usage"].get("prompt_tokens"),
             "glm_completion_tokens": last["usage"].get("completion_tokens"),
             "n_context_assistants": ew["n_context_assistants"], "post_final_continue": False,
             "compaction_after": None, "export_compaction": ew["compaction"]}
        if j < len(main_comps):
            cc = calls[main_comps[j]]
            w["compaction_after"] = {"call": cc["i"], "finish": cc["finish"],
                                     "prompt_tokens": cc["usage"].get("prompt_tokens"),
                                     "completion_tokens": cc["usage"].get("completion_tokens"),
                                     "summary_chars": len(cc["msg"].get("content") or ""),
                                     "summary_reasoning_chars": len(cc["msg"].get("reasoning")
                                                                    or cc["msg"].get("reasoning_content") or "")}
        if k >= 2:
            pm = p_head.get("messages") or []
            if len(pm) < 3 or pm[1].get("role") != "user" or text_of(pm[1].get("content")) != WDWDSF \
                    or pm[-1].get("role") != "user" or text_of(pm[-1].get("content")) != CONTINUE:
                problems.append(f"window {k}: first request is not [system, 'What did we do so far?', ..., 'Continue']")
            try:
                counts = Counter()
                w["context"] = redact_obj(context_messages(pm[1:]), counts)
                w["context_redactions"] = dict(sorted(counts.items()))
            except (ValueError, KeyError, TypeError) as error:
                problems.append(f"window {k}: context: {error!r}"[:300])
            w["post_final_continue"] = windows[-1]["ends_with"] == "final_answer"
        windows.append(w)
    for j, w in enumerate(windows):  # where the tool results of a window's last target are (if it made tool calls)
        if w["ends_with"] == "tool_calls" and w["target_export_indices"]:
            nxt = windows[j + 1] if j + 1 < len(windows) else {}
            tail = (nxt.get("export_compaction") or {}).get("tail") or []
            w["end_tool_results"] = ("next_window_context" if w["target_export_indices"][-1] in tail
                                     else "in_no_sample")
    no_drop = all(not w["dropped"] for w in windows)
    res.update(windows=windows, n_windows=K, window_harness=harness_store,
               route="v2_convert" if K == 1 and no_drop and windows and windows[0]["ends_with"] == "final_answer"
               and len(windows[0]["target_export_indices"]) == len(em) - 1 else "segment")
    res["ok"] = not problems
    return res


# ---------------------------------------------------------------- per record (worker)
def examine_glm(job):
    """v3's examine() (clean rule, compaction_of, flags_of + journal_of / finish_flags, first-request harness, patch
    facts), then the two-source gate and the windows of a clean pass."""
    fact = sv.examine(job)
    fact["key"] = job["key"]
    if fact["status"] != "pass" or not all((fact.get("files") or {}).values()):
        return fact
    attempt = Path(fact["attempt_dir"])
    gen = Path(fact["generation_dir"] or attempt)
    terminal = read_json(attempt / "terminal.json", {}) or {}
    fact["terminal_patch_chars"] = len(terminal.get("model_patch") or "")
    try:
        export = json.loads((gen / "export.json").read_bytes())
        fact["gate"] = gate(gen, export, fact.get("harness"))
    except Exception as error:  # noqa: BLE001 - any failure of the gate excludes the rollout (reported)
        fact["gate"] = {"ok": False, "problems": [f"gate error: {type(error).__name__}: {str(error)[:300]}"],
                        "windows": []}
    return fact


# ---------------------------------------------------------------- task facts, records, old-grader inputs
def key_of(row):
    return f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"


def tier_of(row):  # tier12_accept_compare.py
    if row["accepted_or_strong"]:
        return 1
    return {"partial": 2, "weak": 3}.get(row["audit"], 4)


def task_inputs():
    inventory = {key_of(r): r for r in map(json.loads, INVENTORY.read_text().splitlines()) if r}
    plan = {t["task_id"]: t["run"] for t in json.loads(SFTV3_TASKS.read_text())["tasks"]}
    v3map = {(r["run"], r["task_id"]): k for k, r in inventory.items() if r["sft_v3"] > 0}
    compare_tasks = {t["key"]: t for t in json.loads(ACCEPT_COMPARE.read_text())["tasks"]}
    return inventory, plan, v3map, compare_tasks


def task_fact(row, compare_row):
    tier = tier_of(row)
    because = ("tier1_accepted" if row["status"] == "accepted" else "tier1_strong") if tier == 1 else f"tier{tier}_" + \
        str(row["audit"])
    return {"run": row["run"], "deepswe_target": row.get("target"), "host_repo": row.get("repo"),
            "language": language_of(row), "included_because": because, "accepted": row["status"] == "accepted",
            "task_origin": row["source"], "tier": tier, "task_status": row["status"], "audit": row.get("audit"),
            "reject_reason": row.get("reject_reason"), "task_dir": row.get("task_dir"),
            "glm_verdict": (compare_row or {}).get("glm_verdict"),
            "glm_compare": {k: (compare_row or {}).get(k) for k in ("glm_source", "glm_pass", "glm_graded", "glm_rate",
                                                                    "sc14_pass", "sc14_graded", "kimi_pass",
                                                                    "kimi_graded", "similarity")}}


def load_records(inventory, plan, v3map, compare_tasks):
    """Every tier-1/2 final record -> job; tasks {task key: facts}; skipped sftv3 records outside tier 1/2."""
    jobs, tasks, skipped = [], {}, []
    for batch in BATCHES:
        for n, line in enumerate(FINAL_RECORDS[batch].read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            if batch == "tasksall":
                if r.get("tier") not in (1, 2):
                    continue
                task_key = r["key"]
                assert tier_of(inventory[task_key]) == r["tier"], (task_key, r["tier"])
            else:
                task_key = v3map[(plan[r["task_id"]], r["task_id"])]
                if tier_of(inventory[task_key]) not in (1, 2):
                    skipped.append({"trajectory_id": f"{PREFIX}-{batch}/{r['campaign']}/{r['task_id']}/{r['slot']}",
                                    "batch": batch, "line": n, "task_key": task_key,
                                    "tier": tier_of(inventory[task_key])})
                    continue
            tasks.setdefault(task_key, task_fact(inventory[task_key], compare_tasks.get(task_key)))
            rid = f"{PREFIX}-{batch}/{r['campaign']}/{r['task_id']}/{r['slot']}"
            jobs.append({"batch": batch, "campaign": r["campaign"], "task_id": r["task_id"], "slot": r["slot"],
                         "record": dict(r), "key": rid, "class_key": f"{batch}/{r['campaign']}/{r['task_id']}/{r['slot']}",
                         "task_key": task_key, "results_file": str(FINAL_RECORDS[batch]),
                         "record_source": "final_records", "record_line": n})
    assert len({j["key"] for j in jobs}) == len(jobs), "rollout id collision"
    return jobs, tasks, skipped


def rerun_mappings():
    """(entries, files, unreadable): every re-run mapping entry of the rollout campaigns' results/ dirs, normalized to
    {campaign, task_id, slot, original_attempt_dir, original_status, rerun_campaign, rerun_slot, rerun_attempt_dir,
    rerun_status, rerun_kind, rerun_reason, rerun_source}, and the files read ({path: sha256}). Kinds: "dip" (tasksall
    rerun_dip.py: glmlocal-dip*.json rounds and glmlocal-dip*-parts/*.json mappings, reason = the dip file's rule),
    "infra" (sftv3 rerun_infra.py: glmlocal-infra-rerun1.json, reason = its request) and "sandbox" (the drivers'
    recover(): every results file's sandbox_failure_reruns, reason = the sandbox-side error of the original attempt). A
    file that does not parse (a campaign writing it right now) is listed in unreadable: its re-runs stay unresolved
    (reported)."""
    entries, files, unreadable = [], {}, []

    def add(e, campaign, kind, reason, source, task_id=None):
        rerun = e.get("rerun") or {}
        entries.append({"campaign": e.get("campaign") or campaign, "task_id": e.get("task_id") or task_id,
                        "slot": e.get("slot"), "original_attempt_dir": norm(e.get("original_attempt")),
                        "original_status": e.get("original_status"),
                        "rerun_campaign": e.get("rerun_campaign") or e.get("recovery_campaign"),
                        "rerun_slot": e.get("rerun_slot") or e.get("recovery_slot") or rerun.get("slot"),
                        "rerun_attempt_dir": norm(e.get("rerun_attempt") or e.get("recovery_attempt")
                                                  or rerun.get("attempt_dir")),
                        "rerun_status": e.get("rerun_status") or e.get("recovery_status") or rerun.get("status"),
                        "rerun_kind": kind, "rerun_reason": reason, "rerun_source": source})

    def sandbox(d, campaign, source):
        for task_id, rows in (d.get("sandbox_failure_reruns") or {}).items():
            for e in rows:
                add(e, campaign, "sandbox", "sandbox-side failure of the original attempt "
                    f"({e.get('original_error')}): re-run once by the campaign driver's recover()", source, task_id)

    def name_of(d, path):  # a campaign results file's campaign name ("campaign" holds its probe dir path)
        name = d.get("name") or d.get("campaign")
        return Path(name).name if isinstance(name, str) and name else path.stem[len("glmlocal-"):]

    for batch in BATCHES:
        res = RERUN_RESULTS[batch]
        for path in sorted(res.glob("glmlocal-*.json")):
            data = path.read_bytes()
            files[str(path)] = sha256_bytes(data)
            try:
                d = json.loads(data)
            except ValueError:
                unreadable.append(str(path))
                continue
            source = f"{batch}:results/{path.name}"
            for round_ in d.get("rounds") or []:
                for name, camp in (round_.get("campaigns") or {}).items():
                    for e in camp.get("mapping") or []:
                        add(e, None, "dip", d.get("rule"), source)
                    sandbox(camp, name, source)
            if path.name.startswith("glmlocal-infra-rerun"):
                for name, camp in (d.get("campaigns") or {}).items():
                    for e in camp.get("mapping") or []:
                        add(e, None, "infra", d.get("request"), source)
                    sandbox(camp, name, source)
            elif not d.get("rounds"):
                sandbox(d, name_of(d, path), source)
        for path in sorted(res.glob("glmlocal-dip*-parts/*.json")):
            data = path.read_bytes()
            files[str(path)] = sha256_bytes(data)
            parent = res / (path.parent.name[:-len("-parts")] + ".json")
            try:
                d = json.loads(data)
                rule = json.loads(parent.read_text()).get("rule") if parent.exists() else None
            except ValueError:
                unreadable.append(str(path))
                continue
            source = f"{batch}:results/{path.parent.name}/{path.name}"
            for e in d.get("mapping") or []:
                add(e, None, "dip", rule, source)
            sandbox(d, name_of(d, path), source)
    return entries, files, unreadable


def rerun_provenance(job, fact, entries_by_attempt):
    """The v3 meta re-run fields of a final record: a re-run = its attempt dir lies in another probe campaign; the
    provenance is the one mapping entry with this (campaign, task_id, slot) and this re-run attempt (else unresolved)."""
    rerun = fact.get("probe_campaign") is not None and fact["probe_campaign"] != job["campaign"]
    out = {"rerun": rerun, "rerun_reason": None, "rerun_campaign": None, "rerun_slot": None,
           "original_attempt_dir": None, "original_status": None, "original_last_reply_kind": None,
           "rerun_kind": None, "rerun_source": None, "rerun_provenance": None}
    if not rerun:
        return out
    hits = [e for e in entries_by_attempt.get(fact["attempt_dir"], [])
            if (e["campaign"], e["task_id"], e["slot"]) == (job["campaign"], job["task_id"], job["slot"])]
    if len(hits) != 1:
        out.update(rerun_campaign=fact["probe_campaign"], rerun_provenance=f"unresolved ({len(hits)} mapping entries)")
        return out
    e = hits[0]
    out.update(rerun_reason=e["rerun_reason"], rerun_campaign=e["rerun_campaign"], rerun_slot=e["rerun_slot"],
               original_attempt_dir=e["original_attempt_dir"], original_status=e["original_status"],
               rerun_kind=e["rerun_kind"], rerun_source=e["rerun_source"], rerun_provenance="mapping")
    if e["rerun_campaign"] != fact["probe_campaign"]:
        out["rerun_provenance"] = "mapping (campaign differs from the attempt path)"
    return out


def old_grader_inputs(results_path=REGRADE_RESULTS):
    """(classes bytes, classes passes, results bytes or None, results map)."""
    cbytes = CLASSES.read_bytes()
    classes = json.loads(cbytes)
    rbytes = results_path.read_bytes() if results_path.exists() else None
    results = (json.loads(rbytes).get("results") or {}) if rbytes is not None else {}
    return cbytes, classes, rbytes, results


def old_grader_of(job, fact, classes, results):
    """{grader_kind, class, test_touch_class, ...} of a pass, and the exclusion reason (or None)."""
    c = (classes.get("passes") or {}).get(job["class_key"])
    r = results.get(job["class_key"])
    patch_mismatch = r is not None and r.get("model_patch_sha256") not in (None, fact.get("patch_sha256"))
    if patch_mismatch:  # a re-grade of another model.patch (another attempt): not a re-grade of this pass
        r = None
    regrade = None if r is None else {k: r.get(k) for k in ("reward", "status")}
    if c is None or norm(c.get("attempt_dir")) != fact["attempt_dir"]:
        return {"grader_kind": None, "class": None, "test_touch_class": None, "regrade": regrade,
                "regrade_patch_mismatch": patch_mismatch, "classes_entry": c is not None}, "old_grader_unclassified"
    cls = c["class"]
    og = {"grader_kind": c.get("grader_kind"), "class": cls,
          "test_touch_class": {"n/a": "none", "A": "A", "B": "B"}.get(cls, cls),
          "language": c.get("language"), "why": c.get("why"), "hits": len(c.get("hits") or []), "regrade": regrade,
          "regrade_patch_mismatch": patch_mismatch}
    if cls == "B":
        if r is None:
            return og, "old_grader_not_regraded"
        if r.get("reward") != 1:
            return og, "old_grader_regrade_fail"
    elif r is not None and r.get("reward") is not None and r.get("reward") != 1:
        return og, "old_grader_regrade_fail"
    return og, None


def decide(job, fact, og_reason):
    """The first selection rule a record fails (module docstring), or None when kept."""
    if fact["status"] != "pass":
        return f"final_status_{fact['status']}"
    if not all(fact["files"].values()):
        return "not_clean"
    if fact["evaluation_completed"] is not True or fact["reward"] != 1 or fact["validity_failure"] is not None:
        return "not_clean"
    if og_reason:
        return og_reason
    if fact["journal"]["degenerate_n"]:
        return "degenerate_reply"
    if harness_problem(fact):
        return "harness_unexpected"
    if not (fact.get("gate") or {}).get("ok"):
        return "two_source_mismatch"
    return None


def harness_problem(fact):
    if not fact.get("harness") or fact["harness"]["base_sha256"] not in V1_BASES \
            or fact["harness"]["tools_sha256"] not in V1_TOOLS:
        return "first request"
    for w in (fact.get("gate") or {}).get("windows") or []:
        if w["base_sha256"] not in V1_BASES or w["tools_sha256"] not in V1_TOOLS:
            return f"window {w['segment']}"
    return None


# ---------------------------------------------------------------- the meta written for each kept rollout
_V3_META_OF = sv.meta_of
_TASKS = {}


def glm_meta_of(job, fact, _task_by_id):
    """v3's meta_of() with GLM's task facts (looked up by the unique inventory key, not by task_id; language from
    top100.jsonl for every task) and GLM fields (the re-run provenance replaces v3's record-based re-run fields)."""
    task = _TASKS[job["task_key"]]
    meta = _V3_META_OF(job, fact, task)
    g = fact["gate"]
    meta.update(job["rerun_meta"])
    meta.update({
        "source": SOURCE, "rollout_batch": job["batch"], "task_key": job["task_key"],
        "results_file": job["results_file"], "record_source": "final_records", "record_line": job["record_line"],
        "status_source": f"{job['batch']}_final_records",
        "grader_detail": "graded at rollout time by ATP with the task's installed tests/ (grader.py sha256 in "
                         "grader_sha256); old-grader tasks (DeepSWE's grader.py without the test-path sanitizer) "
                         "carry the static class of model.patch and the current-verifier re-grade (old_grader); no "
                         "LLM teacher audit (flags only)",
        "tier": task["tier"], "task_status": task["task_status"], "audit": task["audit"],
        "reject_reason": task["reject_reason"], "glm_verdict": task["glm_verdict"], "glm_compare": task["glm_compare"],
        "language": task["language"],
        "grader_kind": fact["old_grader"]["grader_kind"], "test_touch_class": fact["old_grader"]["test_touch_class"],
        "old_grader": fact["old_grader"], "regrade": fact["old_grader"]["regrade"],
        "glm": {"route": g["route"], "n_windows": g["n_windows"], "requests": g["requests"], "ok_calls": g["ok_calls"],
                "non200": g["non200"], "compaction_calls": g["compaction_calls"],
                "main_compaction_calls": g["main_compaction_calls"], "subagent_heads": g["subagent_heads"],
                "windows": g["windows"]},
    })
    return meta


def extend_harness(path: Path, kept, facts):
    """Add every window's first-request system variant / tool set / params to write_batch's harness.json."""
    h = json.loads(path.read_text())
    for job in kept:
        g = facts[job["key"]]["gate"]
        run = _TASKS[job["task_key"]]["run"]
        for w in g["windows"]:
            sm = g["window_harness"]["system"][w["system_sha256"]]
            v = h["system_prompt_variants"].get(w["system_sha256"])
            if v is None:
                text = "\n".join(BB["_text"](m) for m in sm)
                parts = BB["split_system"](text)
                v = h["system_prompt_variants"][w["system_sha256"]] = {
                    "system_messages": sm, "chars": len(text),
                    "base_sha256": sha256_bytes(parts["base"].encode("utf-8", "surrogatepass")), "env": parts["env"],
                    "rest_sha256": sha256_bytes(parts["rest"].encode("utf-8", "surrogatepass")),
                    "rest_chars": len(parts["rest"]), "runs": [], "trajectories": 0,
                    "in_v1": w["system_sha256"] in V1H["system_prompt_variants"],
                    "in_v2": w["system_sha256"] in V1H["system_prompt_variants"]
                    or w["system_sha256"] in V2H["system_prompt_variants"]}
            assert v["system_messages"] == sm
            v["window_heads"] = v.get("window_heads", 0) + 1
            if run not in v["runs"]:
                v["runs"].append(run)
            te = h["tool_sets"].setdefault(w["tools_sha256"], {"tools": g["window_harness"]["tools"][w["tools_sha256"]],
                                                                "runs": [], "trajectories": 0,
                                                                "in_v1": w["tools_sha256"] in V1H["tool_sets"]})
            assert te["tools"] == g["window_harness"]["tools"][w["tools_sha256"]]
            te["window_heads"] = te.get("window_heads", 0) + 1
            pe = h["request_params"].setdefault(w["params_sha256"], {
                "params": g["window_harness"]["params"][w["params_sha256"]], "runs": [], "trajectories": 0,
                "in_v1": w["params_sha256"] in V1H["request_params"]})
            pe["window_heads"] = pe.get("window_heads", 0) + 1
    base_v1 = V1_BASES
    variants = h["system_prompt_variants"].values()
    h["comparison_with_v1"] = {  # write_batch's comparison, recomputed over every window head
        "distinct_system_prompts": len(h["system_prompt_variants"]),
        "system_prompts_also_in_v1": sum(v["in_v1"] for v in variants),
        "system_prompts_also_in_v2": sum(v["in_v2"] for v in variants),
        "distinct_system_prompt_bases": len({v["base_sha256"] for v in variants}),
        "bases_not_in_v1": sorted({v["base_sha256"] for v in variants} - base_v1),
        "distinct_tool_sets": len(h["tool_sets"]),
        "tool_sets_not_in_v1": [k for k, v in h["tool_sets"].items() if not v["in_v1"]],
        "distinct_request_params": len(h["request_params"]),
        "request_params_not_in_v1": [k for k, v in h["request_params"].items() if not v["in_v1"]],
    }
    h["window_heads_note"] = ("system_prompt_variants / tool_sets / request_params also hold the first request of every "
                              "window (window_heads = how many windows use it; trajectories = rollouts whose FIRST "
                              "request uses it, as v3)")
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(h, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return h


# ---------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS, help="writes <out>/sources/")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="+", default=[], help="dry run: only these records (KEY or rollout id)")
    ap.add_argument("--only-file", type=Path, help="dry run: --only keys, one per line")
    ap.add_argument("--limit", type=int, default=0, help="dry run: the first N tier-1/2 records of each batch")
    ap.add_argument("--allow-missing-regrade", action="store_true",
                    help="real corpus dir while some class B pass has no re-grade result (excluded as not re-graded)")
    ap.add_argument("--regrade-results", type=Path, default=REGRADE_RESULTS,
                    help="dry run only: another re-grade results file (e.g. a test fixture); the corpus dir refuses it")
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.out.resolve()
    only = list(args.only) + ([x.strip() for x in args.only_file.read_text().splitlines() if x.strip()
                               and not x.startswith("#")] if args.only_file else [])
    only = [x[len(PREFIX) + 1:] if x.startswith(PREFIX + "-") else x for x in only]
    results_path = args.regrade_results.resolve()
    partial = bool(only or args.limit or results_path != REGRADE_RESULTS.resolve())
    if out == CORPUS.resolve():
        if partial:
            raise SystemExit("refusing a partial (--only / --limit / --regrade-results) run into the real corpus dir")
        if (out / "raw_sft").exists():
            raise SystemExit(f"{out / 'raw_sft'} exists: refusing to rewrite the sources of a built corpus")
    os.umask(0o027)
    started = time.time()
    inventory, plan, v3map, compare_tasks = task_inputs()
    jobs, tasks, skipped_not_tier12 = load_records(inventory, plan, v3map, compare_tasks)
    considered_all = len(jobs)
    if only:
        want = set(only)
        unknown = sorted(want - {j["class_key"] for j in jobs})
        if unknown:
            raise SystemExit(f"--only keys not among the tier-1/2 final records: {unknown[:5]}")
        jobs = [j for j in jobs if j["class_key"] in want]
    if args.limit:
        jobs = [j for b in BATCHES for j in [x for x in jobs if x["batch"] == b][:args.limit]]
    _TASKS.update(tasks)
    cbytes, classes, rbytes, results = old_grader_inputs(results_path)
    missing_b = sorted(k for k, v in (classes.get("passes") or {}).items() if v.get("class") == "B" and k not in results)
    if out == CORPUS.resolve() and missing_b and not args.allow_missing_regrade:
        raise SystemExit(f"{results_path}: {'missing' if rbytes is None else 'incomplete'}: {len(missing_b)} of "
                         f"{sum(1 for v in classes['passes'].values() if v.get('class') == 'B')} class B passes have no "
                         "re-grade result yet (they would be excluded as old_grader_not_regraded); wait for the re-grade, "
                         "or pass --allow-missing-regrade")
    print(f"{len(jobs)} records considered (of {considered_all} tier-1/2 final records; partial {partial}); "
          f"classes {classes.get('counts')}; re-grade results: "
          f"{'none' if rbytes is None else len(results)}", flush=True)
    rerun_entries, rerun_files, rerun_unreadable = rerun_mappings()
    entries_by_attempt = defaultdict(list)
    for e in rerun_entries:
        entries_by_attempt[e["rerun_attempt_dir"]].append(e)

    facts = {}
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for n, fact in enumerate(pool.map(examine_glm, jobs, chunksize=1), 1):
            facts[fact["key"]] = fact
            if n % 25 == 0 or n == len(jobs):
                print(f"  examined {n}/{len(jobs)} ({time.time() - started:.0f} s)", flush=True)

    kept, exclusions = [], []
    independent = Counter()
    og_summary = Counter()
    language_mismatch = []
    for job in jobs:
        fact = facts[job["key"]]
        og_reason = None
        if fact["status"] == "pass" and fact.get("attempt_dir"):
            fact["old_grader"], og_reason = old_grader_of(job, fact, classes, results)
            og_language = fact["old_grader"].get("language")
            if og_language is not None and og_language != tasks[job["task_key"]]["language"]:
                language_mismatch.append({"trajectory_id": job["key"], "classes": og_language,
                                          "top100": tasks[job["task_key"]]["language"]})
        job["rerun_meta"] = rerun_provenance(job, fact, entries_by_attempt)
        reason = decide(job, fact, og_reason)
        if fact["status"] == "pass":
            independent["final_pass"] += 1
            clean = all(fact["files"].values()) and fact["evaluation_completed"] is True and fact["reward"] == 1 \
                and fact["validity_failure"] is None
            independent["not_clean"] += not clean
            og = fact.get("old_grader") or {}
            og_summary[f"class_{og.get('class')}"] += 1
            og_summary[f"class_{og.get('class')}_regrade_{(og.get('regrade') or {}).get('reward', 'none')}"] += 1
            independent[og_reason or "old_grader_ok"] += 1
            compacted = bool((fact.get("compaction") or {}).get("compacted"))
            degenerate = bool((fact.get("journal") or {}).get("degenerate_n"))
            independent["compacted"] += compacted
            independent["degenerate_reply"] += degenerate
            independent["compacted_and_degenerate"] += compacted and degenerate
            independent["harness_unexpected"] += bool(harness_problem(fact))
            gate_ok = bool((fact.get("gate") or {}).get("ok"))
            independent["two_source_mismatch"] += clean and not gate_ok
            independent["two_source_ok"] += gate_ok
            if gate_ok:
                g = fact["gate"]
                independent["gate_ok_windows"] += g["n_windows"]
                independent[f"gate_ok_rollouts_with_{g['n_windows']}_windows"] += 1
                independent["gate_ok_route_" + g["route"]] += 1
                for w in g["windows"]:
                    independent["gate_ok_post_final_continue_windows"] += w["post_final_continue"]
                    independent["gate_ok_system_changed_windows"] += w["system_changed_in_window"]
                    independent["gate_ok_system_content_changed_windows"] += w["system_content_changed_targets"] > 0
                    for kind in w["system_change_kinds"]:
                        independent["gate_ok_system_change_kind_" + kind] += 1
                    if w["end_tool_results"]:
                        independent["gate_ok_window_end_tool_results_" + w["end_tool_results"]] += 1
                    for d in w["dropped"]:
                        independent["gate_ok_dropped_" + d["reason"]] += 1
        if reason is None:
            kept.append(job)
            continue
        detail = None
        if reason == "degenerate_reply":
            detail = {"kinds": fact["journal"]["kinds"], "first": fact["journal"]["degenerate"][:3],
                      "ended": fact["journal"]["ended"]}
        elif reason == "two_source_mismatch":
            detail = {"problems": (fact.get("gate") or {}).get("problems", [])[:6]}
        elif reason == "harness_unexpected":
            detail = {"where": harness_problem(fact), "first_request": fact.get("harness")}
        elif reason.startswith("old_grader"):
            detail = fact.get("old_grader")
        elif reason.startswith("final_status"):
            detail = {k: job["record"].get(k) for k in ("f2p", "p2p", "calls", "compacted")}
        elif reason == "not_clean":
            detail = {k: fact.get(k) for k in ("files", "evaluation_completed", "reward", "validity_failure")}
        task = tasks[job["task_key"]]
        exclusions.append({"trajectory_id": job["key"], "rollout_batch": job["batch"], "campaign": job["campaign"],
                           "task_id": job["task_id"], "slot": job["slot"], "task_key": job["task_key"],
                           "tier": task["tier"], "status": fact["status"], "reason": reason,
                           "attempt_dir": norm(job["record"].get("attempt_dir")),
                           "compacted": job["record"].get("compacted"),
                           "grader_kind": (fact.get("old_grader") or {}).get("grader_kind"),
                           "test_touch_class": (fact.get("old_grader") or {}).get("test_touch_class"),
                           "regrade": (fact.get("old_grader") or {}).get("regrade"), "detail": detail})
    kept.sort(key=lambda j: j["key"])

    sources = out / "sources"
    gdir = sources / "glm"
    gdir.mkdir(parents=True, exist_ok=True)
    rdir = sources / "regrade"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "classes.json").write_bytes(cbytes)
    if rbytes is not None:
        (rdir / "results.json").write_bytes(rbytes)
    elif (rdir / "results.json").exists():
        (rdir / "results.json").unlink()
    report_metas, built = [], None
    if kept:
        for job in kept:  # write_batch compares the patch with record["patch_chars"]: the graded patch (terminal.json)
            job["record"]["patch_chars"] = facts[job["key"]]["terminal_patch_chars"]
        sv.meta_of = glm_meta_of
        try:
            built = sv.write_batch(SOURCE, kept, facts, {j["task_id"]: None for j in kept}, gdir, V1H, V2H)
        finally:
            sv.meta_of = _V3_META_OF
        harness = extend_harness(gdir / "harness.json", kept, facts)
        for meta in built["metas"]:  # the trajectories file holds the contexts; the report does not
            for w in meta["glm"]["windows"]:
                w.pop("context", None)
            report_metas.append(meta)
    else:
        for name in ("trajectories.jsonl.gz", "harness.json"):
            if (gdir / name).exists():
                (gdir / name).unlink()
        harness = None

    # task tests of every considered task (inventory key; task_id alone is not unique across runs)
    task_tests = {}
    for job in jobs:
        if job["task_key"] in task_tests:
            continue
        camp = (TASKSALL if job["batch"] == "tasksall" else SFTV3) / "probes/glmlocal" / job["campaign"]
        tt = sv.task_tests(camp, job["task_id"])
        task_tests[job["task_key"]] = {"batch": job["batch"], "campaign": job["campaign"], "task_id": job["task_id"],
                                       **tt}
    (sources / "task_tests.json").write_text(json.dumps(dict(sorted(task_tests.items())), indent=1) + "\n")

    kept_facts = [facts[j["key"]] for j in kept]
    windows = [w for f in kept_facts for w in f["gate"]["windows"]]
    inputs = {norm(str(p)): sha256_file(p) for p in (list(FINAL_RECORDS.values()) + [
        INVENTORY, SFTV3_TASKS, ACCEPT_COMPARE, CLASSES, DESIGN, V3_SOURCES, Path(__file__).resolve(),
        sv.V1_HARNESS, sv.V2_BAIHE_HARNESS,  # rule 5's reference (v1 bases / tool sets) and the in_v2 flags
        TOP100, TEST_PATHS])}  # task language
    inputs.update({norm(p): s for p, s in rerun_files.items()})  # re-run provenance
    inputs[norm(str(results_path))] = sha256_bytes(rbytes) if rbytes is not None else None
    kept_reruns = [j for j in kept if j["rerun_meta"]["rerun"]]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request": REQUEST, "rules": RULES,
        "design": str(DESIGN), "partial": partial, "only": only or None, "limit_per_batch": args.limit or None,
        "inputs_sha256": inputs,
        "v3_sources": {"path": str(V3_SOURCES), "sha256": V3_SOURCES_SHA256,
                       "build_batch_helpers_sha256": sv.BUILD_BATCH_HELPERS_SHA256,
                       "degenerate_block_sha256": sv.DEGENERATE_BLOCK_SHA256,
                       "degenerate_file": str(sv.BATCHES["v4"]["dir"] / "degenerate.py"),
                       "classify_reads_reasoning": CLASSIFY_READS_REASONING,
                       "journal_length_cut": "finish_reason == 'length' (OUTPUT_CAP disabled)"},
        "old_grader_inputs": {"classes": {"path": str(CLASSES), "sha256": sha256_bytes(cbytes),
                                          "copy": "sources/regrade/classes.json", "counts": classes.get("counts"),
                                          "rule": classes.get("rule"), "tools_sha256": classes.get("tools_sha256")},
                              "results": {"path": norm(str(results_path)), "exists": rbytes is not None,
                                          "default_path": results_path == REGRADE_RESULTS.resolve(),
                                          "sha256": sha256_bytes(rbytes) if rbytes is not None else None,
                                          "copy": "sources/regrade/results.json" if rbytes is not None else None,
                                          "n_results": len(results),
                                          "class_b_without_result": len(missing_b),
                                          "updated_at": (json.loads(rbytes).get("updated_at")
                                                         if rbytes is not None else None)}},
        "considered": len(jobs), "considered_all_tier12_records": considered_all,
        "skipped_sftv3_not_tier12": skipped_not_tier12,
        "considered_by_batch": dict(Counter(j["batch"] for j in jobs)),
        "considered_by_tier": dict(Counter(str(tasks[j["task_key"]]["tier"]) for j in jobs)),
        "final_status": dict(Counter(f"{j['batch']}:{facts[j['key']]['status']}" for j in jobs)),
        "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
        "exclusion_counts_by_batch": {b: dict(Counter(e["reason"] for e in exclusions if e["rollout_batch"] == b))
                                      for b in BATCHES},
        "independent_counts_among_final_passes": dict(independent),
        "old_grader_among_final_passes": dict(og_summary),
        "kept": len(kept), "kept_by_batch": dict(Counter(j["batch"] for j in kept)),
        "kept_by_tier": dict(Counter(str(tasks[j["task_key"]]["tier"]) for j in kept)),
        "kept_by_class": dict(Counter(str(facts[j["key"]]["old_grader"]["class"]) for j in kept)),
        "kept_by_campaign": dict(Counter(j["campaign"] for j in kept)),
        "kept_reruns": len(kept_reruns),
        "kept_reruns_by_kind": dict(Counter(str(j["rerun_meta"]["rerun_kind"]) for j in kept_reruns)),
        "kept_reruns_by_campaign": dict(Counter(str(j["rerun_meta"]["rerun_campaign"]) for j in kept_reruns)),
        "reruns_unresolved": [{"trajectory_id": j["key"], "probe_campaign": facts[j["key"]].get("probe_campaign"),
                               "provenance": j["rerun_meta"]["rerun_provenance"]}
                              for j in jobs if j["rerun_meta"]["rerun"]
                              and j["rerun_meta"]["rerun_provenance"] != "mapping"],
        "rerun_mapping_entries": dict(Counter(e["rerun_kind"] for e in rerun_entries)),
        "rerun_mapping_files_unreadable": rerun_unreadable,
        "language_rule": "synth/verifier/test_paths.py repo_language(top100.jsonl, inventory target, inventory repo) "
                         "for every task (regrade/touch_scan.py's rule, the language of classes.json)",
        "languages_considered_tasks": dict(sorted(Counter(str(tasks[k]["language"])
                                                          for k in {j["task_key"] for j in jobs}).items())),
        "language_mismatch_with_classes": language_mismatch,
        "kept_tasks": len({j["task_key"] for j in kept}), "tasks_considered": len({j["task_key"] for j in jobs}),
        "tasks_without_kept": sorted({j["task_key"] for j in jobs} - {j["task_key"] for j in kept}),
        "kept_windows": len(windows),
        "kept_rollouts_by_windows": dict(Counter(str(f["gate"]["n_windows"]) for f in kept_facts)),
        "kept_routes": dict(Counter(f["gate"]["route"] for f in kept_facts)),
        "kept_post_final_continue_windows": sum(w["post_final_continue"] for w in windows),
        "kept_system_changed_windows": sum(w["system_changed_in_window"] for w in windows),
        "kept_system_change_kinds": dict(Counter("+".join(w["system_change_kinds"]) for w in windows
                                                 if w["system_change_kinds"])),
        "kept_system_content_changed_windows": sum(1 for w in windows if w["system_content_changed_targets"]),
        "kept_system_changed_targets": sum(w["system_changed_targets"] for w in windows),
        "kept_system_content_changed_targets": sum(w["system_content_changed_targets"] for w in windows),
        "kept_window_end_tool_results": dict(Counter(str(w["end_tool_results"]) for w in windows)),
        "kept_window_end_results_in_no_sample": [{"trajectory_id": j["key"], "segment": w["segment"],
                                                  "n_segments": w["n_segments"], "last_call_index": w["last_call_index"]}
                                                 for j in kept for w in facts[j["key"]]["gate"]["windows"]
                                                 if w["end_tool_results"] == "in_no_sample"],
        "kept_dropped_turns": dict(Counter(d["reason"] for w in windows for d in w["dropped"])),
        "kept_dropped_turn_list": [{"trajectory_id": j["key"], "segment": w["segment"], **d}
                                   for j in kept for w in facts[j["key"]]["gate"]["windows"] for d in w["dropped"]],
        "flag_counts_kept": sv.flag_counts([facts[j["key"]]["flags"] for j in kept]),
        "exclusions": exclusions,
    }
    if built is not None:
        report.update({
            "redactions_trajectories": {n: built["redactions"].get(n, 0) for n in sv.PATTERN_NAMES},
            "redactions_harness": {n: built["harness_redactions"].get(n, 0) for n in sv.PATTERN_NAMES},
            "harness_comparison_with_v1": harness["comparison_with_v1"],
            "harness_missing": built["harness"]["missing"],
            "first_user_mismatch": built["harness"]["first_user_mismatch"],
            "patch_mismatch_record": [m["trajectory_id"] for m in built["metas"] if not m["patch_matches_record"]],
            "metas": report_metas})
    tmp = gdir / ".selection_report.json.tmp"
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, gdir / "selection_report.json")
    summary = {k: report[k] for k in ("partial", "considered", "final_status", "exclusion_counts",
                                      "independent_counts_among_final_passes", "kept", "kept_by_batch", "kept_by_tier",
                                      "kept_by_class", "kept_windows", "kept_rollouts_by_windows", "kept_routes",
                                      "kept_post_final_continue_windows", "kept_dropped_turns",
                                      "kept_system_change_kinds", "kept_system_content_changed_windows",
                                      "kept_window_end_tool_results", "kept_reruns_by_kind", "reruns_unresolved",
                                      "language_mismatch_with_classes")}
    summary["seconds"] = round(time.time() - started, 1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
