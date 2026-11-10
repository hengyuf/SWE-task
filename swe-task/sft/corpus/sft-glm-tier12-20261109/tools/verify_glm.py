#!/usr/bin/env python3
"""Independent re-check of a GLM-5.3 tier-1/2 SFT corpus build (DESIGN.md "Verification"); shares no code with
sources_glm.py / build_glm.py.

  - rows (v3's verify, imported from sft-v3-20261029/tools/verify_sft_v3.py, sha256 asserted): its re-implementation of
    SFTTokenizer.tokenize_conversation (identity) and SFTDataset.__getitem__ (tc / row: 32-token padding per sample,
    524,289 truncation guard, step-weight spans, loss_mask = weights[1:] with pad / IGNORE masking) on every row of
    every shard: loss-mask sum == sum(counts x weights) per row, unique ids, no generation-stop id among the targets,
    literal BOS targets counted;
  - loss weights (fix 2026-11-10; row() takes its expected mass from the same fields): every sample's system message has
    weight == 1.0 and assistant_weights == the receipt's weight policy (--invalid-policy keep: all 1.0; mask: 0.0 or
    1.0, 0.0 exactly on the targets that call the "invalid" tool), one per assistant_token_counts entry;
  - selection == shards (ids, token_length, training_row_sha256; assistant_tokens == sum(assistant_token_counts),
    n_assistant == their number, masked_assistant_turns == the 0.0 weights), receipt fields (selection_sha256,
    packed_rows, selected_trajectories, shard bytes / rows, shard paths and blend under --data-root, state); a real
    (non dry-run) build must use DESIGN's AGA DATA_ROOT (DATA_ROOT_DEFAULT) unless --allow-other-data-root;
  - per kept rollout, re-derived from the raw attempt files with verify's OWN chain and export code: the final record
    is a pass for that attempt, terminal.json a completed full-reward grade; http.jsonl requests linked into prefix
    chains (a request extends another's messages[1:] by exactly [its response, its tool results]); main windows = the
    chains from the task head and from every "What did we do so far?" ... "Continue" head, one summary-writing call
    between consecutive windows; #windows == #export compaction parts + 1; no branch, no main call outside the windows;
    window 1's head has the system message and tool set of http.jsonl's first request; every target but a window's
    last makes tool calls; the two-source equality (export.json rebuilt per window == the window's last request +
    response); every sample's pieces re-rendered from http.jsonl with the sc14 template (system piece and prompt /
    context piece from the window's first request, located independently; every target turn from its call's response;
    every tool piece from the next request's tool results; build_batch.py's redaction, truncate_history_thinking=False)
    and compared with the shard, its assistant_token_counts == the token counts of the re-rendered assistant pieces and
    its assistant_weights == the policy's; coverage: every main-chain call except summary-writing calls and dropped
    finish "length" replies is a target exactly once across the rollout's samples (an errored export message has no
    call); first / last call index of every selection row; the system-change fields (system_changed_in_window,
    system_change_kinds, system_changed_targets, system_content_changed_targets: own split of the system text into base
    / env / date line / rest) and end_tool_results (the last target's tool call ids among the next window's context
    tool messages); degenerate.py's classify() (exec'd from its source text by verify_sft_v3.py) on every status-200
    reply; the old-grader class and re-grade result (sources/regrade copies, sha256 == the selection report's; a
    re-grade counts only for the model.patch it graded: model_patch_sha256 == sha256(model.patch));
  - completeness: every considered final record is kept or excluded exactly once (a full build: every tier-1/2 final
    record of both batches); every exclusion reason re-derived (fix 2026-11-10): final status, old grader (class,
    re-grade, unclassified), not_clean (terminal.json, files, generation-validity cache), degenerate_reply (classify()
    re-scan), harness_unexpected (v1 base / tool set of the first request and every window head, own code over v1's
    harness.json), two_source_mismatch (the gate's detail present and verify's own chain / export code finds a
    problem); every build-stage drop confirmed (literal '</s>' in the expected targets, or > 524,288 tokens);
  - tools/sim_sft_getitem.py (byte copy of scratch/packing-semantics/tools/sim_sft_getitem.py: the real pinned
    SFTDataset.__getitem__, sft_dataset.py cadbab60) via subprocess on >= --sim-rows rows, preferring rows that hold two
    windows of one rollout: cu_seqlens == cumulative ceil32 lengths, positions reset, no loss position labelled from
    another sample (main and MTP-1), per-sample loss mass == counts x weights, Mamba seq_idx boundaries.
--readme writes the result line into README.md: it replaces VERIFY_RESULT_PLACEHOLDER or the result line of an earlier
verify run ("Result (...): OK." / "Result (...): FAILED. ..."), so a re-run after a failure updates it. Exit status 0
only if all checks hold.

  <vllm python> -I -B tools/verify_glm.py [--out DIR] [--data-root PATH] [--workers 12] [--sim-rows 6]
      [--sim-workers 4] [--readme] [--allow-other-data-root]
"""
import os
import sys

sys.dont_write_bytecode = True
_TMP = os.environ.get("TMPDIR") or "/tmp"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = _TMP + "/hf"
os.environ["HF_DATASETS_CACHE"] = _TMP + "/hf/ds"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
WORK = S / "sft/corpus/glm-tier12-20261109"
CORPUS = S / "sft/corpus/sft-glm-tier12-20261109"
TOOLS = Path(__file__).resolve().parent
DATA_ROOT_DEFAULT = "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/sft-glm-tier12-20261109"
V3_VERIFY = S / "sft/corpus/sft-v3-20261029/tools/verify_sft_v3.py"
V3_VERIFY_SHA256 = "02834abac569a2534be4bcc0f2233d2faff9a6b74aa802c0f14409267b32aa23"  # sft-v3 SHA256SUMS
BUILD_BATCH = S / "export/kimi_pass_batch_20261021/build_batch.py"
BUILD_BATCH_HELPERS_SHA256 = "9b436db30cddf0018557f4326ffa2f7b8b0fcd9be6802a4ba85272838269dd53"
SIM = TOOLS / "sim_sft_getitem.py"
SIM_ORIGINAL = WORK / "scratch/packing-semantics/tools/sim_sft_getitem.py"
SIM_SHA256 = "b1e5257b2fa0057830c12b8ba2e1d7f0395b00caf708868632dda73cb8e09869"
SFT_DATASET_PIN, SFT_TOKENIZER_PIN = "cadbab60", "796e03ac"
PY = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python"
FINAL_RECORDS = {"tasksall": S / "sft/glm-rollouts-tasksall-20261108/analysis/rollouts.jsonl",
                 "sftv3": S / "sft/glm-rollouts-sftv3-20261107/analysis/final-rollouts.jsonl"}
SELECTION = "selection/sft_glm_tier12.jsonl"
SHARDS = [f"raw_sft/part_{k:03d}.mat.jsonl.packed.512k" for k in range(6)]
SEQ = 524288
ALIAS = "/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/"
CANON = "/lustre/fsw/portfolios/nemotron/users/hengyuf/"
GEN = "<|im_start|>assistant\n<think>\n"
TASK_MARK = "What did we do so far?"
CONT = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
SUMMARY_PROMPTS = ("Create a new anchored summary from the conversation history.",
                   "Update the anchored summary below using the conversation history above.")
FIRST_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _load_v3_verify():
    """v3's verify module: tc / row / stop_ids / init and degenerate.py's classify (its DG), from its path."""
    data = V3_VERIFY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == V3_VERIFY_SHA256, "sft-v3 verify_sft_v3.py changed"
    spec = importlib.util.spec_from_file_location("sft_v3_verify", V3_VERIFY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sft_v3_verify"] = module  # workers (fork) pickle its functions by reference
    spec.loader.exec_module(module)
    return module


V3 = _load_v3_verify()
DEGENERATE_BLOCK_SHA256 = "1114c513f188a8b5a4502f103ba5bacc20de8f73de34bf112ea8bd60a8b1c1ad"  # classify() + scan()
assert hashlib.sha256(V3._DEG[V3._DEG.index("def classify(body):"):V3._DEG.index("def fisher(")].encode()).hexdigest() \
    == DEGENERATE_BLOCK_SHA256, "degenerate.py classify / scan changed"
_bb = BUILD_BATCH.read_text()
_bb_block = _bb[_bb.index("def _entropy(text: str)"):
                _bb.index("# ============================================================================================ selection")]
assert hashlib.sha256(_bb_block.encode()).hexdigest() == BUILD_BATCH_HELPERS_SHA256, "build_batch.py helpers changed"
RX = {"re": re, "math": math, "json": json, "Counter": Counter, "Path": Path, "hashlib": hashlib,
      "ENV_MARK": "Here is some useful information about the environment you are running in:",
      "THINK": re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)}
exec(compile(_bb_block, str(BUILD_BATCH), "exec"), RX)  # redact_obj (the corpus's redaction definition), split_system
V1_HARNESS = BUILD_BATCH.parent / "harness.json"  # v1's system bases and tool sets (selection rule 5)
_V1 = json.loads(V1_HARNESS.read_text())
V1_BASES = {v["base_sha256"] for v in _V1["system_prompt_variants"].values()}
V1_TOOLS = set(_V1["tool_sets"])
del _V1
DATE_LINE = re.compile(r"(?m)^[ \t]*Today's date:.*$")
RESULT_LINE = re.compile(r"(?m)^(?:VERIFY_RESULT_PLACEHOLDER|Result \([^)\n]*\): (?:OK|FAILED)\..*)$")


def canon(path):
    return CANON + path[len(ALIAS):] if isinstance(path, str) and path.startswith(ALIAS) else path


def sha256_json(value):  # build_batch.sha256_json's definition (harness.json keys)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8", "surrogatepass")).hexdigest()


def v1_harness_ok(q):
    """A request (read_journal entry with msgs / tools) has v1's system base and tool set (build_batch's harness_of /
    split_system over the redacted system messages and tools, as harness.json is keyed)."""
    system = RX["redact_obj"]([m.get("content") for m in q["msgs"] if m.get("role") == "system"], Counter())
    tools = RX["redact_obj"](q["tools"] or [], Counter())
    base = RX["split_system"]("\n".join(RX["_text"](m) for m in system))["base"]
    return hashlib.sha256(base.encode("utf-8", "surrogatepass")).hexdigest() in V1_BASES and \
        sha256_json(tools) in V1_TOOLS


def system_split(message):
    """Digests of (base, env block without its date line, the date line(s), rest) of a system message, or None."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return None
    parts = RX["split_system"](txt(message.get("content")))
    return tuple(hashlib.sha256(x.encode("utf-8", "surrogatepass")).digest()
                 for x in (parts["base"], DATE_LINE.sub("", parts["env"]), "\n".join(DATE_LINE.findall(parts["env"])),
                           parts["rest"]))


def change_kinds(a, b):
    """The parts in which two requests' system messages differ (empty: equal)."""
    if a["sys"] == b["sys"]:
        return set()
    if a["sys_split"] is None or b["sys_split"] is None:
        return {"other"}
    kinds = {name for name, x, y in zip(("base", "env", "date", "instructions"), a["sys_split"], b["sys_split"])
             if x != y}
    return kinds or {"other"}


def txt(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def no_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode("utf-16", "surrogatepass").decode("utf-16", "replace") if re.search("[\ud800-\udfff]", obj) \
            else obj
    if isinstance(obj, list):
        return [no_surrogates(x) for x in obj]
    if isinstance(obj, dict):
        return {k: no_surrogates(v) for k, v in obj.items()}
    return obj


def clean(obj):
    """The corpus's text pipeline: build_batch redaction, then lone surrogates -> U+FFFD."""
    return no_surrogates(RX["redact_obj"](obj, Counter()))


def mhash(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode("utf-8", "surrogatepass")).digest()


def parse_args_obj(arguments):
    value = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(value, dict):
        raise ValueError("tool call arguments are not a JSON object")
    return value


# ---------------------------------------------------------------- http.jsonl: own parser and chains
def read_journal(path):
    """Every request (hash list of messages[1:], full messages for chain heads, the added messages for linked
    requests) and response; degenerate.py's classify() on every status-200 body."""
    classify = V3.DG["classify"]
    reqs, resp, degenerate, replies = [], {}, [], 0
    with open(path, "rb") as fh:
        for raw in fh:
            head = raw[:4000]
            if b'"event":"request"' in head or b'"event": "request"' in head:
                ev = json.loads(raw)
                if ev.get("event") != "request":
                    continue
                p = ev.get("payload") or {}
                msgs = p.get("messages") or []
                hashes = [mhash(m) for m in msgs[1:]]
                last = msgs[-1] if msgs else {}
                summary_call = not p.get("tools") and last.get("role") == "user" and \
                    txt(last.get("content")).startswith(SUMMARY_PROMPTS)
                link, added = None, None
                if not summary_call:
                    for prev in reversed(reqs):
                        r = resp.get(prev["key"])
                        if prev["summary_call"] or not r or r["message"] is None:
                            continue
                        if prev["n"] < len(msgs) and hashes[:prev["n"] - 1] == prev["hashes"]:
                            link, added = prev["idx"], msgs[prev["n"]:]
                            break
                reqs.append({"idx": len(reqs), "key": (ev.get("call_id"), ev.get("http_attempt")), "n": len(msgs),
                             "hashes": hashes, "summary_call": summary_call, "link": link, "added": added,
                             "msgs": msgs if link is None else None, "tools": p.get("tools") if link is None else None,
                             "tools_hash": mhash(p.get("tools") or []),
                             "sys": mhash(msgs[0]) if msgs else None,
                             "sys_split": None if summary_call or not msgs else system_split(msgs[0]),
                             "m1": txt(msgs[1].get("content")) if len(msgs) > 1 else None})
            elif b'"event":"response"' in head or b'"event": "response"' in head:
                ev = json.loads(raw)
                if ev.get("event") != "response":
                    continue
                r = {"status": ev.get("status"), "message": None, "finish": None}
                if ev.get("status") == 200:
                    try:
                        body = json.loads(ev.get("body") or "{}")
                    except ValueError:
                        body = None
                    if isinstance(body, dict):
                        replies += 1
                        kind, tool = classify(body)
                        if kind:
                            degenerate.append({"reply": replies, "kind": kind, "tool": tool})
                        try:
                            choice = body["choices"][0]
                            r["message"], r["finish"] = choice["message"], choice.get("finish_reason")
                        except (KeyError, IndexError, TypeError):
                            pass
                resp[(ev.get("call_id"), ev.get("http_attempt"))] = r
    for q in reqs:
        r = resp.get(q["key"]) or {}
        q["reply"] = r.get("message") if r.get("status") == 200 else None
        q["finish"] = r.get("finish")
    return reqs, degenerate, replies


def as_request_assistant(reply):
    """A response as OpenCode re-sends it (for the link check)."""
    out = {"content": txt(reply.get("content")), "reasoning": reply.get("reasoning") or reply.get("reasoning_content")
           or "", "calls": [(t.get("id"), (t.get("function") or {}).get("name"),
                            json.dumps(parse_args_obj((t.get("function") or {}).get("arguments")), sort_keys=True))
                           for t in reply.get("tool_calls") or []]}
    return out


def request_assistant(m):
    return {"content": txt(m.get("content")), "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "calls": [(t.get("id"), (t.get("function") or {}).get("name"),
                       json.dumps(parse_args_obj((t.get("function") or {}).get("arguments")), sort_keys=True))
                      for t in m.get("tool_calls") or []]}


def chains(reqs, task_text):
    """Main windows: [(head idx, [member idxs])], the summary-writing call between windows, problems."""
    problems = []
    kids = defaultdict(list)
    for q in reqs:
        if q["link"] is not None and q["reply"] is not None:
            kids[q["link"]].append(q["idx"])
    heads = [q for q in reqs if q["link"] is None and q["reply"] is not None and not q["summary_call"]]
    main = [q for q in heads if (q["n"] == 2 and q["m1"] == task_text)
            or (q["m1"] == TASK_MARK and q["msgs"][-1].get("role") == "user" and txt(q["msgs"][-1].get("content")) == CONT)]
    if not main or main[0]["m1"] != task_text or sum(1 for q in main if q["m1"] == task_text) != 1:
        problems.append("no unique task head first")
    windows = []
    for h in main:
        members, cur = [h["idx"]], h["idx"]
        while kids.get(cur):
            if len(kids[cur]) != 1:
                problems.append(f"request {cur} has {len(kids[cur])} successful successors")
            cur = kids[cur][-1]
            members.append(cur)
        windows.append((h["idx"], members))
    summaries = []
    for (_, a), (hb, _) in zip(windows, windows[1:]):
        between = [q["idx"] for q in reqs if q["summary_call"] and q["reply"] is not None and a[-1] < q["idx"] < hb]
        if not between:
            problems.append(f"no summary-writing call between window ending {a[-1]} and head {hb}")
            summaries.append(None)
        else:
            summaries.append(between[-1])
    in_windows = {i for _, ms in windows for i in ms}
    stray = [q["idx"] for q in reqs if q["reply"] is not None and not q["summary_call"] and q["idx"] not in in_windows
             and q["m1"] in (task_text, TASK_MARK)]
    if stray:
        problems.append(f"main-session calls outside the windows: {stray[:5]}")
    return windows, summaries, problems


def window_messages(reqs, members):
    """The window's last request messages, rebuilt as head messages + every added block."""
    msgs = list(reqs[members[0]]["msgs"])
    for i in members[1:]:
        msgs += reqs[i]["added"]
    return msgs


# ---------------------------------------------------------------- export.json: own window rebuild
def export_reply(m):
    """(sent?, reasoning, visible, [(call id, name, input)], [result text]) of an export assistant message."""
    text, calls, results, n_text = "", [], [], 0
    for p in m.get("parts") or []:
        if p.get("type") == "text" and not p.get("ignored"):
            text, n_text = p.get("text") or "", n_text + 1
        elif p.get("type") == "tool":
            st = p.get("state") or {}
            calls.append((p.get("callID"), p.get("tool"), st.get("input")))
            status = st.get("status")
            results.append(txt(("[Old tool result content cleared]" if (st.get("time") or {}).get("compacted")
                                else st.get("output")) if status == "completed" else st.get("error")
                               if status == "error" else "[Tool execution was interrupted]"))
        elif p.get("type") not in ("step-start", "step-finish", "patch", "snapshot"):
            raise ValueError(f"export part {p.get('type')}")
    if n_text > 1:
        raise ValueError("several text parts")
    first = FIRST_THINK.search(text)
    reasoning = first.group(1) if first else ""
    visible = FIRST_THINK.sub("", text)
    return bool(n_text or calls), reasoning, visible, calls, results


def export_windows(export):
    """[(context as comparable tuples, [(export index, reply)] sent own messages, skipped own indices)] per window."""
    em = export["messages"]
    comp = [i for i, m in enumerate(em) if (m.get("info") or {}).get("role") == "user"
            and any(p.get("type") == "compaction" for p in m.get("parts") or [])]
    index_of = {(m.get("info") or {}).get("id"): i for i, m in enumerate(em)}
    out = []
    for k in range(len(comp) + 1):
        lo = 1 if k == 0 else comp[k - 1] + 3
        hi = comp[k] if k < len(comp) else len(em)
        ctx = []
        if k > 0:
            c = comp[k - 1]
            part = next(p for p in em[c]["parts"] if p.get("type") == "compaction")
            if not part.get("auto") or part.get("overflow"):
                raise ValueError("unsupported compaction part")
            summary = em[c + 1]
            if (summary.get("info") or {}).get("mode") != "compaction":
                raise ValueError("no summary after the compaction marker")
            ctx.append(("user", TASK_MARK))
            sent, r, v, calls, _ = export_reply(summary)
            if sent:
                ctx.append(("assistant", v, r, tuple((cid, n, json.dumps(a, sort_keys=True)) for cid, n, a in calls)))
            start = index_of.get(part.get("tail_start_id")) if part.get("tail_start_id") else None
            for j in range(start, c) if start is not None else []:
                sent, r, v, calls, res = export_reply(em[j])
                if sent:
                    ctx.append(("assistant", v, r, tuple((cid, n, json.dumps(a, sort_keys=True)) for cid, n, a in calls)))
                    ctx += [("tool", cid, x) for (cid, _, _), x in zip(calls, res)]
            cont = em[c + 2]
            if txt("".join(p.get("text") or "" for p in cont.get("parts") or [] if p.get("type") == "text")) != CONT:
                raise ValueError("no Continue message after the summary")
            ctx.append(("user", CONT))
        own, skipped = [], []
        for i in range(lo, hi):
            if (em[i].get("info") or {}).get("role") != "assistant":
                raise ValueError(f"user message inside window {k + 1}")
            reply = export_reply(em[i])
            (own if reply[0] else skipped).append((i, reply) if reply[0] else i)
        out.append({"context": ctx, "own": own, "skipped": skipped})
    return out, len(comp)


def http_comparable(msgs):
    """Request messages[1:] (+ the response) as comparable tuples (system excluded)."""
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            out.append(("user", txt(m.get("content"))))
        elif role == "assistant":
            a = request_assistant(m)
            out.append(("assistant", a["content"], a["reasoning"], tuple(a["calls"])))
        elif role == "tool":
            out.append(("tool", m.get("tool_call_id"), txt(m.get("content"))))
        else:
            out.append((role, txt(m.get("content"))))
    return out


# ---------------------------------------------------------------- expected pieces (sc14 template)
def render(msgs, tools=None, gen=False):
    return V3.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen,
                                      truncate_history_thinking=False)


_PRE = [{"role": "user", "content": ""}]


def target_message(reasoning, visible, calls):
    a = {"role": "assistant"}
    if reasoning.strip():
        a["reasoning_content"], a["content"] = reasoning, visible
    else:
        a["content"] = "<think>\n</think>" + visible
    if calls:
        a["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": args}} for n, args in calls]
    return a


def assistant_piece(a):
    block = render(_PRE + [a])[len(render(_PRE)):]
    assert block.startswith(GEN), "assistant block without the generation prefix"
    return block[len(GEN):]


def tool_piece(a, results):
    base = render(_PRE + [a])
    full = render(_PRE + [a] + [{"role": "tool", "content": r} for r in results], gen=True)
    assert full.startswith(base)
    return full[len(base):]


def context_message(m):
    role = m.get("role")
    if role == "user":
        return {"role": "user", "content": txt(m.get("content"))}
    if role == "tool":
        return {"role": "tool", "content": txt(m.get("content"))}
    if role != "assistant":
        raise ValueError(f"context role {role}")
    a = {"role": "assistant", "content": txt(m.get("content"))}
    reasoning = m.get("reasoning_content") if m.get("reasoning_content") is not None else m.get("reasoning")
    if reasoning:
        a["reasoning_content"] = reasoning
    if m.get("tool_calls"):
        a["tool_calls"] = [{"type": "function", "function": {"name": t["function"]["name"],
                                                             "arguments": parse_args_obj(t["function"]["arguments"])}}
                           for t in m["tool_calls"]]
    return a


def sample_of(shard_path, offset, sid):
    with open(shard_path, "rb") as f:
        f.seek(offset)
        msgs = json.loads(f.readline())["messages"]
    cur, found = [], None
    for m in msgs + [{"role": "system", "step_weight_agent_run_id": None}]:
        if m["role"] == "system":
            if cur and cur[0]["step_weight_agent_run_id"] == sid:
                found = cur
            cur = []
        cur.append(m)
    return found


def task_text_of(export):
    return "\n".join(p.get("text") or "" for p in export["messages"][0].get("parts") or []
                     if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))


def structure(attempt):
    """verify's own reading of one attempt dir: http.jsonl prefix chains (read_journal / chains) paired with the
    export.json windows (export_windows); the link checks, window 1's head has the system message and tool set of
    http.jsonl's first request, and per window: the two-source equality, sent export messages == calls, one tool set, a
    finish "length" reply only as the last call, a target, tool calls on every target but the last, a parsable context
    (k >= 2). Returns (state, problems); state["windows"] is None when the windows cannot be paired."""
    problems = []
    export = json.loads((attempt / "export.json").read_bytes())
    task_text = task_text_of(export)
    reqs, degenerate, replies = read_journal(attempt / "http.jsonl")
    wins, summaries, cprobs = chains(reqs, task_text)
    problems += cprobs
    st = {"export": export, "task_text": task_text, "reqs": reqs, "degenerate": degenerate, "replies": replies,
          "wins": wins, "windows": None}
    ewins, n_comp = export_windows(export)
    K = len(wins)
    if K != n_comp + 1 or len(ewins) != K:
        problems.append(f"{K} http windows vs {n_comp} compaction parts + 1 / {len(ewins)} export windows")
        return st, problems
    for i in [i for _, ms in wins for i in ms[1:]]:  # every link adds exactly [response, its tool results]
        q, prev = reqs[i], reqs[reqs[i]["link"]]
        added = q["added"]
        if not added or added[0].get("role") != "assistant" or \
                request_assistant(added[0]) != as_request_assistant(prev["reply"]):
            problems.append(f"request {i} does not re-send request {prev['idx']}'s response")
        ids = [t.get("id") for t in prev["reply"].get("tool_calls") or []]
        if [m.get("role") for m in added[1:]] != ["tool"] * len(ids) or \
                [m.get("tool_call_id") for m in added[1:]] != ids:
            problems.append(f"request {i}: tool results do not match request {prev['idx']}'s tool calls")
    head1 = reqs[wins[0][0]]
    if (reqs[0]["sys"], reqs[0]["tools_hash"]) != (head1["sys"], head1["tools_hash"]):
        problems.append("window 1's head differs from http.jsonl's first request (system / tools)")
    windows = []
    for k, ((head, members), ew) in enumerate(zip(wins, ewins), 1):
        hq = reqs[head]
        last = reqs[members[-1]]
        # two-source equality (system excluded)
        http_side = http_comparable(window_messages(reqs, members)[1:] + [dict(last["reply"], role="assistant")])
        exp_side = list(ew["context"]) if k > 1 else [("user", task_text)]
        own = ew["own"]
        for q, (_, (_, r, v, calls, res)) in enumerate(own):
            exp_side.append(("assistant", v, r, tuple((cid, n, json.dumps(a, sort_keys=True)) for cid, n, a in calls)))
            if q < len(own) - 1:
                exp_side += [("tool", cid, x) for (cid, _, _), x in zip(calls, res)]
        if exp_side != http_side:
            bad = next((j for j, (a, b) in enumerate(zip(exp_side, http_side)) if a != b),
                       min(len(exp_side), len(http_side)))
            problems.append(f"window {k}: export != http at message {bad} ({len(exp_side)} vs {len(http_side)})")
        if len(own) != len(members):
            problems.append(f"window {k}: {len(own)} sent export messages != {len(members)} calls")
        if any(reqs[i]["tools_hash"] != hq["tools_hash"] for i in members):
            problems.append(f"window {k}: tool set changes")
        # targets: every call except a finish "length" reply (only as the last call)
        targets = [i for i in members if reqs[i]["finish"] != "length"]
        lengths = [i for i in members if reqs[i]["finish"] == "length"]
        if lengths and lengths != [members[-1]]:
            problems.append(f"window {k}: a finish length reply is not the last call")
        if not targets:
            problems.append(f"window {k}: no target")
        for i in targets[:-1]:
            if not reqs[i]["reply"].get("tool_calls"):
                problems.append(f"window {k}: call {i} has no tool calls but is not the window's last target")
        if k > 1:
            try:
                [context_message(m) for m in hq["msgs"][1:]]
            except (ValueError, KeyError, TypeError) as error:
                problems.append(f"window {k}: context: {type(error).__name__}: {error}"[:200])
        windows.append({"k": k, "head": head, "members": members, "targets": targets, "lengths": lengths,
                        "skipped": ew["skipped"]})
    st["windows"] = windows
    return st, problems


def rollout_check(job):
    """Every check of one kept rollout (module docstring); returns (rollout id, problems, facts)."""
    rid, problems = job["rid"], []
    facts = {"windows": None, "targets": 0, "replies": 0, "dropped_length": 0, "errored": 0, "summary_calls": 0,
             "samples_checked": 0, "system_changed_targets": 0, "end_results_in_no_sample": 0}
    try:
        attempt = Path(job["attempt_dir"])
        rec = job["record"]
        if rec is None or rec.get("status") != "pass" or canon(rec.get("attempt_dir")) != str(attempt):
            problems.append("final record is not a pass of this attempt")
        terminal = json.loads((attempt / "terminal.json").read_bytes())
        if terminal.get("evaluation_completed") is not True or terminal.get("reward") != 1:
            problems.append("terminal.json is not a completed full-reward grade")
        patch_sha = sha256_file(attempt / "model.patch")
        if job.get("patch_sha256") != patch_sha:
            problems.append("metadata patch_sha256 != sha256(model.patch)")
        if job.get("regrade_used") and job.get("regrade_patch_sha256") not in (None, patch_sha):
            problems.append("the re-grade used for this pass graded another model.patch")
        st, sprobs = structure(attempt)
        problems += sprobs
        reqs, wins = st["reqs"], st["wins"]
        facts["replies"] = st["replies"]
        if st["degenerate"]:
            problems.append(f"degenerate replies {st['degenerate'][:3]}")
        K = len(wins)
        facts.update(windows=K, summary_calls=sum(1 for q in reqs if q["summary_call"] and q["reply"] is not None))
        if st["windows"] is None:
            return rid, problems, facts
        if [s["segment"] for s in job["samples"]] != list(range(1, K + 1)) or \
                any(s["n_segments"] != K for s in job["samples"]):
            problems.append(f"samples {[(s['segment'], s['n_segments']) for s in job['samples']]} != {K} windows")
            return rid, problems, facts
        export_msgs = st["export"]["messages"]
        for w, sample in zip(st["windows"], job["samples"]):
            k, head, members, targets = w["k"], w["head"], w["members"], w["targets"]
            hq = reqs[head]
            for e in w["skipped"]:
                facts["errored"] += bool((export_msgs[e].get("info") or {}).get("error"))
            facts["dropped_length"] += len(w["lengths"])
            facts["targets"] += len(targets)
            if not targets:
                continue
            if sample.get("first_call_index") is not None and (sample["first_call_index"], sample["last_call_index"]) \
                    != (targets[0], targets[-1]):
                problems.append(f"window {k}: selection call span {(sample['first_call_index'], sample['last_call_index'])}"
                                f" != {(targets[0], targets[-1])}")
            # the system prompt inside the window (the sample uses the window's first request's)
            kinds = set()
            for i in members:
                kinds |= change_kinds(hq, reqs[i])
            changed = [i for i in targets if reqs[i]["sys"] != hq["sys"]]
            expect_sys = {"system_changed_in_window": len({reqs[i]["sys"] for i in members}) > 1,
                          "system_change_kinds": sorted(kinds), "system_changed_targets": len(changed),
                          "system_content_changed_targets": sum(1 for i in changed
                                                                if change_kinds(hq, reqs[i]) - {"date"})}
            got_sys = {key: (sample.get("window") or {}).get(key) for key in expect_sys}
            if got_sys != expect_sys:
                problems.append(f"window {k}: system-change fields {got_sys} != {expect_sys}")
            facts["system_changed_targets"] += len(changed)
            # where the tool results of the window's last target are (the sample itself never holds them)
            last_ids = [t.get("id") for t in reqs[targets[-1]]["reply"].get("tool_calls") or []]
            if not last_ids:
                expect_end = None
            elif k < K:
                in_ctx = {m.get("tool_call_id") for m in reqs[wins[k][0]]["msgs"] if m.get("role") == "tool"}
                expect_end = "next_window_context" if all(i in in_ctx for i in last_ids) else "in_no_sample"
            else:
                expect_end = "in_no_sample"
            if (sample.get("window") or {}).get("end_tool_results") != expect_end:
                problems.append(f"window {k}: end_tool_results {(sample.get('window') or {}).get('end_tool_results')}"
                                f" != {expect_end}")
            facts["end_results_in_no_sample"] += expect_end == "in_no_sample"
            # expected pieces, from http.jsonl only
            system_text = "\n\n".join(clean([m.get("content") for m in hq["msgs"] if m.get("role") == "system"]))
            tools = clean(hq["tools"] or [])
            system = {"role": "system", "content": system_text}
            if k == 1:
                prompt = [{"role": "user", "content": clean(txt(hq["msgs"][1].get("content")))}]
            else:
                prompt = clean([context_message(m) for m in hq["msgs"][1:]])
            sys_piece = render([system], tools)
            gp = render([system] + prompt, tools, gen=True)
            expect = [("system", sys_piece), ("user", gp[len(sys_piece):])]
            weights = []
            for n, i in enumerate(targets):
                reply = reqs[i]["reply"]
                r, v, calls = clean([reply.get("reasoning") or reply.get("reasoning_content") or "",
                                     txt(reply.get("content")),
                                     [[(t.get("function") or {}).get("name"),
                                       parse_args_obj((t.get("function") or {}).get("arguments"))]
                                      for t in reply.get("tool_calls") or []]])
                a = target_message(r, v, [(nm, args) for nm, args in calls])
                expect.append(("assistant", assistant_piece(a)))
                weights.append(0.0 if job.get("invalid_policy") == "mask" and any(nm == "invalid" for nm, _ in calls)
                               else 1.0)
                if n < len(targets) - 1 and calls:
                    nxt = reqs[members[members.index(i) + 1]]
                    results = clean([txt(m.get("content")) for m in nxt["added"][1:]])
                    expect.append(("tool", tool_piece(a, results)))
            sample["expected_literal_eos"] = sum(c.count("</s>") for role, c in expect if role == "assistant")
            if sample.get("dropped"):
                if sample["dropped_reason"] == "assistant_literal_eos" and not sample["expected_literal_eos"]:
                    problems.append(f"window {k}: dropped for a literal </s>, none in its expected targets")
                if sample["dropped_reason"] == "over_524288":
                    n_tok = len(V3.tok("".join(c for _, c in expect), add_special_tokens=False)["input_ids"])
                    if (n_tok + 31) // 32 * 32 <= SEQ:
                        problems.append(f"window {k}: dropped as over 524,288 but has {n_tok} tokens")
                continue
            got = sample_of(sample["shard_path"], sample["offset"], sample["id"])
            if got is None:
                problems.append(f"window {k}: sample {sample['id']} not in its shard row")
                continue
            pieces = [(m["role"], m["content"]) for m in got]
            if pieces != expect:
                bad = next((j for j, (a, b) in enumerate(zip(pieces, expect)) if a != b), min(len(pieces), len(expect)))
                problems.append(f"window {k}: shard sample != http re-render at piece {bad} of {len(pieces)} / "
                                f"{len(expect)} ({pieces[bad][0] if bad < len(pieces) else '-'})")
            if sample["context_tokens"] != len(V3.tok(expect[1][1], add_special_tokens=False)["input_ids"]):
                problems.append(f"window {k}: context_tokens differs")
            # the loss fields of the sample's system message: per-piece token counts and the weight policy
            a_pieces = [c for role, c in expect if role == "assistant"]
            counts = [len(ids) for ids in V3.tok(a_pieces, add_special_tokens=False)["input_ids"]]
            if got[0].get("assistant_token_counts") != counts:
                problems.append(f"window {k}: assistant_token_counts != the token counts of the re-rendered assistant "
                                "pieces")
            if got[0].get("assistant_weights") != weights or got[0].get("weight") != 1.0:
                problems.append(f"window {k}: weight / assistant_weights differ from the weight policy "
                                f"({job.get('invalid_policy')})")
            facts["samples_checked"] += 1
    except Exception as error:  # noqa: BLE001 - reported as a problem of this rollout
        problems.append(f"verify error: {type(error).__name__}: {str(error)[:300]}")
    return rid, problems, facts


def exclusion_check(job):
    """Re-derive the exclusion reason of one excluded final pass from its raw attempt files with verify's own code;
    returns (rollout id, reason, problems)."""
    rid, reason, problems = job["rid"], job["reason"], []
    detail = job.get("detail") or {}
    try:
        attempt = Path(job["attempt_dir"])
        source = V3._source(attempt)  # agent_artifacts_from (never used by GLM attempts) or the attempt itself
        if reason == "not_clean":
            terminal = json.loads((attempt / "terminal.json").read_bytes()) if (attempt / "terminal.json").exists() \
                else {}
            files = (source / "export.json").exists() and (source / "http.jsonl").exists() and \
                any((d / "model.patch").exists() for d in (attempt, source))
            validity = V3._validity(attempt)
            failure = validity.get("failure") if isinstance(validity, dict) else None
            if terminal.get("evaluation_completed") is True and terminal.get("reward") == 1 and files \
                    and failure is None:
                problems.append("not_clean, but terminal.json, the files and the validity cache are clean")
        elif reason == "degenerate_reply":
            _, degenerate, _ = read_journal(source / "http.jsonl")
            if not degenerate:
                problems.append("degenerate_reply, but degenerate.py's classify() finds no degenerate reply")
        elif reason == "harness_unexpected":
            if not detail.get("where"):
                problems.append("harness_unexpected without the failing request in its detail")
            st, _ = structure(source)
            reqs = st["reqs"]
            heads = reqs[:1] + [reqs[h] for h, _ in st["wins"]]
            if heads and all(v1_harness_ok(q) for q in heads):
                problems.append("harness_unexpected, but the first request and every window head have v1's system "
                                "base and tool set")
        elif reason == "two_source_mismatch":
            if not detail.get("problems"):
                problems.append("two_source_mismatch without the gate's problems in its detail")
            try:
                _, sprobs = structure(source)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, StopIteration) as error:
                sprobs = [f"{type(error).__name__}: {error}"]
            if not sprobs:
                problems.append("two_source_mismatch, but verify's own chain / export code finds no problem")
        else:
            problems.append(f"no re-derivation for {reason}")
    except Exception as error:  # noqa: BLE001 - reported
        problems.append(f"verify error: {type(error).__name__}: {str(error)[:300]}")
    return rid, reason, problems


# ---------------------------------------------------------------- sim_sft_getitem.py on sample rows
def run_sim(arg):
    shard, row, out_json, timeout = arg
    proc = subprocess.run([PY, "-I", "-B", str(SIM), "--shard", shard, "--row", str(row), "--out", out_json],
                          capture_output=True, text=True, timeout=timeout,
                          env={k: v for k, v in os.environ.items()})
    if proc.returncode != 0:
        return shard, row, None, proc.stderr[-2000:]
    return shard, row, json.loads(Path(out_json).read_text()), None


def sim_problems(res, expected_ids):
    p = []
    for key in ("cu_seqlens_ok", "position_ids_reset_per_segment", "loss_mask_sum_ok",
                "mamba_seq_idx_changes_match_cu_seqlens"):
        if res.get(key) is not True:
            p.append(f"{key} false")
    for key in ("loss_positions_with_label_from_other_segment", "mtp1_loss_positions_with_label_from_other_segment"):
        if res.get(key) != 0:
            p.append(f"{key} = {res.get(key)}")
    for s in res.get("per_segment_loss_mass") or []:
        if abs(s["loss_mass"] - s["expected"]) >= 0.5:
            p.append(f"segment {s['seg']} loss mass {s['loss_mass']} != {s['expected']}")
    if [s["id"] for s in res.get("per_segment_loss_mass") or []] != expected_ids:
        p.append("segment ids differ from the row's samples")
    if not str(res.get("sft_dataset_sha256", "")).startswith(SFT_DATASET_PIN) or \
            not str(res.get("sft_tokenizer_sha256", "")).startswith(SFT_TOKENIZER_PIN):
        p.append("runtime files are not the pinned ones")
    return p


def data_root_problem(dry_run, data_root, allow_other):
    """A real (non dry-run) build must bake in DESIGN's AGA DATA_ROOT unless explicitly allowed."""
    if dry_run or allow_other or data_root == DATA_ROOT_DEFAULT:
        return None
    return (f"DATA_ROOT {data_root} of a real build is not DESIGN's AGA DATA_ROOT {DATA_ROOT_DEFAULT} (pass "
            "--allow-other-data-root if intended)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CORPUS))
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sim-rows", type=int, default=6)
    ap.add_argument("--sim-workers", type=int, default=4)
    ap.add_argument("--sim-timeout", type=int, default=1800)
    ap.add_argument("--readme", action="store_true")
    ap.add_argument("--allow-other-data-root", action="store_true",
                    help="a real (non dry-run) build whose DATA_ROOT is not DESIGN's AGA DATA_ROOT")
    args = ap.parse_args()
    out = args.out.rstrip("/")
    data_root = args.data_root.rstrip("/")
    started = time.time()
    receipt = json.loads(Path(out, "DATA_PREP_COMPLETE.json").read_text())
    stats = receipt["build_stats"]
    report = json.loads(Path(out, "sources/glm/selection_report.json").read_text())
    policy = stats["invalid_policy"]
    result = {"out": out, "dry_run": stats["dry_run"], "data_root": data_root, "invalid_policy": policy}
    problems = defaultdict(list)
    readme_path = Path(out, "README.md")
    if args.readme:  # the line this run replaces: the placeholder, or an earlier run's result line
        n_slots = len(RESULT_LINE.findall(readme_path.read_text()))
        if n_slots != 1:
            problems["readme"].append(f"README.md has {n_slots} placeholder / result lines (expected exactly 1)")
    root_problem = data_root_problem(stats["dry_run"], data_root, args.allow_other_data_root)
    if root_problem:
        problems["receipt"].append(root_problem)

    # ---- v3's row re-check on every row of every shard
    import datasets  # noqa: E402 (already imported by v3's verify)
    shard_files = sorted(glob.glob(out + "/raw_sft/*.packed.512k"))
    assert [os.path.relpath(f, out) for f in shard_files] == SHARDS, shard_files
    jobs, rows_per_shard = [], []
    for f in shard_files:
        ds = datasets.load_dataset("json", data_files=f, split="all")
        rows_per_shard.append(len(ds))
        for i in range(len(ds)):
            jobs.append((os.path.basename(f), i, ds[i]["messages"]))
    allids, tot, exp, lengths, literal = [], 0.0, 0.0, {}, {}
    pool = get_context("fork").Pool(args.workers, initializer=V3.init)
    for path, ri, nc, s, e, ids, lens, lit in pool.imap_unordered(V3.row, jobs):
        if abs(s - e) >= 0.5:
            problems["rows"].append(f"{path} row {ri}: loss mask {s} != {e}")
        tot += s
        exp += e
        allids += ids
        lengths.update(lens)
        literal.update(lit)
    n_rows = len(jobs)
    del jobs
    if len(allids) != len(set(allids)):
        problems["rows"].append("duplicate sample ids")
    print("rows", n_rows, "samples", len(allids), "loss_mask_sum", tot, "expected", exp, flush=True)
    result.update(rows=n_rows, samples=len(allids), loss_mask_sum=tot, expected=exp)
    V3.init()  # the tokenizer in this process too (stop ids)
    stop_list, bos_id = V3.STOP
    stop_hits = {k: v[0] for k, v in literal.items() if v[0]}
    bos_hits = {k: v[1] for k, v in literal.items() if v[1]}
    if stop_hits:
        problems["rows"].append(f"generation stop ids among targets: {stop_hits}")
    result["assistant_literal"] = {"stop_ids": stop_list, "bos_id": bos_id, "stop_target_samples": stop_hits,
                                   "bos_target_tokens": sum(bos_hits.values()), "bos_target_samples": len(bos_hits)}

    # ---- selection <-> shards, sha / token_length, sample offsets
    sel = [json.loads(line) for line in open(Path(out, SELECTION), encoding="utf-8")]
    if sorted(allids) != sorted(s["agent_run_id"] for s in sel):
        problems["selection"].append("shard samples != selection rows")
    for s in sel:
        if lengths.get(s["agent_run_id"]) != int(s["token_length"]):
            problems["selection"].append(f"{s['agent_run_id']}: token_length")
    shas, where, row_ids, loss_fields = {}, {}, [], {}
    allowed = (1.0,) if policy == "keep" else (0.0, 1.0)
    for f in shard_files:
        with open(f, "rb") as fh:
            ri = 0
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                ms = json.loads(line)["messages"]
                cur, ids = [], []
                for m in ms + [{"role": "system", "_end": 1}]:
                    if m["role"] == "system" and cur:
                        sid = cur[0]["step_weight_agent_run_id"]
                        shas[sid] = hashlib.sha256(json.dumps({"messages": cur}, ensure_ascii=False).encode()).hexdigest()
                        where[sid] = (f, offset, ri, len(ids))
                        ids.append(sid)
                        # the loss fields themselves (v3's row() takes its expected mass from them)
                        s0 = cur[0]
                        aw, cnt = s0.get("assistant_weights"), s0.get("assistant_token_counts")
                        if not (isinstance(s0.get("weight"), float) and s0["weight"] == 1.0
                                and isinstance(aw, list) and isinstance(cnt, list) and len(aw) == len(cnt) > 0
                                and all(isinstance(x, float) and x in allowed for x in aw)
                                and all(isinstance(x, int) and x > 0 for x in cnt)):
                            problems["weights"].append(f"{sid}: weight / assistant_weights / assistant_token_counts "
                                                       f"not of the weight policy ({policy})")
                        if isinstance(aw, list) and isinstance(cnt, list):
                            loss_fields[sid] = (sum(cnt), len(cnt), sum(1 for x in aw if x == 0.0))
                        cur = []
                    cur.append(m)
                row_ids.append((f, ri, ids))
                ri += 1
    for s in sel:
        if shas.get(s["agent_run_id"]) != s["training_row_sha256"]:
            problems["selection"].append(f"{s['agent_run_id']}: training_row_sha256")
        if loss_fields.get(s["agent_run_id"]) != (s["assistant_tokens"], s["n_assistant"], s["masked_assistant_turns"]):
            problems["selection"].append(f"{s['agent_run_id']}: assistant_tokens / n_assistant / masked_assistant_turns "
                                         f"!= the sample's assistant_token_counts / assistant_weights")
        f, _, ri, pos = where.get(s["agent_run_id"], (None, None, None, None))
        if (os.path.basename(f or ""), ri, pos) != (s["shard"] + ".mat.jsonl.packed.512k", s["shard_row"],
                                                   s["row_position"]):
            problems["selection"].append(f"{s['agent_run_id']}: shard / row / position")
        if not s["agent_run_id"].startswith(s["parent_trajectory_id"] + "#seg") or \
                s["agent_run_id"] != f"{s['parent_trajectory_id']}#seg{s['segment']}of{s['n_segments']}":
            problems["selection"].append(f"{s['agent_run_id']}: id / segment fields")
    result["by_batch"] = dict(Counter(s["rollout_batch"] for s in sel))

    # ---- receipt / blend / data root
    try:
        assert receipt["selection_sha256"] == sha256_file(Path(out, SELECTION))
        assert receipt["packed_rows"] == n_rows == sum(rows_per_shard)
        assert receipt["selected_trajectories"] == len(sel) == stats["samples"]
        assert [s["packed_rows"] for s in receipt["shards"]] == rows_per_shard == stats["rows_per_shard"]
        assert [s["bytes"] for s in receipt["shards"]] == [os.path.getsize(f) for f in shard_files]
        assert [s["path"] for s in receipt["shards"]] == [f"{data_root}/{p}" for p in SHARDS]
        assert receipt["blend"] == f"{data_root}/blends/raw_sft.json" and receipt["data_root"] == data_root
        assert receipt["selection"] == f"{data_root}/{SELECTION}"
        blend = json.loads(Path(out, "blends/raw_sft.json").read_text())
        assert blend["train"] == [x for s in receipt["shards"] for x in ("0.16666667", s["path"])]
        assert blend["valid"] is None and blend["test"] is None
        assert receipt["state"] == ("dry-run" if stats["dry_run"] else "complete")
        assert bool(stats["dry_run"]) == bool(report["partial"])
        for b, ps in stats["per_batch"].items():
            assert ps.get("samples", 0) == result["by_batch"].get(b, 0), b
    except AssertionError as error:
        problems["receipt"].append(f"receipt / blend / data root: {error!r}"[:300])

    # ---- old-grader inputs (the copies equal what the selection read)
    og = report["old_grader_inputs"]
    cpath = Path(out, "sources/regrade/classes.json")
    if sha256_file(cpath) != og["classes"]["sha256"]:
        problems["old_grader"].append("sources/regrade/classes.json != the classes the selection read")
    classes = json.loads(cpath.read_text())["passes"]
    rpath = Path(out, "sources/regrade/results.json")
    if og["results"]["exists"] != rpath.exists() or (rpath.exists() and sha256_file(rpath) != og["results"]["sha256"]):
        problems["old_grader"].append("sources/regrade/results.json != the results the selection read")
    results = (json.loads(rpath.read_text()).get("results") or {}) if rpath.exists() else {}

    # ---- metadata, records, completeness
    metas = [json.loads(line) for line in open(Path(out, "metadata.jsonl"), encoding="utf-8")]
    sample_meta = {m["trajectory_id"]: m for m in metas if "parent_trajectory_id" in m}
    excluded = {m["trajectory_id"]: m for m in metas if (m.get("corpus") or {}).get("dropped_reason", "")
                and str(m["corpus"]["dropped_reason"]).startswith("selection:")}
    records = {}
    all_tier12 = set()
    for batch, path in FINAL_RECORDS.items():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rid = f"glm53-{batch}/{r['campaign']}/{r['task_id']}/{r['slot']}"
            records[rid] = r
            if batch == "sftv3" or r.get("tier") in (1, 2):
                all_tier12.add(rid)
    skipped = {x["trajectory_id"] for x in report.get("skipped_sftv3_not_tier12") or []}
    considered = {e["trajectory_id"] for e in report["exclusions"]} | {m["trajectory_id"] for m in report.get("metas", [])}
    kept_rollouts = {m["parent_trajectory_id"] for m in sample_meta.values()}
    if kept_rollouts != {m["trajectory_id"] for m in report.get("metas", [])}:
        problems["completeness"].append("metadata sample parents != the selection report's kept rollouts")
    if set(excluded) != {e["trajectory_id"] for e in report["exclusions"]}:
        problems["completeness"].append("metadata exclusions != the selection report's exclusions")
    if kept_rollouts & set(excluded):
        problems["completeness"].append(f"kept and excluded: {sorted(kept_rollouts & set(excluded))[:3]}")
    if not report["partial"] and considered != all_tier12 - skipped:
        problems["completeness"].append(f"considered != every tier-1/2 final record ({len(considered)} vs "
                                        f"{len(all_tier12 - skipped)})")
    if not considered <= set(records):
        problems["completeness"].append("considered records not in the final records")
    if len(considered) != report["considered"]:
        problems["completeness"].append("selection report considered count")
    xjobs = []  # exclusions re-derived from the attempt files (pool, below)
    known_reasons = {"not_clean", "old_grader_unclassified", "old_grader_not_regraded", "old_grader_regrade_fail",
                     "degenerate_reply", "harness_unexpected", "two_source_mismatch"}
    for rid, m in excluded.items():  # re-checks of the exclusion reasons
        reason = m["reason"]
        r = records.get(rid) or {}
        key = rid[len("glm53-"):]
        c, res = classes.get(key), results.get(key)
        if reason.startswith("final_status_") and r.get("status") != reason[len("final_status_"):]:
            problems["exclusions"].append(f"{rid}: {reason} but record status {r.get('status')}")
        if not reason.startswith("final_status_") and reason not in known_reasons:
            problems["exclusions"].append(f"{rid}: unknown exclusion reason {reason}")
        if reason != "not_clean" and not reason.startswith("final_status_") and r.get("status") != "pass":
            problems["exclusions"].append(f"{rid}: {reason} but not a pass")
        if r.get("status") == "pass" and canon(r.get("attempt_dir")) != canon(m.get("attempt_dir")):
            problems["exclusions"].append(f"{rid}: attempt_dir differs from the final record's")
        if res is not None and res.get("model_patch_sha256") is not None:
            patch = Path(canon(m.get("attempt_dir") or ""), "model.patch")
            if patch.exists() and res["model_patch_sha256"] != sha256_file(patch):
                res = None  # a re-grade of another model.patch: not this pass's
        if reason == "old_grader_unclassified" and not (c is None or canon(c.get("attempt_dir"))
                                                        != canon(m.get("attempt_dir"))):
            problems["exclusions"].append(f"{rid}: old_grader_unclassified, but classes.json classifies this attempt")
        if reason == "old_grader_not_regraded" and not (c and c["class"] == "B" and res is None):
            problems["exclusions"].append(f"{rid}: not_regraded inconsistent")
        if reason == "old_grader_regrade_fail" and not (res is not None and res.get("reward") != 1
                                                         and ((c or {}).get("class") == "B"
                                                              or res.get("reward") is not None)):
            problems["exclusions"].append(f"{rid}: regrade_fail inconsistent")
        if reason in ("not_clean", "degenerate_reply", "harness_unexpected", "two_source_mismatch"):
            xjobs.append({"rid": rid, "reason": reason, "attempt_dir": canon(m.get("attempt_dir")),
                          "detail": m.get("detail")})
    # sample rows of every kept rollout (selection + build drops)
    by_parent = defaultdict(list)
    sel_by_id = {s["agent_run_id"]: s for s in sel}
    window_keys = ("system_changed_in_window", "system_change_kinds", "system_changed_targets",
                   "system_content_changed_targets", "end_tool_results")
    for sid, m in sample_meta.items():
        c = m.get("corpus") or {}
        entry = {"id": sid, "segment": m["segment"], "n_segments": m["n_segments"],
                 "dropped": bool(c.get("dropped")), "dropped_reason": c.get("dropped_reason"),
                 "first_call_index": None, "last_call_index": None, "context_tokens": c.get("context_tokens"),
                 "window": {k: (m.get("window") or {}).get(k) for k in window_keys}}
        if sid in sel_by_id:
            s = sel_by_id[sid]
            f, offset, _, _ = where[sid]
            entry.update(shard_path=f, offset=offset, first_call_index=s["first_call_index"],
                         last_call_index=s["last_call_index"], context_tokens=s["context_tokens"])
            if c.get("dropped"):
                problems["completeness"].append(f"{sid}: dropped in metadata but in the selection")
            if any(s.get(k) != entry["window"][k] for k in window_keys[:4]):
                problems["selection"].append(f"{sid}: system-change fields differ between selection and metadata")
        elif not c.get("dropped"):
            problems["completeness"].append(f"{sid}: neither in the selection nor dropped")
        by_parent[m["parent_trajectory_id"]].append(entry)
    for s in sel:
        if s["agent_run_id"] not in sample_meta:
            problems["completeness"].append(f"{s['agent_run_id']}: no metadata row")
        key = s["parent_trajectory_id"][len("glm53-"):]
        c, res = classes.get(key), results.get(key)
        if res is not None and res.get("model_patch_sha256") not in (
                None, (sample_meta.get(s["agent_run_id"]) or {}).get("patch_sha256")):
            res = None  # a re-grade of another model.patch: not this pass's
        tclass = {"n/a": "none"}.get(c["class"], c["class"]) if c else None
        if c is None or (s["grader_kind"], s["test_touch_class"]) != (c["grader_kind"], tclass):
            problems["old_grader"].append(f"{s['agent_run_id']}: grader kind / class differ from classes.json")
        elif c["class"] == "B" and (res is None or res.get("reward") != 1):
            problems["old_grader"].append(f"{s['agent_run_id']}: class B without a reward-1 re-grade")
        elif res is not None and res.get("reward") is not None and res.get("reward") != 1:
            problems["old_grader"].append(f"{s['agent_run_id']}: a re-grade with reward {res.get('reward')}")
        if (s.get("regrade") or None) != (None if res is None else {k: res.get(k) for k in ("reward", "status")}):
            problems["old_grader"].append(f"{s['agent_run_id']}: regrade field differs from results.json")

    # ---- per kept rollout: independent re-derivation from the raw attempt files
    rjobs = []
    for rid, samples in sorted(by_parent.items()):
        samples.sort(key=lambda x: x["segment"])
        m = sample_meta[samples[0]["id"]]
        res = results.get(rid[len("glm53-"):])
        rjobs.append({"rid": rid, "attempt_dir": canon(m["attempt_dir"]), "record": records.get(rid),
                      "samples": samples, "patch_sha256": m.get("patch_sha256"),
                      "regrade_patch_sha256": (res or {}).get("model_patch_sha256"),
                      "regrade_used": m.get("regrade") is not None, "invalid_policy": policy})
    rfacts = Counter()
    for rid, probs, facts in pool.imap_unordered(rollout_check, rjobs, chunksize=1):
        if probs:
            problems["rollouts"].append({rid: probs[:6]})
        rfacts["rollouts"] += 1
        for k in ("windows", "targets", "replies", "dropped_length", "errored", "summary_calls", "samples_checked",
                  "system_changed_targets", "end_results_in_no_sample"):
            rfacts[k] += facts.get(k) or 0
    xfacts = Counter()
    for rid, reason, probs in pool.imap_unordered(exclusion_check, xjobs, chunksize=1):
        if probs:
            problems["exclusions"].append({rid: probs[:4]})
        else:
            xfacts[reason] += 1
    pool.close()
    pool.join()
    result["rollouts"] = dict(rfacts)
    result["exclusions_rederived"] = {"jobs": len(xjobs), "confirmed": dict(xfacts),
                                      "by_reason": dict(Counter(m["reason"] for m in excluded.values()))}
    build_drops = {sid: (m.get("corpus") or {}).get("dropped_reason") for sid, m in sample_meta.items()
                   if (m.get("corpus") or {}).get("dropped")}
    result["build_drops"] = build_drops

    # ---- sim_sft_getitem.py (the real pinned SFTDataset.__getitem__) on sample rows
    sim_res = {"rows": [], "problems": []}
    if sha256_file(SIM) != SIM_SHA256 or (SIM_ORIGINAL.exists() and sha256_file(SIM_ORIGINAL) != SIM_SHA256):
        problems["sim"].append("tools/sim_sft_getitem.py is not the pinned copy")
    parent = {s["agent_run_id"]: s["parent_trajectory_id"] for s in sel}
    two = [(f, ri, ids) for f, ri, ids in row_ids if max(Counter(parent[i] for i in ids).values()) >= 2]
    multi = [(f, ri, ids) for f, ri, ids in row_ids if len(ids) >= 2 and (f, ri, ids) not in two]
    single = [(f, ri, ids) for f, ri, ids in row_ids if len(ids) == 1]
    pick = []
    for group, n in ((two, 4), (multi, 2), (single, 1)):
        for item in group:
            if len([p for p in pick if p in group]) >= n:
                break
            pick.append(item)
    for item in row_ids:
        if len(pick) >= args.sim_rows:
            break
        if item not in pick:
            pick.append(item)
    sim_dir = Path(_TMP, "verify-sim")
    sim_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.sim_workers) as tpe:
        futs = [tpe.submit(run_sim, (f, ri, str(sim_dir / f"{os.path.basename(f)}-{ri}.json"), args.sim_timeout))
                for f, ri, _ in pick]
        for (f, ri, ids), fut in zip(pick, futs):
            shard, row, res, err = fut.result()
            entry = {"shard": os.path.basename(shard), "row": row, "samples": len(ids),
                     "windows_of_one_rollout": max(Counter(parent[i] for i in ids).values())}
            if res is None:
                problems["sim"].append(f"{entry['shard']} row {row}: sim failed: {err}")
            else:
                p = sim_problems(res, ids)
                entry.update(cu_seqlens=res["cu_seqlens"], loss_positions=res["loss_positions"],
                             cross=res["loss_positions_with_label_from_other_segment"],
                             mtp_cross=res["mtp1_loss_positions_with_label_from_other_segment"],
                             seconds=res["seconds"]["total"], problems=p)
                if p:
                    problems["sim"].append(f"{entry['shard']} row {row}: {p}")
            sim_res["rows"].append(entry)
            print("sim", json.dumps(entry)[:300], flush=True)
    sim_res["rows_with_two_windows_of_one_rollout_available"] = len(two)
    result["sim"] = sim_res
    ok = not any(problems.values())
    result["problems"] = {k: v[:30] for k, v in problems.items() if v}
    result["ok"] = ok
    result["seconds"] = round(time.time() - started, 1)
    print(json.dumps(result, indent=1, default=str))
    if args.readme and not problems["readme"]:
        text = readme_path.read_text()
        sims = sim_res["rows"]
        line = (f"Result ({time.strftime('%Y-%m-%d %H:%M %Z')}): {'OK' if ok else 'FAILED'}. {n_rows} rows, "
                f"{len(allids)} samples ({', '.join(f'{k} {v}' for k, v in result['by_batch'].items())}), no truncation, "
                f"loss-mask sum {tot:,.0f} == sum(counts x weights) {exp:,.0f}; every sample's weight 1.0 and "
                f"assistant_weights of the `{policy}` policy; selection == shards, every training_row_sha256 / "
                f"token_length / shard position / assistant token count matches; receipt and blend under `{data_root}`"
                f"{' (DESIGN AGA DATA_ROOT)' if data_root == DATA_ROOT_DEFAULT else ' (NOT the DESIGN AGA DATA_ROOT)'}; "
                f"{rfacts['rollouts']} kept rollouts re-derived from their attempt files ({rfacts['windows']} windows, "
                f"{rfacts['samples_checked']} samples re-rendered from http.jsonl and equal to the shards incl. their "
                f"assistant_token_counts and weights, {rfacts['targets']:,} target calls each a target exactly once "
                f"({rfacts['system_changed_targets']:,} made with another system prompt than their sample's, as the "
                f"system-change fields say), {rfacts['dropped_length']} finish-length and {rfacts['errored']} errored "
                f"replies never targets, {rfacts['end_results_in_no_sample']} window-end tool results in no sample, "
                f"{rfacts['summary_calls']} summary-writing calls not samples, export == http for every window, "
                f"{rfacts['replies']:,} replies re-scanned with degenerate.py's classify(): none degenerate); "
                f"old-grader classes and re-grades consistent; every considered final record kept or excluded once "
                f"({len(considered)}), exclusions re-derived from the attempt files: "
                + (", ".join(f"{k} {v}" for k, v in sorted(xfacts.items())) or "none needed")
                + f"; generation-stop targets: {len(stop_hits)}, literal BOS targets {sum(bos_hits.values())} in "
                f"{len(bos_hits)} samples; build drops: "
                + (", ".join(f"`{k}` ({v})" for k, v in build_drops.items()) or "none")
                + f"; the real SFTDataset.__getitem__ (sft_dataset.py {SFT_DATASET_PIN}) on {len(sims)} rows ("
                f"{sum(1 for x in sims if x['windows_of_one_rollout'] >= 2)} holding two or more windows of one "
                f"rollout): cu_seqlens == cumulative ceil32 lengths, 0 loss positions labelled from another sample "
                f"(main and MTP-1), per-sample loss mass == counts x weights, Mamba seq_idx boundaries == cu_seqlens."
                + ("" if ok else f" PROBLEMS: {json.dumps(result['problems'], default=str)[:1500]}"))
        line = line.replace("\n", " ")
        tmp = readme_path.with_name(".README.md.tmp")
        tmp.write_text(RESULT_LINE.sub(lambda _: line, text, count=1))
        os.replace(tmp, readme_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
