#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): the "missing </think>" problem per model, from the model calls of each slot's final
attempt (http.jsonl response events) and, for the 10/20-epoch fleets, the counting parser's vLLM log lines.

    python3 -I -B think_analysis.py [--models sftv2e10 sftv2e20 sftv2 sc14] [--procs 16]
      -> TA/analysis/think/{summary.json, calls-<model>.jsonl}

Join: vLLM's response id is "chatcmpl-" + ChatCompletionRequest.request_id (no X-Request-Id header on this path), and
the counting parser logs request.request_id, so a log line maps to the response whose body id is "chatcmpl-<id>".
Per model call (200 responses of the final attempts): logged recovered (unclosed think + <tool_call>: the lenient rule
ended the reasoning there), logged unclosed-no-tool (chars=0 = EOS right after <think>), logged tool-in-closed-think
(only workers whose parser logs that counter), and post hoc: "<tool_call>" inside the returned reasoning (a closed
think holding a tool call), empty output (no reasoning, no content, no tool call), 1-token stop (completion_tokens 1,
finish stop), and the candidate signature "reasoning ends with a newline".

2026-10-27: + the empty-think-prefix fleets etp5 / etp10 (counting parser, logged like 10/20 ep). Per call also
prompt_tokens and latency_s (response minus request journal timestamp of the same call_id and http_attempt: the
server time of the call as the rollout saw it, incl. vLLM queueing); per slot the status codes of non-200 responses
(summary "non200_by_status"). Existing fields are unchanged.
    python3 -I -B think_analysis.py --models etp5 etp10 --procs 12 > run-etp.log

2026-11-04: copied to analysis/v3etp20/think for v3etp20 (SFT-v3 empty-think-prefix 20 ep, iter284; counting parser,
fleet dir fleets/v3etp20). Only OUT changed; per-call logic unchanged.
    python3 -I -B think_analysis.py --models v3etp20 --procs 16 > run-v3etp20.log
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import collections
import json
from pathlib import Path
import re

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/think"  # 2026-11-04 copy for v3etp20: writes here only (the original analysis/think is read-only)
FLEET = {"sftv2": "sftv2", "sc14": "base", "sftv2e10": "sftv2e10", "sftv2e20": "sftv2e20", "etp5": "etp5",
         "etp10": "etp10"}  # model -> fleet dir; a model not listed uses its own name (e.g. etp20)
LOG_LINE = re.compile(r"TA_THINK_(RECOVERED|UNCLOSED_NO_TOOL|TOOL_IN_CLOSED_THINK) request_id=(\S+)"
                      r"(?: reasoning_chars=(\d+)| chars=(\d+))?")
COUNTS_LINE = re.compile(r"TA_THINK_COUNTS pid=(\d+) (.*)$")
REQ_CALL = re.compile(rb'"call_id": ?"([^"]+)"')
REQ_ATTEMPT = re.compile(rb'"http_attempt": ?(\d+)')
REQ_TS = re.compile(rb'"timestamp": ?([0-9.eE+-]+)')


def slots(model):
    out = []
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for record in records:
                out.append({"model": model, "group": data["group"], "task_id": task_id, "slot": record["slot"],
                            "status": record["status"], "attempt_dir": record.get("attempt_dir"),
                            "patch_chars": record.get("patch_chars")})
    return out


def scan_attempt(slot):
    """Model calls (200 responses) of one attempt's http.jsonl."""
    calls, requests, non200 = [], 0, 0
    request_ts = {}  # (call_id, http_attempt) -> journal timestamp of the request (2026-10-27: per-call latency)
    non200_status = collections.Counter()
    path = Path(slot["attempt_dir"]) / "http.jsonl" if slot.get("attempt_dir") else None
    if not path or not path.exists():
        return slot, calls, {"requests": 0, "non200": 0, "http_missing": True, "non200_status": {}}
    with open(path, "rb") as handle:
        for raw in handle:
            if b'"event":"request"' in raw[:2000]:
                requests += 1
                head = raw[:2000]
                cid, att, ts = REQ_CALL.search(head), REQ_ATTEMPT.search(head), REQ_TS.search(head)
                if cid and ts:
                    request_ts[(cid.group(1).decode(), int(att.group(1)) if att else None)] = float(ts.group(1))
                continue
            if b'"event":"response"' not in raw[:2000] and b'"event": "response"' not in raw[:2000]:
                continue
            event = json.loads(raw)
            if event.get("event") != "response":
                continue
            body = event.get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = None
            if event.get("status") != 200 or not isinstance(body, dict) or not body.get("choices"):
                non200 += 1
                non200_status[str(event.get("status"))] += 1
                continue
            choice = body["choices"][0]
            message = choice.get("message") or {}
            reasoning = message.get("reasoning")
            if reasoning is None:
                reasoning = message.get("reasoning_content")
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            usage = body.get("usage") or {}
            sent = request_ts.get((event.get("call_id"), event.get("http_attempt")))
            calls.append({"id": body.get("id"), "finish": choice.get("finish_reason"),
                          "completion_tokens": usage.get("completion_tokens"),
                          "reasoning_len": len(reasoning) if isinstance(reasoning, str) else None,
                          "reasoning_tail": reasoning[-40:] if isinstance(reasoning, str) else None,
                          "reasoning_nl_end": isinstance(reasoning, str) and reasoning.endswith("\n"),
                          "tool_in_reasoning": isinstance(reasoning, str) and "<tool_call>" in reasoning,
                          "content_len": len(content) if isinstance(content, str) else None,
                          "content_head": content[:40] if isinstance(content, str) else None,
                          "tool_calls": len(tool_calls), "http_attempt": event.get("http_attempt"),
                          "ts": event.get("timestamp"), "prompt_tokens": usage.get("prompt_tokens"),
                          "latency_s": round(event["timestamp"] - sent, 3)
                          if sent is not None and isinstance(event.get("timestamp"), (int, float)) else None})
    return slot, calls, {"requests": requests, "non200": non200, "http_missing": False,
                         "non200_status": dict(non200_status)}


def parse_logs(fleet):
    """{request_id: (kind, chars, log)}, per-log parser version, last TA_THINK_COUNTS per (log, pid)."""
    marks, versions, counts = {}, {}, {}
    for log in sorted((TA / "fleets" / fleet / "logs").glob("vllm-*.log")):
        new_version = False
        with open(log, "rb") as handle:
            for raw in handle:
                if b"TA_THINK_" not in raw:
                    continue
                line = raw.decode("utf-8", "replace").rstrip("\n")
                match = LOG_LINE.search(line)
                if match:
                    kind, request_id, rchars, chars = match.groups()
                    marks[request_id] = (kind, int(rchars or chars or 0), log.name)
                    continue
                match = COUNTS_LINE.search(line)
                if match:
                    fields = dict(item.split("=") for item in match.group(2).split())
                    if "closed_tool_in_think" in fields:
                        new_version = True
                    counts[(log.name, match.group(1))] = {k: int(v) for k, v in fields.items()}
        versions[log.name] = "with_closed_tool_counter" if new_version else (
            "early_or_unknown" if any(key[0] == log.name for key in counts) else "no_counts_line")
    return marks, versions, counts


def attempt_ids(path):
    """Response body ids of one http.jsonl (any attempt), for the log-to-response mapping check."""
    ids = set()
    with open(path, "rb") as handle:
        for raw in handle:
            if b'"event":"response"' not in raw[:2000]:
                continue
            body = json.loads(raw).get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = None
            if isinstance(body, dict) and isinstance(body.get("id"), str):
                ids.add(body["id"])
    return ids


SIGNATURES = {
    # name: (logged kind it should identify, predicate on a call)
    # main's hypothesis: the model wrote "...\n<tool_call>" without </think>, so the reasoning returned ends with \n
    "nl_end": ("RECOVERED", lambda r: bool(r["reasoning_nl_end"])),
    "nl_end_with_tool_call": ("RECOVERED", lambda r: bool(r["reasoning_nl_end"]) and r["tool_calls"] > 0),
    # a bare unclosed "<tool_call>": the lenient parser returns reasoning None (a closed "</think><tool_call>" gives "")
    "reasoning_none_with_tool_call": ("RECOVERED", lambda r: r["reasoning_len"] is None and r["tool_calls"] > 0),
    "none_or_nl_end_with_tool_call": ("RECOVERED", lambda r: r["tool_calls"] > 0 and (
        r["reasoning_len"] is None or bool(r["reasoning_nl_end"]))),
    # unclosed think without a tool call: all output is reasoning, no content, no tool call
    "notool_no_content": ("UNCLOSED_NO_TOOL", lambda r: r["tool_calls"] == 0 and not r["content_len"]),
    "notool_no_content_reasoning_text": ("UNCLOSED_NO_TOOL", lambda r: r["tool_calls"] == 0 and not r["content_len"]
                                         and bool(r["reasoning_len"])),
}


def rate(n, d):
    return round(100.0 * n / d, 2) if d else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["sftv2e10", "sftv2e20", "sftv2", "sc14"])
    parser.add_argument("--procs", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for model in args.models:
        model_slots = slots(model)
        marks, versions, counts = parse_logs(FLEET.get(model, model))
        with ProcessPoolExecutor(args.procs) as pool:
            scanned = list(pool.map(scan_attempt, model_slots, chunksize=2))
        rows = []
        per_slot = []
        id_seen = set()
        for slot, calls, meta in scanned:
            flags = collections.Counter()
            for call in calls:
                rid = (call["id"] or "")[len("chatcmpl-"):] if (call["id"] or "").startswith("chatcmpl-") else call["id"]
                mark = marks.get(rid)
                id_seen.add(rid)
                call["log_kind"] = mark[0] if mark else None
                call["log_chars"] = mark[1] if mark else None
                call["log_file"] = mark[2] if mark else None
                call["empty_output"] = (not call["reasoning_len"]) and (not call["content_len"]) and not call["tool_calls"]
                call["one_token_stop"] = call["completion_tokens"] == 1 and call["finish"] == "stop"
                flags["recovered"] += call["log_kind"] == "RECOVERED"
                flags["unclosed_no_tool"] += call["log_kind"] == "UNCLOSED_NO_TOOL"
                flags["unclosed_no_tool_empty"] += call["log_kind"] == "UNCLOSED_NO_TOOL" and call["log_chars"] == 0
                flags["logged_tool_in_closed_think"] += call["log_kind"] == "TOOL_IN_CLOSED_THINK"
                flags["tool_in_reasoning"] += call["tool_in_reasoning"]
                flags["empty_output"] += call["empty_output"]
                flags["one_token_stop"] += call["one_token_stop"]
                flags["calls"] += 1
                rows.append(dict(call, task_id=slot["task_id"], slot=slot["slot"], group=slot["group"],
                                 status=slot["status"]))
            per_slot.append({"task_id": slot["task_id"], "slot": slot["slot"], "status": slot["status"],
                             "requests": meta["requests"], "non200": meta["non200"],
                             "non200_status": meta.get("non200_status") or {},
                             "http_missing": meta["http_missing"], **flags})
        with open(OUT / f"calls-{model}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        n_calls = len(rows)
        n_slots = len(per_slot)
        keys = ("recovered", "unclosed_no_tool", "unclosed_no_tool_empty", "logged_tool_in_closed_think",
                "tool_in_reasoning", "empty_output", "one_token_stop")
        out = {"slots": n_slots, "slots_without_http": sum(s["http_missing"] for s in per_slot),
               "calls": n_calls, "calls_per_slot_mean": round(n_calls / n_slots, 1) if n_slots else None,
               "requests": sum(s["requests"] for s in per_slot), "non200_responses": sum(s["non200"] for s in per_slot),
               "non200_by_status": dict(sum((collections.Counter(s["non200_status"]) for s in per_slot),
                                            collections.Counter())),
               "non200_slots": [f"{s['task_id']}/{s['slot']}: {s['non200_status']}" for s in per_slot if s["non200"]],
               "per_call_pct": {k: rate(sum(s.get(k, 0) for s in per_slot), n_calls) for k in keys},
               "per_call_n": {k: sum(s.get(k, 0) for s in per_slot) for k in keys},
               "rollouts_with_ge1_pct": {k: rate(sum(1 for s in per_slot if s.get(k)), n_slots) for k in keys},
               "rollouts_with_ge1_n": {k: sum(1 for s in per_slot if s.get(k)) for k in keys}}
        if marks:
            all_paths = sorted((TA / "probes" / model).glob(f"group-*/models/{model}/attempts/*/attempt-*/http.jsonl"))
            with ProcessPoolExecutor(args.procs) as pool:
                all_ids = set().union(*pool.map(attempt_ids, all_paths, chunksize=2)) if all_paths else set()
            stripped = {i[len("chatcmpl-"):] for i in all_ids if i.startswith("chatcmpl-")}
            out["mapping_check"] = {"attempts_scanned": len(all_paths), "response_ids": len(all_ids),
                                    "response_ids_with_chatcmpl_prefix": len(stripped),
                                    "log_ids": len(marks), "log_ids_found_in_any_attempt": sum(r in stripped for r in marks)}
            post = [r for r in rows if r["tool_in_reasoning"]]
            out["tool_in_reasoning_vs_log"] = {
                "posthoc_calls": len(post),
                "posthoc_with_log_TOOL_IN_CLOSED_THINK": sum(r["log_kind"] == "TOOL_IN_CLOSED_THINK" for r in post),
                "log_TOOL_IN_CLOSED_THINK_calls": sum(r["log_kind"] == "TOOL_IN_CLOSED_THINK" for r in rows),
                "log_TOOL_IN_CLOSED_THINK_posthoc_true": sum(r["log_kind"] == "TOOL_IN_CLOSED_THINK" and r["tool_in_reasoning"]
                                                            for r in rows)}
            kinds = collections.Counter(kind for kind, _, _ in marks.values())
            matched = collections.Counter(kind for rid, (kind, _, _) in marks.items() if rid in id_seen)
            out["log_lines"] = dict(kinds)
            out["log_lines_matched_to_final_attempt_calls"] = dict(matched)
            out["log_versions"] = collections.Counter(versions.values())
            # parser-version coverage of the final-attempt calls that carry a log mark
            out["counts_lines_last_per_process"] = {f"{log}:{pid}": c for (log, pid), c in sorted(counts.items())}
            # signature validation vs the logged recoveries (every call of the final attempts)
            for name, (kind, sig) in SIGNATURES.items():
                tp = sum(1 for r in rows if sig(r) and r["log_kind"] == kind)
                fp = sum(1 for r in rows if sig(r) and r["log_kind"] != kind)
                fn = sum(1 for r in rows if not sig(r) and r["log_kind"] == kind)
                out.setdefault("signature_validation", {})[name] = {
                    "target": kind, "tp": tp, "fp": fp, "fn": fn, "precision": rate(tp, tp + fp),
                    "recall": rate(tp, tp + fn)}
            shape = lambda r: ("tool" if r["tool_calls"] else "notool",  # noqa: E731
                               "r=None" if r["reasoning_len"] is None else ("r=''" if r["reasoning_len"] == 0 else
                                                                            ("r..\\n" if r["reasoning_nl_end"] else "r..text")),
                               "c=None" if r["content_len"] is None else ("c=''" if r["content_len"] == 0 else "c=text"))
            out["shape_recovered"] = {" ".join(k): v for k, v in collections.Counter(
                shape(r) for r in rows if r["log_kind"] == "RECOVERED").most_common()}
            out["shape_unclosed_no_tool"] = {" ".join(k): v for k, v in collections.Counter(
                shape(r) for r in rows if r["log_kind"] == "UNCLOSED_NO_TOOL").most_common()}
            out["shape_unlogged"] = {" ".join(k): v for k, v in collections.Counter(
                shape(r) for r in rows if r["log_kind"] is None).most_common()}
            out["recovered_log_reasoning_chars"] = dict(collections.Counter(
                "0" if not r["log_chars"] else ("1-9" if r["log_chars"] < 10 else ">=10") for r in rows
                if r["log_kind"] == "RECOVERED"))
        # signature counts (any model): per call and rollouts with >= 1
        by_slot = collections.defaultdict(list)
        for r in rows:
            by_slot[(r["task_id"], r["slot"])].append(r)
        out["signature_counts"] = {name: {"calls": sum(1 for r in rows if sig(r)),
                                          "per_call_pct": rate(sum(1 for r in rows if sig(r)), n_calls),
                                          "rollouts_with_ge1": sum(1 for calls in by_slot.values() if any(sig(r) for r in calls)),
                                          "rollouts_with_ge1_pct": rate(sum(1 for calls in by_slot.values()
                                                                            if any(sig(r) for r in calls)), n_slots)}
                                   for name, (_, sig) in SIGNATURES.items()}
        summary[model] = out
        summary_path.write_text(json.dumps(summary, indent=1, default=dict) + "\n")
        print(model, json.dumps(out, default=dict)[:3000], flush=True)


if __name__ == "__main__":
    main()
