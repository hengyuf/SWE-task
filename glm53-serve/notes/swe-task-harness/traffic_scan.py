#!/usr/bin/env python3
"""Per-call traffic of GLM-5.3 rollouts, from ATP's per-attempt HTTP journals (read-only on every input).

    /usr/bin/python3 -I -B traffic_scan.py scan [--workers 6]          # hub baselines -> cache/{tr2,tdw}/<hash>.json
    /usr/bin/python3 -I -B traffic_scan.py scan --label local1 --results R.json [R2.json ...]   # e.g. the local smoke
    /usr/bin/python3 -I -B traffic_scan.py scan --label local1 --probe-dir P [P2 ...]          # no results file yet
    /usr/bin/python3 -I -B traffic_scan.py report [--labels tr2,tdw,local1]   # -> traffic-stats.json (+ stdout)

Hub sources (graded slots only, final attempt of each slot):
  tr2 = swe-task/sft/teacher-rollouts-v2-20261025  results/glm-group-{0..7}.json  (50 synthetic tasks x 5,
        OpenCode context 1,048,576 = never compacts, per-call max_tokens 131,072)
  tdw = swe-task/sft/teacher-deepswe-20261030      dataset/glm.jsonl.gz           (DeepSWE v1.1, 264 graded slots,
        OpenCode context 262,144 = compaction at 242,144, per-call max_tokens 131,072)
--results: driver results files in the teacher-rollouts-v2 shape ({"records": {task_id: [{slot, status,
attempt_dir}]}}). --probe-dir: an ATP probe campaign; every slot's newest attempt-<n> with an http.jsonl is used and its
status is approximated from terminal.json (reward 1 -> pass, evaluation_completed -> fail, else infra).
http.jsonl: one "request" event (full chat payload) and one "response" event (status, body) per upstream HTTP attempt,
paired by (call_id, http_attempt). Nothing outside this script's directory is written.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
TR2 = S / "sft/teacher-rollouts-v2-20261025"
TDW = S / "sft/teacher-deepswe-20261030"
COMPACTION_SYS = "You are an anchored context summarization assistant"
TOOL_ERR = ("was called with invalid arguments", "Invalid input", "unavailable tool", "Invalid JSON", "invalid_type")
LEAK = ("<tool_call>", "<arg_key>", "<arg_value>", "</think>", "<think>", "<|observation|>", "<|user|>",
        "<|assistant|>", "[PAD]")


def slots():
    """[(campaign, task_id, slot, status, attempt_dir)] of the graded GLM slots of both hub campaigns."""
    out = []
    for path in sorted((TR2 / "results").glob("glm-group-*.json")):
        out += results_slots("tr2", path)
    with gzip.open(TDW / "dataset/glm.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            out.append(("tdw", row["task_id"], row["slot"], row["status"], row["attempt_dir"]))
    return out


def results_slots(label, path):
    data = json.loads(Path(path).read_text())
    return [(label, task_id, record["slot"], record["status"], record["attempt_dir"])
            for task_id, records in data["records"].items() for record in records if record.get("attempt_dir")]


def probe_slots(label, probe_dir):
    """Newest attempt with an http.jsonl per slot of an ATP probe campaign (status approximated, see docstring)."""
    probe_dir = Path(probe_dir)
    manifest = json.loads((probe_dir / "manifest.json").read_text())
    names = {task["_ng_task_index"]: task["task_id"] for task in manifest["tasks"]}
    out = []
    for slot_dir in sorted(probe_dir.glob("models/*/attempts/*-*")):
        attempts = sorted((d for d in slot_dir.glob("attempt-*") if (d / "http.jsonl").exists()),
                          key=lambda d: int(d.name.split("-")[1]))
        if not attempts:
            continue
        attempt = attempts[-1]
        try:
            terminal = json.loads((attempt / "terminal.json").read_text())
        except (OSError, ValueError):
            terminal = {}
        status = ("pass" if terminal.get("reward") == 1 else
                  "fail" if terminal.get("evaluation_completed") else "infra")
        index, repeat = slot_dir.name.split("-")
        alias = slot_dir.parents[1].name
        out.append((label, names.get(int(index), index), f"{alias}-r{repeat}", status, str(attempt)))
    return out


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else json.dumps(content)


def scan_attempt(item):
    campaign, task_id, slot, status, attempt_dir = item
    key = hashlib.sha1(attempt_dir.encode()).hexdigest()[:16]
    target = CACHE / campaign / f"{key}.json"
    if target.exists():
        return str(target)
    requests, calls, statuses = {}, [], collections.Counter()
    arg_types, error_bodies = collections.Counter(), collections.Counter()
    path = Path(attempt_dir) / "http.jsonl"
    with open(path, "rb") as stream:
        for raw in stream:
            head = raw[:2500]
            is_request = b'"event":"request"' in head or b'"event": "request"' in head
            is_response = b'"event":"response"' in head or b'"event": "response"' in head
            if not (is_request or is_response):
                continue
            event = json.loads(raw)
            if event.get("event") == "request":
                payload = event.get("payload") or {}
                messages = payload.get("messages") or []
                system = _text(messages[0].get("content")) if messages and messages[0].get("role") == "system" else ""
                kind = "compaction" if system.startswith(COMPACTION_SYS) else (
                    "main" if system.startswith("You are opencode") else "other:" + system[:48])
                # messages after the last assistant turn = what this call adds (tool results / user turn)
                tail_start = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1) + 1
                tail = messages[tail_start:]
                tool_errors = sum(1 for m in tail if m.get("role") == "tool"
                                  and any(s in _text(m.get("content"))[:4000] for s in TOOL_ERR))
                history_reasoning = sum(1 for m in messages if m.get("role") == "assistant"
                                        and (m.get("reasoning_content") or m.get("reasoning")))
                history_think_in_content = sum(1 for m in messages if m.get("role") == "assistant"
                                               and "</think>" in _text(m.get("content")))
                requests[(event["call_id"], event.get("http_attempt"))] = {
                    "t0": event["timestamp"], "kind": kind, "nmsg": len(messages),
                    "ntools": len(payload.get("tools") or []), "max_tokens": payload.get("max_tokens"),
                    "stream": bool(payload.get("stream")), "req_bytes": len(raw), "tool_errors": tool_errors,
                    "hist_reasoning_field": history_reasoning, "hist_think_in_content": history_think_in_content,
                    "params": sorted(k for k in payload if k not in ("messages", "tools")),
                    "reasoning_effort": payload.get("reasoning_effort"), "temperature": payload.get("temperature"),
                    "top_p": payload.get("top_p"), "tool_choice": payload.get("tool_choice")}
                continue
            status_code = event.get("status")
            statuses[str(status_code)] += 1
            request = requests.pop((event["call_id"], event.get("http_attempt")), None) or {}
            row = dict(request, t1=event["timestamp"], status=status_code)
            row["lat"] = (event["timestamp"] - request["t0"]) if request else None
            if status_code == 200:
                try:
                    body = json.loads(event.get("body") or "{}")
                except ValueError:
                    body = {}
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                content = content if isinstance(content, str) else json.dumps(content)
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                row.update(
                    pt=usage.get("prompt_tokens"), ct=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    out=usage.get("completion_tokens"), rt=details.get("reasoning_tokens"),
                    acc=details.get("accepted_prediction_tokens"), rej=details.get("rejected_prediction_tokens"),
                    fin=choice.get("finish_reason"), ntc=len(message.get("tool_calls") or []),
                    clen=len(content), rlen=len(reasoning),
                    rkey=("reasoning_content" if message.get("reasoning_content") else
                          "reasoning" if message.get("reasoning") else None),
                    leak=[s for s in LEAK if s in content],
                    upstream=event.get("upstream_transport"))
                bad_args = 0
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    try:
                        args = json.loads(function.get("arguments") or "{}")
                    except ValueError:
                        bad_args += 1
                        arg_types[f"{function.get('name')}:<unparsable>"] += 1
                        continue
                    for name, value in (args.items() if isinstance(args, dict) else []):
                        arg_types[f"{function.get('name')}:{name}:{type(value).__name__}"] += 1
                row["bad_args_json"] = bad_args
            else:
                # first 160 chars of a non-200 body (e.g. vLLM's 400 "maximum context length"), no request content
                error_bodies[f"{status_code}: {(event.get('body') or '')[:160]}"] += 1
            calls.append(row)
    result = {"campaign": campaign, "task_id": task_id, "slot": slot, "status": status, "attempt_dir": attempt_dir,
              "statuses": dict(statuses), "unanswered_requests": len(requests), "calls": calls,
              "tool_arg_types": dict(arg_types), "error_bodies": dict(error_bodies.most_common(20))}
    try:  # ATP's verdict on the agent timeout (progress.agent_timeout's first source)
        validity = json.loads((Path(attempt_dir) / "generation-validity-v2-r7.json").read_text())
        result["budget_timeout"] = validity.get("budget_timeout")
    except (OSError, ValueError):
        result["budget_timeout"] = None
    for name in ("started.json", "terminal.json"):
        try:
            result[name.split(".")[0] + "_mtime"] = (Path(attempt_dir) / name).stat().st_mtime
        except OSError:
            result[name.split(".")[0] + "_mtime"] = None
    try:
        export = json.loads((Path(attempt_dir) / "export.json").read_text())
        result["export_compactions"] = sum(1 for m in export.get("messages") or []
                                           if (m.get("info") or {}).get("mode") == "compaction"
                                           or (m.get("info") or {}).get("summary") is True)
    except (OSError, ValueError):
        result["export_compactions"] = None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name("." + target.name + ".tmp")
    tmp.write_text(json.dumps(result, separators=(",", ":")))
    os.replace(tmp, target)
    return str(target)


def scan(workers, label=None, results=(), probe_dirs=()):
    from concurrent.futures import ProcessPoolExecutor
    if label:
        items = [item for path in results for item in results_slots(label, path)]
        items += [item for path in probe_dirs for item in probe_slots(label, path)]
    else:
        items = slots()
    print(f"{len(items)} graded slots", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(scan_attempt, items, chunksize=1):
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(items)}", flush=True)
    print("scan complete", flush=True)


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))
    return values[index]


def dist(values, digits=0):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    fmt = (lambda v: round(v, digits)) if digits else (lambda v: int(round(v)))
    return {"n": len(values), "mean": fmt(statistics.mean(values)), "p10": fmt(pct(values, 10)),
            "p50": fmt(pct(values, 50)), "p90": fmt(pct(values, 90)), "p99": fmt(pct(values, 99)),
            "max": fmt(max(values))}


def report(labels=("tr2", "tdw")):
    out = {}
    for campaign in labels:
        files = sorted((CACHE / campaign).glob("*.json"))
        attempts = [json.loads(path.read_text()) for path in files]
        arg_types, error_bodies = collections.Counter(), collections.Counter()
        for attempt in attempts:
            arg_types.update(attempt.get("tool_arg_types") or {})
            error_bodies.update(attempt.get("error_bodies") or {})
        overflow_rollouts = sum(1 for attempt in attempts if any(
            "maximum context length" in body for body in (attempt.get("error_bodies") or {})))
        ok_calls, per_rollout, statuses, kinds = [], [], collections.Counter(), collections.Counter()
        finish, rkeys, leaks, others = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        params, efforts = collections.Counter(), collections.Counter()
        reasoning_kept_pairs = reasoning_dropped_pairs = 0
        for attempt in attempts:
            calls = [c for c in attempt["calls"] if c.get("status") == 200 and c.get("pt") is not None]
            statuses.update(attempt["statuses"])
            for c in calls:
                kinds[c.get("kind", "?") if not str(c.get("kind", "")).startswith("other:") else "other"] += 1
                if str(c.get("kind", "")).startswith("other:"):
                    others[c["kind"]] += 1
                finish[c.get("fin")] += 1
                rkeys[c.get("rkey")] += 1
                for s in c.get("leak") or []:
                    leaks[s] += 1
                params[",".join(c.get("params") or [])] += 1
                efforts[str(c.get("reasoning_effort"))] += 1
            ok_calls += [dict(c, _attempt=attempt["attempt_dir"], _status=attempt["status"]) for c in calls]
            main = sorted((c for c in calls if c.get("kind") == "main"), key=lambda c: c["t0"])
            for a, b in zip(main, main[1:]):
                if b["nmsg"] > a["nmsg"] and a.get("rt"):
                    delta = b["pt"] - a["pt"]
                    if delta >= a["out"] - 8:
                        reasoning_kept_pairs += 1
                    elif delta < a["out"] - a["rt"] + 8 and b["pt"] < a["pt"] + a["out"]:
                        reasoning_dropped_pairs += 1
            if not calls:
                continue
            t0 = min(c["t0"] for c in calls if c.get("t0"))
            t1 = max(c["t1"] for c in calls)
            wall = t1 - t0
            busy = sum(c["lat"] for c in calls if c.get("lat") is not None)
            # time-weighted live context: each call's prompt+output is held until the next call starts
            ordered = sorted(calls, key=lambda c: c["t0"])
            weighted = 0.0
            for a, b in zip(ordered, ordered[1:]):
                weighted += (a["pt"] + (a["out"] or 0)) * max(0.0, b["t0"] - a["t0"])
            live_ctx = weighted / max(1e-9, ordered[-1]["t0"] - ordered[0]["t0"]) if len(ordered) > 1 else ordered[0]["pt"]
            compaction_calls = [c for c in calls if c.get("kind") == "compaction"]
            per_rollout.append({
                "status": attempt["status"], "calls": len(calls),
                "http_attempts": sum(attempt["statuses"].values()),
                "non200": sum(v for k, v in attempt["statuses"].items() if k != "200"),
                "prompt_sum": sum(c["pt"] for c in calls), "cached_sum": sum(c.get("ct") or 0 for c in calls),
                "out_sum": sum(c["out"] or 0 for c in calls), "reason_sum": sum(c.get("rt") or 0 for c in calls),
                "max_prompt": max(c["pt"] for c in calls), "max_ctx": max(c["pt"] + (c["out"] or 0) for c in calls),
                "wall_min": wall / 60, "busy_frac": busy / wall if wall > 0 else None,
                "calls_per_min": len(calls) / (wall / 60) if wall > 0 else None,
                "mean_prompt": statistics.mean(c["pt"] for c in calls), "live_ctx": live_ctx,
                "compaction_calls": len(compaction_calls),
                "export_compactions": attempt.get("export_compactions"),
                "slot_wall_min": ((attempt["terminal_mtime"] - attempt["started_mtime"]) / 60
                                  if attempt.get("terminal_mtime") and attempt.get("started_mtime") else None),
                "tool_errors": sum(c.get("tool_errors") or 0 for c in calls),
                "over_131072": sum(c["pt"] > 131072 for c in calls),
                "over_262144_minus_out": sum(c["pt"] + 131072 > 262144 for c in calls)})
        main_calls = [c for c in ok_calls if c.get("kind") == "main"]
        comp_calls = [c for c in ok_calls if c.get("kind") == "compaction"]
        n = len(ok_calls)
        big_out = [c for c in ok_calls if (c["out"] or 0) >= 1000 and c.get("lat")]
        acc = sum(c.get("acc") or 0 for c in ok_calls)
        rej = sum(c.get("rej") or 0 for c in ok_calls)
        stats = {
            "rollouts": len(per_rollout),
            "rollout_status": dict(collections.Counter(r["status"] for r in per_rollout)),
            "calls_200": n, "http_status_counts": dict(statuses),
            "call_kinds": dict(kinds), "other_system_prompts": dict(others.most_common(10)),
            "request_param_sets": dict(params.most_common(5)), "reasoning_effort_values": dict(efforts),
            "finish_reasons": dict(finish), "response_reasoning_key": {str(k): v for k, v in rkeys.items()},
            "content_leak_markers": dict(leaks),
            "per_rollout": {
                "calls": dist([r["calls"] for r in per_rollout]),
                "http_attempts_incl_retries": dist([r["http_attempts"] for r in per_rollout]),
                "non200_replies": dist([r["non200"] for r in per_rollout]),
                "prompt_tokens_sum": dist([r["prompt_sum"] for r in per_rollout]),
                "uncached_prompt_tokens_sum": dist([r["prompt_sum"] - r["cached_sum"] for r in per_rollout]),
                "completion_tokens_sum": dist([r["out_sum"] for r in per_rollout]),
                "reasoning_tokens_sum": dist([r["reason_sum"] for r in per_rollout]),
                "max_prompt_tokens": dist([r["max_prompt"] for r in per_rollout]),
                "max_prompt_plus_output": dist([r["max_ctx"] for r in per_rollout]),
                "mean_prompt_tokens": dist([r["mean_prompt"] for r in per_rollout]),
                "time_weighted_live_context_tokens": dist([r["live_ctx"] for r in per_rollout]),
                "llm_span_wall_min": dist([r["wall_min"] for r in per_rollout], 1),
                "slot_wall_min_started_to_terminal": dist([r["slot_wall_min"] for r in per_rollout], 1),
                "llm_busy_fraction": dist([r["busy_frac"] for r in per_rollout], 3),
                "calls_per_min": dist([r["calls_per_min"] for r in per_rollout], 2),
                "tool_arg_errors": dist([r["tool_errors"] for r in per_rollout]),
                "compaction_calls": dist([r["compaction_calls"] for r in per_rollout]),
            },
            "rollouts_with_compaction_call": sum(r["compaction_calls"] > 0 for r in per_rollout),
            "rollouts_with_export_compaction": sum(bool(r["export_compactions"]) for r in per_rollout),
            "rollouts_max_prompt_over_242144": sum(r["max_prompt"] > 242144 for r in per_rollout),
            "rollouts_max_prompt_over_131072": sum(r["max_prompt"] > 131072 for r in per_rollout),
            "rollouts_max_prompt_over_262144": sum(r["max_prompt"] > 262144 for r in per_rollout),
            "rollouts_max_ctx_over_393216": sum(r["max_ctx"] > 393216 for r in per_rollout),
            "per_call_all": {
                "prompt_tokens": dist([c["pt"] for c in ok_calls]),
                "cached_tokens": dist([c.get("ct") for c in ok_calls]),
                "uncached_prompt_tokens": dist([c["pt"] - (c.get("ct") or 0) for c in ok_calls]),
                "completion_tokens": dist([c["out"] for c in ok_calls]),
                "reasoning_tokens": dist([c.get("rt") for c in ok_calls]),
                "hub_latency_s": dist([c.get("lat") for c in ok_calls], 1),
                "hub_decode_tok_per_s_calls_ge_1000_out": dist([c["out"] / c["lat"] for c in big_out if c["lat"] > 0], 1),
                "messages_per_request": dist([c.get("nmsg") for c in ok_calls]),
                "request_bytes": dist([c.get("req_bytes") for c in ok_calls]),
                "tool_calls_per_response": dist([c.get("ntc") for c in ok_calls], 2),
            },
            "per_call_main": {
                "prompt_tokens": dist([c["pt"] for c in main_calls]),
                "completion_tokens": dist([c["out"] for c in main_calls]),
            },
            "compaction_calls": {
                "count": len(comp_calls), "share_of_calls": round(len(comp_calls) / n, 5) if n else None,
                "prompt_tokens": dist([c["pt"] for c in comp_calls]),
                "cached_tokens": dist([c.get("ct") for c in comp_calls]),
                "completion_tokens": dist([c["out"] for c in comp_calls]),
                "hub_latency_s": dist([c.get("lat") for c in comp_calls], 1),
            },
            "share_calls_prompt_over": {str(t): round(sum(c["pt"] > t for c in ok_calls) / n, 4) if n else None
                                        for t in (65536, 131072, 200000, 242144, 262144)},
            "calls_prompt_plus_131072_over_262144": sum(c["pt"] + 131072 > 262144 for c in ok_calls),
            "share_calls_out_over": {str(t): round(sum((c["out"] or 0) > t for c in ok_calls) / n, 5) if n else None
                                     for t in (4096, 16384, 32768, 65536, 131071)},
            "max_completion_tokens_seen": max((c["out"] or 0) for c in ok_calls) if ok_calls else None,
            "token_totals": {
                "prompt": sum(c["pt"] for c in ok_calls), "cached": sum(c.get("ct") or 0 for c in ok_calls),
                "completion": sum(c["out"] or 0 for c in ok_calls),
                "reasoning": sum(c.get("rt") or 0 for c in ok_calls),
                "cached_share": round(sum(c.get("ct") or 0 for c in ok_calls) / max(1, sum(c["pt"] for c in ok_calls)), 4),
                "reasoning_share_of_completion": round(sum(c.get("rt") or 0 for c in ok_calls)
                                                       / max(1, sum(c["out"] or 0 for c in ok_calls)), 4)},
            "hub_mtp": {"accepted": acc, "rejected": rej,
                        "acceptance": round(acc / (acc + rej), 4) if acc + rej else None,
                        "accepted_per_completion_token": round(acc / max(1, sum(c["out"] or 0 for c in ok_calls)), 4)},
            "history_reasoning_check": {
                "pairs_prompt_grew_by_full_output (reasoning kept in history)": reasoning_kept_pairs,
                "pairs_prompt_grew_less_than_output (reasoning dropped?)": reasoning_dropped_pairs},
            "history_reasoning_sent_as_field_calls": sum(1 for c in ok_calls if c.get("hist_reasoning_field")),
            "history_think_in_content_calls": sum(1 for c in ok_calls if c.get("hist_think_in_content")),
            "tool_arg_error_tool_results": sum(c.get("tool_errors") or 0 for c in ok_calls),
            "tool_calls_with_unparsable_arguments": sum(c.get("bad_args_json") or 0 for c in ok_calls),
            "tool_arg_types_top": dict(arg_types.most_common(40)),
            "non200_bodies_top": dict(error_bodies.most_common(10)),
            "rollouts_with_context_length_400": overflow_rollouts,
            "rollouts_budget_timeout": sum(1 for attempt in attempts if attempt.get("budget_timeout")),
        }
        out[campaign] = stats
    target = HERE / ("traffic-stats.json" if tuple(labels) == ("tr2", "tdw") else
                     "traffic-stats-" + "-".join(labels) + ".json")
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--label", help="cache/<label>/ for --results / --probe-dir inputs (default: the hub baselines)")
    s.add_argument("--results", nargs="*", default=[])
    s.add_argument("--probe-dir", nargs="*", default=[])
    r = sub.add_parser("report")
    r.add_argument("--labels", default="tr2,tdw")
    args = parser.parse_args()
    if args.cmd == "scan":
        if (args.results or args.probe_dir) and not args.label:
            parser.error("--results / --probe-dir need --label")
        if args.label in ("tr2", "tdw"):
            parser.error("tr2 / tdw are the hub baselines' labels")
        scan(args.workers, args.label, args.results, args.probe_dir)
    else:
        report(tuple(label.strip() for label in args.labels.split(",") if label.strip()))


if __name__ == "__main__":
    sys.exit(main())
