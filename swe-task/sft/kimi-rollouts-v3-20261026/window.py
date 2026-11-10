#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): effects of the 262,144-token window per rollout (read-only on all inputs).

    /usr/bin/python3 -I -B window.py [--sources kimi3 kimi] [--workers 16]  -> analysis/window-<source>.jsonl + summary

Per slot (the final attempt, as in features.py):
  hub replies (http.jsonl "response" events = every upstream reply, main session, subagents and compaction calls):
    statuses; non-200 bodies classified as context_overflow / budget / rate_limit / other; replies with
    finish_reason "length" or completion_tokens >= 131072 (cut at the output cap); max and p50 prompt tokens per call;
    calls with prompt > 242,144 (OpenCode's compaction point at limit.input 262,144 - 20,000); LiteLLM usage.cost sum;
    degenerate replies ([PAD] / chat-template special tokens in the text, or no tool call and no text at all) and whether
    the LAST reply was one (no tool call: OpenCode ended the session on it)
  OpenCode session (export.json = the main session): compactions (assistant messages with mode "compaction" / summary)
    and the prompt size that triggered each (the previous assistant step's input + cache read tokens); after the LAST
    compaction: assistant turns, action turns, git commit commands, whether the session ended with a no-tool message, the
    last message's finish and error name; any assistant message error (name) anywhere; "early stop after compaction" =
    the session ended within 3 action turns of a compaction without a later git commit
Sources: kimi3 (this batch, sources.py) and kimi (the previous Kimi-K3 rollouts of the 50 tasks, features/kimi.jsonl).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(HERE))
import sources  # noqa: E402

COMPACT_AT = 262144 - 20000
OUTPUT_CAP = 131072
OVERFLOW = re.compile(r"(?i)context[ _-]?length|maximum context|context[ _]window|too many tokens|prompt is too long|"
                      r"exceeds? the (?:model'?s? )?(?:maximum|context)|input.{0,40}too long|max_model_len")
BUDGET = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|max budget")
COMMIT = re.compile(r"\bgit\b[^\n|;&]*\bcommit\b")
SPECIAL = re.compile(r"<\|(?:close|im_end|im_start|im_middle|tool_calls?_section_(?:begin|end)|tool_call_(?:begin|end)|tool_call_argument_begin)\|>")


def slot_list(source):
    if source == "kimi":
        return [json.loads(line) for line in (HERE / "features" / "kimi.jsonl").read_text().splitlines() if line.strip()]
    return sources.slots(source)


def replies(attempt: Path) -> dict:
    out = {"replies": 0, "statuses": {}, "errors": collections.Counter(), "error_samples": [], "length_cut": 0,
           "prompt_max": 0, "prompts": [], "over_compact_point": 0, "cost": 0.0, "degenerate_replies": 0,
           "last_reply_degenerate": False}
    path = attempt / "http.jsonl"
    if not path.exists():
        return out
    with open(path, "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            status = str(event.get("status"))
            out["replies"] += 1
            out["statuses"][status] = out["statuses"].get(status, 0) + 1
            body = event.get("body") or ""
            if status != "200":
                kind = ("context_overflow" if OVERFLOW.search(body) else "budget" if BUDGET.search(body)
                        else "rate_limit" if status == "429" else "other")
                out["errors"][f"{status}:{kind}"] += 1
                if len(out["error_samples"]) < 3 and status != "429":
                    out["error_samples"].append(f"{status} {body[:240]}")
                continue
            try:
                data = json.loads(body)
            except ValueError:
                continue
            usage = data.get("usage") or {}
            prompt = usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            choice = (data.get("choices") or [{}])[0] or {}
            finish = choice.get("finish_reason")
            message = choice.get("message") or {}
            text = str(message.get("content") or "") + str(message.get("reasoning_content") or message.get("reasoning") or "")
            calls = message.get("tool_calls") or []
            garbage = "[PAD]" in text + "".join(str((c.get("function") or {}).get("arguments")) for c in calls) \
                or bool(SPECIAL.search(text))
            degenerate = garbage or (not calls and not text.strip())
            out["degenerate_replies"] += degenerate
            out["last_reply_degenerate"] = degenerate and not calls
            out["prompts"].append(prompt)
            out["prompt_max"] = max(out["prompt_max"], prompt)
            out["over_compact_point"] += prompt > COMPACT_AT
            out["length_cut"] += finish == "length" or completion >= OUTPUT_CAP
            if isinstance(usage.get("cost"), (int, float)):
                out["cost"] += usage["cost"]
    prompts = out.pop("prompts")
    out["prompt_p50"] = statistics.median(prompts) if prompts else None
    out["errors"] = dict(out["errors"])
    return out


def session(attempt: Path) -> dict:
    try:
        export = json.loads((attempt / "export.json").read_text())
    except (OSError, ValueError):
        return {"export": False}
    messages = export.get("messages") or []
    turns = []  # main-session assistant steps, compactions marked
    previous_prompt = None
    for message in messages:
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        tokens = info.get("tokens") or {}
        prompt = (tokens.get("input") or 0) + ((tokens.get("cache") or {}).get("read") or 0)
        compaction = info.get("mode") == "compaction" or info.get("summary") is True
        tools = [p for p in message.get("parts") or [] if p.get("type") == "tool"]
        commands = [((p.get("state") or {}).get("input") or {}).get("command") or "" for p in tools]
        error = info.get("error")
        turns.append({"compaction": compaction, "trigger_prompt": previous_prompt if compaction else None,
                      "tools": len(tools), "commit": any(COMMIT.search(c) for c in commands if isinstance(c, str)),
                      "finish": info.get("finish"),
                      "error": (error.get("name") or "error") if isinstance(error, dict) else (str(error) if error else None)})
        if not compaction:
            previous_prompt = prompt
    compactions = [i for i, t in enumerate(turns) if t["compaction"]]
    out = {"export": True, "turns": sum(not t["compaction"] for t in turns), "compactions": len(compactions),
           "compaction_trigger_prompts": [t["trigger_prompt"] for t in turns if t["compaction"]],
           "errors": [t["error"] for t in turns if t["error"]],
           "last_finish": turns[-1]["finish"] if turns else None, "last_error": turns[-1]["error"] if turns else None,
           "ended_notool": bool(turns) and not turns[-1]["tools"] and not turns[-1]["compaction"],
           "commit_any": any(t["commit"] for t in turns)}
    if compactions:
        tail = turns[compactions[-1] + 1:]
        actions_after = sum(t["tools"] > 0 for t in tail)
        out.update({"turns_after_last_compaction": len(tail), "action_turns_after_last_compaction": actions_after,
                    "commit_after_last_compaction": any(t["commit"] for t in tail),
                    "early_stop_after_compaction": actions_after <= 3 and not any(t["commit"] for t in tail),
                    "ended_on_compaction": not tail})
    return out


def analyse(slot):
    attempt = Path(slot["attempt_dir"]) if slot.get("attempt_dir") else None
    row = {key: slot.get(key) for key in ("source", "task_id", "slot", "status", "reward", "attempt_dir")}
    if attempt is None:
        return row
    row.update(hub=replies(attempt), session=session(attempt))
    try:
        terminal = json.loads((attempt / "terminal.json").read_text())
    except (OSError, ValueError):
        terminal = {}
    row["agent_error"] = terminal.get("opencode_run_error_type") or terminal.get("failure_reason")
    row["patch_chars"] = (attempt / "model.patch").stat().st_size if (attempt / "model.patch").exists() else 0
    return row


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def summarize(rows):
    ex = [r for r in rows if (r.get("session") or {}).get("export")]
    graded = [r for r in rows if r.get("status") in ("pass", "fail")]
    comp = [r for r in ex if r["session"]["compactions"]]
    nocomp = [r for r in ex if not r["session"]["compactions"]]
    errors = collections.Counter()
    for r in rows:
        errors.update((r.get("hub") or {}).get("errors") or {})
    session_errors = collections.Counter(e for r in ex for e in r["session"]["errors"])
    trigger = [p for r in comp for p in r["session"]["compaction_trigger_prompts"] if p]
    graded_c = [r for r in comp if r["status"] in ("pass", "fail")]
    graded_n = [r for r in nocomp if r["status"] in ("pass", "fail")]
    return {
        "rollouts": len(rows), "graded": len(graded), "infra": len(rows) - len(graded), "with_export": len(ex),
        "pass_at_1": pct(sum(r["status"] == "pass" for r in graded), len(graded)),
        "pct_rollouts_compacted": pct(len(comp), len(ex)),
        "compactions_per_rollout_mean": round(statistics.fmean([r["session"]["compactions"] for r in ex]), 2) if ex else None,
        "compactions_per_compacted_rollout_mean": round(statistics.fmean([r["session"]["compactions"] for r in comp]), 2)
        if comp else None,
        "compactions_hist": dict(sorted(collections.Counter(r["session"]["compactions"] for r in ex).items())),
        "compaction_trigger_prompt_median": statistics.median(trigger) if trigger else None,
        "pass_at_1_compacted": pct(sum(r["status"] == "pass" for r in graded_c), len(graded_c)),
        "graded_compacted": len(graded_c),
        "pass_at_1_not_compacted": pct(sum(r["status"] == "pass" for r in graded_n), len(graded_n)),
        "graded_not_compacted": len(graded_n),
        "pct_compacted_with_commit_after_last_compaction": pct(sum(r["session"].get("commit_after_last_compaction", False)
                                                                   for r in comp), len(comp)),
        "pct_compacted_with_any_commit": pct(sum(r["session"]["commit_any"] for r in comp), len(comp)),
        "pct_not_compacted_with_any_commit": pct(sum(r["session"]["commit_any"] for r in nocomp), len(nocomp)),
        "compacted_early_stop": sum(r["session"].get("early_stop_after_compaction", False) for r in comp),
        "compacted_ended_on_compaction": sum(r["session"].get("ended_on_compaction", False) for r in comp),
        "compacted_ended_notool": sum(r["session"]["ended_notool"] for r in comp),
        "action_turns_after_last_compaction_median": statistics.median(
            [r["session"]["action_turns_after_last_compaction"] for r in comp]) if comp else None,
        "hub_replies": sum((r.get("hub") or {}).get("replies", 0) for r in rows),
        "hub_errors": dict(errors),
        "rollouts_with_context_overflow": sum(any(k.endswith("context_overflow") for k in (r.get("hub") or {}).get("errors", {}))
                                              for r in rows),
        "calls_cut_at_output_cap": sum((r.get("hub") or {}).get("length_cut", 0) for r in rows),
        "rollouts_with_cut_call": sum(bool((r.get("hub") or {}).get("length_cut")) for r in rows),
        "calls_over_compaction_point": sum((r.get("hub") or {}).get("over_compact_point", 0) for r in rows),
        "max_prompt_tokens": max(((r.get("hub") or {}).get("prompt_max", 0) for r in rows), default=None),
        "pct_rollouts_max_prompt_over_242k": pct(sum((r.get("hub") or {}).get("prompt_max", 0) > COMPACT_AT for r in rows),
                                                 len(rows)),
        "degenerate_hub_replies": sum((r.get("hub") or {}).get("degenerate_replies", 0) for r in rows),
        "rollouts_ended_by_degenerate_reply": sum(bool((r.get("hub") or {}).get("last_reply_degenerate")) for r in rows),
        "pass_at_1_excluding_degenerate_end": pct(sum(r["status"] == "pass" for r in graded
                                                      if not (r.get("hub") or {}).get("last_reply_degenerate")),
                                                  sum(1 for r in graded if not (r.get("hub") or {}).get("last_reply_degenerate"))),
        "session_errors": dict(session_errors),
        "agent_errors": dict(collections.Counter(str(r.get("agent_error")) for r in rows if r.get("agent_error"))),
        "cost_reported_usd_mean": round(statistics.fmean([(r.get("hub") or {}).get("cost", 0) for r in rows]), 3) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["kimi3", "kimi"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks", nargs="*", help="restrict to these task ids")
    args = parser.parse_args()
    summary = {}
    for source in args.sources:
        slots = [s for s in slot_list(source) if not args.tasks or s["task_id"] in set(args.tasks)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(analyse, slots, chunksize=2))
        path = HERE / "analysis" / f"window-{source}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary[source] = summarize(rows)
    (HERE / "analysis" / "window-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
