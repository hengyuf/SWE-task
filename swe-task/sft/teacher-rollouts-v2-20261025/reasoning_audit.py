#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-26): is an "unreasoned" teacher turn really unreasoned, or was reasoning lost between
the hub and the OpenCode export? Audits the raw hub replies (ATP's http.jsonl "response" events, status 200) of the
final attempts that features.py analysed (features/<source>.jsonl attempt_dir). Read-only.

Per reply: tool calls (message.tool_calls), reasoning (message.reasoning_content / reasoning, non-empty after strip),
visible content, usage.completion_tokens and, when the hub reports it, completion_tokens_details.reasoning_tokens.
Per source: share of tool-call replies that carry reasoning (compare: features' pct_action_turns_with_reasoning over
the same attempts), and for tool-call replies WITHOUT reasoning text: share/size of hub-reported reasoning tokens
(hidden reasoning would show here) and completion tokens vs the visible output (content + tool-call JSON, chars / 4),
with the same excess on reasoned replies ((reasoning + visible chars) / 4) as the control for format overhead.

    /usr/bin/python3 -I -B reasoning_audit.py [--sources glm deepseek] [--campaigns full] [--workers 16]
      -> prints one JSON per source; --out PATH writes them all
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent


def attempt_replies(attempt: str) -> list:
    """[(tool_calls, has_reasoning, has_content, completion_tokens, reported_reasoning or None, visible_chars,
    reasoning_chars)]."""
    out = []
    path = Path(attempt) / "http.jsonl"
    if not path.exists():
        return out
    with open(path, errors="replace") as stream:
        for line in stream:
            if '"event":"response"' not in line[:1500]:
                continue
            try:
                event = json.loads(line)
                if event.get("status") != 200:
                    continue
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            choices = body.get("choices") or []
            if not choices:
                continue
            message = choices[0].get("message") or {}
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []
            usage = body.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            visible = len(content) + sum(len(json.dumps(call.get("function") or {})) for call in calls)
            out.append((len(calls), bool(str(reasoning).strip()), bool(str(content).strip()),
                        usage.get("completion_tokens") or 0, details.get("reasoning_tokens"), visible,
                        len(str(reasoning))))
    return out


def summarize(source: str, rows: list, replies: list) -> dict:
    tool = [r for r in replies if r[0]]
    unreasoned = [r for r in tool if not r[1]]
    reasoned = [r for r in tool if r[1]]
    reported = [r for r in unreasoned if r[4] is not None]
    ex = [r for r in rows if r.get("export") and r.get("turns")]

    def med(xs):
        return statistics.median(xs) if xs else None

    return {
        "source": source, "attempts": len(rows), "replies_200": len(replies), "tool_call_replies": len(tool),
        "pct_tool_call_replies_with_reasoning": round(100 * len(reasoned) / len(tool), 1) if tool else None,
        "export_pct_action_turns_with_reasoning": round(100 * sum(r["action_with_reasoning"] for r in ex) /
                                                        max(1, sum(r["action_turns"] for r in ex)), 1),
        "export_action_turns": sum(r["action_turns"] for r in ex),
        "unreasoned_tool_replies": len(unreasoned),
        "unreasoned_with_reported_reasoning_field": len(reported),
        "unreasoned_reported_reasoning_tokens_gt0_pct": round(100 * sum((r[4] or 0) > 0 for r in reported) / len(reported), 1)
        if reported else None,
        "unreasoned_reported_reasoning_tokens_mean": round(statistics.fmean([r[4] or 0 for r in reported]), 1)
        if reported else None,
        "reasoned_reported_reasoning_tokens_median": med([r[4] for r in reasoned if r[4] is not None]),
        "completion_tokens_median_unreasoned": med([r[3] for r in unreasoned]),
        "completion_tokens_median_reasoned": med([r[3] for r in reasoned]),
        "visible_tokens_est_median_unreasoned": med([r[5] / 4 for r in unreasoned]),
        "excess_completion_tokens_median_unreasoned": med([r[3] - r[5] / 4 for r in unreasoned]),
        # control: the same estimate on reasoned replies, (reasoning + visible chars) / 4; equal medians = format overhead
        "excess_completion_tokens_median_reasoned": med([r[3] - (r[5] + r[6]) / 4 for r in reasoned]),
        "pct_unreasoned_with_visible_content": round(100 * sum(r[2] for r in unreasoned) / len(unreasoned), 1)
        if unreasoned else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["glm", "deepseek"])
    parser.add_argument("--campaigns", choices=("full", "pilot", "all"), default="full")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out")
    args = parser.parse_args()
    results = []
    for source in args.sources:
        rows = [json.loads(line) for line in (HERE / "features" / f"{source}.jsonl").read_text().splitlines() if line]
        if args.campaigns != "all":
            rows = [r for r in rows if ((r.get("campaign") or "").endswith("-pilot")) == (args.campaigns == "pilot")]
        attempts = [r["attempt_dir"] for r in rows if r.get("attempt_dir")]
        with ProcessPoolExecutor(args.workers) as pool:
            replies = [reply for chunk in pool.map(attempt_replies, attempts, chunksize=2) for reply in chunk]
        results.append(summarize(source, rows, replies))
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
