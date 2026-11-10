#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): classify every successful hub reply by how it ends (read-only).

    /usr/bin/python3 -I -B stops.py [--campaigns sel50-g0 ...] [--previous] [--out FILE]

For each 200 "response" event of each attempt's http.jsonl (main session, subagents and compaction calls alike):
finish_reason, tool calls (count), visible content chars, reasoning chars (reasoning_content / reasoning). Classes:
  tool          >= 1 tool call
  stop_text     no tool call, visible content (a normal final message, or text-only turn)
  stop_reasoning_only   no tool call, no visible content, reasoning present   <- the agent loop ends here
  stop_empty    no tool call, nothing
Per attempt: the class of its LAST reply and the counts; replies containing "[PAD]" (a degenerate generation seen on
2026-10-26: a reasoning stream that turns into [PAD] runs, then finish "stop"); replies whose usage carries speculative-
decoding counters (completion_tokens_details.rejected_prediction_tokens). "degenerate ending" = the last reply has no
tool call and is either empty or contains [PAD]. --previous scans the previous Kimi rollouts of the 50 tasks
(features/kimi.jsonl attempt dirs) instead.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent


def classify(body):
    choice = (body.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    tools = len(message.get("tool_calls") or [])
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if tools:
        kind = "tool"
    elif content.strip():
        kind = "stop_text"
    elif str(reasoning).strip():
        kind = "stop_reasoning_only"
    else:
        kind = "stop_empty"
    arguments = "".join(str((call.get("function") or {}).get("arguments") or "") for call in message.get("tool_calls") or [])
    pads = (content + str(reasoning) + arguments).count("[PAD]")
    details = ((body.get("usage") or {}).get("completion_tokens_details") or {})
    return kind, choice.get("finish_reason"), len(content), len(str(reasoning)), pads, "rejected_prediction_tokens" in details


def attempt_stats(attempt: Path):
    kinds = collections.Counter()
    pad_replies = pad_tokens = spec = 0
    last = None
    with open(attempt / "http.jsonl", "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                event = json.loads(raw)
                if event.get("status") != 200:
                    continue
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            kind, finish, content, reasoning, pads, speculative = classify(body)
            kinds[kind] += 1
            pad_replies += pads > 0
            pad_tokens += pads
            spec += speculative
            last = {"kind": kind, "finish": finish, "content_chars": content, "reasoning_chars": reasoning, "pads": pads,
                    "completion_tokens": (body.get("usage") or {}).get("completion_tokens")}
    return {"attempt": str(attempt), "kinds": dict(kinds), "last": last, "pad_replies": pad_replies,
            "pad_tokens": pad_tokens, "speculative_usage_replies": spec, "terminal": (attempt / "terminal.json").exists()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="*")
    parser.add_argument("--previous", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.previous:
        rows = [json.loads(line) for line in (HERE / "features" / "kimi.jsonl").read_text().splitlines() if line.strip()]
        attempts = [Path(r["attempt_dir"]) for r in rows if r.get("attempt_dir")]
    else:
        names = args.campaigns or sorted(p.name for p in (HERE / "probes" / "kimi").iterdir())
        attempts = [a for name in names for a in sorted((HERE / "probes" / "kimi" / name).glob("models/*/attempts/*/attempt-*"))
                    if (a / "http.jsonl").exists()]
    stats = [attempt_stats(a) for a in attempts]
    total = collections.Counter()
    for s in stats:
        total.update(s["kinds"])
    finished = [s for s in stats if s["terminal"]]
    summary = {"attempts": len(stats), "finished": len(finished), "replies": dict(total),
               "pct_replies_stop_reasoning_only": round(100 * total["stop_reasoning_only"] / max(1, sum(total.values())), 2),
               "attempts_with_reasoning_only_stop": sum(bool(s["kinds"].get("stop_reasoning_only")) for s in stats),
               "finished_last_reply": dict(collections.Counter((s["last"] or {}).get("kind") for s in finished)),
               "replies_with_pad": sum(s["pad_replies"] for s in stats),
               "attempts_with_pad_reply": sum(bool(s["pad_replies"]) for s in stats),
               "finished_ending_on_degenerate_reply": sum(1 for s in finished if (s["last"] or {}).get("kind") != "tool" and (
                   (s["last"] or {}).get("pads") or (s["last"] or {}).get("kind") == "stop_empty")),
               "replies_with_speculative_usage_fields": sum(s["speculative_usage_replies"] for s in stats),
               "examples": [s["attempt"] for s in finished if (s["last"] or {}).get("kind") == "stop_reasoning_only"][:5]}
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "attempts": stats}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
