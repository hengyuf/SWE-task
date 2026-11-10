#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): every status-200 reply of every teacher-rollouts-v2 GLM-5.3 / DeepSeek-V4-Pro
final attempt (the slots brokenreply.py --validate-tr2 reads), forward, read-only; writes analysis/tr2-reply-scan.json.

    /usr/bin/python3 -I -B setup/tr2_reply_scan.py [--workers 8]

Per reply: degenerate.classify() (KR4's broken-reply definition: [PAD] / <|[a-z_]+|> special token / empty), tool call
or not, finish reason, completion tokens. Also counted: replies that used the whole output budget (completion_tokens
>= 131072, or finish "length") = runaway generations, and replies with no tool call whose visible content is empty but
reasoning is not (reasoning-only stops: they end the session, or a task sub-session, without a degenerate kind).
Per model: totals, rates per 1000 replies, attempts with any such reply, and the attempts where one of them is the
LAST reply (the session ended there).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parents[1]
TR2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
OUTPUT_CAP = 131072


def _degenerate():
    spec = importlib.util.spec_from_file_location("tdw_degenerate", TD / "degenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(attempt: str) -> dict:
    classify = _degenerate().classify
    out = {"attempt": attempt, "replies": 0, "kinds": {}, "kinds_without_tool": {}, "runaway": 0,
           "reasoning_only_stops": 0, "last": None, "runaway_details": []}
    path = Path(attempt) / "http.jsonl"
    if not path.exists():
        return out
    with open(path, "rb") as stream:
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
            kind, tool = classify(body)
            choice = (body.get("choices") or [{}])[0] or {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
            completion = (body.get("usage") or {}).get("completion_tokens") or 0
            runaway = completion >= OUTPUT_CAP or choice.get("finish_reason") == "length"
            reasoning_only = not tool and not content.strip() and bool(reasoning.strip())
            out["replies"] += 1
            if kind:
                out["kinds"][kind] = out["kinds"].get(kind, 0) + 1
                if not tool:
                    out["kinds_without_tool"][kind] = out["kinds_without_tool"].get(kind, 0) + 1
            out["runaway"] += runaway
            out["reasoning_only_stops"] += reasoning_only
            if runaway:
                out["runaway_details"].append({"finish": choice.get("finish_reason"), "completion": completion,
                                               "tool": tool, "content_chars": len(content),
                                               "reasoning_chars": len(reasoning), "kind": kind})
            out["last"] = {"kind": kind, "tool": tool, "runaway": runaway, "reasoning_only": reasoning_only,
                           "finish": choice.get("finish_reason")}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = {"source": str(TR2), "output_cap": OUTPUT_CAP, "models": {}}
    for model in ("glm", "deepseek"):
        rows = []
        for path in sorted((TR2 / "results").glob(f"{model}-*.json")):
            for task_id, records in json.loads(path.read_text())["records"].items():
                for record in records:
                    if record.get("attempt_dir"):
                        rows.append((task_id, record["slot"], record["status"], record["attempt_dir"]))
        with ProcessPoolExecutor(args.workers) as pool:
            scans = list(pool.map(scan, [row[3] for row in rows]))
        replies = sum(s["replies"] for s in scans)
        kinds = {}
        for s in scans:
            for kind, count in s["kinds"].items():
                kinds[kind] = kinds.get(kind, 0) + count
        degenerate = sum(kinds.values())
        runaway = sum(s["runaway"] for s in scans)
        ended = [{"task_id": row[0], "slot": row[1], "status": row[2], "attempt": row[3], "last": s["last"],
                  "runaway_details": s["runaway_details"]}
                 for row, s in zip(rows, scans) if s["last"] and (s["last"]["kind"] and not s["last"]["tool"]
                                                                  or s["last"]["runaway"] or s["last"]["reasoning_only"])]
        report["models"][model] = {
            "attempts": len(scans), "replies": replies, "degenerate_replies": degenerate, "by_kind": kinds,
            "degenerate_per_1000": round(1000 * degenerate / replies, 3) if replies else None,
            "degenerate_without_tool": sum(sum(s["kinds_without_tool"].values()) for s in scans),
            "attempts_with_degenerate": sum(bool(s["kinds"]) for s in scans),
            "runaway_replies": runaway, "runaway_per_1000": round(1000 * runaway / replies, 3) if replies else None,
            "attempts_with_runaway": sum(bool(s["runaway"]) for s in scans),
            "reasoning_only_stops": sum(s["reasoning_only_stops"] for s in scans),
            "attempts_ended_by_degenerate_runaway_or_reasoning_only": ended,
            "runaway_examples": [d | {"attempt": s["attempt"]} for s in scans for d in s["runaway_details"]][:20]}
    out = TD / "analysis" / "tr2-reply-scan.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({m: {k: v for k, v in e.items() if k not in ("runaway_examples",)}
                      for m, e in report["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
