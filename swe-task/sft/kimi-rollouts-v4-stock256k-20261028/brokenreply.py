#!/usr/bin/env python3
"""kimi-rollouts-v4 (2026-10-28): "was this rollout ENDED by a broken Kimi reply?" -- the detector behind the approved
broken-reply re-runs (user, 2026-10-28: "这次若遇到被坏回复终止的 slot, 可以重跑1遍"). Read-only.

Definition (degenerate.py's, unchanged): every status-200 "response" event of an attempt's http.jsonl is classified by
degenerate.classify() (this module loads degenerate.py from this directory by path and calls that very function):
  pad      "[PAD]" in the reply's content / reasoning / tool-call arguments
  special  a chat-template special token (<|...|>) in it, and no [PAD]
  empty    no tool call, no content, no reasoning
A rollout is ENDED by a broken reply when its LAST status-200 reply (with a parsable body, in journal order: exactly
degenerate.py's replies[-1]) is degenerate AND has no tool call -- OpenCode ends the session on a reply without a tool
call, so the session stopped there. A rollout that merely contains a degenerate reply somewhere earlier (and went on)
is NOT broken-ended.

Only the tail of http.jsonl is read (the journal is read backwards line by line until the last 200 reply is found), so
one check costs a few hundred KB instead of the whole ~100 MB journal; it returns the same reply as degenerate.py's
forward scan (same head test, same JSON / status / body filters, applied from the end).

    /usr/bin/python3 -I -B brokenreply.py ATTEMPT_DIR [ATTEMPT_DIR ...]      # one JSON line per attempt
    /usr/bin/python3 -I -B brokenreply.py --validate-v3 [--forward] [--out FILE]   # offline check against v3 (read-only)

--validate-v3 runs the detector on the final attempt of every slot that kimi-rollouts-v3 analysed (window-kimi3.jsonl:
sel50 g0-4, 250 slots; window-ctrl-old.jsonl: ctrl-old, 20 slots) and compares it with v3's own per-slot label
hub.last_reply_degenerate (window.py's forward scan; its aggregate equals degenerate.py's rollouts_ended_by_degenerate_
reply 70/250 + 4/20, analysis/degenerate-compare.json): confusion matrix, precision, recall, and the disagreements.
With --forward it also runs degenerate.py's own forward scan (degenerate.scan(): every line of the journal) on each of
those attempts and checks that the tail reader returns the same last reply (timestamp, kind, tool call) and the same
"ended" verdict.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
V3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026")


def _load_degenerate():
    spec = importlib.util.spec_from_file_location("kr4_degenerate", HERE / "degenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEGENERATE = _load_degenerate()
classify = DEGENERATE.classify  # the definition: degenerate.py's classify(), not a copy


def _reverse_lines(path: Path, block: int = 1 << 20):
    """The lines of a file, last first (without the trailing newline; empty lines skipped)."""
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        tail = b""
        while position > 0:
            step = min(block, position)
            position -= step
            stream.seek(position)
            data = stream.read(step) + tail
            lines = data.split(b"\n")
            tail = lines[0]  # may continue in the previous block
            for line in reversed(lines[1:]):
                if line:
                    yield line
        if tail:
            yield tail


def last_reply(attempt) -> dict:
    """{kind, tool, timestamp, call_id, finish} of the last status-200 reply of an attempt (degenerate.scan()'s
    replies[-1]), or {} when the attempt has no journal or no such reply."""
    path = Path(attempt) / "http.jsonl"
    if not path.exists():
        return {}
    for raw in _reverse_lines(path):
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
        return {"kind": kind, "tool": tool, "timestamp": event.get("timestamp") or 0.0,
                "call_id": event.get("call_id"), "finish": choice.get("finish_reason")}
    return {}


def broken_end(attempt) -> dict:
    """{"ended": bool, "kind": pad|special|empty|None, "replies_seen": bool, ...} for one attempt directory:
    ended = the last 200 reply is degenerate and carries no tool call (degenerate.py's "ended")."""
    if not attempt:
        return {"ended": False, "kind": None, "replies_seen": False}
    last = last_reply(attempt)
    return {"ended": bool(last and last.get("kind") and not last.get("tool")), "kind": last.get("kind"),
            "tool": last.get("tool"), "finish": last.get("finish"), "replies_seen": bool(last),
            "last_reply_at": last.get("timestamp")}


def validate_v3(out=None, forward=False) -> dict:
    """Detector vs kimi-rollouts-v3's recorded labels (read-only on v3)."""
    rows = []
    for name in ("window-kimi3.jsonl", "window-ctrl-old.jsonl"):
        for line in (V3 / "analysis" / name).read_text().splitlines():
            if line.strip():
                rows.append((name, json.loads(line)))
    tp = fp = fn = tn = 0
    disagreements = []
    by_kind = {}
    missing = 0
    checks = {"attempts": 0, "same_last_reply": 0, "same_verdict": 0}
    for source, row in rows:
        label = bool((row.get("hub") or {}).get("last_reply_degenerate"))
        if not row.get("attempt_dir") or not (Path(row["attempt_dir"]) / "http.jsonl").exists():
            missing += 1
            if label:
                fn += 1
                disagreements.append({"source": source, "task_id": row["task_id"], "slot": row["slot"],
                                      "label": label, "detected": None, "reason": "no journal"})
            else:
                tn += 1  # no journal = no reply = nothing to flag (v3 labelled these False as well)
            continue
        found = broken_end(row["attempt_dir"])
        if forward:
            _, replies = DEGENERATE.scan(row["attempt_dir"])
            last, tail = (replies[-1] if replies else None), last_reply(row["attempt_dir"])
            same = (last is None and not tail) or bool(last and tail and (last[0], last[1], last[2]) ==
                                                     (tail["timestamp"], tail["kind"], tail["tool"]))
            forward_ended = bool(last and last[1] and not last[2])
            checks["same_last_reply"] += same
            checks["same_verdict"] += forward_ended == found["ended"]
            checks["attempts"] += 1
        if found["ended"]:
            by_kind[found["kind"]] = by_kind.get(found["kind"], 0) + 1
        if found["ended"] and label:
            tp += 1
        elif found["ended"]:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
        if found["ended"] != label:
            disagreements.append({"source": source, "task_id": row["task_id"], "slot": row["slot"], "label": label,
                                  "detected": found, "attempt_dir": row["attempt_dir"]})
    report = {
        "labels": "kimi-rollouts-v3 analysis/window-kimi3.jsonl + window-ctrl-old.jsonl hub.last_reply_degenerate "
                  "(final attempt of every graded/infra slot; 70/250 + 4/20 = degenerate-compare.json's "
                  "rollouts_ended_by_degenerate_reply)",
        "slots": len(rows), "no_journal": missing,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "flagged_by_kind": by_kind, "disagreements": disagreements}
    if forward:
        report["forward_scan_check"] = dict(checks, note="degenerate.scan() (whole journal, forward) vs this tail "
                                                         "reader on the same attempts")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempts", nargs="*")
    parser.add_argument("--validate-v3", action="store_true")
    parser.add_argument("--forward", action="store_true", help="--validate-v3: also compare with a forward scan")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.validate_v3:
        print(json.dumps(validate_v3(args.out, args.forward), indent=1))
        return
    for attempt in args.attempts:
        print(json.dumps({"attempt": attempt, **broken_end(attempt)}))


if __name__ == "__main__":
    main()
