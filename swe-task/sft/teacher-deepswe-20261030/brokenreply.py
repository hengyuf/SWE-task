#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30) copy of kimi-rollouts-v4's brokenreply.py: classify(), last_reply() and broken_end()
are KR4's code unchanged (checked against KR4 with ast in setup/offline_checks.py); --validate-tr2 is new (the
validation of the same detector on teacher-rollouts-v2's GLM-5.3 and DeepSeek-V4-Pro journals, read-only). KR4
docstring follows.

kimi-rollouts-v4 (2026-10-28): "was this rollout ENDED by a broken Kimi reply?" -- the detector behind the approved
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
    /usr/bin/python3 -I -B brokenreply.py --validate-tr2 [--forward-sample N] [--review FILE] [--out FILE]

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
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402

HERE = Path(__file__).resolve().parent
V3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026")
TR2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")


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


# ---------------------------------------------------------------- teacher-deepswe: validation on GLM / DeepSeek (new)
FULLWIDTH = re.compile("<\uff5c|\uff5c>")  # DeepSeek-style special-token delimiters (<｜...｜>), not in SPECIAL
BROAD_SPECIAL = re.compile(r"<\|[^|<>\n]{1,40}\|>")  # any <|...|>, wider than degenerate.SPECIAL (<|[a-z_]+|>)


def _text_of(message: dict) -> tuple:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    calls = message.get("tool_calls") or []
    arguments = "".join(str((c.get("function") or {}).get("arguments") or "") for c in calls)
    return content, reasoning, calls, arguments


def last_reply_detail(attempt) -> dict:
    """The reply last_reply() selects (same reader, same filters), with what a reviewer needs: finish reason, stop class
    (tool / stop_text / stop_reasoning_only / stop_empty, as stops.py), sizes, [PAD] / special-token hits (the
    classifier's pattern, a broader <|...|> pattern and DeepSeek's fullwidth delimiters) and excerpts."""
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
        content, reasoning, calls, arguments = _text_of(choice.get("message") or {})
        text = content + reasoning + arguments
        stop_class = ("tool" if calls else "stop_text" if content.strip() else
                      "stop_reasoning_only" if reasoning.strip() else "stop_empty")
        return {"kind": kind, "tool": tool, "finish": choice.get("finish_reason"), "stop_class": stop_class,
                "timestamp": event.get("timestamp"), "content_chars": len(content), "reasoning_chars": len(reasoning),
                "tool_calls": len(calls), "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
                "pad": "[PAD]" in text, "special_hits": sorted(set(DEGENERATE.SPECIAL.findall(text)))[:10],
                "broad_special_hits": sorted(set(BROAD_SPECIAL.findall(text)))[:10],
                "fullwidth_hits": len(FULLWIDTH.findall(text)),
                "content_head": content[:500], "content_tail": content[-300:] if len(content) > 500 else "",
                "reasoning_head": reasoning[:300], "reasoning_tail": reasoning[-300:] if len(reasoning) > 300 else ""}
    return {}


def _order_key(attempt: str) -> str:
    return hashlib.sha1(os.path.realpath(attempt).encode()).hexdigest()


def validate_tr2(out=None, forward_sample=12, review=None) -> dict:
    """The detector on every slot of teacher-rollouts-v2's GLM-5.3 / DeepSeek-V4-Pro campaigns (results/<model>-*.json:
    group-0..7 = 50 sel50 tasks x 5, and the 2-task pilot; final attempt of each slot; read-only). There are no labels
    for these models, so the report gives: flagged slots (ended by a broken reply) by kind and status; the distribution
    of how every session ended (stop class, finish reason); every unflagged ending that could hide a broken reply
    (no tool call with no / tiny visible text, finish other than stop / tool_calls, [PAD], <|...|> or fullwidth
    special-token text anywhere in the last reply); a deterministic sample of ordinary unflagged endings; and, on the
    flagged + suspicious + sample attempts (up to --forward-sample ordinary ones per model), degenerate.scan()'s forward
    reading of the whole journal (same last reply? any degenerate reply earlier in the session?). A --review file (the
    reviewer's reading of the excerpts and the per-model decision) is merged in as "manual_review" / "decision"."""
    report = {"source": str(TR2), "definition": "brokenreply.broken_end (KR4): last status-200 reply degenerate "
              "(degenerate.classify: [PAD] / <|[a-z_]+|> special token / empty) and without a tool call", "models": {}}
    for model in ("glm", "deepseek"):
        rows = []
        for path in sorted((TR2 / "results").glob(f"{model}-*.json")):
            data = json.loads(path.read_text())
            for task_id, records in data["records"].items():
                for record in records:
                    rows.append({"campaign": path.stem, "task_id": task_id, "slot": record["slot"],
                                 "status": record["status"], "attempt_dir": record.get("attempt_dir")})
        for row in rows:
            attempt = row["attempt_dir"]
            row["journal"] = bool(attempt) and (Path(attempt) / "http.jsonl").exists()
            if not row["journal"]:
                continue
            row["verdict"] = broken_end(attempt)
            row["last"] = last_reply_detail(attempt)
        seen = [r for r in rows if r["journal"] and r["last"]]
        flagged = [r for r in seen if r["verdict"]["ended"]]
        suspicious = [r for r in seen if not r["verdict"]["ended"] and (
            (r["last"]["stop_class"] != "tool" and (r["last"]["content_chars"] < 40))
            or r["last"]["finish"] not in ("stop", "tool_calls") or r["last"]["pad"] or r["last"]["broad_special_hits"]
            or r["last"]["fullwidth_hits"] or r["last"]["stop_class"] in ("stop_reasoning_only", "stop_empty"))]
        ordinary = sorted((r for r in seen if not r["verdict"]["ended"] and r not in suspicious),
                          key=lambda r: _order_key(r["attempt_dir"]))
        sample = ordinary[:forward_sample]
        forward = {"attempts": 0, "same_last_reply": 0, "same_verdict": 0, "earlier_degenerate_replies": {},
                   "attempts_with_earlier_degenerate": 0, "replies": 0}
        for r in flagged + suspicious + sample:
            _, replies = DEGENERATE.scan(r["attempt_dir"])
            last = replies[-1] if replies else None
            tail = r["last"]
            same = bool(last and tail and (last[0], last[1], last[2]) == (tail["timestamp"], tail["kind"], tail["tool"]))
            forward["attempts"] += 1
            forward["replies"] += len(replies)
            forward["same_last_reply"] += same
            forward["same_verdict"] += bool(last and last[1] and not last[2]) == r["verdict"]["ended"]
            earlier = [kind for _, kind, _ in replies[:-1] if kind]
            for kind in earlier:
                forward["earlier_degenerate_replies"][kind] = forward["earlier_degenerate_replies"].get(kind, 0) + 1
            forward["attempts_with_earlier_degenerate"] += bool(earlier)
            r["forward_replies"] = len(replies)
            r["forward_earlier_degenerate"] = earlier

        def excerpt(r):
            return {k: r[k] for k in ("campaign", "task_id", "slot", "status", "attempt_dir")} | {
                "verdict": r["verdict"], "last": r["last"], "forward_replies": r.get("forward_replies"),
                "forward_earlier_degenerate": r.get("forward_earlier_degenerate")}

        def counter(values):
            out = {}
            for value in values:
                out[str(value)] = out.get(str(value), 0) + 1
            return dict(sorted(out.items()))

        report["models"][model] = {
            "slots": len(rows), "with_journal": sum(r["journal"] for r in rows), "with_last_reply": len(seen),
            "status": counter(r["status"] for r in rows),
            "flagged_ended_by_broken_reply": len(flagged),
            "flagged_by_kind": counter(r["verdict"]["kind"] for r in flagged),
            "flagged_by_status": counter(r["status"] for r in flagged),
            "last_reply_stop_class": counter(r["last"]["stop_class"] for r in seen),
            "last_reply_finish": counter(r["last"]["finish"] for r in seen),
            "last_reply_kind_any": counter(r["last"]["kind"] for r in seen),
            "last_reply_with_tool_call_but_degenerate": sum(1 for r in seen if r["last"]["kind"] and r["last"]["tool"]),
            "suspicious_unflagged": len(suspicious), "ordinary_unflagged": len(ordinary),
            "forward_scan": forward,
            "flagged": [excerpt(r) for r in flagged],
            "suspicious_unflagged_endings": [excerpt(r) for r in suspicious],
            "sample_unflagged_endings": [excerpt(r) for r in sample]}
    if review:
        reviewed = json.loads(Path(review).read_text())
        report["manual_review"] = reviewed.get("manual_review")
        report["decision"] = reviewed.get("decision")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(out).with_name("." + Path(out).name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempts", nargs="*")
    parser.add_argument("--validate-v3", action="store_true")
    parser.add_argument("--forward", action="store_true", help="--validate-v3: also compare with a forward scan")
    parser.add_argument("--validate-tr2", action="store_true")
    parser.add_argument("--forward-sample", type=int, default=12)
    parser.add_argument("--review", help="--validate-tr2: reviewer notes + decision JSON to merge")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.validate_v3:
        print(json.dumps(validate_v3(args.out, args.forward), indent=1))
        return
    if args.validate_tr2:
        report = validate_tr2(args.out, args.forward_sample, args.review)
        print(json.dumps({model: {k: v for k, v in entry.items() if not isinstance(v, list)}
                          for model, entry in report["models"].items()}, indent=1, ensure_ascii=False))
        return
    for attempt in args.attempts:
        print(json.dumps({"attempt": attempt, **broken_end(attempt)}))


if __name__ == "__main__":
    main()
