#!/usr/bin/env python3
"""kimi-rollouts-v5 final summary (2026-10-29): degenerate Kimi-K3 hub replies in this batch (official2 tasks, stock
prompt, 262,144 window, 2026-10-29) vs the BUILD-TIME Kimi probes of the same 33 tasks (synth pipeline, 1M window,
2026-10-23/24), read-only; writes analysis/degenerate-<out>.json (summary) and analysis/degenerate-<out>-attempts.jsonl
(one row per attempt, used by phase_report.py).

    /usr/bin/python3 -I -B degenerate.py [--out v5-vs-build] [--workers 8]

classify(), scan() and the statistics helpers (fisher, binom_tail_ge, clopper_pearson, ratio_ci, rate_block,
boot_diff, compare) are kimi-rollouts-v3's code, byte-identical (the 2026-10-26 file is backup-pre-final-20261029/
degenerate.py): brokenreply.py, hence driver.py's broken-reply re-run rule, loads this file by path and calls classify(),
and brokenreply.py --validate-v3 --forward calls scan(). Only the arms, the per-attempt rows and the summary are new.

Every status-200 "response" event of every attempt (all attempts, including ones later retried: each reply is one
sample of the hub's behaviour) is classified:
  pad      "[PAD]" in the reply's content / reasoning / tool-call arguments
  special  a chat-template special token (<|close|>, <|im_end|>, <|tool_call...|> ...) in it, and no [PAD]
  empty    no tool call, no content, no reasoning (e.g. a 5-token empty stop)
"degenerate" = any of the three. A degenerate reply without a tool call ends the OpenCode session it belongs to: when
it is the attempt's LAST 200 reply the rollout ended there ("ended", brokenreply.py's verdict); earlier, it ended a task
subagent, whose broken text became the task tool's output in the main session ("ended_subsession"). A degenerate reply
WITH a tool call let the session go on with the broken text in its history.
Arms:
  v5     every attempt of probes/kimi/official2-g*/ (the 165 first attempts and the -brk re-runs; the 1-rollout pilot is
         reported apart, as "pilot")
  build  every attempt of the 165 build-time Kimi slots (sources.py kimi5-build: all_attempts, each generation dir once)
Rollout sets: v5 first attempts (the slots' original rollouts), v5 re-runs, v5 final slots, build graded slots (the
attempts score() graded).
Statistics (as v3): per-reply rate with a two-sided Fisher exact test, an exact 95% CI of the rate ratio, a 95%
bootstrap CI of the rate difference resampling rollouts within each arm; per rollout: rollouts with >= 1 degenerate
reply and rollouts ended by one, Fisher. Also per task, per kind, per local calendar day.
Per-attempt rows: replies, the degenerate ones (reply index, kind, tool call, finish, completion tokens, time), ended,
ended_subsession, non-200 statuses, LiteLLM usage.cost sum (and the replies carrying it), prompt / cached / completion
tokens, the largest prompt of one call, calls over 242,144 prompt tokens (OpenCode's compaction point at limit.input
262,144 - 20,000), finish_reason "length" cuts, first / last reply time, the leg's Slurm job (started.json).
Historical references (same classification): previous Kimi on the 50 v2 tasks, Oct 17-21: 0 in 27,163 replies; default-
setting Kimi probes of official2 builds, Oct 24 07:04-13:07: 11 [PAD] in 10,134 (1.09 / 1000); kimi-rollouts-v3 phase 1
(Oct 26): 78 in 21,538 (3.62 / 1000), its 1M control 5 in 1,338 (3.74 / 1000).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
SPECIAL = re.compile(r"<\|[a-z_]+\|>")
CTRL = "ctrl-old"
COMPACT_AT = 262144 - 20000
OUTPUT_CAP = 131072
HISTORICAL = {
    "previous_kimi_50_tasks_oct17_21": {"replies": 27163, "degenerate": 0, "per_1000_replies": 0.0},
    "default_settings_official2_builds_oct24": {"replies": 10134, "degenerate": 11, "per_1000_replies": 1.09},
    "kimi3_phase1_oct26_sentence_256k": {"replies": 21538, "degenerate": 78, "per_1000_replies": 3.62},
    "kimi3_ctrl_old_oct26_1M": {"replies": 1338, "degenerate": 5, "per_1000_replies": 3.74}}


def classify(body):
    choice = (body.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    text = content + reasoning + "".join(str((c.get("function") or {}).get("arguments") or "") for c in calls)
    if "[PAD]" in text:
        kind = "pad"
    elif SPECIAL.search(text):
        kind = "special"
    elif not calls and not content.strip() and not reasoning.strip():
        kind = "empty"
    else:
        kind = None
    return kind, bool(calls)


def scan(attempt):
    """[(timestamp, kind or None, has_tool_call)] for every 200 reply of one attempt."""
    out = []
    path = Path(attempt) / "http.jsonl"
    if not path.exists():
        return attempt, out
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
            out.append((event.get("timestamp") or 0.0, kind, tool))
    return attempt, out


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables at most as likely as the observed one)."""
    row1, col1, n = a + b, a + c, a + b + c + d

    def logp(x):
        return (math.lgamma(col1 + 1) - math.lgamma(x + 1) - math.lgamma(col1 - x + 1)
                + math.lgamma(n - col1 + 1) - math.lgamma(row1 - x + 1) - math.lgamma(n - col1 - row1 + x + 1)
                - math.lgamma(n + 1) + math.lgamma(row1 + 1) + math.lgamma(n - row1 + 1))
    low, high = max(0, row1 + col1 - n), min(row1, col1)
    observed = logp(a)
    return min(1.0, sum(math.exp(logp(x)) for x in range(low, high + 1) if logp(x) <= observed + 1e-9))


def binom_tail_ge(x, n, p):
    """P(X >= x) for X ~ Binomial(n, p), exactly (log-space terms)."""
    if x <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + k * math.log(p) + (n - k) * math.log(1 - p)) for k in range(x, n + 1))


def clopper_pearson(x, n):
    """Exact 95% CI of a binomial proportion by bisection on the exact tails."""
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if x == 0 else solve(lambda q: binom_tail_ge(x, n, q) < 0.025)
    upper = 1.0 if x == n else solve(lambda q: 1 - binom_tail_ge(x + 1, n, q) > 0.025)
    return lower, upper


def ratio_ci(events_ctrl, replies_ctrl, events_new, replies_new):
    """Exact 95% CI of the per-reply rate ratio new/control: conditional on the total number of events, the control's
    share is binomial (Clopper-Pearson), pi = r_c e_c / (r_c e_c + r_n e_n) -> ratio = (e_c / e_n) (1 - pi) / pi."""
    n, x = events_ctrl + events_new, events_ctrl
    if n == 0:
        return None
    lo_pi, hi_pi = clopper_pearson(x, n)
    scale = replies_ctrl / replies_new
    upper = scale * (1 - lo_pi) / lo_pi if lo_pi > 0 else float("inf")
    lower = scale * (1 - hi_pi) / hi_pi if hi_pi < 1 else 0.0
    return [round(lower, 2), round(upper, 2)]


def rate_block(rollouts):
    replies = sum(len(r["replies"]) for r in rollouts)
    kinds = {k: sum(sum(1 for _, kind, _ in r["replies"] if kind == k) for r in rollouts) for k in ("pad", "special", "empty")}
    bad = sum(kinds.values())
    return {"rollouts": len(rollouts), "replies": replies, "degenerate": bad, **kinds,
            "per_1000_replies": round(1000 * bad / replies, 2) if replies else None}


def boot_diff(ctrl, new, n=4000, seed=20261026):
    """95% CI of (new rate - control rate) per 1000 replies, resampling rollouts within each arm."""
    rng = random.Random(seed)
    def counts(rs):
        return [(sum(1 for _, k, _ in r["replies"] if k), len(r["replies"])) for r in rs if r["replies"]]
    c, w = counts(ctrl), counts(new)
    if not c or not w:
        return None
    stats = []
    for _ in range(n):
        cs = [c[rng.randrange(len(c))] for _ in c]
        ws = [w[rng.randrange(len(w))] for _ in w]
        cr = sum(x for x, _ in cs) / max(1, sum(y for _, y in cs))
        wr = sum(x for x, _ in ws) / max(1, sum(y for _, y in ws))
        stats.append(1000 * (wr - cr))
    stats.sort()
    return [round(stats[int(0.025 * n)], 2), round(stats[int(0.975 * n) - 1], 2)]


def compare(ctrl, new):
    a, b = rate_block(ctrl), rate_block(new)
    out = {"control": a, "new": b,
           "fisher_p_per_reply": fisher(a["degenerate"], a["replies"] - a["degenerate"],
                                        b["degenerate"], b["replies"] - b["degenerate"]),
           "bootstrap95_new_minus_control_per_1000": boot_diff(ctrl, new)}
    if b["per_1000_replies"] is not None and a["per_1000_replies"]:
        out["rate_ratio_new_over_control"] = round(b["per_1000_replies"] / a["per_1000_replies"], 2)
        out["rate_ratio_exact95"] = ratio_ci(a["degenerate"], a["replies"], b["degenerate"], b["replies"])
    for label, key in (("with_degenerate_reply", "any"), ("ended_by_degenerate_reply", "ended")):
        ca = sum(r[key] for r in ctrl if r["final"])
        na = sum(r[key] for r in new if r["final"])
        cn = sum(1 for r in ctrl if r["final"])
        nn = sum(1 for r in new if r["final"])
        out[f"rollouts_{label}"] = {"control": f"{ca}/{cn}", "new": f"{na}/{nn}",
                                    "fisher_p": fisher(ca, cn - ca, na, nn - na) if cn and nn else None}
    return out



# ---------------------------------------------------------------- v5 (2026-10-29): arms, per-attempt rows, summary
def _sources():
    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
    sys.path.insert(0, str(HERE))
    import sources
    return sources


def scan_full(job):
    """One attempt's replies: (key, row) with the classified 200 replies (as scan()) plus usage and statuses."""
    key, journal_dir = job
    row = {"replies": [], "degenerate": [], "statuses": {}, "cost": 0.0, "cost_calls": 0, "prompt_tokens": 0,
           "cached_tokens": 0, "completion_tokens": 0, "prompt_max": 0, "over_compact_point": 0, "length_cut": 0,
           "journal": (Path(journal_dir) / "http.jsonl").exists()}
    if not row["journal"]:
        return key, row
    with open(Path(journal_dir) / "http.jsonl", "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            status = str(event.get("status"))
            row["statuses"][status] = row["statuses"].get(status, 0) + 1
            if event.get("status") != 200:
                continue
            try:
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            kind, tool = classify(body)
            stamp = event.get("timestamp") or 0.0
            row["replies"].append((stamp, kind, tool))
            usage = (body.get("usage") or {}) if isinstance(body, dict) else {}
            prompt = usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            choice = (body.get("choices") or [{}])[0] or {}
            if kind:
                row["degenerate"].append({"index": len(row["replies"]), "kind": kind, "tool": tool,
                                          "finish": choice.get("finish_reason"), "completion_tokens": completion,
                                          "at": stamp})
            if isinstance(usage.get("cost"), (int, float)):
                row["cost"] += usage["cost"]
                row["cost_calls"] += 1
            row["prompt_tokens"] += prompt
            row["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            row["completion_tokens"] += completion
            row["prompt_max"] = max(row["prompt_max"], prompt)
            row["over_compact_point"] += prompt > COMPACT_AT
            row["length_cut"] += choice.get("finish_reason") == "length" or completion >= OUTPUT_CAP
    return key, row


def attempt_jobs():
    """[(arm, attempt dir, generation dir, meta)] of every v5 attempt (official2 campaigns + pilot) and every build-time
    attempt of the 165 build slots (each generation dir once)."""
    sources = _sources()
    meta = {}
    for slot in sources.slots("kimi5-all"):
        meta[str(Path(slot["attempt_dir"]).resolve())] = {
            "task_id": slot["task_id"], "slot": slot["slot"], "role": slot["role"], "final": slot["final"],
            "status": slot["status"], "campaign": slot["campaign"], "graded": slot["status"] in ("pass", "fail"),
            "ended_by_broken_reply_results": slot["ended_by_broken_reply"]}
    jobs = []
    for attempt in sorted((HERE / "probes" / "kimi").glob("*/models/kimi/attempts/*/attempt-*")):
        attempt = attempt.resolve()
        campaign = attempt.parents[4].name
        if not (campaign.startswith("official2-g") or campaign == "pilot"):
            continue
        info = dict(meta.get(str(attempt)) or {})
        if not info:  # pilot or an attempt outside the results (none expected): task from the campaign manifest
            manifest = json.loads((attempt.parents[4] / "manifest.json").read_text())
            index, repeat = attempt.parent.name.split("-")
            info = {"task_id": next(t["task_id"] for t in manifest["tasks"] if str(t["_ng_task_index"]) == index),
                    "slot": f"kimi-r{repeat}", "role": "pilot" if campaign == "pilot" else "unrecorded",
                    "final": False, "status": None, "campaign": campaign, "graded": False}
        jobs.append(("pilot" if campaign == "pilot" else "v5", attempt, attempt, info))
    seen = set()
    for slot in sources.slots("kimi5-build"):
        for attempt in slot["all_attempts"]:
            generation = sources.generation_dir(attempt).resolve()
            if generation in seen:
                continue
            seen.add(generation)
            final = str(Path(attempt).resolve()) == str(Path(slot["attempt_dir"]).resolve())
            jobs.append(("build", Path(attempt).resolve(), generation, {
                "task_id": slot["task_id"], "slot": slot["slot"], "role": "build", "final": final,
                "status": slot["status"] if final else "retried", "campaign": slot["campaign"],
                "graded": final and slot["status"] in ("pass", "fail")}))
    return jobs


def rollout_rows(jobs, scanned):
    rows = []
    for arm, attempt, generation, info in jobs:
        data = scanned[str(generation)]
        replies = data["replies"]
        last = replies[-1] if replies else None
        degenerate = data["degenerate"]
        started = {}
        try:
            started = json.loads((attempt / "started.json").read_text())
        except (OSError, ValueError):
            pass
        rows.append({"arm": arm, "attempt": str(attempt), "generation_dir": str(generation), **info,
                     "job": str(started.get("job_id") or ""), "replies_n": len(replies),
                     "degenerate_n": len(degenerate), "degenerate": degenerate,
                     "kinds": dict(collections.Counter(d["kind"] for d in degenerate)),
                     "degenerate_with_tool": sum(d["tool"] for d in degenerate),
                     "ended": bool(last and last[1] and not last[2]),
                     "ended_subsession": sum(1 for d in degenerate if not d["tool"] and d["index"] < len(replies)),
                     "last_kind": last[1] if last else None,
                     "statuses": data["statuses"], "non200": sum(v for k, v in data["statuses"].items() if k != "200"),
                     **{k: data[k] for k in ("cost", "cost_calls", "prompt_tokens", "cached_tokens", "completion_tokens",
                                             "prompt_max", "over_compact_point", "length_cut", "journal")},
                     "first_at": replies[0][0] if replies else None, "last_at": replies[-1][0] if replies else None,
                     "_replies": replies})
    return rows


def as_compare(rows, final_key):
    """compare()'s rollout dicts: replies, final (the rollout-level set), any, ended."""
    return [{"replies": r["_replies"], "final": bool(r.get(final_key)), "any": r["degenerate_n"] > 0,
             "ended": r["ended"]} for r in rows]


def rollout_counts(rows) -> dict:
    n = len(rows)
    return {"rollouts": n, "with_degenerate_reply": sum(r["degenerate_n"] > 0 for r in rows),
            "ended_by_degenerate_reply": sum(r["ended"] for r in rows),
            "subsession_ended_by_degenerate_reply": sum(r["ended_subsession"] > 0 for r in rows),
            "with_degenerate_reply_with_tool_call": sum(r["degenerate_with_tool"] > 0 for r in rows),
            "ended_and_failed": sum(r["ended"] and r.get("status") == "fail" for r in rows),
            "ended_and_passed": sum(r["ended"] and r.get("status") == "pass" for r in rows),
            "pct_ended": round(100 * sum(r["ended"] for r in rows) / n, 1) if n else None}


def per_day(rows) -> dict:
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for stamp, kind, _ in r["_replies"]:
            day = time.strftime("%Y-%m-%d", time.localtime(stamp)) if stamp else "?"
            out[day][0] += 1
            out[day][1] += bool(kind)
    return {day: {"replies": n, "degenerate": d, "per_1000_replies": round(1000 * d / n, 2) if n else None}
            for day, (n, d) in sorted(out.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="v5-vs-build")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = attempt_jobs()
    unique = sorted({str(generation) for _, _, generation, _ in jobs})
    with ProcessPoolExecutor(args.workers) as pool:
        scanned = dict(pool.map(scan_full, [(g, g) for g in unique], chunksize=1))
    rows = rollout_rows(jobs, scanned)
    v5 = [r for r in rows if r["arm"] == "v5"]
    build = [r for r in rows if r["arm"] == "build"]
    pilot = [r for r in rows if r["arm"] == "pilot"]
    first = [r for r in v5 if r["role"] == "first"]
    reruns = [r for r in v5 if r["role"] == "rerun"]
    v5_final = [r for r in v5 if r["final"]]
    build_graded = [r for r in build if r["final"]]
    # compare(control, new): per reply over every attempt of each arm; per rollout over the "final" set given
    for r in v5:
        r["_first"] = r["role"] == "first"
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arms": {"v5": "probes/kimi/official2-g*/ (first attempts + -brk re-runs), 262,144 window, 2026-10-29",
                 "build": "the synth pipeline's probe campaigns of the same 33 tasks (5 Kimi slots each, every attempt),"
                          " 1M window, 2026-10-23/24"},
        "attempts": {"v5": len(v5), "v5_first": len(first), "v5_reruns": len(reruns), "build": len(build),
                     "build_graded_slots": len(build_graded), "pilot": len(pilot)},
        "windows": {name: [min((r["first_at"] for r in rs if r["first_at"]), default=None),
                           max((r["last_at"] for r in rs if r["last_at"]), default=None)]
                    for name, rs in (("v5", v5), ("build", build))},
        "per_reply": {"v5": rate_block(as_compare(v5, "final")), "v5_first": rate_block(as_compare(first, "final")),
                      "v5_reruns": rate_block(as_compare(reruns, "final")), "build": rate_block(as_compare(build, "final")),
                      "pilot": rate_block(as_compare(pilot, "final"))},
        "v5_vs_build": compare(as_compare(build, "final"), as_compare(v5, "_first")),
        "rollouts": {"v5_first_attempts": rollout_counts(first), "v5_reruns": rollout_counts(reruns),
                     "v5_final_slots": rollout_counts(v5_final), "v5_all_graded": rollout_counts(v5),
                     "build_graded_slots": rollout_counts(build_graded), "pilot": rollout_counts(pilot)},
        "per_day": {"v5": per_day(v5), "build": per_day(build)},
        "historical_reference": HISTORICAL,
    }
    report["v5_vs_build"]["note"] = ("control = build (every attempt; rollout-level fields over the 165 graded build "
                                     "slots), new = v5 (every attempt; rollout-level fields over the 165 first attempts)")
    tasks = sorted({r["task_id"] for r in v5 + build})
    report["per_task"] = {}
    for task in tasks:
        entry = {}
        for name, rs in (("v5_first", [r for r in first if r["task_id"] == task]),
                         ("v5_reruns", [r for r in reruns if r["task_id"] == task]),
                         ("build", [r for r in build_graded if r["task_id"] == task])):
            entry[name] = {"rollouts": len(rs), "replies": sum(r["replies_n"] for r in rs),
                           "degenerate": sum(r["degenerate_n"] for r in rs),
                           "with_degenerate": sum(r["degenerate_n"] > 0 for r in rs),
                           "ended": sum(r["ended"] for r in rs),
                           "ended_slots": [r["slot"] for r in rs if r["ended"]],
                           "subsession_ended_slots": [r["slot"] for r in rs if r["ended_subsession"]],
                           "with_tool_slots": [r["slot"] for r in rs if r["degenerate_with_tool"]]}
        report["per_task"][task] = entry
    # the detector's verdict here vs driver.py's recorded verdict (results files), for the v5 first attempts + re-runs
    mismatch = [r["attempt"] for r in v5 if "ended_by_broken_reply_results" in r
                and bool(r["ended_by_broken_reply_results"]) != r["ended"]]
    report["check_ended_vs_results"] = {"attempts": sum("ended_by_broken_reply_results" in r for r in v5),
                                        "mismatches": mismatch}
    out = HERE / "analysis" / f"degenerate-{args.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    tmp.replace(out)
    rows_path = HERE / "analysis" / f"degenerate-{args.out}-attempts.jsonl"
    tmp = rows_path.with_name("." + rows_path.name + ".tmp")
    tmp.write_text("".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in rows))
    tmp.replace(rows_path)
    print(json.dumps({k: report[k] for k in ("attempts", "per_reply", "rollouts", "check_ended_vs_results")}, indent=1))
    print(json.dumps({k: v for k, v in report["v5_vs_build"].items() if k not in ("control", "new")}, indent=1))


if __name__ == "__main__":
    main()
