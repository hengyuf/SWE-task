#!/usr/bin/env python3
"""kimi-rollouts-v4 final summary (2026-10-29): degenerate ("broken") Kimi-K3 hub replies in this batch (194 v2 tasks x 5,
stock prompt, 262,144 window, 2026-10-28/29) vs kimi-rollouts-v3's sentence arm (sel50, 256k + one reasoning sentence,
2026-10-26) and the earlier 1M Kimi-K3 rollouts of the same 50 sel50 tasks (stock prompt, 2026-10-17/21), read-only;
writes analysis/degenerate-<out>.json (summary) and analysis/degenerate-<out>-attempts.jsonl (one row per attempt, used
by phase_report.py: degenerate replies, usage.cost, tokens, prompt sizes, non-200 replies).

    /usr/bin/python3 -I -B degenerate.py [--out v4-final] [--workers 16]

classify(), scan() and the statistics helpers (fisher, binom_tail_ge, clopper_pearson, ratio_ci, rate_block,
boot_diff, compare) are kimi-rollouts-v3's code, byte-identical (the pre-final file, itself byte-identical to v3's, is
backup-pre-final-20261029/degenerate.py): brokenreply.py, hence driver.py's broken-reply re-run rule and progress.py,
loads this file by path and calls classify(), and brokenreply.py --validate-v3 --forward calls scan(). Only the arms,
the per-attempt rows and the summary are new (they follow kimi-rollouts-v5's final degenerate.py).

Every status-200 "response" event of every attempt (all attempts, including ones later retried: each reply is one
sample of the hub's behaviour) is classified:
  pad      "[PAD]" in the reply's content / reasoning / tool-call arguments
  special  a chat-template special token (<|close|>, <|im_end|>, <|tool_call...|> ...) in it, and no [PAD]
  empty    no tool call, no content, no reasoning (e.g. a 5-token empty stop)
"degenerate" = any of the three. A degenerate reply without a tool call ends the OpenCode session it belongs to: when
it is the attempt's LAST 200 reply the rollout ended there ("ended", brokenreply.py's verdict); earlier, it ended a task
subagent (or another side call), whose broken text became the task tool's output in the main session
("ended_subsession"). A degenerate reply WITH a tool call let the session go on with the broken text in its history.
Arms (attempt dirs):
  v4      every attempt of probes/kimi/{sel50,rest47,baihe}-g<K>[-brk<n>]/: the 970 first attempts (role "first"), the
          broken-reply re-runs (role "rerun") and the attempts ATP retried after an infra error (role "retried")
  pilot   v4's two 1-rollout pilots (reported apart)
  v3      every attempt of V3/probes/kimi/sel50-g<K>[-rec<n>]/ (role "final" = the slot's record in V3/results, else
          "retried")
  ctrl-old  every attempt of V3/probes/kimi/ctrl-old/ (v3's 1M stock control, 4 tasks x 5; reference only)
  1m      every attempt of the 250 earlier 1M Kimi-K3 slots of the sel50 tasks (sources.py kimi-1m all_attempts; role
          "final" = the attempt teacher-rollouts-v2 analysed, else "retried")
Rollout sets: v4 first attempts (the slots' original rollouts; by taskset), v4 re-runs, v4 final slots, v3 final slots,
1M final slots. Comparisons on the 50 sel50 tasks (compare(control, new) as v3: per reply over every attempt of each
arm except v4's broken-reply re-runs, which have no counterpart in the other arms; per rollout over v4's 250 sel50 first
attempts and the other arm's 250 final slots): v4 vs v3, v4 vs 1M, v3 vs 1M. Also v4 by taskset, per local calendar
day, per task.
Per-attempt rows: replies, the degenerate ones (reply index, kind, tool call, finish, completion tokens, time), ended,
ended_subsession, statuses and non-200 replies classified as window.py does (context_overflow / budget / rate_limit /
other), LiteLLM usage.cost sum (and the replies carrying it), prompt / cached / completion tokens, the largest prompt of
one call, calls over 242,144 prompt tokens (OpenCode's compaction point at limit.input 262,144 - 20,000), finish_reason
"length" cuts (or completion >= the 131,072 output cap), first / last reply time, the leg's Slurm job (started.json).
Historical references (same classification): previous Kimi on the 50 v2 tasks, Oct 17-21: 0 in 27,163 replies (final
attempts); default-setting Kimi probes of official2 builds, Oct 24 07:04-13:07: 11 [PAD] in 10,134 (1.09 / 1000);
kimi-rollouts-v3 phase 1 (Oct 26): 78 in 21,538 (3.62 / 1000), its 1M control 5 in 1,338 (3.74 / 1000);
kimi-rollouts-v5 (official2 tasks, stock + 256k, Oct 29): 9 in 12,935 (0.70 / 1000).
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
V3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026")
OVERFLOW = re.compile(r"(?i)context[ _-]?length|maximum context|context[ _]window|too many tokens|prompt is too long|"
                      r"exceeds? the (?:model'?s? )?(?:maximum|context)|input.{0,40}too long|max_model_len")  # window.py's
BUDGET = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|max budget")  # window.py's
HISTORICAL = {
    "previous_kimi_50_tasks_oct17_21": {"replies": 27163, "degenerate": 0, "per_1000_replies": 0.0},
    "default_settings_official2_builds_oct24": {"replies": 10134, "degenerate": 11, "per_1000_replies": 1.09},
    "kimi3_phase1_oct26_sentence_256k": {"replies": 21538, "degenerate": 78, "per_1000_replies": 3.62},
    "kimi3_ctrl_old_oct26_1M": {"replies": 1338, "degenerate": 5, "per_1000_replies": 3.74},
    "kimi5_official2_oct29_stock_256k": {"replies": 12935, "degenerate": 9, "per_1000_replies": 0.70}}


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


# ---------------------------------------------------------------- v4 final (2026-10-29): arms, per-attempt rows, summary
def _sources():
    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
    sys.path.insert(0, str(HERE))
    import sources
    return sources


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def generation_dir(attempt) -> Path:
    """Where an attempt's generation (http.jsonl, export.json) lives: started.json agent_artifacts_from (a regrade
    attempt reuses an earlier attempt's generation), else the attempt itself."""
    started = _read_json(Path(attempt) / "started.json", {}) or {}
    return Path(started.get("agent_artifacts_from") or attempt).resolve()


def scan_full(job):
    """One journal: (key, row) with the classified 200 replies (exactly as scan(): same head test, same filters, same
    classify()) plus usage, statuses and the non-200 replies classified as window.py does."""
    key, journal_dir = job
    row = {"replies": [], "degenerate": [], "statuses": {}, "errors": {}, "cost": 0.0, "cost_calls": 0,
           "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "prompt_max": 0, "over_compact_point": 0,
           "length_cut": 0, "journal": (Path(journal_dir) / "http.jsonl").exists()}
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
                text = str(event.get("body") or "")
                label = status + ":" + ("context_overflow" if OVERFLOW.search(text) else "budget" if BUDGET.search(text)
                                        else "rate_limit" if status == "429" else "other")
                row["errors"][label] = row["errors"].get(label, 0) + 1
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


def _task_of(attempt: Path, cache: dict) -> str:
    """task id of .../<campaign>/models/kimi/attempts/<index>-<repeat>/attempt-<n> from the campaign manifest."""
    campaign_dir = attempt.parents[4]
    if campaign_dir not in cache:
        manifest = json.loads((campaign_dir / "manifest.json").read_text())
        cache[campaign_dir] = {str(t["_ng_task_index"]): t["task_id"] for t in manifest["tasks"]}
    return cache[campaign_dir][attempt.parent.name.split("-")[0]]


def attempt_jobs():
    """[(arm, attempt dir, generation dir, meta)] of every attempt of every arm (see the module docstring)."""
    sources = _sources()
    manifests = {}
    jobs = []
    # v4 and its pilots
    meta = {}
    for slot in sources.slots("kimi4-all"):
        meta[str(Path(slot["attempt_dir"]).resolve())] = {
            "task_id": slot["task_id"], "slot": slot["slot"], "role": slot["role"], "final": slot["final"],
            "status": slot["status"], "campaign": slot["campaign"], "taskset": slot["taskset"],
            "graded": slot["status"] in ("pass", "fail"), "ended_by_broken_reply_results": slot["ended_by_broken_reply"]}
    for attempt in sorted((HERE / "probes" / "kimi").glob("*/models/kimi/attempts/*/attempt-*")):
        attempt = attempt.resolve()
        campaign = attempt.parents[4].name
        taskset = campaign.split("-g")[0]
        if campaign in ("pilot", "baihe-pilot"):
            arm = "pilot"
        elif taskset in ("sel50", "rest47", "baihe"):
            arm = "v4"
        else:
            continue
        info = dict(meta.get(str(attempt)) or {})
        if not info:  # an attempt ATP retried after an infra error, or a pilot
            info = {"task_id": _task_of(attempt, manifests), "slot": f"kimi-r{attempt.parent.name.split('-')[1]}",
                    "role": "pilot" if arm == "pilot" else "retried", "final": arm == "pilot", "status": None,
                    "campaign": campaign, "taskset": "pilot" if arm == "pilot" else taskset, "graded": False}
            if arm == "pilot":
                record = next(iter((_read_json(HERE / "results" / f"kimi-{campaign}.json", {}) or {})
                                   .get("records", {}).values()), [{}])[0]
                info.update(status=record.get("status"), graded=record.get("status") in ("pass", "fail"),
                            final=str(Path(record.get("attempt_dir") or "").resolve()) == str(attempt))
        jobs.append((arm, attempt, generation_dir(attempt), info))
    # v3: sel50 sentence arm and the 1M control
    for source, arm in (("kimi3", "v3"), ("ctrl-old", "ctrl-old")):
        finals = {}
        for slot in sources.slots(source):
            if slot.get("attempt_dir"):
                finals[str(Path(slot["attempt_dir"]).resolve())] = slot
        prefix = "sel50-g" if arm == "v3" else CTRL
        for attempt in sorted((V3 / "probes" / "kimi").glob(f"{prefix}*/models/kimi/attempts/*/attempt-*")):
            attempt = attempt.resolve()
            campaign = attempt.parents[4].name
            if arm == "ctrl-old" and campaign != CTRL:
                continue
            slot = finals.get(str(attempt))
            info = {"task_id": slot["task_id"] if slot else _task_of(attempt, manifests),
                    "slot": slot["slot"] if slot else f"kimi-r{attempt.parent.name.split('-')[1]}",
                    "role": "final" if slot else "retried", "final": bool(slot),
                    "status": slot["status"] if slot else None, "campaign": campaign, "taskset": "sel50" if arm == "v3"
                    else "ctrl-old", "graded": bool(slot) and slot["status"] in ("pass", "fail")}
            jobs.append((arm, attempt, generation_dir(attempt), info))
    # the earlier 1M Kimi-K3 rollouts of the sel50 tasks (every attempt of the 250 slots)
    for slot in sources.slots("kimi-1m"):
        final_dir = str(Path(slot["attempt_dir"]).resolve()) if slot.get("attempt_dir") else None
        for attempt in slot["all_attempts"]:
            attempt = Path(attempt).resolve()
            final = str(attempt) == final_dir
            jobs.append(("1m", attempt, generation_dir(attempt), {
                "task_id": slot["task_id"], "slot": slot["slot"], "role": "final" if final else "retried",
                "final": final, "status": slot["status"] if final else None, "campaign": slot["campaign"],
                "taskset": "sel50", "graded": final and slot["status"] in ("pass", "fail")}))
    return jobs


def rollout_rows(jobs, scanned):
    rows = []
    seen = set()
    for arm, attempt, generation, info in jobs:
        data = scanned[str(generation)]
        replies = data["replies"]
        last = replies[-1] if replies else None
        degenerate = data["degenerate"]
        started = _read_json(attempt / "started.json", {}) or {}
        duplicate = str(generation) in seen  # a regrade of a generation already counted (rates use it once)
        seen.add(str(generation))
        rows.append({"arm": arm, "attempt": str(attempt), "generation_dir": str(generation), **info,
                     "duplicate_generation": duplicate, "job": str(started.get("job_id") or ""),
                     "replies_n": len(replies), "degenerate_n": len(degenerate), "degenerate": degenerate,
                     "kinds": dict(collections.Counter(d["kind"] for d in degenerate)),
                     "degenerate_with_tool": sum(d["tool"] for d in degenerate),
                     "ended": bool(last and last[1] and not last[2]),
                     "ended_subsession": sum(1 for d in degenerate if not d["tool"] and d["index"] < len(replies)),
                     "last_kind": last[1] if last else None,
                     "statuses": data["statuses"], "errors": data["errors"],
                     "non200": sum(v for k, v in data["statuses"].items() if k != "200"),
                     **{k: data[k] for k in ("cost", "cost_calls", "prompt_tokens", "cached_tokens", "completion_tokens",
                                             "prompt_max", "over_compact_point", "length_cut", "journal")},
                     "first_at": replies[0][0] if replies else None, "last_at": replies[-1][0] if replies else None,
                     "_replies": replies})
    return rows


def as_compare(rows, final_key):
    """compare()'s rollout dicts: replies (each generation once), final (the rollout-level set), any, ended."""
    return [{"replies": r["_replies"], "final": bool(r.get(final_key)), "any": r["degenerate_n"] > 0,
             "ended": r["ended"]} for r in rows if not r["duplicate_generation"]]


def rollout_counts(rows) -> dict:
    n = len(rows)
    graded = [r for r in rows if r.get("graded")]
    return {"rollouts": n, "graded": len(graded), "with_degenerate_reply": sum(r["degenerate_n"] > 0 for r in rows),
            "ended_by_degenerate_reply": sum(r["ended"] for r in rows),
            "ended_kinds": dict(collections.Counter(r["last_kind"] for r in rows if r["ended"])),
            "subsession_ended_by_degenerate_reply": sum(r["ended_subsession"] > 0 for r in rows),
            "with_degenerate_reply_with_tool_call": sum(r["degenerate_with_tool"] > 0 for r in rows),
            "ended_and_failed": sum(r["ended"] and r.get("status") == "fail" for r in rows),
            "ended_and_passed": sum(r["ended"] and r.get("status") == "pass" for r in rows),
            "pct_ended": round(100 * sum(r["ended"] for r in rows) / n, 1) if n else None,
            "pct_with_degenerate_reply": round(100 * sum(r["degenerate_n"] > 0 for r in rows) / n, 1) if n else None}


def per_day(rows) -> dict:
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["duplicate_generation"]:
            continue
        for stamp, kind, _ in r["_replies"]:
            day = time.strftime("%Y-%m-%d", time.localtime(stamp)) if stamp else "?"
            out[day][0] += 1
            out[day][1] += bool(kind)
    return {day: {"replies": n, "degenerate": d, "per_1000_replies": round(1000 * d / n, 2) if n else None}
            for day, (n, d) in sorted(out.items())}


def errors_of(rows) -> dict:
    total = collections.Counter()
    for r in rows:
        if not r["duplicate_generation"]:
            total.update(r["errors"])
    return dict(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="v4-final")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    jobs = attempt_jobs()
    unique = sorted({str(generation) for _, _, generation, _ in jobs})
    with ProcessPoolExecutor(args.workers) as pool:
        scanned = dict(pool.map(scan_full, [(g, g) for g in unique], chunksize=1))
    rows = rollout_rows(jobs, scanned)
    arm = collections.defaultdict(list)
    for r in rows:
        arm[r["arm"]].append(r)
    v4, v3, one_m, ctrl, pilot = arm["v4"], arm["v3"], arm["1m"], arm["ctrl-old"], arm["pilot"]
    first = [r for r in v4 if r["role"] == "first"]
    reruns = [r for r in v4 if r["role"] == "rerun"]
    retried = [r for r in v4 if r["role"] == "retried"]
    originals = [r for r in v4 if r["role"] != "rerun"]  # first attempts + ATP-retried attempts (no re-runs)
    final = [r for r in v4 if r["final"]]
    for r in v4:
        r["_first"] = r["role"] == "first"
    tasksets = ("sel50", "rest47", "baihe")
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arms": {"v4": "probes/kimi/{sel50,rest47,baihe}-g<K>[-brk<n>]/ (first attempts, ATP-retried attempts, "
                       "broken-reply re-runs), stock prompt, 262,144 window, 2026-10-28/29",
                 "v3": "V3/probes/kimi/sel50-g<K>[-rec<n>]/ (every attempt), 256k + one reasoning sentence, 2026-10-26",
                 "1m": "the 250 earlier Kimi-K3 slots of the sel50 tasks (synth build probes, every attempt), 1M window, "
                       "stock prompt, 2026-10-17/21",
                 "ctrl-old": "V3/probes/kimi/ctrl-old/ (v3's 1M stock control, 4 tasks x 5), reference only",
                 "pilot": "v4's two pilots"},
        "attempts": {"v4": len(v4), "v4_first": len(first), "v4_reruns": len(reruns), "v4_retried": len(retried),
                     "v4_final_slots": len(final), "v3": len(v3), "v3_final_slots": sum(r["final"] for r in v3),
                     "1m": len(one_m), "1m_final_slots": sum(r["final"] for r in one_m), "ctrl-old": len(ctrl),
                     "pilot": len(pilot),
                     "duplicate_generations": sum(r["duplicate_generation"] for r in rows),
                     "attempts_without_journal": sum(not r["journal"] for r in rows)},
        "windows": {name: [min((r["first_at"] for r in rs if r["first_at"]), default=None),
                           max((r["last_at"] for r in rs if r["last_at"]), default=None)]
                    for name, rs in (("v4", v4), ("v3", v3), ("1m", one_m), ("ctrl-old", ctrl))},
        "per_reply": {
            "v4_all_attempts": rate_block(as_compare(v4, "final")),
            "v4_originals_no_reruns": rate_block(as_compare(originals, "final")),
            "v4_first_attempts": rate_block(as_compare(first, "final")),
            "v4_reruns": rate_block(as_compare(reruns, "final")),
            "v4_retried_attempts": rate_block(as_compare(retried, "final")),
            "v4_final_slots": rate_block(as_compare(final, "final")),
            **{f"v4_{ts}_originals_no_reruns": rate_block(as_compare([r for r in originals if r["taskset"] == ts], "final"))
               for ts in tasksets},
            **{f"v4_{ts}_first_attempts": rate_block(as_compare([r for r in first if r["taskset"] == ts], "final"))
               for ts in tasksets},
            **{f"v4_{ts}_all_attempts": rate_block(as_compare([r for r in v4 if r["taskset"] == ts], "final"))
               for ts in tasksets},
            "v3_all_attempts": rate_block(as_compare(v3, "final")),
            "v3_final_slots": rate_block(as_compare([r for r in v3 if r["final"]], "final")),
            "1m_all_attempts": rate_block(as_compare(one_m, "final")),
            "1m_final_slots": rate_block(as_compare([r for r in one_m if r["final"]], "final")),
            "ctrl-old_all_attempts": rate_block(as_compare(ctrl, "final")),
            "pilot": rate_block(as_compare(pilot, "final"))},
        "rollouts": {
            "v4_first_attempts": rollout_counts(first),
            **{f"v4_{ts}_first_attempts": rollout_counts([r for r in first if r["taskset"] == ts]) for ts in tasksets},
            "v4_reruns": rollout_counts(reruns),
            **{f"v4_{ts}_reruns": rollout_counts([r for r in reruns if r["taskset"] == ts]) for ts in tasksets},
            "v4_retried_attempts": rollout_counts(retried),
            "v4_final_slots": rollout_counts(final),
            **{f"v4_{ts}_final_slots": rollout_counts([r for r in final if r["taskset"] == ts]) for ts in tasksets},
            "v4_all_graded": rollout_counts([r for r in v4 if r["role"] in ("first", "rerun")]),
            "v3_final_slots": rollout_counts([r for r in v3 if r["final"]]),
            "1m_final_slots": rollout_counts([r for r in one_m if r["final"]]),
            "ctrl-old_final_slots": rollout_counts([r for r in ctrl if r["final"]]),
            "pilot": rollout_counts(pilot)},
        "non200_replies": {"v4": errors_of(v4), "v3": errors_of(v3), "1m": errors_of(one_m), "ctrl-old": errors_of(ctrl)},
        "per_day": {"v4": per_day(v4), "v3": per_day(v3), "1m": per_day(one_m)},
        "historical_reference": HISTORICAL,
    }
    sel50_v4 = [r for r in originals if r["taskset"] == "sel50"]
    report["comparisons"] = {
        "sel50_v4_vs_v3": compare(as_compare(v3, "final"), as_compare(sel50_v4, "_first")),
        "sel50_v4_vs_1m": compare(as_compare(one_m, "final"), as_compare(sel50_v4, "_first")),
        "sel50_v3_vs_1m": compare(as_compare(one_m, "final"), as_compare(v3, "final")),
        "all_v4_vs_v3": compare(as_compare(v3, "final"), as_compare(originals, "_first")),
        "note": "compare(control, new): per reply over every attempt of each arm (v4 without its broken-reply re-runs, "
                "which no other arm has); per rollout over v4's first attempts (250 on sel50, 970 overall) and the "
                "control's final slots (250); Fisher two-sided; bootstrap resamples rollouts within each arm"}
    # per task (v4 first attempts / re-runs; v3 and 1M final slots on the sel50 tasks)
    per_task = collections.defaultdict(dict)
    for name, rs in (("v4_first", first), ("v4_reruns", reruns), ("v3", [r for r in v3 if r["final"]]),
                     ("1m", [r for r in one_m if r["final"]])):
        for r in rs:
            entry = per_task[r["task_id"]].setdefault(name, {"rollouts": 0, "replies": 0, "degenerate": 0,
                                                            "with_degenerate": 0, "ended": 0, "ended_slots": []})
            entry["rollouts"] += 1
            entry["replies"] += r["replies_n"]
            entry["degenerate"] += r["degenerate_n"]
            entry["with_degenerate"] += r["degenerate_n"] > 0
            entry["ended"] += r["ended"]
            if r["ended"]:
                entry["ended_slots"].append(r["slot"])
    for task, entry in per_task.items():
        entry["taskset"] = next((r["taskset"] for r in first if r["task_id"] == task), "sel50")
    report["per_task"] = dict(sorted(per_task.items()))
    hist = collections.Counter(per_task[t].get("v4_first", {}).get("ended", 0) for t in per_task if "v4_first" in per_task[t])
    report["v4_tasks_by_broken_endings_in_first_attempts"] = dict(sorted(hist.items()))
    # the detector's verdict here vs driver.py's recorded verdict (results files): v4 first attempts + re-runs
    checked = [r for r in v4 if "ended_by_broken_reply_results" in r]
    report["check_ended_vs_results"] = {
        "attempts": len(checked),
        "mismatches": [r["attempt"] for r in checked if bool(r["ended_by_broken_reply_results"]) != r["ended"]]}
    out = HERE / "analysis" / f"degenerate-{args.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    tmp.replace(out)
    rows_path = HERE / "analysis" / f"degenerate-{args.out}-attempts.jsonl"
    tmp = rows_path.with_name("." + rows_path.name + ".tmp")
    tmp.write_text("".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in rows))
    tmp.replace(rows_path)
    print(json.dumps({k: report[k] for k in ("attempts", "per_reply", "check_ended_vs_results")}, indent=1))
    print(json.dumps({k: (v if k == "note" else {kk: vv for kk, vv in v.items() if kk not in ("control", "new")})
                      for k, v in report["comparisons"].items()}, indent=1))


if __name__ == "__main__":
    main()
