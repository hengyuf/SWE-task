#!/usr/bin/env python3
"""kimi-rollouts-v4 (2026-10-28): the 30-minute progress report (user, 2026-10-28: "每30min 汇报进度在一个markdown里").
Each run rewrites PROGRESS.md in this directory:
  1. header: the request, the configuration, the start time, the total (970 = 194 v2 tasks x 5);
  2. a progress table that gains ONE row per run (rows are kept in PROGRESS-rows.json next to it; a run within
     --min-gap-min minutes of the previous row replaces that row instead of adding one, so a repeated run does not
     duplicate rows);
  3. the current per-campaign status table.
Read-only on probes/ and results/ (no ATP code is called: ATP's slot_state writes validity caches into attempt dirs).
Writes only PROGRESS.md, PROGRESS-rows.json and progress-cache.json (per-attempt facts of finished attempts, which never
change: grade, broken ending, compactions, duration), plus key_budget.py's budget-log.jsonl lines (2 tiny hub calls,
--no-budget skips them). Slurm: one `squeue` (--no-squeue skips it).

Definitions (per original slot of the 18 campaigns; the pilots are reported in the header only):
  graded          a finished attempt with evaluation_completed (and no recorded generation failure); the slot's
                  first such attempt, as ATP's slot_state; for a completed campaign its results file is used instead
  broken-ended    the graded attempt's session was ENDED by a broken Kimi reply (brokenreply.py = degenerate.py's
                  definition: the last 200 reply is [PAD] / special-token / empty and has no tool call)
  pass@1 raw      passes / graded, first attempts (infra-recovered slots count with their recovery rollout)
  excl. broken    the same over graded first attempts that were not broken-ended
  re-runs         the one approved re-run of each broken-ended slot (reruns/<campaign>.json, <campaign>-brk<n>):
                  planned / done / pending, outcomes pass / fail / broken again (graded, ended broken again) / infra
  after re-run    pass@1 over resolved slots: graded first attempts that need no re-run, plus broken-ended slots whose
                  re-run is done (its grade; an infra re-run keeps the original); pending re-runs are left out
  compacted       graded rollouts (first attempts and re-runs) whose OpenCode session compacted at least once
  infra/review    slots final as infra in a results file + slots whose latest attempt ended without a grade in a
                  running campaign (awaiting ATP's retry or review)
  in flight       started attempts without an ending, in campaigns whose kr4- job is RUNNING
  spend           NVAPI_KEY_NEW_2 spend now - its spend at the first v4 leg start (key_budget.py); NVAPI_KEY_NEW_3 the
                  same from the first v4 leg that started on it (NEW_3 is shared with the official3 queue: upper bound)
  $/rollout       spend / (finished rollouts + 0.5 x in flight); finished = graded + final infra + done re-runs
  projection      remaining = slots not yet final + pending re-runs + expected re-runs (broken-ended rate so far, v3's
                  28% before 20 graded); rate = finished per hour over the last 3 h once >= 10 finished there, else
                  concurrency / median rollout hours; cost = spend + remaining x $/rollout (v3's $3.13 before 30 finished)

    /usr/bin/python3 -I -B progress.py [--no-budget] [--no-squeue] [--min-gap-min 10]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
TA = S / "sft/train-acc-v2-20261024"
BAIHE_TSV = S / "sft/baihe-snapshot-access-20261026/baihe_v2_tasks.tsv"
PROBES = HERE / "probes" / "kimi"
OUT = HERE / "PROGRESS.md"
ROWS = HERE / "PROGRESS-rows.json"
CACHE = HERE / "progress-cache.json"
GROUPS = {"sel50": 5, "rest47": 5, "baihe": 8}
CONCURRENCY = {"sel50": 8, "rest47": 8, "baihe": 4}  # submit.sh defaults (the campaigns' settings.json win)
REPEATS = 5
TOTAL = 970
V3_USD_PER_ROLLOUT = 3.13      # kimi-rollouts-v3 phase 1 (analysis/phase1-report.md, memory)
V3_BROKEN_ENDED_RATE = 0.28     # v3: 70/250 sessions ended by a broken reply
PRIOR_ROLLOUT_HOURS = 1.1       # v3 median agent wall 55 min + grading
REQUEST = ("在之前v2 上的所有task 上,用kimi 在每个生成的task 上分别rollout 5遍, context window 设置为256k, prompt 保持不变"
           "(不要加reason before action 之类的prompt) ... 每30min 汇报进度在一个markdown里")
RERUN_RULE = "这次若遇到被坏回复终止的 slot, 可以重跑1遍"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"kr4_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BROKEN = _load("brokenreply")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def campaigns() -> dict:
    """{campaign: {taskset, tasks}} for the 18 campaigns (driver.py's task order and grouping)."""
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    pool = json.loads((TA / "pool.json").read_text())["tasks"]
    chosen = {t["task_id"] for t in tasks}
    sel50 = [t["task_id"] for t in tasks]
    rest47 = [t["task_id"] for t in pool if t["task_id"] not in chosen]
    lines = BAIHE_TSV.read_text().splitlines()
    baihe = [line.split("\t")[0] for line in lines[1:] if line.strip()]
    sets = {"sel50": sel50, "rest47": rest47, "baihe": baihe}
    assert (len(sel50), len(rest47), len(baihe)) == (50, 47, 97)
    return {f"{ts}-g{g}": {"taskset": ts, "tasks": ids[g::GROUPS[ts]]} for ts, ids in sets.items()
            for g in range(GROUPS[ts])}


# ---------------------------------------------------------------- per-attempt facts (cached once finished)
def attempt_facts(attempt: Path, cache: dict) -> dict:
    """{ended: bool, completed, status, reward, broken, kind, compactions, duration_s} of one attempt dir."""
    terminal = attempt / "terminal.json"
    validity = attempt / "generation-validity-v2-r7.json"
    try:
        signature = [terminal.stat().st_mtime, validity.stat().st_mtime if validity.exists() else None]
    except OSError:
        failure = (attempt / "failure.json").exists() or (attempt / "interrupted.json").exists() or \
            (attempt / "dispatch-failure.json").exists()
        return {"ended": failure, "completed": False, "started": (attempt / "started.json").exists()}
    key = str(attempt)
    entry = cache.get(key)
    if entry and entry.get("signature") == signature:
        return entry
    result = read_json(terminal, {}) or {}
    failure = (read_json(validity, {}) or {}).get("failure") if validity.exists() else None
    completed = result.get("evaluation_completed") is True and failure is None
    entry = {"signature": signature, "ended": True, "started": True, "completed": completed,
             "reward": result.get("reward"),
             "status": ("pass" if result.get("reward") == 1 else "fail") if completed else None}
    if completed:
        verdict = BROKEN.broken_end(attempt)
        entry.update(broken=verdict["ended"], kind=verdict["kind"])
        export = read_json(attempt / "export.json", {}) or {}
        entry["compactions"] = sum(1 for m in export.get("messages") or []
                                   if (m.get("info") or {}).get("role") == "assistant"
                                   and ((m.get("info") or {}).get("mode") == "compaction"
                                        or (m.get("info") or {}).get("summary") is True))
        started = (read_json(attempt / "started.json", {}) or {}).get("started_at")
        entry["duration_s"] = round(signature[0] - started, 1) if started else None
        entry["finished_at"] = signature[0]
    cache[key] = entry
    return entry


def slot_view(slot_dir: Path, cache: dict) -> dict:
    """ATP-like view of one slot from its attempt dirs: graded (first completed attempt) | active | ended | pending."""
    attempts = sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not attempts:
        return {"state": "pending"}
    for attempt in attempts:
        facts = attempt_facts(attempt, cache)
        if facts.get("completed"):
            return {"state": "graded", "attempt": str(attempt), **facts}
    facts = attempt_facts(attempts[-1], cache)
    if facts.get("ended"):
        return {"state": "ended", "attempt": str(attempts[-1])}
    return {"state": "active" if facts.get("started") else "pending", "attempt": str(attempts[-1])}


def facts_of(attempt_dir, cache: dict) -> dict:
    return attempt_facts(Path(attempt_dir), cache) if attempt_dir else {}


def scan(probe_dir: Path, cache: dict) -> dict:
    """{(task_id, slot): view} of a campaign dir (empty before its manifest exists)."""
    manifest = read_json(probe_dir / "manifest.json")
    if not manifest:
        return {}
    out = {}
    for task in manifest["tasks"]:
        for repeat in range(manifest["settings"]["repeats"]):
            slot_dir = probe_dir / "models" / "kimi" / "attempts" / f"{task['_ng_task_index']}-{repeat}"
            out[(task["task_id"], f"kimi-r{repeat}")] = slot_view(slot_dir, cache)
    return out


# ---------------------------------------------------------------- Slurm / budget
def squeue():
    try:
        out = subprocess.run(["squeue", "--noheader", "-u", os.environ.get("USER", "hengyuf"),
                              "--format=%i|%j|%T|%M|%P"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    jobs = collections.defaultdict(list)
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 5 and parts[1].startswith("kr4-"):
            jobs[parts[1][4:]].append({"id": parts[0], "state": parts[2], "time": parts[3], "part": parts[4]})
    return jobs


def key_budget(key):
    try:
        out = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "key_budget.py"), "--key-name", key,
                              "--tag", "progress"], capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return {"key": key, "remaining": None}


def legs():
    path = HERE / "legs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# ---------------------------------------------------------------- one campaign
def campaign_status(name: str, info: dict, cache: dict, jobs, leg_rows: list) -> dict:
    probe = PROBES / name
    result = read_json(HERE / "results" / f"kimi-{name}.json")
    settings = read_json(probe / "settings.json", {}) or {}
    cap = settings.get("max_concurrency_per_model") or CONCURRENCY[info["taskset"]]
    mine = (jobs or {}).get(name, [])
    running = any(j["state"] == "RUNNING" for j in mine)
    row = {"campaign": name, "taskset": info["taskset"], "slots": len(info["tasks"]) * REPEATS, "cap": cap,
           "graded": 0, "pass": 0, "broken": 0, "broken_pass": 0, "clean_graded": 0, "clean_pass": 0,
           "infra_final": 0, "ended_ungraded": 0, "active": 0, "in_flight": 0, "compacted": 0, "graded_rollouts": 0,
           "rerun_planned": 0, "rerun_done": 0, "rerun_pending": 0,
           "rerun_outcomes": collections.Counter(), "after_graded": 0, "after_pass": 0, "finished_times": [],
           "durations": [], "result": bool(result), "jobs": [f"{j['id']}:{j['state']}:{j['time']}:{j['part']}" for j in mine],
           "legs": sum(1 for leg in leg_rows if leg.get("event") == "start" and leg.get("campaign") == name),
           "recovery_campaigns": sorted(p.name for p in PROBES.glob(f"{name}-*")) if PROBES.exists() else []}

    def add_rollout(facts):
        if facts.get("completed"):
            row["graded_rollouts"] += 1
            row["compacted"] += bool(facts.get("compactions"))
            if facts.get("finished_at"):
                row["finished_times"].append(facts["finished_at"])
            if facts.get("duration_s"):
                row["durations"].append(facts["duration_s"])

    # first attempts (originals; infra-recovered slots with their recovery rollout once the results exist)
    if result:
        first = result.get("first_attempt_records") or result.get("records") or {}
        firsts = [(t, r) for t, recs in first.items() for r in recs]
        for task_id, record in firsts:
            facts = facts_of(record.get("attempt_dir"), cache)
            if record["status"] in ("pass", "fail"):
                row["graded"] += 1
                row["pass"] += record["status"] == "pass"
                broken = bool(record.get("ended_by_broken_reply"))
                row["broken"] += broken
                row["broken_pass"] += broken and record["status"] == "pass"
                if not broken:
                    row["clean_graded"] += 1
                    row["clean_pass"] += record["status"] == "pass"
                    row["after_graded"] += 1
                    row["after_pass"] += record["status"] == "pass"
                add_rollout(facts if facts.get("completed") else {"completed": True})
            else:
                row["infra_final"] += 1
        for task_id, entries in (result.get("broken_reply_reruns") or {}).items():
            for entry in entries:
                row["rerun_planned"] += 1
                row["rerun_done"] += 1
                row["rerun_outcomes"][entry["outcome"]] += 1
                facts = facts_of(entry.get("rerun_attempt"), cache)
                if entry["outcome"] == "infra":  # the original stays
                    row["after_graded"] += 1
                    row["after_pass"] += entry["original_status"] == "pass"
                else:
                    row["after_graded"] += 1
                    row["after_pass"] += entry["rerun_status"] == "pass"
                    add_rollout(facts if facts.get("completed") else {"completed": True})
        return row

    views = scan(probe, cache)
    for (task_id, slot), view in views.items():
        if view["state"] == "graded":
            row["graded"] += 1
            row["pass"] += view["status"] == "pass"
            row["broken"] += bool(view.get("broken"))
            row["broken_pass"] += bool(view.get("broken")) and view["status"] == "pass"
            if not view.get("broken"):
                row["clean_graded"] += 1
                row["clean_pass"] += view["status"] == "pass"
                row["after_graded"] += 1
                row["after_pass"] += view["status"] == "pass"
            add_rollout(view)
        elif view["state"] == "ended":
            row["ended_ungraded"] += 1
        elif view["state"] == "active":
            row["active"] += 1
            row["in_flight"] += running
    # recovery campaigns of a campaign whose results are not written yet (rec: infra recovery; brk: re-runs)
    for rec in [p for p in row["recovery_campaigns"] if "-rec" in p]:
        for view in scan(PROBES / rec, cache).values():
            if view["state"] == "active":
                row["active"] += 1
                row["in_flight"] += running
            add_rollout(view)
    plan = read_json(HERE / "reruns" / f"{name}.json")
    original = {key: view for key, view in views.items()}
    if plan:
        brk_views = {}
        for entry in plan["slots"]:
            if entry["rerun_campaign"] not in brk_views:
                brk_views[entry["rerun_campaign"]] = scan(PROBES / entry["rerun_campaign"], cache)
            view = brk_views[entry["rerun_campaign"]].get((entry["task_id"], entry["rerun_slot"]), {"state": "pending"})
            row["rerun_planned"] += 1
            if view["state"] == "graded":
                row["rerun_done"] += 1
                outcome = "broken_again" if view.get("broken") else view["status"]
                row["rerun_outcomes"][outcome] += 1
                row["after_graded"] += 1
                row["after_pass"] += view["status"] == "pass"
                add_rollout(view)
            else:
                row["rerun_pending"] += 1
                if view["state"] == "active":
                    row["active"] += 1
                    row["in_flight"] += running
    else:  # not planned yet: every broken-ended graded slot will get its re-run
        pending = sum(1 for view in original.values() if view["state"] == "graded" and view.get("broken"))
        row["rerun_planned"] += pending
        row["rerun_pending"] += pending
    return row


# ---------------------------------------------------------------- report
def pct(a, b, digits=1):
    return f"{100.0 * a / b:.{digits}f}%" if b else "-"


def ratio(a, b):
    return f"{100.0 * a / b:.1f}" if b else "-"


def fmt_time(epoch):
    return time.strftime("%m-%d %H:%M", time.localtime(epoch)) if epoch else "-"


def aggregate(rows):
    keys = ("slots", "graded", "pass", "broken", "broken_pass", "clean_graded", "clean_pass", "infra_final",
            "ended_ungraded", "active", "in_flight", "compacted", "graded_rollouts", "rerun_planned", "rerun_done",
            "rerun_pending", "after_graded", "after_pass", "legs")
    out = {key: sum(r[key] for r in rows) for key in keys}
    out["rerun_outcomes"] = sum((r["rerun_outcomes"] for r in rows), collections.Counter())
    out["finished_times"] = [t for r in rows for t in r["finished_times"]]
    out["durations"] = [d for r in rows for d in r["durations"]]
    out["cap"] = sum(r["cap"] for r in rows)
    return out


def projection(by_ts: dict, now: float, unit: float, start: float) -> dict:
    out = {}
    durations = [d for agg in by_ts.values() for d in agg["durations"]]
    hours = statistics.median(durations) / 3600 if len(durations) >= 5 else PRIOR_ROLLOUT_HOURS
    total_remaining = 0.0
    for ts, agg in by_ts.items():
        final = agg["graded"] + agg["infra_final"]
        remaining_first = max(0, agg["slots"] - final)
        rate_broken = agg["broken"] / agg["graded"] if agg["graded"] >= 20 else V3_BROKEN_ENDED_RATE
        remaining = remaining_first + agg["rerun_pending"] + remaining_first * rate_broken
        recent = [t for t in agg["finished_times"] if t >= now - 3 * 3600]
        if len(recent) >= 10 and now - start >= 3 * 3600:
            rate, basis = len(recent) / 3.0, "last 3 h"
        else:
            rate, basis = agg["cap"] / hours, f"cap {agg['cap']} / {hours:.2f} h"
        eta = now + 3600 * remaining / rate if remaining and rate else now
        out[ts] = {"remaining": remaining, "rate_per_h": rate, "basis": basis, "eta": eta}
        total_remaining += remaining
    out["total_remaining"] = total_remaining
    out["eta"] = max((v["eta"] for k, v in out.items() if isinstance(v, dict)), default=now)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--no-squeue", action="store_true")
    parser.add_argument("--min-gap-min", type=float, default=10.0)
    args = parser.parse_args()
    now = time.time()
    cache = read_json(CACHE, {}) or {}
    jobs = None if args.no_squeue else squeue()
    leg_rows = legs()
    info = campaigns()
    rows = [campaign_status(name, data, cache, jobs, leg_rows) for name, data in info.items()]
    by_ts = {ts: aggregate([r for r in rows if r["taskset"] == ts]) for ts in GROUPS}
    total = aggregate(rows)

    # start time and spend
    starts = [leg for leg in leg_rows if leg.get("event") == "start"]
    main_starts = [leg for leg in starts if leg.get("campaign") in info]
    start_at = min((time.mktime(time.strptime(leg["at"][:19], "%Y-%m-%dT%H:%M:%S")) for leg in main_starts),
                   default=None)
    spend = {}
    for key in ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3"):
        on_key = [leg for leg in starts if leg.get("hub_key") == key and (leg.get("budget") or {}).get("spend") is not None]
        baseline = on_key[0]["budget"]["spend"] if on_key else None
        if args.no_budget or baseline is None:
            spend[key] = {"baseline": baseline, "now": None, "v4": 0.0 if baseline is None else None,
                          "remaining": None}
            continue
        record = key_budget(key)
        now_spend = record.get("spend")
        spend[key] = {"baseline": baseline, "now": now_spend, "remaining": record.get("remaining"),
                      "v4": round(now_spend - baseline, 2) if now_spend is not None else None}
    spent = sum(v["v4"] or 0.0 for v in spend.values())
    finished = total["graded"] + total["infra_final"] + total["rerun_done"]
    effective = finished + 0.5 * total["in_flight"]
    budget_known = all(v["v4"] is not None for v in spend.values())
    measured = finished >= 30 and effective > 0 and spent > 0 and budget_known
    unit = spent / effective if measured else V3_USD_PER_ROLLOUT
    unit_basis = "measured" if measured else "v3 prior"
    proj = projection(by_ts, now, unit, start_at or now)
    projected_cost = spent + max(0.0, proj["total_remaining"] - 0.5 * total["in_flight"]) * unit

    outcomes = total["rerun_outcomes"]
    row = {
        "at": now, "time": time.strftime("%m-%d %H:%M", time.localtime(now)),
        "graded": {ts: [by_ts[ts]["graded"], by_ts[ts]["slots"]] for ts in GROUPS},
        "graded_total": total["graded"], "pass": total["pass"],
        "pass_at_1_raw": ratio(total["pass"], total["graded"]),
        "pass_at_1_excl_broken": ratio(total["clean_pass"], total["clean_graded"]),
        "broken": total["broken"], "broken_pct": pct(total["broken"], total["graded"]),
        "rerun_pending": total["rerun_pending"], "rerun_done": total["rerun_done"],
        "rerun_outcomes": {k: outcomes.get(k, 0) for k in ("pass", "fail", "broken_again", "infra")},
        "pass_at_1_after_rerun": ratio(total["after_pass"], total["after_graded"]),
        "compacted": total["compacted"], "graded_rollouts": total["graded_rollouts"],
        "infra_final": total["infra_final"], "ended_ungraded": total["ended_ungraded"],
        "in_flight": total["in_flight"], "active": total["active"], "legs": total["legs"],
        "spend": {k: v["v4"] for k, v in spend.items()}, "spent": round(spent, 2),
        "usd_per_rollout": round(unit, 3), "usd_basis": unit_basis,
        "eta": proj["eta"], "projected_cost": round(projected_cost),
        "jobs_known": jobs is not None}
    history = read_json(ROWS, []) or []
    if history and now - history[-1]["at"] < 60 * args.min_gap_min:
        history[-1] = row
    else:
        history.append(row)
    write_atomic(ROWS, json.dumps(history, indent=1) + "\n")
    write_atomic(CACHE, json.dumps(cache) + "\n")

    # ---- markdown
    pilot = {name: read_json(HERE / "results" / f"kimi-{name}.json") for name in ("pilot", "baihe-pilot")}
    verify = {name: sorted(HERE.glob(f"analysis/verify-{name}-*.json")) for name in ("pilot", "baihe-pilot")}
    lines = [
        "# Kimi-K3 v4 rollouts: stock prompt + 256k window, all 194 v2 tasks x 5",
        "",
        f"_Generated by progress.py at {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(now))}. "
        "The file is rewritten on every run; the table keeps one row per run (PROGRESS-rows.json)._",
        "",
        f"- **Request (hengyuf, 2026-10-28, verbatim):** {REQUEST}",
        f"- **Broken-reply re-runs (hengyuf, 2026-10-28, verbatim):** {RERUN_RULE} -> each slot whose session a broken "
        "reply ended gets exactly one re-run (`<campaign>-brk<n>`); a graded re-run replaces the original; both are kept.",
        "- **Config:** Kimi-K3 (`nvidia/moonshotai/kimi-k3`, reasoning_effort max) via OpenCode 1.17.11 with its stock "
        "system prompt (no added sentence); OpenCode limit.context = limit.input = 262,144, limit.output = 131,072; "
        "harness, graders and sandboxes as in kimi-rollouts-v3; hub key NVAPI_KEY_NEW_2, then NVAPI_KEY_NEW_3; "
        "sandbox credential CELL3_SANDBOX_API_KEY (hengyuf tasks), BAIHE_SANDBOX_API_KEY (Baihe tasks).",
        "- **Campaigns:** " + "; ".join(
            f"{ts} g0-{GROUPS[ts] - 1} (max in flight per campaign: "
            f"{'/'.join(str(c) for c in sorted({r['cap'] for r in rows if r['taskset'] == ts}))})" for ts in GROUPS)
        + " (sel50 / rest47: hengyuf's sandbox account; baihe: Baihe's).",
        f"- **Start:** {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(start_at)) if start_at else 'not started'}; "
        f"**total:** {TOTAL} rollouts (sel50 250, rest47 235, baihe 485) + re-runs.",
    ]
    for name in ("pilot", "baihe-pilot"):
        data = pilot[name]
        if data:
            record = next(iter(next(iter(data["records"].values()))), {})
            checks = read_json(verify[name][-1], {}) if verify[name] else {}
            lines.append(f"- **{name}:** {record.get('status')} (F2P {record.get('f2p')}, P2P {record.get('p2p')}), "
                         f"broken-ended {record.get('ended_by_broken_reply')}; verify_pilot "
                         f"{'all checks OK' if checks.get('all_ok') else ('FAILED: ' + ', '.join(k for k, v in (checks.get('checks') or {}).items() if not v)) if checks else 'not run'}")
        else:
            lines.append(f"- **{name}:** {'running' if (jobs or {}).get(name) else 'no result yet'}")
    lines += [
        "",
        "## Progress (one row per report)",
        "",
        "| time | graded sel50 · rest47 · baihe | graded / 970 | passes | pass@1 raw | pass@1 excl. broken-ended "
        "| broken-ended | re-runs pending / done | re-run pass / fail / broken again / infra | pass@1 first → after re-run "
        "| compacted | infra / awaiting review | in flight | legs | spend NEW_2 / NEW_3 | $/rollout | ETA | proj. cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in history:
        g = r["graded"]
        o = r["rerun_outcomes"]
        lines.append(
            f"| {r['time']} | {g['sel50'][0]}/{g['sel50'][1]} · {g['rest47'][0]}/{g['rest47'][1]} · "
            f"{g['baihe'][0]}/{g['baihe'][1]} | {r['graded_total']} | {r['pass']} | {r['pass_at_1_raw']} | "
            f"{r['pass_at_1_excl_broken']} | {r['broken']} ({r['broken_pct']}) | {r['rerun_pending']} / {r['rerun_done']} | "
            f"{o['pass']} / {o['fail']} / {o['broken_again']} / {o['infra']} | {r['pass_at_1_raw']} → "
            f"{r['pass_at_1_after_rerun']} | {r['compacted']} of {r['graded_rollouts']} | {r['infra_final']} / "
            f"{r['ended_ungraded']} | {r['in_flight'] if r['jobs_known'] else str(r['active']) + '?'} | {r['legs']} | "
            f"${r['spend'].get('NVAPI_KEY_NEW_2') if r['spend'].get('NVAPI_KEY_NEW_2') is not None else '?'} / "
            f"${r['spend'].get('NVAPI_KEY_NEW_3') if r['spend'].get('NVAPI_KEY_NEW_3') is not None else '?'} | "
            f"${r['usd_per_rollout']:.2f} ({r['usd_basis']}) | {fmt_time(r['eta'])} | ${r['projected_cost']:,} |")
    lines += [
        "",
        "pass@1 values are percentages. pass@1 first = original rollouts; after re-run = resolved slots with each "
        "broken-ended slot replaced by its graded re-run (pending re-runs left out). broken-ended = the session was "
        "ended by a broken Kimi reply (brokenreply.py, degenerate.py's definition). in flight with '?' = squeue "
        "unavailable (count of started attempts).",
        "",
        "## Campaigns now",
        "",
        "| campaign | slots | cap | graded | pass | pass@1 | broken-ended | re-runs planned / done | in flight | "
        "infra / awaiting review | compacted | legs | jobs | results |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['campaign']} | {r['slots']} | {r['cap']} | {r['graded']} | {r['pass']} | {ratio(r['pass'], r['graded'])} "
            f"| {r['broken']} | {r['rerun_planned']} / {r['rerun_done']} | {r['in_flight']} | {r['infra_final']} / "
            f"{r['ended_ungraded']} | {r['compacted']} | {r['legs']} | {' '.join(r['jobs']) or '-'} | "
            f"{'yes' if r['result'] else '-'} |")
    lines += ["", "## Projection", ""]
    for ts in GROUPS:
        p = proj[ts]
        lines.append(f"- {ts}: {p['remaining']:.0f} rollouts to go (incl. expected re-runs) at {p['rate_per_h']:.1f}/h "
                     f"({p['basis']}) -> ~{fmt_time(p['eta'])}")
    budget_note = ", ".join(f"{k}: v4 ${v['v4']} (remaining ${v['remaining']})" for k, v in spend.items()
                            if v.get("v4") is not None)
    lines += [f"- cost: ${spent:,.2f} spent so far ({budget_note or 'no budget data'}); ~${unit:.2f}/rollout "
              f"({unit_basis}); projected total ~${projected_cost:,.0f}; projected finish ~{fmt_time(proj['eta'])} "
              "(leg restarts not modelled)", ""]
    write_atomic(OUT, "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("graded",)}, default=str))


if __name__ == "__main__":
    main()
