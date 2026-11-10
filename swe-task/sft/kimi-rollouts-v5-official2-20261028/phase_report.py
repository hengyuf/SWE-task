#!/usr/bin/env python3
"""kimi-rollouts-v5 final report (2026-10-29): official2 accepted + strong tasks x 5, Kimi-K3, stock prompt, 262,144
window, compared per task with the build-time Kimi probes of the same tasks (1M window). Read-only on every input;
writes analysis/<out>.json, analysis/<out>.md (tables generated from the JSON) and analysis/sft-usable-passes.jsonl.
(The v3 phase-1 report this replaces is backup-pre-final-20261029/phase_report.py.)

    /usr/bin/python3 -I -B phase_report.py [--out final-report]

Inputs (made first, all in this directory):
  results/kimi-official2-g<K>.json            driver.py: first_attempt_records, records (final slots), broken_reply_reruns
  analysis/features/kimi5-all.jsonl           features.py: every graded v5 rollout (165 first attempts + re-runs)
  analysis/features/kimi5-build.jsonl         features.py: the 165 build-time Kimi slots (1M window)
  analysis/degenerate-v5-vs-build.json        degenerate.py: per-reply / per-rollout broken-reply statistics
  analysis/degenerate-v5-vs-build-attempts.jsonl  degenerate.py: per attempt (degenerate replies, usage.cost, tokens)
  official2-tasks.json, each task's build metrics.json (teacher_clean_rate, student_rate, similarity), legs.jsonl,
  probes/kimi/<campaign>/kr5-legs.jsonl, budget-log.jsonl, PROGRESS-rows.json (progress.py's last row, cross-check)
Definitions:
  pass@1 first        passes / graded over the 165 first attempts (each slot's original rollout)
  pass@1 after re-run the 165 final slots: a slot whose first attempt a broken reply ENDED is its one re-run (user's
                      rule for v4, applied by main to v5: "这次若遇到被坏回复终止的 slot, 可以重跑1遍"; both kept)
  broken-ended        the attempt's last 200 hub reply is degenerate ([PAD] / special token / empty) without a tool call
                      (brokenreply.py = degenerate.py classify()); driver.py's recorded verdict, re-checked by
                      degenerate.py (0 mismatches required)
  compaction          OpenCode compaction messages in the main session (export.json); also the largest main-session
                      prompt (input + cache read tokens of one step) against OpenCode's compaction point 242,144
  reasoning           features.py: % of action turns (assistant steps with >= 1 tool call) that carry non-empty
                      reasoning, pooled; 95% CI by task bootstrap; paired task bootstrap for v5 - build
  cost                LiteLLM-reported usage.cost of every 200 reply of every v5 attempt (degenerate.py), by the hub
                      key of the attempt's leg (started.json job -> legs.jsonl start hub_key); keys' spend deltas are
                      not used (NVAPI_KEY_NEW_2 is shared with kimi-rollouts-v4)
  build comparison    the synth pipeline's 5 Kimi probe slots per task (sources.py kimi5-build: same model, effort max,
                      OpenCode 1.17.11, stock prompt, 1M window), graded as score() did (rollouts/index.md; all 165
                      match the attempt dirs); teacher_clean_rate = metrics.json (audited, cheating-free passes / graded)
  SFT-usable pass     a graded pass with a clean grade (evaluation completed, reward 1, every F2P and P2P test passing,
                      no verifier / agent error, a non-empty patch and an exported session) and no broken reply (no
                      degenerate hub reply anywhere in the attempt, sub-sessions included, and no [PAD] / special-token
                      text in the main session's assistant parts or tool outputs). No LLM cheating audit was run on v5
                      (the build ran one on its own passes: teacher_audit.json).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
COMPACT_AT = 262144 - 20000
KEYS = ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
RERUN_RULE = "这次若遇到被坏回复终止的 slot, 可以重跑1遍"
REQUEST = ("这组rollout 跑完之后,再给official的第二批(找rank2-5 repo 的那一批) 里的accepted tasks 和标记为strong的task "
           "每个task 额外做5个kimi 的256 k context rollout")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def resolved(path):
    return str(Path(path).resolve()) if path else None


def pct(a, b, digits=1):
    return round(100.0 * a / b, digits) if b else None


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def fraction(text):
    """'a/b' -> (a, b) ints, else (None, None)."""
    try:
        a, b = str(text).split("/")
        return int(a), int(b)
    except (ValueError, TypeError):
        return None, None


# ---------------------------------------------------------------- task-level bootstrap (as v3's phase_report.py)
def by_task(rows, num, den):
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        out[r["task_id"]][0] += num(r)
        out[r["task_id"]][1] += den(r)
    return out


def boot(rows, num, den, n=2000, seed=20261029):
    values = list(by_task(rows, num, den).values())
    if not values:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        d = sum(v[1] for v in sample)
        if d:
            stats.append(100.0 * sum(v[0] for v in sample) / d)
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 1), round(stats[int(0.975 * len(stats)) - 1], 1)] if stats else None


def paired_boot(a_rows, b_rows, num, den, n=2000, seed=20261030):
    """Pooled ratio(a) - pooled ratio(b) in points, 95% CI resampling the shared tasks."""
    a, b = by_task(a_rows, num, den), by_task(b_rows, num, den)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        da, db = sum(a[t][1] for t in sample), sum(b[t][1] for t in sample)
        if da and db:
            stats.append(100.0 * sum(a[t][0] for t in sample) / da - 100.0 * sum(b[t][0] for t in sample) / db)
    stats.sort()
    point = 100.0 * sum(a[t][0] for t in tasks) / sum(a[t][1] for t in tasks) - \
        100.0 * sum(b[t][0] for t in tasks) / sum(b[t][1] for t in tasks)
    return {"diff_points": round(point, 1), "ci95": [round(stats[int(0.025 * len(stats))], 1),
                                                     round(stats[int(0.975 * len(stats)) - 1], 1)],
            "tasks": len(tasks)}


def graded(r):
    return r.get("status") in ("pass", "fail")


def passed(r):
    return r.get("status") == "pass"


def pass_block(rows):
    g = [r for r in rows if graded(r)]
    p = sum(passed(r) for r in g)
    per_task = by_task(g, passed, lambda r: 1)
    return {"rollouts": len(rows), "graded": len(g), "pass": p, "pass_at_1": pct(p, len(g)),
            "ci95_task_bootstrap": boot(g, passed, lambda r: 1),
            "tasks": len(per_task), "tasks_any_pass": sum(1 for v in per_task.values() if v[0] > 0),
            "tasks_all_pass": sum(1 for v in per_task.values() if v[0] == v[1]),
            "tasks_no_pass": sum(1 for v in per_task.values() if v[0] == 0),
            "mean_task_pass_rate": round(100 * statistics.fmean([v[0] / v[1] for v in per_task.values()]), 1)
            if per_task else None}


def reasoning(rows):
    ex = [r for r in rows if r.get("export") and r.get("turns")]
    A = sum(r["action_turns"] for r in ex)
    T = sum(r["turns"] for r in ex)
    rtok = [x for r in ex for x in r["reasoning_tokens"]]
    runs = [r["longest_unreasoned_action_run"] for r in ex]
    return {
        "rollouts": len(rows), "with_export": len(ex), "action_turns": A,
        "pct_action_turns_with_reasoning": pct(sum(r["action_with_reasoning"] for r in ex), A),
        "ci95_pct_action_turns_with_reasoning": boot(ex, lambda r: r["action_with_reasoning"], lambda r: r["action_turns"]),
        "rollout_mean_pct_action_reasoned": round(statistics.fmean(
            [100 * r["action_with_reasoning"] / r["action_turns"] for r in ex if r["action_turns"]]), 1) if ex else None,
        "pct_rollouts_all_actions_reasoned": pct(sum(r["all_actions_reasoned"] for r in ex), len(ex)),
        "pct_first_action_reasoned": pct(sum(r["first_action_reasoning"] for r in ex), len(ex)),
        "longest_unreasoned_run_median": statistics.median(runs) if runs else None,
        "pct_action_turns_with_visible_text": pct(sum(r["action_with_visible"] for r in ex), A),
        "pct_action_turns_bare": pct(sum(r["action_bare"] for r in ex), A),
        "reasoning_tokens_per_reasoning_turn_median": statistics.median(rtok) if rtok else None,
        "reasoning_tokens_per_turn_incl_zero": round(sum(rtok) / T, 1) if T else None,
        "turns_per_rollout_median": statistics.median([r["turns"] for r in ex]) if ex else None,
        "tool_calls_per_rollout_median": statistics.median([r["tool_calls"] for r in ex]) if ex else None,
        "agent_wall_min_median": round(statistics.median([r["agent_wall_s"] for r in ex if r.get("agent_wall_s")]) / 60, 1)
        if any(r.get("agent_wall_s") for r in ex) else None,
        "pct_rollouts_with_task_subagent": pct(sum(r["task_calls"] > 0 for r in ex), len(ex)),
    }


def window(rows, attempts):
    ex = [r for r in rows if r.get("export")]
    prompts = [r["prompt_max_main"] for r in ex]
    att = [attempts.get(resolved(r.get("export_dir") or r.get("attempt_dir"))) or {} for r in rows]
    return {"rollouts": len(rows), "compacted_rollouts": sum(r.get("compactions", 0) > 0 for r in ex),
            "compactions": sum(r.get("compactions", 0) for r in ex),
            "main_prompt_max_median": statistics.median(prompts) if prompts else None,
            "main_prompt_max_p90": quantile(prompts, 0.9), "main_prompt_max_max": max(prompts, default=None),
            "rollouts_main_prompt_over_200k": sum(r["steps_over_200k"] > 0 for r in ex),
            "rollouts_main_prompt_over_242144": sum(r["steps_over_242144"] > 0 for r in ex),
            "hub_calls_over_242144_any_session": sum(a.get("over_compact_point", 0) for a in att),
            "hub_prompt_max_any_session": max((a.get("prompt_max", 0) for a in att), default=None),
            "length_cut_replies": sum(a.get("length_cut", 0) for a in att),
            "non200_replies": sum(a.get("non200", 0) for a in att),
            "non200_statuses": dict(sum((collections.Counter(a.get("statuses") or {}) for a in att),
                                        collections.Counter()) - collections.Counter({"200": 10 ** 12})),
            "final_finish": dict(collections.Counter(r.get("final_finish") for r in ex)),
            "message_errors": dict(collections.Counter(e for r in ex for e in r.get("message_errors") or []))}


def cost_block(attempt_rows):
    costs = [a["cost"] for a in attempt_rows]
    prompt = sum(a["prompt_tokens"] for a in attempt_rows)
    return {"attempts": len(attempt_rows), "usd": round(sum(costs), 2),
            "usd_per_rollout_mean": round(statistics.fmean(costs), 3) if costs else None,
            "usd_per_rollout_median": round(statistics.median(costs), 3) if costs else None,
            "replies": sum(a["replies_n"] for a in attempt_rows),
            "replies_with_cost": sum(a["cost_calls"] for a in attempt_rows),
            "prompt_Mtok": round(prompt / 1e6, 2), "cached_pct": pct(sum(a["cached_tokens"] for a in attempt_rows), prompt),
            "completion_Mtok": round(sum(a["completion_tokens"] for a in attempt_rows) / 1e6, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="final-report")
    parser.add_argument("--degenerate", default="v5-vs-build")
    args = parser.parse_args()
    task_file = read_json(HERE / "official2-tasks.json")
    tasks = task_file["tasks"]
    order = {t["task_id"]: i for i, t in enumerate(tasks)}
    group_of = {t["task_id"]: i % 4 for i, t in enumerate(tasks)}
    results = {int(p.stem.rsplit("-g", 1)[1]): read_json(p) for p in sorted(HERE.glob("results/kimi-official2-g*.json"))}
    v5 = read_jsonl(ANALYSIS / "features" / "kimi5-all.jsonl")
    build = read_jsonl(ANALYSIS / "features" / "kimi5-build.jsonl")
    degenerate = read_json(ANALYSIS / f"degenerate-{args.degenerate}.json", {})
    attempts_rows = read_jsonl(ANALYSIS / f"degenerate-{args.degenerate}-attempts.jsonl")
    attempts = {}
    for a in attempts_rows:
        attempts[a["generation_dir"]] = a
        attempts.setdefault(a["attempt"], a)
    legs = read_jsonl(HERE / "legs.jsonl")
    key_of_job = {str(leg["job"]): leg.get("hub_key") for leg in legs if leg.get("event") == "start"}
    for path in HERE.glob("probes/kimi/*/kr5-legs.jsonl"):
        for leg in read_jsonl(path):
            key_of_job.setdefault(str(leg.get("job")), leg.get("hub_key"))

    first = [r for r in v5 if r.get("role") == "first"]
    reruns = [r for r in v5 if r.get("role") == "rerun"]
    final = [r for r in v5 if r.get("final")]
    first_clean = [r for r in first if not r.get("ended_by_broken_reply")]
    build_graded = [r for r in build if graded(r)]
    deg_rows = {resolved(a["attempt"]): a for a in attempts_rows}

    def deg_of(r):
        return attempts.get(resolved(r.get("export_dir") or r.get("attempt_dir"))) or deg_rows.get(resolved(r.get("attempt_dir"))) or {}

    build_ended = [r for r in build_graded if deg_of(r).get("ended")]
    build_clean = [r for r in build_graded if not deg_of(r).get("ended")]
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request": REQUEST,
              "rerun_rule": RERUN_RULE, "tasks": len(tasks)}

    # ---- configuration (from the results files and the legs)
    any_result = next(iter(results.values()))
    report["config"] = {
        "model_id": any_result.get("model_id"), "spec": any_result.get("spec"),
        "context_window": any_result.get("context_window"), "prompt": any_result.get("prompt"),
        "system_prompt": any_result.get("system_prompt"), "sandbox_credential": any_result.get("sandbox_credential"),
        "max_concurrency_per_campaign": any_result.get("max_concurrency"),
        "campaigns": {f"official2-g{g}": {"job": r.get("job"), "tasks": len(r.get("tasks") or []),
                                          "hub_keys_by_campaign": {c: sorted({x["hub_key"] for x in v})
                                                                   for c, v in (r.get("hub_keys_by_campaign") or {}).items()},
                                          "started": time.strftime("%m-%d %H:%M", time.localtime(r["started_at"]))
                                          if r.get("started_at") else None, "finished": r.get("finished_at")}
                      for g, r in sorted(results.items())},
        "build_probe_settings": {"context_window": sorted({r.get("context_window") for r in build}, key=str),
                                 "model_id": sorted({r.get("model_id") for r in build}, key=str),
                                 "reasoning_effort": sorted({r.get("reasoning_effort") for r in build}, key=str),
                                 "opencode_version": sorted({r.get("opencode_version") for r in build}, key=str),
                                 "index_md_match": f"{sum(bool(r.get('index_match')) for r in build)}/{len(build)}",
                                 "window": degenerate.get("windows", {}).get("build")}}

    # ---- pass@1
    report["pass_at_1"] = {
        "v5_first_attempts": pass_block(first),
        "v5_first_attempts_excluding_broken_ended": pass_block(first_clean),
        "v5_after_rerun_final_slots": pass_block(final),
        "build_1M": pass_block(build),
        "build_1M_excluding_broken_ended": pass_block(build_clean),
        "paired_v5_final_minus_build": paired_boot(final, build, passed, graded),
        "paired_v5_first_minus_build": paired_boot(first, build, passed, graded),
        "paired_v5_first_excl_broken_minus_build_excl_broken": paired_boot(first_clean, build_clean, passed, graded),
        "by_why": {why: {"v5_first": pass_block([r for r in first if task_why(tasks, r) == why]),
                         "v5_final": pass_block([r for r in final if task_why(tasks, r) == why]),
                         "build": pass_block([r for r in build if task_why(tasks, r) == why])}
                   for why in sorted({t["why"] for t in tasks})},
        "by_run": {run: {"tasks": sum(t["run"] == run for t in tasks),
                         "v5_first": pass_block([r for r in first if r["run"] == run])["pass_at_1"],
                         "v5_final": pass_block([r for r in final if r["run"] == run])["pass_at_1"],
                         "build": pass_block([r for r in build if r["run"] == run])["pass_at_1"]}
                   for run in sorted({t["run"] for t in tasks if any(r["run"] == t["run"] for r in final)})}}

    # ---- broken replies and re-runs
    rerun_entries = []
    for g, r in sorted(results.items()):
        for task_id, entries in (r.get("broken_reply_reruns") or {}).items():
            for e in entries:
                original = deg_rows.get(resolved(e["original_attempt"])) or {}
                rerun = deg_rows.get(resolved(e.get("rerun_attempt"))) or {}
                first_rec = next(x for x in r["first_attempt_records"][task_id] if x["slot"] == e["slot"])
                final_rec = next(x for x in r["records"][task_id] if x["slot"] == e["slot"])
                rerun_entries.append({
                    "campaign": f"official2-g{g}", "task_id": task_id, "slot": e["slot"],
                    "original_status": e["original_status"], "original_f2p": first_rec.get("f2p"),
                    "original_patch_chars": first_rec.get("patch_chars"),
                    "original_kind": e.get("original_last_reply_kind"), "original_replies": original.get("replies_n"),
                    "broken_reply_completion_tokens": (original.get("degenerate") or [{}])[-1].get("completion_tokens"),
                    "rerun_campaign": e["rerun_campaign"], "rerun_slot": e["rerun_slot"],
                    "rerun_status": e.get("rerun_status"), "rerun_f2p": final_rec.get("f2p") if e.get("replaced") else None,
                    "rerun_broken": e.get("rerun_ended_by_broken_reply"), "outcome": e.get("outcome"),
                    "replaced": e.get("replaced"), "rerun_replies": rerun.get("replies_n"),
                    "original_attempt": e["original_attempt"], "rerun_attempt": e.get("rerun_attempt")})
    outcomes = collections.Counter(e["outcome"] for e in rerun_entries)
    report["broken_replies"] = {
        "broken_ended_first_attempts": sum(bool(r.get("ended_by_broken_reply")) for r in first),
        "pct_broken_ended_first_attempts": pct(sum(bool(r.get("ended_by_broken_reply")) for r in first), len(first)),
        "broken_ended_kinds": dict(collections.Counter(r.get("last_reply_kind") for r in first if r.get("ended_by_broken_reply"))),
        "broken_ended_that_failed": sum(bool(r.get("ended_by_broken_reply")) and r["status"] == "fail" for r in first),
        "reruns": len(rerun_entries),
        "rerun_outcomes": {k: outcomes.get(k, 0) for k in ("pass", "fail", "broken_again", "infra")},
        "reruns_broken_again": sum(bool(e["rerun_broken"]) for e in rerun_entries),
        "third_attempts": 0, "rerun_list": rerun_entries,
        "detector_check": degenerate.get("check_ended_vs_results"),
        "per_reply": degenerate.get("per_reply"), "rollouts": degenerate.get("rollouts"),
        "v5_vs_build": {k: v for k, v in (degenerate.get("v5_vs_build") or {}).items() if k not in ("control", "new")},
        "per_day": degenerate.get("per_day"), "historical_reference": degenerate.get("historical_reference"),
        "subsession_broken": [{"task_id": a["task_id"], "slot": a["slot"], "status": a.get("status"),
                               "arm": a["arm"], "degenerate": a["degenerate"]}
                              for a in attempts_rows if a.get("ended_subsession") and a["arm"] in ("v5", "build")
                              and (a["arm"] == "v5" or a.get("final"))],
        "build_with_tool_call_broken": [{"task_id": a["task_id"], "slot": a["slot"], "status": a.get("status")}
                                        for a in attempts_rows if a["arm"] == "build" and a.get("final")
                                        and a.get("degenerate_with_tool")]}

    # ---- the 262,144 window
    report["window"] = {"v5_all_graded": window(v5, attempts), "v5_first_attempts": window(first, attempts),
                        "v5_reruns": window(reruns, attempts), "build_1M": window(build, attempts),
                        "compaction_point": COMPACT_AT}

    # ---- reasoning before actions
    v5_no_degenerate = [r for r in v5 if not deg_of(r).get("degenerate_n")]
    build_no_degenerate = [r for r in build if not deg_of(r).get("degenerate_n")]
    num, den = (lambda r: r["action_with_reasoning"]), (lambda r: r["action_turns"])
    report["reasoning"] = {
        "v5_all_graded": reasoning(v5), "v5_first_attempts": reasoning(first), "v5_reruns": reasoning(reruns),
        "v5_final_slots": reasoning(final), "v5_without_degenerate_rollouts": reasoning(v5_no_degenerate),
        "build_1M": reasoning(build), "build_without_degenerate_rollouts": reasoning(build_no_degenerate),
        "paired_v5_final_minus_build": paired_boot([r for r in final if r.get("export")],
                                                   [r for r in build if r.get("export")], num, den),
        "paired_v5_first_minus_build": paired_boot([r for r in first if r.get("export")],
                                                   [r for r in build if r.get("export")], num, den),
        "paired_v5_minus_build_both_without_degenerate": paired_boot(v5_no_degenerate, build_no_degenerate, num, den)}

    # ---- cost by key (v5's own replies)
    v5_attempts = [a for a in attempts_rows if a["arm"] == "v5"]
    pilot_attempts = [a for a in attempts_rows if a["arm"] == "pilot"]
    by_key = collections.defaultdict(list)
    for a in v5_attempts + pilot_attempts:
        by_key[key_of_job.get(a.get("job")) or "unknown"].append(a)
    budget = [row for row in read_jsonl(HERE / "budget-log.jsonl") if row.get("key") in KEYS and row.get("status") == 200]
    last_budget = {key: next((row for row in reversed(budget) if row["key"] == key), None) for key in KEYS}
    progress_rows = read_json(HERE / "PROGRESS-rows.json", []) or []
    report["cost"] = {
        "by_key": {key: {"batch_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["arm"] == "v5"), 2),
                         "first_attempts_usd": round(sum(a["cost"] for a in by_key.get(key, [])
                                                         if a["arm"] == "v5" and a.get("role") == "first"), 2),
                         "reruns_usd": round(sum(a["cost"] for a in by_key.get(key, [])
                                                 if a["arm"] == "v5" and a.get("role") == "rerun"), 2),
                         "pilot_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["arm"] == "pilot"), 2),
                         "attempts": len(by_key.get(key, [])),
                         "key_remaining_last_check": (last_budget.get(key) or {}).get("remaining"),
                         "key_last_check_at": (last_budget.get(key) or {}).get("at")}
                   for key in list(KEYS) + sorted(k for k in by_key if k not in KEYS)},
        "jobs_to_keys": {job: key for job, key in sorted(key_of_job.items())},
        "v5_all_attempts": cost_block(v5_attempts),
        "v5_first_attempts": cost_block([a for a in v5_attempts if a.get("role") == "first"]),
        "v5_reruns": cost_block([a for a in v5_attempts if a.get("role") == "rerun"]),
        "pilot": cost_block(pilot_attempts),
        "build_1M_all_attempts_reference": cost_block([a for a in attempts_rows if a["arm"] == "build"]),
        "usd_per_graded_slot_after_rerun": round(sum(a["cost"] for a in v5_attempts) / max(1, len(final)), 3),
        "progress_py_last_row": {k: (progress_rows[-1] if progress_rows else {}).get(k)
                                 for k in ("time", "spend", "spent", "pilot_cost", "usd_per_rollout")}}

    # ---- per task
    per_task = []
    for task in tasks:
        tid = task["task_id"]
        metrics = read_json(task["candidate_metrics"][0], {}) or {}
        t_first = [r for r in first if r["task_id"] == tid]
        t_final = [r for r in final if r["task_id"] == tid]
        t_reruns = [r for r in reruns if r["task_id"] == tid]
        t_build = [r for r in build if r["task_id"] == tid]
        t_v5_att = [deg_of(r) for r in v5 if r["task_id"] == tid]
        t_build_att = [deg_of(r) for r in t_build]
        rs_v5, rs_b = reasoning([r for r in v5 if r["task_id"] == tid]), reasoning(t_build)
        per_task.append({
            "task_id": tid, "run": task["run"], "why": task["why"], "title": task.get("title"),
            "language": task.get("language"), "group": group_of[tid],
            "build_pass": sum(passed(r) for r in t_build), "build_graded": sum(graded(r) for r in t_build),
            "build_teacher_clean_rate": metrics.get("teacher_clean_rate"), "build_student_rate": metrics.get("student_rate"),
            "build_similarity": metrics.get("similarity"),
            "build_broken_ended": sum(bool(deg_of(r).get("ended")) for r in t_build),
            "v5_first_pass": sum(passed(r) for r in t_first), "v5_first_graded": sum(graded(r) for r in t_first),
            "v5_broken_ended": sum(bool(r.get("ended_by_broken_reply")) for r in t_first),
            "v5_broken_slots": [r["slot"] for r in t_first if r.get("ended_by_broken_reply")],
            "v5_rerun_outcomes": [f"{r['slot']}:{r['status']}" for r in t_reruns],
            "v5_final_pass": sum(passed(r) for r in t_final), "v5_final_graded": sum(graded(r) for r in t_final),
            "diff_final_minus_build": sum(passed(r) for r in t_final) - sum(passed(r) for r in t_build),
            "v5_pct_action_reasoned": rs_v5["pct_action_turns_with_reasoning"],
            "build_pct_action_reasoned": rs_b["pct_action_turns_with_reasoning"],
            "v5_turns_median": rs_v5["turns_per_rollout_median"], "build_turns_median": rs_b["turns_per_rollout_median"],
            "v5_compacted": sum(r.get("compactions", 0) > 0 for r in v5 if r["task_id"] == tid),
            "v5_main_prompt_max": max((r.get("prompt_max_main", 0) for r in v5 if r["task_id"] == tid), default=None),
            "build_main_prompt_max": max((r.get("prompt_max_main", 0) for r in t_build), default=None),
            "v5_usd_per_rollout": round(statistics.fmean([a.get("cost", 0.0) for a in t_v5_att]), 2) if t_v5_att else None,
            "build_usd_per_rollout": round(statistics.fmean([a.get("cost", 0.0) for a in t_build_att]), 2)
            if t_build_att else None})
    report["per_task"] = per_task
    diffs = [t["diff_final_minus_build"] for t in per_task]
    report["per_task_summary"] = {
        "tasks_v5_final_higher": sum(d > 0 for d in diffs), "tasks_equal": sum(d == 0 for d in diffs),
        "tasks_v5_final_lower": sum(d < 0 for d in diffs), "diff_hist": dict(sorted(collections.Counter(diffs).items())),
        "build_selection_note": "accepted tasks needed teacher_clean_rate >= 0.6 at build time (task_builder.score), so "
                                "the build-time Kimi rates of these tasks are selected upward (regression to the mean "
                                "is expected in any re-run); the strong non-accepted task (risor) also had 0.6"}

    # ---- SFT-usable passes
    usable, excluded = [], []
    for r in v5:
        if not passed(r):
            continue
        a = deg_of(r)
        f2p, f2p_total = fraction(r.get("f2p"))
        p2p, p2p_total = fraction(r.get("p2p"))
        reasons = []
        if r.get("reward") != 1:
            reasons.append(f"reward {r.get('reward')}")
        if r.get("evaluation_completed") is not True:
            reasons.append("evaluation not completed")
        if f2p is None or f2p != f2p_total or not f2p_total:
            reasons.append(f"F2P {r.get('f2p')}")
        if p2p is None or p2p != p2p_total:
            reasons.append(f"P2P {r.get('p2p')}")
        if r.get("verifier_error"):
            reasons.append("verifier error")
        if r.get("agent_error"):
            reasons.append(f"agent error {r.get('agent_error')}")
        if not r.get("export"):
            reasons.append("no session export")
        if not r.get("patch_chars"):
            reasons.append("empty patch")
        if not a:
            reasons.append("no hub journal scanned")
        if a.get("degenerate_n"):
            kinds = ",".join(f"{d['kind']}@reply{d['index']}/{a.get('replies_n')}"
                             + ("" if d["tool"] else " no-tool") for d in a["degenerate"])
            reasons.append(f"broken reply in the attempt ({kinds}; "
                           + ("ended a task subagent" if a.get("ended_subsession") else "main session") + ")")
        if r.get("broken_text_parts") or r.get("broken_tool_outputs"):
            reasons.append(f"[PAD]/special text in the main session (assistant parts {r.get('broken_text_parts')}, "
                           f"tool outputs {r.get('broken_tool_outputs')})")
        if r.get("ended_by_broken_reply"):
            reasons.append("ended by a broken reply")
        entry = {"task_id": r["task_id"], "run": r["run"], "slot": r["slot"], "role": r["role"],
                 "campaign": r["campaign"], "final": r.get("final"), "f2p": r.get("f2p"), "p2p": r.get("p2p"),
                 "patch_chars": r.get("patch_chars"), "turns": r.get("turns"), "tool_calls": r.get("tool_calls"),
                 "compactions": r.get("compactions"), "main_prompt_max": r.get("prompt_max_main"),
                 "pct_action_reasoned": pct(r.get("action_with_reasoning", 0), r.get("action_turns", 0)),
                 "attempt_dir": r["attempt_dir"], "export": str(Path(r.get("export_dir") or r["attempt_dir"]) / "export.json")}
        (excluded if reasons else usable).append(dict(entry, reasons=reasons) if reasons else entry)
    usable.sort(key=lambda e: (order[e["task_id"]], e["slot"], e["role"]))
    per_task_usable = collections.Counter(e["task_id"] for e in usable)
    report["sft_usable"] = {
        "criteria": "graded pass; evaluation completed; reward 1; every F2P and P2P test passing; no verifier or agent "
                    "error; non-empty patch; exported session; no degenerate hub reply anywhere in the attempt (sub-"
                    "sessions included) and no [PAD] / special-token text in the main session. No LLM cheating audit.",
        "passes": len(usable) + len(excluded), "usable": len(usable), "excluded": excluded,
        "usable_by_role": dict(collections.Counter(e["role"] for e in usable)),
        "tasks_with_usable": len(per_task_usable),
        "tasks_without_usable": [t["task_id"] for t in tasks if not per_task_usable.get(t["task_id"])],
        "usable_per_task_hist": dict(sorted(collections.Counter(per_task_usable.get(t["task_id"], 0)
                                                                for t in tasks).items())),
        "usable_compacted": sum(bool(e.get("compactions")) for e in usable),
        "list": usable}
    for t in per_task:
        t["sft_usable"] = per_task_usable.get(t["task_id"], 0)

    out = ANALYSIS / f"{args.out}.json"
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(out)
    sft = ANALYSIS / "sft-usable-passes.jsonl"
    tmp = sft.with_name("." + sft.name + ".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in usable))
    tmp.replace(sft)
    md = ANALYSIS / f"{args.out}.md"
    tmp = md.with_name("." + md.name + ".tmp")
    tmp.write_text(markdown(report))
    tmp.replace(md)
    print(json.dumps({"pass_at_1": {k: v for k, v in report["pass_at_1"].items() if k not in ("by_why", "by_run")},
                      "broken": {k: report["broken_replies"][k] for k in ("broken_ended_first_attempts", "reruns",
                                                                         "rerun_outcomes", "detector_check")},
                      "window": report["window"]["v5_all_graded"], "build_window": report["window"]["build_1M"],
                      "reasoning": {k: report["reasoning"][k]["pct_action_turns_with_reasoning"]
                                    for k in report["reasoning"] if not k.startswith("paired")},
                      "reasoning_paired": {k: v for k, v in report["reasoning"].items() if k.startswith("paired")},
                      "cost_by_key": report["cost"]["by_key"], "sft_usable": report["sft_usable"]["usable"],
                      "sft_excluded": report["sft_usable"]["excluded"]}, indent=1, ensure_ascii=False))


def task_why(tasks, row):
    return next((t["why"] for t in tasks if t["task_id"] == row["task_id"]), None)


def markdown(report) -> str:
    p = report["pass_at_1"]
    b = report["broken_replies"]
    w = report["window"]
    rs = report["reasoning"]
    c = report["cost"]
    s = report["sft_usable"]

    def fmt_pass(block):
        return f"{block['pass']}/{block['graded']} = {block['pass_at_1']}% (CI {block['ci95_task_bootstrap']})"

    lines = [f"# kimi-rollouts-v5 final report (phase_report.py, {report['generated_at']})", "",
             f"{report['tasks']} official2 tasks x 5, Kimi-K3 (effort max), OpenCode stock prompt, 262,144 window; "
             "build-time comparison = the synth pipeline's 5 Kimi probe slots per task (1M window).", "",
             "## pass@1", "", "| set | v5 (256k) | build (1M) |", "|---|---|---|",
             f"| first attempts / build slots | {fmt_pass(p['v5_first_attempts'])} | {fmt_pass(p['build_1M'])} |",
             f"| excluding broken-ended | {fmt_pass(p['v5_first_attempts_excluding_broken_ended'])} | "
             f"{fmt_pass(p['build_1M_excluding_broken_ended'])} |",
             f"| after the one re-run (final slots) | {fmt_pass(p['v5_after_rerun_final_slots'])} | (no re-runs) |",
             f"| tasks with >= 1 pass / all 5 / none | {p['v5_after_rerun_final_slots']['tasks_any_pass']} / "
             f"{p['v5_after_rerun_final_slots']['tasks_all_pass']} / {p['v5_after_rerun_final_slots']['tasks_no_pass']} | "
             f"{p['build_1M']['tasks_any_pass']} / {p['build_1M']['tasks_all_pass']} / {p['build_1M']['tasks_no_pass']} |",
             "", f"paired task bootstrap, v5 final - build: {json.dumps(p['paired_v5_final_minus_build'])}; "
             f"v5 first - build: {json.dumps(p['paired_v5_first_minus_build'])}; both excluding broken-ended: "
             f"{json.dumps(p['paired_v5_first_excl_broken_minus_build_excl_broken'])}", "",
             "## Broken replies and re-runs", "",
             f"broken-ended first attempts {b['broken_ended_first_attempts']}/165 ({b['pct_broken_ended_first_attempts']}%), "
             f"kinds {json.dumps(b['broken_ended_kinds'])}, all failed: {b['broken_ended_that_failed']}; re-runs "
             f"{b['reruns']}: {json.dumps(b['rerun_outcomes'])}; detector check {json.dumps(b['detector_check'])}", "",
             "| campaign | task | slot | original | broken reply (completion tokens) | re-run | re-run status (F2P) |",
             "|---|---|---|---|---|---|---|"]
    for e in b["rerun_list"]:
        lines.append(f"| {e['campaign']} | {e['task_id']} | {e['slot']} | {e['original_status']} (F2P {e['original_f2p']}, "
                     f"patch {e['original_patch_chars']} chars, {e['original_replies']} replies) | {e['original_kind']} "
                     f"({e['broken_reply_completion_tokens']}) | {e['rerun_campaign']} {e['rerun_slot']} | "
                     f"{e['rerun_status']} ({e['rerun_f2p']}) |")
    pr = b.get("per_reply") or {}
    vb = b.get("v5_vs_build") or {}
    lines += ["", f"per reply: v5 {pr.get('v5', {}).get('degenerate')}/{pr.get('v5', {}).get('replies')} = "
              f"{pr.get('v5', {}).get('per_1000_replies')}/1000; build {pr.get('build', {}).get('degenerate')}/"
              f"{pr.get('build', {}).get('replies')} = {pr.get('build', {}).get('per_1000_replies')}/1000; Fisher p "
              f"{round(vb.get('fisher_p_per_reply', 0), 3) if vb.get('fisher_p_per_reply') is not None else None}, "
              f"rate ratio {vb.get('rate_ratio_new_over_control')} (exact 95% {vb.get('rate_ratio_exact95')}); rollouts "
              f"ended {json.dumps(vb.get('rollouts_ended_by_degenerate_reply'))}; with any "
              f"{json.dumps(vb.get('rollouts_with_degenerate_reply'))}", "",
              "## 262,144 window", "", "| | v5 all graded | build (1M) |", "|---|---|---|"]
    for k in ("rollouts", "compacted_rollouts", "main_prompt_max_median", "main_prompt_max_p90", "main_prompt_max_max",
              "rollouts_main_prompt_over_200k", "rollouts_main_prompt_over_242144", "hub_calls_over_242144_any_session",
              "hub_prompt_max_any_session", "length_cut_replies", "non200_statuses"):
        lines.append(f"| {k} | {w['v5_all_graded'].get(k)} | {w['build_1M'].get(k)} |")
    lines += ["", "## Reasoning before actions", "", "| metric | v5 all graded | v5 first | v5 final | build (1M) |",
              "|---|---|---|---|---|"]
    for k in ("rollouts", "action_turns", "pct_action_turns_with_reasoning", "ci95_pct_action_turns_with_reasoning",
              "rollout_mean_pct_action_reasoned", "pct_rollouts_all_actions_reasoned", "pct_first_action_reasoned",
              "pct_action_turns_with_visible_text", "pct_action_turns_bare", "reasoning_tokens_per_reasoning_turn_median",
              "turns_per_rollout_median", "tool_calls_per_rollout_median", "agent_wall_min_median",
              "pct_rollouts_with_task_subagent"):
        lines.append(f"| {k} | {rs['v5_all_graded'].get(k)} | {rs['v5_first_attempts'].get(k)} | "
                     f"{rs['v5_final_slots'].get(k)} | {rs['build_1M'].get(k)} |")
    lines += ["", f"paired v5 final - build: {json.dumps(rs['paired_v5_final_minus_build'])}; without degenerate "
              f"rollouts: {json.dumps(rs['paired_v5_minus_build_both_without_degenerate'])}", "",
              "## Cost (LiteLLM usage.cost of v5's own replies)", "", "| key | batch | first attempts | re-runs | pilot | "
              "key remaining (last check) |", "|---|---|---|---|---|---|"]
    for key, v in c["by_key"].items():
        lines.append(f"| {key} | ${v['batch_usd']} | ${v['first_attempts_usd']} | ${v['reruns_usd']} | ${v['pilot_usd']} | "
                     f"{v['key_remaining_last_check']} ({v['key_last_check_at']}) |")
    lines += ["", f"v5 all attempts {json.dumps(c['v5_all_attempts'])}", f"build reference {json.dumps(c['build_1M_all_attempts_reference'])}",
              f"progress.py last row {json.dumps(c['progress_py_last_row'], ensure_ascii=False)}", "",
              "## Per task (v5 vs build-time 1M probes)", "",
              "| task | run | why | build pass | build clean rate | build broken-ended | v5 first pass | v5 broken-ended -> re-run "
              "| v5 final pass | final - build | reasoned % v5 / build | turns v5 / build | $/rollout v5 / build | max prompt v5 | SFT-usable |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in report["per_task"]:
        lines.append(f"| {t['task_id']} | {t['run'].removeprefix('official2-')} | {t['why']} | {t['build_pass']}/{t['build_graded']} "
                     f"| {t['build_teacher_clean_rate']} | {t['build_broken_ended']} | {t['v5_first_pass']}/{t['v5_first_graded']} | "
                     f"{', '.join(t['v5_rerun_outcomes']) or '-'} | {t['v5_final_pass']}/{t['v5_final_graded']} | "
                     f"{t['diff_final_minus_build']:+d} | {t['v5_pct_action_reasoned']} / {t['build_pct_action_reasoned']} | "
                     f"{t['v5_turns_median']} / {t['build_turns_median']} | {t['v5_usd_per_rollout']} / "
                     f"{t['build_usd_per_rollout']} | {t['v5_main_prompt_max']} | {t['sft_usable']} |")
    ps = report["per_task_summary"]
    lines += ["", f"tasks: v5 final higher {ps['tasks_v5_final_higher']}, equal {ps['tasks_equal']}, lower "
              f"{ps['tasks_v5_final_lower']}; diff histogram {json.dumps(ps['diff_hist'])}. {ps['build_selection_note']}.", "",
              f"## SFT-usable passes: {s['usable']} of {s['passes']}", "", s["criteria"], ""]
    for e in s["excluded"]:
        lines.append(f"- excluded: {e['task_id']} {e['slot']} ({e['role']}, {e['campaign']}): {'; '.join(e['reasons'])}")
    lines += ["", "| task | usable passes (slot, role) |", "|---|---|"]
    grouped = collections.defaultdict(list)
    for e in s["list"]:
        grouped[e["task_id"]].append(e["slot"] + ("" if e["role"] == "first" else f" ({e['role']}, {e['campaign']})"))
    for t in report["per_task"]:
        lines.append(f"| {t['task_id']} | {', '.join(grouped.get(t['task_id'], [])) or '-'} |")
    lines += ["", "Full list with attempt dirs and export paths: analysis/sft-usable-passes.jsonl.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
