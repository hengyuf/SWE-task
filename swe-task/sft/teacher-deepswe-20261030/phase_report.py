#!/usr/bin/env python3
"""kimi-rollouts-v4 final report (2026-10-29): Kimi-K3 on all 194 v2 SFT tasks x 5 (sel50 / rest47 / baihe), OpenCode
stock prompt, 262,144 window, one approved re-run per slot ended by a broken reply; compared on the 50 sel50 tasks with
kimi-rollouts-v3's sentence arm (256k + one reasoning sentence) and the earlier 1M Kimi-K3 rollouts (stock prompt).
Read-only on every input; writes analysis/<out>.json, analysis/<out>.md (tables generated from the JSON) and
analysis/sft-usable-passes.jsonl. (The v3 phase-1 report this replaces is backup-pre-final-20261029/phase_report.py;
the structure follows kimi-rollouts-v5's final phase_report.py.)

    /usr/bin/python3 -I -B phase_report.py [--out final-report] [--degenerate v4-final]

Inputs (made first, all in this directory unless noted):
  results/kimi-<taskset>-g<K>.json            driver.py: first_attempt_records, records (final slots), broken_reply_reruns
  analysis/features/kimi4-all.jsonl           features.py: every graded v4 rollout (970 first attempts + re-runs)
  analysis/features/kimi3.jsonl               features.py: v3's sentence arm (sel50, 250 slots; V3 read-only)
  analysis/features/kimi-1m.jsonl             features.py: the earlier 1M Kimi-K3 rollouts of the sel50 tasks (250 slots)
  analysis/degenerate-<d>.json                degenerate.py: per-reply / per-rollout broken-reply statistics, comparisons
  analysis/degenerate-<d>-attempts.jsonl      degenerate.py: per attempt (degenerate replies, usage.cost, tokens, sizes)
  legs.jsonl, probes/kimi/<campaign>/kr4-legs.jsonl (job -> hub key NAME), budget-log.jsonl (key spend, by NAME only),
  PROGRESS-rows.json (progress.py's last row, cross-check), kimi-rollouts-v5's analysis/final-report.json (read-only:
  v5's own journal cost on the shared key NVAPI_KEY_NEW_2, for the key-spend reconciliation)
Definitions:
  first attempts      each slot's original rollout (970; sel50 250, rest47 235, baihe 485)
  after re-run        the 970 final slots: a slot whose first attempt a broken reply ENDED is its one re-run (user,
                      2026-10-28: "这次若遇到被坏回复终止的 slot, 可以重跑1遍"; both records kept in the results files)
  broken-ended        the attempt's last 200 hub reply is degenerate ([PAD] / special token / empty) without a tool call
                      (brokenreply.py = degenerate.py classify()); driver.py's recorded verdict, re-checked by
                      degenerate.py (0 mismatches required); v3 / 1M: degenerate.py's verdict on their final attempts
  pass@1 excl.        passes / graded over the rollouts that were not broken-ended
  reasoning           features.py: % of action turns (assistant steps with >= 1 tool call) that carry non-empty
                      reasoning, pooled; 95% CI by task bootstrap; paired task bootstrap for the differences
  compaction          OpenCode compaction messages in the main session (export.json; compaction point 242,144 = limit.input
                      262,144 - 20,000); "truncation": replies cut at the 131,072 output cap (finish_reason length) and
                      context-overflow refusals (hub non-200 bodies, window.py's pattern); for the 1M arm also the
                      rollouts whose main-session prompt passed 242,144 (they would have been compacted at 256k)
  broken replies      degenerate.py: per 1000 status-200 replies over every attempt of an arm (sub-sessions and ATP-retried
                      attempts included, each generation once; v4's re-runs reported apart), and per rollout
  cost                LiteLLM-reported usage.cost of every 200 reply of every v4 attempt (degenerate.py), by the hub key
                      NAME of the attempt's leg (started.json job -> legs.jsonl / kr4-legs.jsonl hub_key). The key's
                      spend delta (progress.py) is NOT v4's cost: NVAPI_KEY_NEW_2 also paid kimi-rollouts-v5 (2026-10-29
                      02:09-10:32); the reconciliation is reported
  SFT-usable pass     a graded pass with a clean grade (evaluation completed, reward 1, every F2P and P2P test passing,
                      no verifier / agent error, a non-empty patch and an exported session) and no broken reply (no
                      degenerate hub reply anywhere in the attempt, sub-sessions included, and no [PAD] / special-token
                      text in the main session's assistant parts, nor in its tool outputs when the attempt had a
                      degenerate reply, i.e. a broken sub-session's text). Tool-output matches in an attempt WITHOUT any
                      degenerate hub reply are repository / command text (e.g. zsa's own source contains "<|highlight|>"),
                      not a Kimi reply: kept, and flagged in "notes". No LLM cheating audit was run.
Caveat (selection): the 194 v2 tasks are tasks with Kimi passes in the earlier 1M rollouts (kimi-pass-v2 selection) and
sel50 is a random 50 of the hengyuf ones, so the 1M arm's pass@1 on sel50 is measured on the rollouts that selected the
tasks (its pass@5 is 100% by construction) and is biased upward against any re-run.
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
V5_REPORT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028/"
                 "analysis/final-report.json")
COMPACT_AT = 262144 - 20000
KEYS = ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
TASKSETS = ("sel50", "rest47", "baihe")
RERUN_RULE = "这次若遇到被坏回复终止的 slot, 可以重跑1遍"
REQUEST = ("在之前v2 上的所有task 上,用kimi 在每个生成的task 上分别rollout 5遍, context window 设置为256k, prompt 保持不变"
           "(不要加reason before action 之类的prompt)")
ARMS = {"v4": "v4: stock prompt, 256k (2026-10-28/29)",
        "v3": "v3 sentence arm: 256k + reasoning sentence (2026-10-26)",
        "1m": "earlier Kimi-K3: 1M, stock prompt (2026-10-17/21)"}


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


# ---------------------------------------------------------------- task-level bootstrap (as v3's / v5's phase_report.py)
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


def with_turns(rows):
    return [r for r in rows if r.get("export") and r.get("turns")]


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
    ex = with_turns(rows)
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
        "longest_unreasoned_run_p90": quantile(runs, 0.9),
        "pct_action_turns_with_visible_text": pct(sum(r["action_with_visible"] for r in ex), A),
        "pct_action_turns_bare": pct(sum(r["action_bare"] for r in ex), A),
        "reasoning_tokens_per_reasoning_turn_median": statistics.median(rtok) if rtok else None,
        "reasoning_tokens_per_reasoning_turn_mean": round(statistics.fmean(rtok), 1) if rtok else None,
        "reasoning_tokens_per_turn_incl_zero": round(sum(rtok) / T, 1) if T else None,
        "turns_per_rollout_median": statistics.median([r["turns"] for r in ex]) if ex else None,
        "tool_calls_per_rollout_median": statistics.median([r["tool_calls"] for r in ex]) if ex else None,
        "agent_wall_min_median": round(statistics.median([r["agent_wall_s"] for r in ex if r.get("agent_wall_s")]) / 60, 1)
        if any(r.get("agent_wall_s") for r in ex) else None,
        "pct_rollouts_git_commit": pct(sum(bool(r["git_commit_calls"]) for r in ex), len(ex)),
        "pct_rollouts_with_task_subagent": pct(sum(r["task_calls"] > 0 for r in ex), len(ex)),
    }


def window_block(rows, deg_of):
    ex = [r for r in rows if r.get("export")]
    comp = [r for r in ex if r.get("compactions", 0) > 0]
    nocomp = [r for r in ex if not r.get("compactions")]
    prompts = [r.get("prompt_max_main", 0) for r in ex]
    att = [deg_of(r) for r in rows]
    trig = [p for r in comp for p in r.get("compaction_trigger_prompts") or [] if p]
    gc, gn = [r for r in comp if graded(r)], [r for r in nocomp if graded(r)]
    overflow = [sum(v for k, v in (a.get("errors") or {}).items() if k.endswith("context_overflow")) for a in att]
    return {
        "rollouts": len(rows), "with_export": len(ex),
        "compacted_rollouts": len(comp), "pct_rollouts_compacted": pct(len(comp), len(ex)),
        "compactions": sum(r.get("compactions", 0) for r in ex),
        "compactions_hist": {str(k): v for k, v in sorted(collections.Counter(r.get("compactions", 0) for r in ex).items())},
        "compaction_trigger_prompt_median": statistics.median(trig) if trig else None,
        "pass_at_1_compacted": pct(sum(passed(r) for r in gc), len(gc)), "graded_compacted": len(gc),
        "pass_at_1_not_compacted": pct(sum(passed(r) for r in gn), len(gn)), "graded_not_compacted": len(gn),
        "compacted_with_commit_after_last_compaction": sum(bool(r.get("commit_after_last_compaction")) for r in comp),
        "compacted_ended_on_compaction": sum(bool(r.get("ended_on_compaction")) for r in comp),
        "compacted_early_stop": sum((r.get("action_turns_after_last_compaction") or 0) <= 3
                                    and not r.get("commit_after_last_compaction") for r in comp),
        "action_turns_after_last_compaction_median": statistics.median(
            [r.get("action_turns_after_last_compaction") or 0 for r in comp]) if comp else None,
        "main_prompt_max_median": statistics.median(prompts) if prompts else None,
        "main_prompt_max_p90": quantile(prompts, 0.9), "main_prompt_max_max": max(prompts, default=None),
        "rollouts_main_prompt_over_200k": sum(r.get("steps_over_200k", 0) > 0 for r in ex),
        "rollouts_main_prompt_over_242144": sum(r.get("steps_over_242144", 0) > 0 for r in ex),
        "pct_rollouts_main_prompt_over_242144": pct(sum(r.get("steps_over_242144", 0) > 0 for r in ex), len(ex)),
        "hub_calls_over_242144_any_session": sum(a.get("over_compact_point", 0) for a in att),
        "rollouts_with_hub_call_over_242144": sum(a.get("over_compact_point", 0) > 0 for a in att),
        "hub_prompt_max_any_session": max((a.get("prompt_max", 0) for a in att), default=None),
        "length_cut_replies": sum(a.get("length_cut", 0) for a in att),
        "rollouts_with_length_cut": sum(a.get("length_cut", 0) > 0 for a in att),
        "context_overflow_replies": sum(overflow), "rollouts_with_context_overflow": sum(x > 0 for x in overflow),
        "non200_by_kind": dict(sum((collections.Counter(a.get("errors") or {}) for a in att), collections.Counter())),
        "final_finish": dict(collections.Counter(str(r.get("final_finish")) for r in ex)),
        "message_errors": dict(collections.Counter(e for r in ex for e in r.get("message_errors") or []))}


def broken_block(rows, deg_of, ended_of):
    """Broken replies of a rollout set: per reply (each attempt's own journal) and per rollout; pass@1 raw / excl."""
    att = [deg_of(r) for r in rows]
    replies = sum(a.get("replies_n", 0) for a in att)
    bad = sum(a.get("degenerate_n", 0) for a in att)
    kinds = sum((collections.Counter(a.get("kinds") or {}) for a in att), collections.Counter())
    g = [r for r in rows if graded(r)]
    clean = [r for r in g if not ended_of(r)]
    ended = [r for r in rows if ended_of(r)]
    return {"rollouts": len(rows), "replies": replies, "degenerate_replies": bad, "kinds": dict(kinds),
            "per_1000_replies": round(1000 * bad / replies, 2) if replies else None,
            "rollouts_with_degenerate_reply": sum(a.get("degenerate_n", 0) > 0 for a in att),
            "pct_rollouts_with_degenerate_reply": pct(sum(a.get("degenerate_n", 0) > 0 for a in att), len(rows)),
            "rollouts_ended_by_degenerate_reply": len(ended), "pct_rollouts_ended": pct(len(ended), len(rows)),
            "ended_kinds": dict(collections.Counter(deg_of(r).get("last_kind") for r in ended)),
            "ended_and_failed": sum(r.get("status") == "fail" for r in ended),
            "ended_and_passed": sum(r.get("status") == "pass" for r in ended),
            "subsession_ended": sum(a.get("ended_subsession", 0) > 0 for a in att),
            "with_tool_call_degenerate": sum(a.get("degenerate_with_tool", 0) > 0 for a in att),
            "pass_at_1_raw": pct(sum(passed(r) for r in g), len(g)), "graded": len(g),
            "pass_at_1_excluding_broken_ended": pct(sum(passed(r) for r in clean), len(clean)),
            "graded_excluding_broken_ended": len(clean)}


def cost_block(attempt_rows, rollouts=None):
    costs = [a["cost"] for a in attempt_rows]
    prompt = sum(a["prompt_tokens"] for a in attempt_rows)
    out = {"attempts": len(attempt_rows), "usd": round(sum(costs), 2),
           "usd_per_attempt_mean": round(statistics.fmean(costs), 3) if costs else None,
           "usd_per_attempt_median": round(statistics.median(costs), 3) if costs else None,
           "replies": sum(a["replies_n"] for a in attempt_rows),
           "replies_with_cost": sum(a["cost_calls"] for a in attempt_rows),
           "prompt_Mtok": round(prompt / 1e6, 2), "cached_pct": pct(sum(a["cached_tokens"] for a in attempt_rows), prompt),
           "completion_Mtok": round(sum(a["completion_tokens"] for a in attempt_rows) / 1e6, 3)}
    if rollouts:
        out["usd_per_rollout"] = round(sum(costs) / rollouts, 3)
        out["rollouts"] = rollouts
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="final-report")
    parser.add_argument("--degenerate", default="v4-final")
    args = parser.parse_args()
    results = {p.stem.removeprefix("kimi-"): read_json(p) for p in sorted(HERE.glob("results/kimi-*-g*.json"))}
    v4 = read_jsonl(ANALYSIS / "features" / "kimi4-all.jsonl")
    v3 = read_jsonl(ANALYSIS / "features" / "kimi3.jsonl")
    one_m = read_jsonl(ANALYSIS / "features" / "kimi-1m.jsonl")
    degenerate = read_json(ANALYSIS / f"degenerate-{args.degenerate}.json", {})
    attempts_rows = read_jsonl(ANALYSIS / f"degenerate-{args.degenerate}-attempts.jsonl")
    by_attempt = {}
    for a in attempts_rows:
        by_attempt.setdefault(a["attempt"], a)
        by_attempt.setdefault(a["generation_dir"], a)
    legs = read_jsonl(HERE / "legs.jsonl")
    key_of_job = {str(leg["job"]): leg.get("hub_key") for leg in legs if leg.get("event") == "start"}
    for path in HERE.glob("probes/kimi/*/kr4-legs.jsonl"):
        for leg in read_jsonl(path):
            key_of_job.setdefault(str(leg.get("job")), leg.get("hub_key"))

    def deg_of(r):
        return by_attempt.get(resolved(r.get("export_dir") or r.get("attempt_dir"))) or {}

    def ended_v4(r):
        return bool(r.get("ended_by_broken_reply"))

    def ended_deg(r):
        return bool(deg_of(r).get("ended"))

    task_order = {}
    for name, data in results.items():
        for task in data["tasks"]:
            task_order.setdefault(task, (TASKSETS.index(data["taskset"]), len(task_order)))
    first = [r for r in v4 if r.get("role") == "first"]
    reruns = [r for r in v4 if r.get("role") == "rerun"]
    final = [r for r in v4 if r.get("final")]
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request": REQUEST, "rerun_rule": RERUN_RULE,
              "arms": ARMS}

    # ---- configuration (results files, legs)
    any_result = next(iter(results.values()))
    starts = [leg for leg in legs if leg.get("event") == "start"]
    main_starts = [leg for leg in starts if leg.get("campaign") in results]
    report["config"] = {
        "model_id": any_result.get("model_id"), "spec": any_result.get("spec"),
        "context_window": any_result.get("context_window"), "prompt": any_result.get("prompt"),
        "system_prompt": any_result.get("system_prompt"),
        "sandbox_credential_by_taskset": {ts: sorted({r.get("sandbox_credential") for r in results.values()
                                                      if r["taskset"] == ts}) for ts in TASKSETS},
        "max_concurrency_by_taskset": {ts: sorted({r.get("max_concurrency") for r in results.values()
                                                   if r["taskset"] == ts}) for ts in TASKSETS},
        "tasks_by_taskset": {ts: sum(len(r["tasks"]) for r in results.values() if r["taskset"] == ts) for ts in TASKSETS},
        "campaigns": {name: {"jobs": sorted({x["job"] for v in (r.get("hub_keys_by_campaign") or {}).values() for x in v}),
                             "hub_keys": sorted({x["hub_key"] for v in (r.get("hub_keys_by_campaign") or {}).values()
                                                 for x in v}),
                             "legs": sum(1 for leg in main_starts if leg["campaign"] == name),
                             "rerun_campaigns": sorted(c for c in (r.get("hub_keys_by_campaign") or {}) if c != name),
                             "finished": r.get("finished_at")} for name, r in results.items()},
        "legs_main": len(main_starts), "legs_pilot": len(starts) - len(main_starts),
        "first_leg_start": min((leg["at"] for leg in main_starts), default=None),
        "last_results_written": max((r.get("finished_at") or "" for r in results.values()), default=None),
        "hub_keys_used_by_legs": dict(collections.Counter(leg.get("hub_key") for leg in starts))}

    # ---- (1) the 50 sel50 tasks: v4 vs v3 sentence arm vs earlier 1M
    s_first = [r for r in first if r["taskset"] == "sel50"]
    s_final = [r for r in final if r["taskset"] == "sel50"]
    s_reruns = [r for r in reruns if r["taskset"] == "sel50"]
    sel = {
        "v4_first": s_first, "v4_after_rerun": s_final, "v3": v3, "1m": one_m}
    sel_clean = {"v4_first": [r for r in s_first if not ended_v4(r)],
                 "v4_after_rerun": [r for r in s_final if not ended_v4(r)],
                 "v3": [r for r in v3 if not ended_deg(r)], "1m": [r for r in one_m if not ended_deg(r)]}
    num, den = (lambda r: r["action_with_reasoning"]), (lambda r: r["action_turns"])
    no_deg = {k: [r for r in rows if not deg_of(r).get("degenerate_n")] for k, rows in sel.items()}
    comparisons = degenerate.get("comparisons") or {}
    report["sel50"] = {
        "tasks": len({r["task_id"] for r in s_first}),
        "pass_at_1": {k: pass_block(rows) for k, rows in sel.items()},
        "pass_at_1_excluding_broken_ended": {k: pass_block(rows) for k, rows in sel_clean.items()},
        "paired_pass": {
            "v4_first_minus_1m": paired_boot(s_first, one_m, passed, graded),
            "v4_after_rerun_minus_1m": paired_boot(s_final, one_m, passed, graded),
            "v4_first_minus_v3": paired_boot(s_first, v3, passed, graded),
            "v4_after_rerun_minus_v3": paired_boot(s_final, v3, passed, graded),
            "v4_first_excl_minus_v3_excl": paired_boot(sel_clean["v4_first"], sel_clean["v3"], passed, graded),
            "v4_first_excl_minus_1m": paired_boot(sel_clean["v4_first"], one_m, passed, graded),
            "v3_minus_1m": paired_boot(v3, one_m, passed, graded)},
        "reasoning": {k: reasoning(rows) for k, rows in sel.items()},
        "reasoning_without_degenerate_rollouts": {k: reasoning(rows) for k, rows in no_deg.items()},
        "paired_reasoning": {
            "v4_first_minus_1m": paired_boot(with_turns(s_first), with_turns(one_m), num, den),
            "v4_first_minus_v3": paired_boot(with_turns(s_first), with_turns(v3), num, den),
            "v3_minus_1m": paired_boot(with_turns(v3), with_turns(one_m), num, den),
            "v4_after_rerun_minus_1m": paired_boot(with_turns(s_final), with_turns(one_m), num, den),
            "v4_minus_1m_both_without_degenerate": paired_boot(with_turns(no_deg["v4_first"]), with_turns(no_deg["1m"]),
                                                               num, den),
            "v4_minus_v3_both_without_degenerate": paired_boot(with_turns(no_deg["v4_first"]), with_turns(no_deg["v3"]),
                                                               num, den)},
        "window": {k: window_block(rows, deg_of) for k, rows in sel.items()},
        "broken": {"v4_first": broken_block(s_first, deg_of, ended_v4),
                   "v4_reruns": broken_block(s_reruns, deg_of, ended_v4),
                   "v4_after_rerun": broken_block(s_final, deg_of, ended_v4),
                   "v3": broken_block(v3, deg_of, ended_deg), "1m": broken_block(one_m, deg_of, ended_deg)},
        "broken_per_reply_all_attempts": {
            "v4_originals_no_reruns": (degenerate.get("per_reply") or {}).get("v4_sel50_originals_no_reruns"),
            "v4_all_attempts": (degenerate.get("per_reply") or {}).get("v4_sel50_all_attempts"),
            "v3_all_attempts": (degenerate.get("per_reply") or {}).get("v3_all_attempts"),
            "1m_all_attempts": (degenerate.get("per_reply") or {}).get("1m_all_attempts")},
        "broken_tests": {k: {kk: vv for kk, vv in v.items() if kk not in ("control", "new")}
                         for k, v in comparisons.items() if k.startswith("sel50_")},
        "cost": {"v4_first": cost_block([deg_of(r) for r in s_first if deg_of(r)], len(s_first)),
                 "v3": cost_block([deg_of(r) for r in v3 if deg_of(r)], len(v3)),
                 "1m": cost_block([deg_of(r) for r in one_m if deg_of(r)], len(one_m))}}

    # ---- (2) the full v4 set, by taskset
    rerun_entries = []
    for name, data in results.items():
        for task_id, entries in (data.get("broken_reply_reruns") or {}).items():
            for e in entries:
                first_rec = next(x for x in data["first_attempt_records"][task_id] if x["slot"] == e["slot"])
                final_rec = next(x for x in data["records"][task_id] if x["slot"] == e["slot"])
                original = by_attempt.get(resolved(e["original_attempt"])) or {}
                rerun = by_attempt.get(resolved(e.get("rerun_attempt"))) or {}
                rerun_entries.append({
                    "campaign": name, "taskset": data["taskset"], "task_id": task_id, "slot": e["slot"],
                    "original_status": e["original_status"], "original_f2p": first_rec.get("f2p"),
                    "original_patch_chars": first_rec.get("patch_chars"), "original_kind": e.get("original_last_reply_kind"),
                    "original_replies": original.get("replies_n"),
                    "broken_reply_completion_tokens": (original.get("degenerate") or [{}])[-1].get("completion_tokens"),
                    "rerun_campaign": e["rerun_campaign"], "rerun_slot": e["rerun_slot"],
                    "rerun_status": e.get("rerun_status"), "rerun_f2p": final_rec.get("f2p") if e.get("replaced") else None,
                    "rerun_broken": e.get("rerun_ended_by_broken_reply"), "rerun_kind": rerun.get("last_kind"),
                    "outcome": e.get("outcome"), "replaced": e.get("replaced"), "rerun_replies": rerun.get("replies_n"),
                    "original_attempt": e["original_attempt"], "rerun_attempt": e.get("rerun_attempt")})
    rerun_entries.sort(key=lambda e: (TASKSETS.index(e["taskset"]), e["campaign"], e["task_id"], e["slot"]))
    full = {}
    for ts in list(TASKSETS) + ["all"]:
        ts_first = [r for r in first if ts == "all" or r["taskset"] == ts]
        ts_final = [r for r in final if ts == "all" or r["taskset"] == ts]
        ts_reruns = [r for r in reruns if ts == "all" or r["taskset"] == ts]
        ts_entries = [e for e in rerun_entries if ts == "all" or e["taskset"] == ts]
        outcomes = collections.Counter(e["outcome"] for e in ts_entries)
        ts_attempts = [a for a in attempts_rows if a["arm"] == "v4" and (ts == "all" or a["taskset"] == ts)]
        originals = [a for a in ts_attempts if a["role"] != "rerun" and not a.get("duplicate_generation")]
        full[ts] = {
            "tasks": len({r["task_id"] for r in ts_first}), "slots": len(ts_first),
            "pass_at_1_first": pass_block(ts_first),
            "pass_at_1_first_excluding_broken_ended": pass_block([r for r in ts_first if not ended_v4(r)]),
            "pass_at_1_after_rerun": pass_block(ts_final),
            "pass_at_1_after_rerun_excluding_broken_ended": pass_block([r for r in ts_final if not ended_v4(r)]),
            "broken_first": broken_block(ts_first, deg_of, ended_v4),
            "broken_reruns": broken_block(ts_reruns, deg_of, ended_v4),
            "broken_final": broken_block(ts_final, deg_of, ended_v4),
            "broken_per_reply_originals_no_reruns": {
                "replies": sum(a["replies_n"] for a in originals), "degenerate": sum(a["degenerate_n"] for a in originals),
                "per_1000_replies": round(1000 * sum(a["degenerate_n"] for a in originals)
                                          / max(1, sum(a["replies_n"] for a in originals)), 2)},
            "reruns": len(ts_entries),
            "rerun_outcomes": {k: outcomes.get(k, 0) for k in ("pass", "fail", "broken_again", "infra")},
            "rerun_replaced": sum(bool(e["replaced"]) for e in ts_entries),
            "reasoning_first": reasoning(ts_first), "reasoning_final": reasoning(ts_final),
            "window_first": window_block(ts_first, deg_of), "window_all_graded": window_block(ts_first + ts_reruns, deg_of)}
    report["v4_by_taskset"] = full
    outcomes = collections.Counter(e["outcome"] for e in rerun_entries)
    report["reruns"] = {
        "reruns": len(rerun_entries), "outcomes": {k: outcomes.get(k, 0) for k in ("pass", "fail", "broken_again", "infra")},
        "replaced": sum(bool(e["replaced"]) for e in rerun_entries), "third_attempts": 0,
        "rerun_campaigns": sorted({e["rerun_campaign"] for e in rerun_entries}),
        "original_kinds": dict(collections.Counter(e["original_kind"] for e in rerun_entries)),
        "originals_that_passed": [{k: e[k] for k in ("campaign", "task_id", "slot", "original_f2p", "outcome")}
                                  for e in rerun_entries if e["original_status"] == "pass"],
        "broken_again": [{k: e[k] for k in ("campaign", "task_id", "slot", "rerun_kind", "rerun_status", "rerun_f2p")}
                         for e in rerun_entries if e["outcome"] == "broken_again"],
        "tasks_with_reruns_hist": dict(sorted(collections.Counter(
            collections.Counter(e["task_id"] for e in rerun_entries).values()).items())),
        "list": rerun_entries,
        "detector_check": degenerate.get("check_ended_vs_results")}

    # ---- (3) cost by key (v4's own replies; LiteLLM usage.cost)
    v4_attempts = [a for a in attempts_rows if a["arm"] == "v4"]
    pilot_attempts = [a for a in attempts_rows if a["arm"] == "pilot"]
    by_key = collections.defaultdict(list)
    for a in v4_attempts + pilot_attempts:
        by_key[key_of_job.get(str(a.get("job"))) or "unknown"].append(a)
    budget = [row for row in read_jsonl(HERE / "budget-log.jsonl") if row.get("key") in KEYS and row.get("status") == 200]
    last_budget = {key: next((row for row in reversed(budget) if row["key"] == key), None) for key in KEYS}
    progress_rows = read_json(HERE / "PROGRESS-rows.json", []) or []
    first_start = {key: next((leg for leg in starts if leg.get("hub_key") == key and (leg.get("budget") or {}).get("spend")
                              is not None), None) for key in KEYS}
    v5 = read_json(V5_REPORT, {}) or {}
    v5_new2 = ((v5.get("cost") or {}).get("by_key") or {}).get("NVAPI_KEY_NEW_2") or {}
    total_v4 = sum(a["cost"] for a in v4_attempts)
    total_pilot = sum(a["cost"] for a in pilot_attempts)
    delta = {}
    for key in KEYS:
        start, last = first_start.get(key), last_budget.get(key)
        if start and last:
            delta[key] = {"from": start["budget"]["at"], "spend_from": round(start["budget"]["spend"], 2),
                          "to": last["at"], "spend_to": round(last["spend"], 2),
                          "delta": round(last["spend"] - start["budget"]["spend"], 2), "remaining": last.get("remaining")}
        else:
            delta[key] = {"delta": 0.0 if not start else None, "note": "no v4 leg started on this key" if not start else
                          "no budget record", "remaining_last_check": (last or {}).get("remaining")}
    v5_total = (v5_new2.get("batch_usd") or 0) + (v5_new2.get("pilot_usd") or 0)
    report["cost"] = {
        "by_key": {key: {"usd_total": round(sum(a["cost"] for a in by_key.get(key, [])), 2),
                         "first_attempts_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["role"] == "first"), 2),
                         "reruns_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["role"] == "rerun"), 2),
                         "retried_attempts_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["role"] == "retried"), 2),
                         "pilots_usd": round(sum(a["cost"] for a in by_key.get(key, []) if a["arm"] == "pilot"), 2),
                         "attempts": len(by_key.get(key, [])),
                         "graded_rollouts": sum(a["role"] in ("first", "rerun") for a in by_key.get(key, [])),
                         "usd_per_graded_rollout": round(sum(a["cost"] for a in by_key.get(key, []) if a["arm"] == "v4")
                                                         / max(1, sum(a["role"] in ("first", "rerun")
                                                                      for a in by_key.get(key, []))), 3)
                         if by_key.get(key) else None}
                   for key in list(KEYS) + sorted(k for k in by_key if k not in KEYS)},
        "jobs_to_keys": dict(sorted(key_of_job.items())),
        "batch_usd": round(total_v4, 2), "pilots_usd": round(total_pilot, 2), "total_usd": round(total_v4 + total_pilot, 2),
        "all_attempts": cost_block(v4_attempts),
        "first_attempts": cost_block([a for a in v4_attempts if a["role"] == "first"]),
        "reruns": cost_block([a for a in v4_attempts if a["role"] == "rerun"]),
        "retried_attempts": cost_block([a for a in v4_attempts if a["role"] == "retried"]),
        "pilots": cost_block(pilot_attempts),
        "usd_per_graded_rollout": round(total_v4 / max(1, len(first) + len(reruns)), 3),
        "usd_per_final_slot": round(total_v4 / max(1, len(final)), 3),
        "by_taskset": {ts: dict(cost_block([a for a in v4_attempts if a["taskset"] == ts]),
                                slots=sum(r["taskset"] == ts for r in final),
                                usd_per_final_slot=round(sum(a["cost"] for a in v4_attempts if a["taskset"] == ts)
                                                         / max(1, sum(r["taskset"] == ts for r in final)), 3),
                                usd_per_first_attempt_mean=round(statistics.fmean(
                                    [a["cost"] for a in v4_attempts if a["taskset"] == ts and a["role"] == "first"]), 3))
                       for ts in TASKSETS},
        "key_spend_delta": delta,
        "reconciliation_NVAPI_KEY_NEW_2": {
            "key_delta": (delta.get("NVAPI_KEY_NEW_2") or {}).get("delta"),
            "v4_journal_usd_incl_pilots": round(total_v4 + total_pilot, 2),
            "v5_journal_usd_incl_pilot": round(v5_total, 2),
            "residual": round(((delta.get("NVAPI_KEY_NEW_2") or {}).get("delta") or 0) - total_v4 - total_pilot - v5_total, 2),
            "note": "kimi-rollouts-v5 (official2, 2026-10-29 02:09-10:32) used the same key; its journal cost is from its "
                    "analysis/final-report.json (read-only)"},
        "progress_py_last_row": {k: (progress_rows[-1] if progress_rows else {}).get(k)
                                 for k in ("time", "spend", "spent", "usd_per_rollout")}}

    # ---- (4) SFT-usable passes
    usable, excluded = [], []
    for r in v4:
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
            kinds = ",".join(f"{d['kind']}@reply{d['index']}/{a.get('replies_n')}" + ("" if d["tool"] else " no-tool")
                             for d in a["degenerate"])
            reasons.append(f"broken reply in the attempt ({kinds}; "
                           + ("ended a sub-session" if a.get("ended_subsession") else "main session") + ")")
        notes = []
        if r.get("broken_text_parts") or (r.get("broken_tool_outputs") and a.get("degenerate_n")):
            reasons.append(f"[PAD]/special text in the main session (assistant parts {r.get('broken_text_parts')}, "
                           f"tool outputs {r.get('broken_tool_outputs')})")
        elif r.get("broken_tool_outputs"):
            notes.append(f"[PAD]/special-token text in {r.get('broken_tool_outputs')} tool output(s) of the main session, "
                         "but no degenerate hub reply in the attempt: repository / command text, kept")
        if r.get("ended_by_broken_reply"):
            reasons.append("ended by a broken reply")
        entry = {"taskset": r["taskset"], "task_id": r["task_id"], "run": r.get("run"), "slot": r["slot"],
                 "role": r["role"], "campaign": r["campaign"], "final": r.get("final"), "f2p": r.get("f2p"),
                 "p2p": r.get("p2p"), "patch_chars": r.get("patch_chars"), "turns": r.get("turns"),
                 "tool_calls": r.get("tool_calls"), "compactions": r.get("compactions"),
                 "main_prompt_max": r.get("prompt_max_main"),
                 "pct_action_reasoned": pct(r.get("action_with_reasoning", 0), r.get("action_turns", 0)),
                 "usd": round(a.get("cost", 0.0), 3) if a else None,
                 "attempt_dir": r["attempt_dir"], "export": str(Path(r.get("export_dir") or r["attempt_dir"]) / "export.json")}
        if notes:
            entry["notes"] = notes
        (excluded if reasons else usable).append(dict(entry, reasons=reasons) if reasons else entry)
    usable.sort(key=lambda e: (task_order.get(e["task_id"], (9, 0)), e["slot"], e["role"]))
    per_task_usable = collections.Counter(e["task_id"] for e in usable)
    tasks_of = {ts: sorted({r["task_id"] for r in first if r["taskset"] == ts}) for ts in TASKSETS}
    reason_kinds = collections.Counter()
    for e in excluded:
        for reason in e["reasons"]:
            reason_kinds[reason.split(" (")[0].split(" ")[0] if reason.startswith(("F2P", "P2P", "reward")) else
                         reason.split(" (")[0]] += 1
    report["sft_usable"] = {
        "criteria": "graded pass; evaluation completed; reward 1; every F2P and P2P test passing; no verifier or agent "
                    "error; non-empty patch; exported session; no degenerate hub reply anywhere in the attempt (sub-"
                    "sessions included) and no [PAD] / special-token text in the main session's assistant parts (nor in "
                    "its tool outputs when the attempt had a degenerate reply). Tool-output matches without any "
                    "degenerate reply are repository text: kept, flagged in notes. No LLM cheating audit.",
        "passes": len(usable) + len(excluded), "usable": len(usable), "excluded_count": len(excluded),
        "kept_with_notes": [{k: e[k] for k in ("taskset", "task_id", "slot", "role", "campaign", "notes")}
                            for e in usable if e.get("notes")],
        "excluded_reasons": dict(reason_kinds),
        "by_taskset": {ts: {"tasks": len(tasks_of[ts]),
                            "passes_all_graded": sum(1 for r in v4 if r["taskset"] == ts and passed(r)),
                            "passes_final_slots": sum(1 for r in final if r["taskset"] == ts and passed(r)),
                            "usable": sum(e["taskset"] == ts for e in usable),
                            "usable_first_attempts": sum(e["taskset"] == ts and e["role"] == "first" for e in usable),
                            "usable_reruns": sum(e["taskset"] == ts and e["role"] == "rerun" for e in usable),
                            "usable_without_compaction": sum(e["taskset"] == ts and not e.get("compactions")
                                                             for e in usable),
                            "excluded": sum(e["taskset"] == ts for e in excluded),
                            "tasks_with_usable": sum(1 for t in tasks_of[ts] if per_task_usable.get(t)),
                            "tasks_without_usable": [t for t in tasks_of[ts] if not per_task_usable.get(t)],
                            "usable_per_task_hist": {str(k): v for k, v in sorted(collections.Counter(
                                per_task_usable.get(t, 0) for t in tasks_of[ts]).items())}}
                       for ts in TASKSETS},
        "usable_without_compaction": sum(not e.get("compactions") for e in usable),
        "excluded": excluded, "list": usable}

    # ---- (5) per task
    per_task = []
    v3_by_task = collections.defaultdict(list)
    for r in v3:
        v3_by_task[r["task_id"]].append(r)
    m_by_task = collections.defaultdict(list)
    for r in one_m:
        m_by_task[r["task_id"]].append(r)
    for task in sorted({r["task_id"] for r in first}, key=lambda t: task_order.get(t, (9, 0))):
        t_first = [r for r in first if r["task_id"] == task]
        t_final = [r for r in final if r["task_id"] == task]
        t_all = [r for r in v4 if r["task_id"] == task]
        t_entries = [e for e in rerun_entries if e["task_id"] == task]
        rs = reasoning(t_first)
        per_task.append({
            "task_id": task, "taskset": t_first[0]["taskset"], "run": t_first[0].get("run"),
            "campaign": t_first[0]["campaign"],
            "first_pass": sum(passed(r) for r in t_first), "first_graded": sum(graded(r) for r in t_first),
            "broken_ended_slots": [r["slot"] for r in t_first if ended_v4(r)],
            "rerun_outcomes": [f"{e['slot']}:{e['outcome']}" for e in t_entries],
            "final_pass": sum(passed(r) for r in t_final), "final_graded": sum(graded(r) for r in t_final),
            "v3_pass": sum(passed(r) for r in v3_by_task.get(task, [])) if task in v3_by_task else None,
            "v3_graded": sum(graded(r) for r in v3_by_task.get(task, [])) if task in v3_by_task else None,
            "1m_pass": sum(passed(r) for r in m_by_task.get(task, [])) if task in m_by_task else None,
            "1m_graded": sum(graded(r) for r in m_by_task.get(task, [])) if task in m_by_task else None,
            "pct_action_reasoned_first": rs["pct_action_turns_with_reasoning"],
            "turns_median_first": rs["turns_per_rollout_median"],
            "compacted_rollouts": sum(r.get("compactions", 0) > 0 for r in t_all),
            "main_prompt_max": max((r.get("prompt_max_main", 0) for r in t_all), default=None),
            "usd_per_rollout": round(statistics.fmean([deg_of(r).get("cost", 0.0) for r in t_all]), 2) if t_all else None,
            "sft_usable": per_task_usable.get(task, 0)})
    report["per_task"] = per_task

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
    s = report["sel50"]
    print(json.dumps({"sel50_pass": {k: (v["pass"], v["graded"], v["pass_at_1"], v["ci95_task_bootstrap"])
                                     for k, v in s["pass_at_1"].items()},
                      "sel50_pass_excl": {k: (v["pass"], v["graded"], v["pass_at_1"])
                                          for k, v in s["pass_at_1_excluding_broken_ended"].items()},
                      "sel50_paired_pass": s["paired_pass"],
                      "sel50_reasoning": {k: (v["pct_action_turns_with_reasoning"], v["ci95_pct_action_turns_with_reasoning"])
                                          for k, v in s["reasoning"].items()},
                      "sel50_paired_reasoning": s["paired_reasoning"],
                      "sel50_window": {k: {kk: v[kk] for kk in ("pct_rollouts_compacted", "compacted_rollouts",
                                                                "rollouts_main_prompt_over_242144", "length_cut_replies",
                                                                "context_overflow_replies")}
                                       for k, v in s["window"].items()},
                      "sel50_broken": {k: {kk: v[kk] for kk in ("replies", "degenerate_replies", "per_1000_replies",
                                                                "rollouts_ended_by_degenerate_reply", "pct_rollouts_ended")}
                                       for k, v in s["broken"].items()},
                      "full": {ts: {"first": (v["pass_at_1_first"]["pass"], v["pass_at_1_first"]["graded"],
                                              v["pass_at_1_first"]["pass_at_1"]),
                                    "excl": v["pass_at_1_first_excluding_broken_ended"]["pass_at_1"],
                                    "after": (v["pass_at_1_after_rerun"]["pass"], v["pass_at_1_after_rerun"]["pass_at_1"]),
                                    "reruns": v["rerun_outcomes"]} for ts, v in full.items()},
                      "cost": {k: report["cost"][k] for k in ("batch_usd", "pilots_usd", "total_usd", "usd_per_graded_rollout",
                                                              "usd_per_final_slot", "reconciliation_NVAPI_KEY_NEW_2")},
                      "sft": {ts: {k: v[k] for k in ("usable", "usable_first_attempts", "usable_reruns",
                                                     "usable_without_compaction", "excluded", "tasks_with_usable")}
                              for ts, v in report["sft_usable"]["by_taskset"].items()}},
                     indent=1, ensure_ascii=False))


def fmt_pass(block):
    return f"{block['pass']}/{block['graded']} = {block['pass_at_1']}% (CI {block['ci95_task_bootstrap']})"


def fmt_diff(d):
    return f"{d['diff_points']:+.1f} (CI {d['ci95'][0]:+.1f} to {d['ci95'][1]:+.1f}, {d['tasks']} tasks)" if d else "-"


def markdown(report) -> str:
    s = report["sel50"]
    full = report["v4_by_taskset"]
    c = report["cost"]
    u = report["sft_usable"]
    rr = report["reruns"]
    cols = ("v4_first", "v4_after_rerun", "v3", "1m")
    names = {"v4_first": "v4 first attempts", "v4_after_rerun": "v4 after re-run", "v3": "v3 sentence (256k)",
             "1m": "earlier Kimi (1M)"}
    lines = [f"# kimi-rollouts-v4 final report (phase_report.py, {report['generated_at']})", "",
             f"Request (hengyuf, 2026-10-28): {report['request']}. Re-run rule: {report['rerun_rule']}.", "",
             f"## 1. The 50 sel50 tasks: {', '.join(report['arms'].values())}", "",
             "| metric | " + " | ".join(names[k] for k in cols) + " |", "|---|" + "---|" * len(cols)]
    rows = [
        ("pass@1 raw", lambda k: fmt_pass(s["pass_at_1"][k])),
        ("pass@1 excl. broken-ended", lambda k: fmt_pass(s["pass_at_1_excluding_broken_ended"][k])),
        ("tasks >=1 pass / 5 of 5 / none", lambda k: f"{s['pass_at_1'][k]['tasks_any_pass']} / "
                                                     f"{s['pass_at_1'][k]['tasks_all_pass']} / {s['pass_at_1'][k]['tasks_no_pass']}"),
        ("% action turns with reasoning", lambda k: f"{s['reasoning'][k]['pct_action_turns_with_reasoning']} "
                                                    f"(CI {s['reasoning'][k]['ci95_pct_action_turns_with_reasoning']})"),
        ("  same, rollouts without broken replies", lambda k: str(s["reasoning_without_degenerate_rollouts"][k]
                                                                  ["pct_action_turns_with_reasoning"])),
        ("compacted rollouts", lambda k: f"{s['window'][k]['compacted_rollouts']}/{s['window'][k]['with_export']} "
                                         f"({s['window'][k]['pct_rollouts_compacted']}%)"),
        ("main prompt > 242,144 (rollouts)", lambda k: f"{s['window'][k]['rollouts_main_prompt_over_242144']} "
                                                       f"({s['window'][k]['pct_rollouts_main_prompt_over_242144']}%)"),
        ("largest main-session prompt", lambda k: str(s["window"][k]["main_prompt_max_max"])),
        ("replies cut at output cap / context-overflow refusals", lambda k: f"{s['window'][k]['length_cut_replies']} / "
                                                                            f"{s['window'][k]['context_overflow_replies']}"),
        ("broken replies / replies (final attempts)", lambda k: f"{s['broken'][k]['degenerate_replies']}/"
                                                                f"{s['broken'][k]['replies']} = "
                                                                f"{s['broken'][k]['per_1000_replies']} per 1000"),
        ("rollouts ended by a broken reply", lambda k: f"{s['broken'][k]['rollouts_ended_by_degenerate_reply']}/"
                                                       f"{s['broken'][k]['rollouts']} ({s['broken'][k]['pct_rollouts_ended']}%)"),
        ("rollouts with any broken reply", lambda k: f"{s['broken'][k]['rollouts_with_degenerate_reply']} "
                                                     f"({s['broken'][k]['pct_rollouts_with_degenerate_reply']}%)"),
        ("turns per rollout (median)", lambda k: str(s["reasoning"][k]["turns_per_rollout_median"])),
        ("agent wall min (median)", lambda k: str(s["reasoning"][k]["agent_wall_min_median"]))]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(k) for k in cols) + " |")
    lines += ["", "Paired task bootstrap (points): pass@1 " + "; ".join(f"{k} {fmt_diff(v)}" for k, v in s["paired_pass"].items()),
              "", "% action turns with reasoning: " + "; ".join(f"{k} {fmt_diff(v)}" for k, v in s["paired_reasoning"].items()),
              "", "Broken replies, every attempt (per 1000 replies): " + "; ".join(
                  f"{k} {v.get('degenerate')}/{v.get('replies')} = {v.get('per_1000_replies')}"
                  for k, v in s["broken_per_reply_all_attempts"].items() if v),
              "", "Tests: " + json.dumps(s["broken_tests"]),
              "", "Cost on sel50 (LiteLLM usage.cost of the analysed attempts): " + "; ".join(
                  f"{k} ${v['usd']} ({v['usd_per_rollout']}/rollout)" for k, v in s["cost"].items()),
              "", "## 2. v4 full set by taskset", "",
              "| | sel50 | rest47 | baihe | all |", "|---|---|---|---|---|"]
    keys = list(TASKSETS) + ["all"]
    table = [
        ("tasks / slots", lambda v: f"{v['tasks']} / {v['slots']}"),
        ("pass@1 first attempts", lambda v: fmt_pass(v["pass_at_1_first"])),
        ("pass@1 first, excl. broken-ended", lambda v: fmt_pass(v["pass_at_1_first_excluding_broken_ended"])),
        ("pass@1 after the one re-run", lambda v: fmt_pass(v["pass_at_1_after_rerun"])),
        ("tasks >=1 pass / 5 of 5 / none (after re-run)",
         lambda v: f"{v['pass_at_1_after_rerun']['tasks_any_pass']} / {v['pass_at_1_after_rerun']['tasks_all_pass']} / "
                   f"{v['pass_at_1_after_rerun']['tasks_no_pass']}"),
        ("broken-ended first attempts", lambda v: f"{v['broken_first']['rollouts_ended_by_degenerate_reply']} "
                                                  f"({v['broken_first']['pct_rollouts_ended']}%) "
                                                  f"{json.dumps(v['broken_first']['ended_kinds'])}"),
        ("re-runs: pass / fail / broken again / infra", lambda v: f"{v['reruns']}: " + " / ".join(
            str(v["rerun_outcomes"][k]) for k in ("pass", "fail", "broken_again", "infra"))),
        ("broken replies per 1000 (first attempts)", lambda v: f"{v['broken_first']['degenerate_replies']}/"
                                                               f"{v['broken_first']['replies']} = "
                                                               f"{v['broken_first']['per_1000_replies']}"),
        ("rollouts with any broken reply (first)", lambda v: f"{v['broken_first']['rollouts_with_degenerate_reply']} "
                                                             f"({v['broken_first']['pct_rollouts_with_degenerate_reply']}%)"),
        ("% action turns with reasoning (first)", lambda v: f"{v['reasoning_first']['pct_action_turns_with_reasoning']} "
                                                            f"(CI {v['reasoning_first']['ci95_pct_action_turns_with_reasoning']})"),
        ("compacted (first attempts)", lambda v: f"{v['window_first']['compacted_rollouts']} "
                                                 f"({v['window_first']['pct_rollouts_compacted']}%)"),
        ("pass@1 compacted / not (first)", lambda v: f"{v['window_first']['pass_at_1_compacted']} / "
                                                     f"{v['window_first']['pass_at_1_not_compacted']}"),
        ("length cuts / context overflows (all graded)", lambda v: f"{v['window_all_graded']['length_cut_replies']} / "
                                                                   f"{v['window_all_graded']['context_overflow_replies']}"),
        ("turns / agent wall min (median, first)", lambda v: f"{v['reasoning_first']['turns_per_rollout_median']} / "
                                                             f"{v['reasoning_first']['agent_wall_min_median']}")]
    for label, fn in table:
        lines.append(f"| {label} | " + " | ".join(fn(full[k]) for k in keys) + " |")
    lines += ["", f"## 3. Broken-reply re-runs: {rr['reruns']} ({json.dumps(rr['outcomes'])}), replaced {rr['replaced']}, "
              f"third attempts {rr['third_attempts']}; detector check {json.dumps(rr['detector_check'])}", "",
              "| campaign | task | slot | original (F2P, kind) | re-run | outcome (F2P) |", "|---|---|---|---|---|---|"]
    for e in rr["list"]:
        lines.append(f"| {e['campaign']} | {e['task_id']} | {e['slot']} | {e['original_status']} ({e['original_f2p']}, "
                     f"{e['original_kind']}) | {e['rerun_campaign']} {e['rerun_slot']} | {e['outcome']} ({e['rerun_f2p']}) |")
    lines += ["", "## 4. Cost (LiteLLM usage.cost of v4's own replies, by the hub key of each attempt's leg)", "",
              "| key | total | first attempts | re-runs | ATP-retried attempts | pilots | graded rollouts | $/graded rollout |",
              "|---|---|---|---|---|---|---|---|"]
    for key, v in c["by_key"].items():
        lines.append(f"| {key} | ${v['usd_total']} | ${v['first_attempts_usd']} | ${v['reruns_usd']} | "
                     f"${v['retried_attempts_usd']} | ${v['pilots_usd']} | {v['graded_rollouts']} | "
                     f"{v['usd_per_graded_rollout']} |")
    lines += ["", f"batch ${c['batch_usd']} + pilots ${c['pilots_usd']} = ${c['total_usd']}; per graded rollout "
              f"${c['usd_per_graded_rollout']}; per final slot ${c['usd_per_final_slot']}",
              f"by taskset: " + json.dumps({ts: {k: v[k] for k in ("usd", "usd_per_final_slot", "usd_per_first_attempt_mean",
                                                                   "cached_pct")} for ts, v in c["by_taskset"].items()}),
              f"key spend delta: {json.dumps(c['key_spend_delta'])}",
              f"reconciliation NEW_2: {json.dumps(c['reconciliation_NVAPI_KEY_NEW_2'], ensure_ascii=False)}",
              f"progress.py last row: {json.dumps(c['progress_py_last_row'])}", "",
              f"## 5. SFT-usable passes: {u['usable']} of {u['passes']} graded passes", "", u["criteria"], "",
              "| taskset | tasks | final passes | usable (first / re-run) | usable w/o compaction | excluded | tasks with usable |"
              " usable per task |", "|---|---|---|---|---|---|---|---|"]
    for ts, v in u["by_taskset"].items():
        lines.append(f"| {ts} | {v['tasks']} | {v['passes_final_slots']} | {v['usable']} ({v['usable_first_attempts']} / "
                     f"{v['usable_reruns']}) | {v['usable_without_compaction']} | {v['excluded']} | "
                     f"{v['tasks_with_usable']} | {json.dumps(v['usable_per_task_hist'])} |")
    lines += ["", f"excluded reasons: {json.dumps(u['excluded_reasons'])}", ""]
    for e in u["excluded"]:
        if e["role"] == "first" and e.get("final") is False:
            continue  # the broken-ended originals that a re-run replaced
        lines.append(f"- excluded: {e['taskset']} {e['task_id']} {e['slot']} ({e['role']}, {e['campaign']}): "
                     f"{'; '.join(e['reasons'])}")
    for e in u.get("kept_with_notes") or []:
        lines.append(f"- kept (note): {e['taskset']} {e['task_id']} {e['slot']} ({e['role']}, {e['campaign']}): "
                     f"{'; '.join(e['notes'])}")
    lines += ["", "## 6. Per task", "",
              "| task | taskset | first pass | broken-ended | re-runs | final pass | v3 | 1M | reasoned % | compacted | "
              "max prompt | $/rollout | SFT-usable |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in report["per_task"]:
        lines.append(f"| {t['task_id']} | {t['taskset']} | {t['first_pass']}/{t['first_graded']} | "
                     f"{' '.join(t['broken_ended_slots']) or '-'} | {' '.join(t['rerun_outcomes']) or '-'} | "
                     f"{t['final_pass']}/{t['final_graded']} | "
                     f"{'-' if t['v3_pass'] is None else str(t['v3_pass']) + '/' + str(t['v3_graded'])} | "
                     f"{'-' if t['1m_pass'] is None else str(t['1m_pass']) + '/' + str(t['1m_graded'])} | "
                     f"{t['pct_action_reasoned_first']} | {t['compacted_rollouts']} | {t['main_prompt_max']} | "
                     f"{t['usd_per_rollout']} | {t['sft_usable']} |")
    lines += ["", "Full SFT-usable list with attempt dirs and export paths: analysis/sft-usable-passes.jsonl.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
