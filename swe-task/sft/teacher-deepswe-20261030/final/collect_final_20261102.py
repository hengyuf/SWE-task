"""One-off patch (collect phase, 2026-11-02 PST): collect.py and final/verify_collection.py for the FINAL collection
after the user's GLM cancel (OPERATOR_NOTE entry headed "2026-11-02 ~02:00 PST (main) -- USER DECISION, CANCEL GLM";
hengyuf, verbatim: "cancel the DeepSWE teacher rollout for GLM.").

State at the time: the 5 Kimi and 5 DeepSeek campaigns are complete (results/*.json); the 5 GLM campaigns have no
results file (main wrote babysit/glm-*/STOP and cancelled their legs at 01:46:45 PST). collect.py (window-12 version,
sha256 63e63b46...) was written for the 2026-10-31 partial collection (NEW_3 floor stop, no campaign finished); this
patch:
  collect.py
  - USER_CANCELLED: a model whose wave campaigns without results file are "cancelled_by_user" (main's STOP file must
    exist: check user_cancel_stop_missing); model_status per model (complete / partial_cancelled_by_user / partial).
  - own_view(): new slot state "interrupted" (latest attempt ended only by ATP's interrupted.json, i.e. its leg was
    stopped: main's GLM pause / wave-2 stop); in a cancelled campaign an attempt without end marker becomes
    "cancelled_in_flight" (in flight at the user's cancel; checked: ATP "active", started before the cancel), with its
    start time, last file change and sandbox id (agent-sandbox.json descriptor.sandbox_id).
  - summarize(): interrupted / cancelled_in_flight / stopped counts, ATP retries (slots with > 1 attempt), pass@5.
  - uncovered verdicts: yes / no / undetermined ("open" only for slots of an unfinished campaign; a complete
    campaign's final infra slot is final).
  - Kimi vs the 1M export: wave split, pass@5, all-5, the uncovered base <= 2/10 yes lists.
  - cost: legs per hub key NAME (tdw-legs.jsonl of every campaign), NVAPI_KEY_NEW_2 readings (setup, before / after
    the hub's 10-31 spend reset, last) next to the NEW_3 ones.
  - per_task.csv: + <model>_interrupted, <model>_cancelled_in_flight, <model>_model_status.
  - FINAL-REPORT.md: render_report() rewritten for "Kimi and DeepSeek complete, GLM PARTIAL (cancelled by the user)".
  final/verify_collection.py
  - the new states in the ATP / progress.py comparisons (progress.py "active" = the in-flight cancelled slots);
  - complete campaigns vs their results files (pass / fail / infra, records = first_attempt_records, no re-run);
  - GLM: STOP files, MANIFEST model_status.

    python3 collect_final_20261102.py collect SRC DST
    python3 collect_final_20261102.py verify  SRC DST

Every replacement is an exact text that must occur once. The caller checks DST (py_compile, unit checks) and mv's it
over the source (no job runs either file).
"""
import sys
from pathlib import Path

which, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
text = src.read_text()


def sub(old, new, count=1):
    global text
    assert text.count(old) == count, (old[:120], text.count(old))
    text = text.replace(old, new)


def replace_function(name, end_marker, new):
    """Replace `def name(` .. the text before end_marker (both must occur once)."""
    global text
    start = text.index(f"def {name}(")
    assert text.count(f"def {name}(") == 1
    end = text.index(end_marker, start)
    text = text[:start] + new + text[end:]


SUMMARIZE = '''def summarize(entries: list, base: dict) -> dict:
    """Numbers of a set of wave slots (slot ledger entries)."""
    graded = [e for e in entries if e["status"] in ("pass", "fail")]
    passes = sum(e["status"] == "pass" for e in graded)
    infra = [e for e in entries if e["state"] in INFRA_STATES]
    cancelled = [e for e in entries if e["state"] == "cancelled"]
    interrupted = [e for e in entries if e["state"] == "interrupted"]
    in_flight = [e for e in entries if e["state"] == "cancelled_in_flight"]
    not_started = [e for e in entries if e["state"] == "not_started"]
    active = [e for e in entries if e["state"] == "active"]
    clean = [e for e in graded if not e.get("ended_by_broken_reply")]
    no_timeout = [e for e in graded if not e.get("ended_by_agent_timeout")]
    by_task = collections.defaultdict(list)
    for e in entries:
        by_task[e["task_id"]].append(e)
    task_pass = {t: sum(x["status"] == "pass" for x in v) for t, v in by_task.items()}
    task_graded = {t: sum(x["status"] in ("pass", "fail") for x in v) for t, v in by_task.items()}
    started_tasks = [t for t in by_task if task_graded[t] > 0]
    complete_tasks = [t for t in by_task if task_graded[t] == REPEATS]
    graded_cost = [e["cost_usd"] for e in graded if e.get("cost_usd") is not None]
    graded_slot_cost = [e["slot_cost_usd"] for e in graded if e.get("slot_cost_usd") is not None]
    all_cost = sum(e.get("slot_cost_usd") or 0.0 for e in entries)
    base_rate = (sum((base.get(e["task_id"]) or {}).get("base_k10") or 0 for e in graded) / (10.0 * len(graded))
                 if graded else None)
    task_means = [task_pass[t] / task_graded[t] for t in started_tasks]
    tasks_any_pass = sum(task_pass[t] > 0 for t in started_tasks)
    return {
        "planned": len(entries), "graded": len(graded), "pass": passes, "fail": len(graded) - passes,
        "infra": len(infra), "infra_needs_review": sum(e.get("atp_status") == "needs_review" for e in infra),
        "cancelled": len(cancelled), "interrupted": len(interrupted), "cancelled_in_flight": len(in_flight),
        "stopped": len(cancelled) + len(interrupted) + len(in_flight),
        "not_started": len(not_started), "active": len(active),
        "retried_slots": sum((e.get("attempts") or 0) > 1 for e in entries),
        "pass_at_1_raw": pct(passes, len(graded) + len(infra)),
        "pass_at_1_excl_infra": pct(passes, len(graded)),
        "pass_at_1_excl_broken": pct(sum(e["status"] == "pass" for e in clean), len(clean)),
        "broken_ended": len(graded) - len(clean),
        "agent_timeouts": len(graded) - len(no_timeout),
        "agent_timeouts_passed": sum(e["status"] == "pass" for e in graded if e.get("ended_by_agent_timeout")),
        "pass_at_1_excl_timeouts": pct(sum(e["status"] == "pass" for e in no_timeout), len(no_timeout)),
        "runaways": sum(bool(e.get("ended_by_output_cap_runaway")) for e in graded),
        "compacted": sum(bool(e.get("compactions")) for e in graded),
        "pass_no_compaction": sum(e["status"] == "pass" and not e.get("compactions") for e in graded),
        "pass_clean": sum(e["status"] == "pass" and not e.get("ended_by_broken_reply")
                          and not e.get("ended_by_agent_timeout") for e in graded),
        "compaction_rate": pct(sum(bool(e.get("compactions")) for e in graded), len(graded)),
        "tasks": len(by_task), "tasks_started": len(started_tasks), "tasks_complete": len(complete_tasks),
        "tasks_any_pass": tasks_any_pass, "pass_at_5_rate": pct(tasks_any_pass, len(started_tasks)),
        "tasks_complete_any_pass": sum(task_pass[t] > 0 for t in complete_tasks),
        "tasks_complete_all_pass": sum(task_pass[t] == REPEATS for t in complete_tasks),
        "task_mean_pass_rate": pct(sum(task_means), len(task_means)),
        "base_pass_at_1_same_slots": pct(base_rate, 1) if base_rate is not None else "-",
        "cost_graded": round(sum(graded_cost), 2),
        "cost_per_graded_rollout": round(statistics.mean(graded_cost), 3) if graded_cost else None,
        "cost_per_graded_slot_all_attempts": round(statistics.mean(graded_slot_cost), 3) if graded_slot_cost else None,
        "cost_all_attempts": round(all_cost, 2),
        "cost_all_per_graded": round(all_cost / len(graded), 3) if graded else None}


'''

RENDER = r'''def render_report(summary, campaign_state, tasks, waves, manifest, counts, checks, ledger, by_model_task, per_teacher,
                  any_yes, any_open, verdicts, uncovered, rows_1m, n_table, states_meta) -> list:
    complete = manifest["complete"]
    ov = summary["overview"]
    status = summary["model_status"]
    done = [m for m in MODELS if status[m] == "complete"]
    partial = [m for m in MODELS if status[m] != "complete"]
    cancelled_models = [m for m in MODELS if status[m] == "partial_cancelled_by_user"]
    tag = {m: ("complete" if status[m] == "complete" else "PARTIAL, cancelled by the user"
               if status[m] == "partial_cancelled_by_user" else "PARTIAL") for m in MODELS}
    label = {m: (m if status[m] == "complete" else m + " (PARTIAL)") for m in MODELS}
    wave_ledger = [e for e in ledger if e["campaign"] != "pilot"]
    title = "# DeepSWE v1.1 teacher rollouts: final report"
    if not complete:
        title += f" ({', '.join(done) or 'no model'} complete; {', '.join(partial)} PARTIAL)"
    lines = [title, "", f"_Generated by collect.py at {time.strftime('%Y-%m-%d %H:%M %Z')}. Sources: results/*.json for "
             "complete campaigns and the probe dirs of the others (read-only). Every planned slot with its state: "
             "dataset/slots.jsonl._", ""]
    for model in cancelled_models:
        info = summary["user_cancelled"][model]
        b, w1, w2 = ov[model], summary["by_wave"][f"{model} w1"], summary["by_wave"][f"{model} w2"]
        ended = b["interrupted"] + b["cancelled"] + b["infra"]
        lines += [
            f"> **{model.upper()} IS PARTIAL: CANCELLED BY THE USER.** {info['source']}: hengyuf, verbatim: "
            f"\"{info['decision']}\" {info['how']}",
            f"> - {model}: **{b['graded']} of {b['planned']} slots graded** (wave 1: {w1['graded']} of {w1['planned']}; "
            f"wave 2: {w2['graded']} of {w2['planned']}). Not graded: {b['cancelled_in_flight']} in flight at the cancel "
            f"(cut off, not graded), {ended} that had ended ungraded earlier and were never retried "
            f"({b['interrupted']} interrupted by an earlier stop, {b['cancelled']} cancelled by the 2026-10-31 "
            f"budget-floor stop, {b['infra']} infra endings), and {b['not_started']} never started.",
            f"> - Every {model} number below counts only its graded slots and is labelled PARTIAL. Within each campaign, "
            f"ATP admits slots in a fixed (task, repeat) order, so {model}'s graded slots are the early tasks of each "
            f"campaign: {b['tasks_started']} of 113 tasks have at least one graded slot and {b['tasks_complete']} have "
            f"all 5. Like-for-like numbers: \"Matched task sets\"."]
    if not complete:
        for model in done:
            b = ov[model]
            lines.append(f"> - {model}: complete (all 5 campaigns finished; {b['planned']} of {b['planned']} slots "
                         f"final: {b['graded']} graded, {b['infra']} final infra).")
        stopped = [f"{m} {n}" for (m, n), s in campaign_state.items() if s == "stopped"]
        never = [f"{m} {n}" for (m, n), s in campaign_state.items() if s == "not_started"]
        if stopped:
            lines.append(f"> - Campaigns stopped before finishing (not cancelled by the user): {', '.join(stopped)}.")
        if never:
            lines.append(f"> - Campaigns never started: {', '.join(never)}.")
        lines += ["> - A row that adds models up (\"all\") mixes complete and PARTIAL counts and says so; no pass rate "
                  "is pooled across models.", ""]
    legs = summary.get("legs_by_key") or {}
    leg_text = "; ".join(f"{key}: {v['legs']} legs ({v['pilot_legs']} pilot, {v['wave_legs']} wave), started "
                         f"{v['first']} .. {v['last']}" for key, v in sorted(legs.items()))
    lines += [f"- **Request (hengyuf, 2026-10-30, verbatim):** {REQUEST}",
              "- **Settings:** kimi = `nvidia/moonshotai/kimi-k3`, glm = `nvidia/zai-org/glm-5.3`, deepseek = "
              "`nvidia/deepseek-ai/deepseek-v4-pro` (reasoning_effort max, temperature 1.0, top_p 0.95); OpenCode 1.17.11 "
              "stock system prompt; limit.context = limit.input = 262,144 (compaction at 242,144), limit.output = "
              "131,072; ATP agent timeout 10,800 s; each task's own DeepSWE grader; hengyuf's OpenSandbox credential "
              "(CELL3_SANDBOX_API_KEY). Campaigns: per model wave 1 (79 tasks) in 3 groups and wave 2 (34 tasks) in 2 "
              "groups, 5 rollouts per task; in flight per campaign at most kimi 8, glm 8 (wave 2: 4), deepseek 5, "
              "lowered further by ATP's adaptive admission.",
              "- **Hub keys (key names only):** the 3 pilot legs (2026-10-30 21:35 PDT) ran on NVAPI_KEY_NEW_2; the "
              "9 wave-1 legs of 10-30 23:07 PDT ran on NVAPI_KEY_NEW_3 (OPERATOR_NOTE 10-30 21:45, user decision) until "
              "its $1,000 floor stop on 10-31 10:30-10:40 PDT; every leg from 10-31 18:29 PDT ran on NVAPI_KEY_NEW_2 "
              "(OPERATOR_NOTE 10-31 18:15, user decision; floors $1,700 start / $1,500 guard). Legs per key "
              f"(tdw-legs.jsonl): {leg_text or 'none recorded'}. Each hub reply's cost is attributed to the leg that "
              "was running when the reply arrived, so to that leg's key.",
              "- **Definitions:** graded = the slot's final record is pass or fail: for a complete campaign, driver.py's "
              "results/<model>-<campaign>.json, which holds ATP's grade of the slot's graded attempt; for a campaign "
              "without a results file, ATP's slot_state \"completed\", cross-checked with this script's own view. Main's "
              "decision adds one graded fail (GRADED_FAIL_OVERRIDES: kimi w2-g1 13-0). **pass@1 raw** = passes / "
              "(graded + slots that ended on an infra error), so infra counts as a fail. Slots stopped before they "
              "ended on their own are left out: cancelled or interrupted by a stop, or in flight at the user's cancel. "
              "Never-started slots are also left out. **excl. broken-ended** = passes / graded slots not ended by a "
              "broken reply (brokenreply.py). **excl. infra** = passes / graded. **excl. agent timeouts** = passes / "
              "graded slots not ended by ATP's 10,800 s agent timeout; those slots stay graded (OPERATOR_NOTE "
              "2026-10-31 02:15). **pass@5** = tasks with at least one pass, out of the tasks with at least one graded "
              "slot; it is exact for a task with all 5 graded and a lower bound otherwise. **Compaction rate** = "
              "graded rollouts whose OpenCode session compacted at least once (export.json). **$ per graded "
              "rollout** = mean hub cost of a graded slot's final attempt. The cost comes from the journal: LiteLLM "
              "usage.cost when reported, else prices.json per token. **all spend / graded** = the cost of every "
              "attempt (graded, infra, stopped) / graded slots. **Base** = the base student swonlysc-step0014, from "
              "eval runs base-step0-rep5-out131k and its rerun1, resolved out of 5 each = k/10 (tasks/base_k10.json).",
              ""]

    lines += ["## Completeness", "",
              "| model | status | planned slots | graded | pass | fail | infra (ATP: retry / needs_review) | stopped "
              "before the end: cancelled / interrupted / in flight at the user's cancel | not started | campaigns "
              "complete / cancelled by the user / stopped / never started | tasks with >= 1 graded / all 5 graded |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        b = ov[model]
        states = collections.Counter(s for (m, _), s in campaign_state.items() if m == model)
        lines.append(f"| {model} | {tag[model]} | {b['planned']} | {b['graded']} | {b['pass']} | {b['fail']} | "
                     f"{b['infra']} ({b['infra'] - b['infra_needs_review']} / {b['infra_needs_review']}) | "
                     f"{b['cancelled']} / {b['interrupted']} / {b['cancelled_in_flight']} | {b['not_started']} | "
                     f"{states['complete']} / {states['cancelled_by_user']} / {states['stopped']} / "
                     f"{states['not_started']} | {b['tasks_started']} / {b['tasks_complete']} of 113 |")
    tot = {k: sum(ov[m][k] for m in MODELS) for k in ("planned", "graded", "pass", "infra", "cancelled", "interrupted",
                                                       "cancelled_in_flight", "not_started")}
    lines += [f"| all three ({'mixes complete and PARTIAL counts' if not complete else 'complete'}) | | "
              f"{tot['planned']} | {tot['graded']} | {tot['pass']} | {tot['graded'] - tot['pass']} | {tot['infra']} | "
              f"{tot['cancelled']} / {tot['interrupted']} / {tot['cancelled_in_flight']} | {tot['not_started']} | | |",
              ""]

    lines += ["## Overview per model", "",
              "| model | status | graded | pass | pass@1 raw | pass@1 excl. broken-ended | pass@1 excl. infra | agent "
              "timeouts (passed) | pass@1 excl. agent timeouts | base pass@1, same slots | pass@5: tasks >= 1 pass / tasks "
              "graded | tasks all 5 graded: >= 1 pass / 5 of 5 | broken-ended | runaways | compaction rate | passes "
              "without compaction | $ per graded rollout | all spend / graded |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        b = ov[model]
        lines.append(f"| {model} | {tag[model]} | {b['graded']} | {b['pass']} | {b['pass_at_1_raw']} | "
                     f"{b['pass_at_1_excl_broken']} | {b['pass_at_1_excl_infra']} | {b['agent_timeouts']} "
                     f"({b['agent_timeouts_passed']}) | {b['pass_at_1_excl_timeouts']} | {b['base_pass_at_1_same_slots']} "
                     f"| {b['tasks_any_pass']} / {b['tasks_started']} ({b['pass_at_5_rate']}%) | "
                     f"{b['tasks_complete_any_pass']} / {b['tasks_complete_all_pass']} of {b['tasks_complete']} | "
                     f"{b['broken_ended']} | {b['runaways']} | {b['compaction_rate']}% ({b['compacted']}) | "
                     f"{b['pass_no_compaction']} of {b['pass']} | {money(b['cost_per_graded_rollout'])} | "
                     f"{money(b['cost_all_per_graded'])} |")
    notes = []
    for over in manifest.get("status_overrides") or []:
        b = ov[over["model"]]
        notes.append(f"{over['model']} counts {over['campaign']} {over['task_id']} r{over['repeat']} as a graded fail "
                     f"(GRADED_FAIL_OVERRIDES; ATP leaves it needs_review). In ATP's own view {over['model']} has "
                     f"{b['graded'] - 1} graded and pass@1 {pct(b['pass'], b['graded'] - 1)} "
                     f"({pct(b['pass'] - b['agent_timeouts_passed'], b['graded'] - 1 - b['agent_timeouts'])} without "
                     "the agent timeouts), as PROGRESS.md shows.")
    for model in partial:
        notes.append(f"{model} is PARTIAL: its pass@5 is a lower bound on the {ov[model]['tasks_started'] - ov[model]['tasks_complete']} "
                     "tasks with 1-4 graded slots, and its numbers cover a different task mix than the complete "
                     "models.")
    lines += [""] + [f"- {note}" for note in notes] + [
        "- Base = the base student (swonlysc-step0014, 10 rollouts per task) on the same tasks, weighted by the "
        "graded slots.", ""]

    lines += ["## Wave 1 (79 uncovered tasks) vs wave 2 (34 covered tasks)", "",
              "| model x wave | status | tasks | planned | graded | pass@1 raw | excl. broken | excl. infra | excl. agent "
              "timeouts | base, same slots | pass@5: tasks >= 1 pass / graded | tasks all 5 graded | compaction | $ per "
              "graded rollout | hub cost (all attempts) | campaigns |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for wave in (1, 2):
            b = summary["by_wave"][f"{model} w{wave}"]
            states = collections.Counter(s for (m, n), s in campaign_state.items()
                                         if m == model and n.startswith(f"w{wave}"))
            state_text = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(states.items()))
            lines.append(f"| {model} w{wave} | {tag[model]} | {b['tasks']} | {b['planned']} | {b['graded']} | "
                         f"{b['pass_at_1_raw']} | {b['pass_at_1_excl_broken']} | {b['pass_at_1_excl_infra']} | "
                         f"{b['pass_at_1_excl_timeouts']} | {b['base_pass_at_1_same_slots']} | {b['tasks_any_pass']} / "
                         f"{b['tasks_started']} | {b['tasks_complete']} | {b['compaction_rate']}% | "
                         f"{money(b['cost_per_graded_rollout'])} | {money(b['cost_all_attempts'])} | {state_text} |")
    lines.append("")

    lines += ["## Per campaign", "",
              "| model | campaign | state | planned | graded | pass | pass@1 excl. infra | infra | stopped before the end "
              "| not started | tasks all 5 graded | ATP retries | hub cost |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for name in CAMPAIGNS:
            b = summary["by_campaign"][f"{model} {name}"]
            lines.append(f"| {model} | {name} | {b['state'].replace('_', ' ')} | {b['planned']} | {b['graded']} | "
                         f"{b['pass']} | {b['pass_at_1_excl_infra']} | {b['infra']} | {b['stopped']} | "
                         f"{b['not_started']} | {b['tasks_complete']} | {b['retried_slots']} | "
                         f"{money(b['cost_all_attempts'])} |")
    lines.append("")

    lines += ["## Matched task sets (like-for-like across models)", ""]
    for name, data in summary["matched"].items():
        lines += [f"**Tasks where {name}: {data['tasks']}.**", ""]
        if not data["tasks"]:
            lines += ["(none)", ""]
            continue
        lines += ["| model | graded slots | pass@1 excl. infra | task-mean pass rate | tasks >= 1 pass | base, same "
                  "slots |", "|---|---|---|---|---|---|"]
        for model in MODELS:
            b = data["per_model"][model]
            lines.append(f"| {label[model]} | {b['graded']} | {b['pass_at_1_excl_infra']} | {b['task_mean_pass_rate']} | "
                         f"{b['tasks_any_pass']} / {b['tasks_started']} | {b['base_pass_at_1_same_slots']} |")
        lines.append("")

    lines += ["## pass@1 (excl. infra) per language", "",
              "Cell = pass@1 over the model's graded slots (graded slots; base pass@1 on the same slots).", "",
              "| language | " + " | ".join(label[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for language, per_model in summary["by_language"].items():
        ids = [t for t in tasks if tasks[t]["language"] == language]
        cells = [f"{per_model[m]['pass_at_1_excl_infra']} ({per_model[m]['graded']}; base "
                 f"{per_model[m]['base_pass_at_1_same_slots']})" for m in MODELS]
        lines.append(f"| {language} ({len(ids)} tasks) | " + " | ".join(cells) + " |")
    lines.append("")

    u = summary["uncovered"]
    lines += ["## Uncovered tasks with base <= 2/10 and a teacher >= 3/5", "",
              f"Wave 1 = the 79 tasks with no accepted synthetic task. {u['tasks']} of them have base_k10 <= 2, meaning "
              "the base student resolved at most 2 of its 10 rollouts. A teacher reaches >= 3/5 when at least 3 of its "
              "5 slots pass. For a complete model every task is **yes** or **no**. A complete campaign's final infra "
              "slot counts as not passed. For PARTIAL GLM: **yes** = at least 3 passes among its graded slots, which "
              "is final. **no** = fewer than 3 passes even if every ungraded slot had passed. **undetermined** = "
              "neither, because the user's cancel came before enough slots ran.", "",
              "| teacher | status | yes | no | undetermined |", "|---|---|---|---|---|"]
    for model in MODELS:
        d = u["per_teacher"][model]
        lines.append(f"| {model} | {tag[model]} | {d['yes']} | {d['no']} | {d['undetermined']} |")
    lines.append(f"| any complete teacher ({' or '.join(u['complete_teachers']) or 'none'}) | complete | "
                 f"{len(u['complete_teachers_yes'])} | {u['tasks'] - len(u['complete_teachers_yes'])} | 0 |")
    lines.append(f"| any teacher (with {', '.join(partial) or 'no'} PARTIAL) | | {len(any_yes)} | "
                 f"{u['tasks'] - len(any_yes) - len(any_open)} | {len(any_open)} (no yes; at least one undetermined) |")
    lines += ["", f"Tasks with at least one teacher at >= 3/5 ({len(any_yes)}):", "",
              "| task | language | base_k10 | " + " | ".join(f"{label[m]} pass / graded" for m in MODELS) + " |",
              "|---|---|---|" + "---|" * len(MODELS)]
    for task_id in any_yes:
        cells = []
        for model in MODELS:
            v = verdicts[task_id][model]
            cells.append(f"{v['pass']} / {v['graded']}" + (" **yes**" if v["verdict"] == "yes" else
                                                            " (undetermined)" if v["verdict"] == "undetermined" else ""))
        lines.append(f"| {task_id} | {tasks[task_id]['language']} | {tasks[task_id]['base_k10']} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    for model in MODELS:
        ids = u["per_teacher_yes"][model]
        lines.append(f"- {label[model]} yes ({len(ids)}): {', '.join(ids) or 'none'}.")
        if u["per_teacher_undetermined"][model]:
            lines.append(f"- {label[model]} undetermined ({len(u['per_teacher_undetermined'][model])}): "
                         + ", ".join(f"{t} ({verdicts[t][model]['pass']} / {verdicts[t][model]['graded']} graded)"
                                     for t in u["per_teacher_undetermined"][model]) + ".")
    no_yes = u["no_teacher_yes"]
    lines += [f"- No teacher at >= 3/5 ({len(no_yes)}): " + ", ".join(
        f"{t} (base {tasks[t]['base_k10']}; " + ", ".join(f"{m} {verdicts[t][m]['pass']}/{verdicts[t][m]['graded']}"
                                                         for m in MODELS) + ")" for t in no_yes) + ".",
              "- Per task and teacher: analysis/collect-summary.json (uncovered.verdicts).", ""]

    k = summary["kimi_vs_1m"]
    lines += ["## Kimi here (256k context) vs the 1M-context Kimi export (context only)", "",
              f"Reference: {KIMI_1M_NOTE}. It uses a different pipeline, context limit and serving period, so this is "
              "context, not a controlled comparison. Per task: analysis/collect-kimi-vs-1m.csv.", ""]
    if k.get("tasks_with_kimi_graded"):
        lines += ["| | Kimi here (256k) | Kimi 1M export |", "|---|---|---|",
                  f"| tasks / slots | {k['tasks_with_kimi_graded']} / {k['slots']} | {k['kimi1m_tasks']} / "
                  f"{k['kimi1m_scored_total']} |",
                  f"| avg@5 (pass@1), all tasks | {k['kimi256k_pass_at_1']} ({k['kimi256k_pass']} / {k['slots']}) | "
                  f"{k['kimi1m_all_tasks_avg']} ({k['kimi1m_correct_total']} / {k['kimi1m_scored_total']}) |",
                  f"| wave 1 (79 uncovered tasks) | {k['wave1_256k']} | {k['kimi1m_wave1_avg']} |",
                  f"| wave 2 (34 covered tasks) | {k['wave2_256k']} | {k['kimi1m_wave2_avg']} |",
                  f"| pass@5: tasks >= 1 pass (tasks with all 5 graded here) | {k['complete_256k_any_pass']} of "
                  f"{k['complete_tasks']} | {k['complete_1m_any_pass']} of {k['complete_tasks']} |",
                  f"| tasks with 5 of 5 passing | {k['all5_256k']} | {k['all5_1m']} |",
                  f"| uncovered base <= 2/10 tasks at >= 3/5 | {len(k['uncovered_yes_256k'])} of {u['tasks']} | "
                  f"{len(k['uncovered_yes_1m'])} of {u['tasks']} |", "",
                  f"- Without the {k['slots'] - k['slots_excl_timeouts']} graded agent timeouts here "
                  f"({ov['kimi']['agent_timeouts_passed']} of them passed; the numerator keeps every pass): "
                  f"{k['kimi256k_pass_at_1_excl_timeouts']} over {k['slots_excl_timeouts']} slots vs "
                  f"{k['kimi1m_same_slots_excl_timeouts']} for the 1M export on the same tasks, weighted the same way.",
                  f"- Per task (rate here vs the 1M rate): higher here {k['tasks_256k_better']}, lower "
                  f"{k['tasks_256k_worse']}, tied {k['tasks_tied']}. Task-mean: {k['kimi256k_task_mean']} vs "
                  f"{k['kimi1m_task_mean']}.",
                  "- Uncovered base <= 2/10 tasks at >= 3/5 for one of the two only: here only: "
                  + (", ".join(t for t in k["uncovered_yes_256k"] if t not in k["uncovered_yes_1m"]) or "none")
                  + "; 1M only: " + (", ".join(t for t in k["uncovered_yes_1m"] if t not in k["uncovered_yes_256k"])
                                     or "none") + ".", ""]
    else:
        lines += ["(no Kimi slot graded)", ""]

    for model in cancelled_models:
        mine = [e for e in wave_ledger if e["model"] == model]
        b = ov[model]
        lines += [f"## {model.upper()}: what the user's cancel left (PARTIAL)", ""]
        flight = [e for e in mine if e["state"] == "cancelled_in_flight"]
        lines.append(f"- In flight at the cancel ({len(flight)}; cut off, not graded; their dirs have no end marker and "
                     "ATP's slot_state says \"active\"): " + ("; ".join(
                         f"{e['campaign']} {e['task_id']} r{e['repeat']} (attempt {Path(e['attempt_dir']).parent.name}/"
                         f"{Path(e['attempt_dir']).name}, started {(e.get('reason') or {}).get('started_at')}, last file "
                         f"change {(e.get('reason') or {}).get('last_file_change')}, sandbox {e.get('sandbox_id')})"
                         for e in flight) or "none") + ". Main deletes their sandboxes (OPERATOR_NOTE 2026-11-02 "
                     "item 4).")
        earlier = [e for e in mine if e["state"] in ("interrupted", "cancelled", "infra")]
        lines.append(f"- Ended ungraded before the cancel and never retried ({len(earlier)}): " + ("; ".join(
            f"{e['campaign']} {e['task_id']} r{e['repeat']} ({e['state']}; ATP {e.get('atp_status')}; "
            + (f"interrupted ({(e.get('reason') or {}).get('interrupted_reason')}), "
               if e["state"] == "interrupted" else "")
            + reason_text(e.get("reason")) + ")" for e in earlier) or "none") + ".")
        per_campaign = collections.Counter(e["campaign"] for e in mine if e["state"] == "not_started")
        lines.append(f"- Never started: {b['not_started']} ("
                     + ", ".join(f"{n} {per_campaign[n]}" for n in CAMPAIGNS) + ").")
        task_graded = collections.Counter(e["task_id"] for e in mine if e["status"] in ("pass", "fail"))
        lines.append(f"- Tasks: {sum(v == REPEATS for v in task_graded.values())} with all 5 graded, "
                     f"{sum(0 < v < REPEATS for v in task_graded.values())} with 1-4 graded, "
                     f"{113 - len(task_graded)} with none.")
        broken = [e for e in mine if e.get("ended_by_broken_reply")]
        lines.append(f"- Broken-ended graded slots: {len(broken)}" + ((": " + "; ".join(
            f"{e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in broken)
            + ". Its one-time re-run (brk) was planned for the end of the campaign; the cancel removed it, so the slot "
              "counts as it stands (flagged ended_by_broken_reply).") if broken else "."))
        ungraded_cost = sum(e.get("slot_cost_usd") or 0.0 for e in mine if e["status"] not in ("pass", "fail"))
        lines.append(f"- Hub spend of the {model} attempts that ended ungraded or were cut off: {money(ungraded_cost)} "
                     f"of {money(summary['cost'][model]['all_attempts'])} (all {model} wave attempts).")
        lines.append("")

    lines += ["## Re-runs, ATP retries, agent timeouts, broken replies", ""]
    reruns = summary.get("rerun_dirs") or []
    lines.append("- Broken-reply re-runs (`<campaign>-brk<n>`) and budget-refusal recoveries (`<campaign>-rec<n>`): "
                 + (f"probe dirs {', '.join(reruns)}" if reruns else "**none ran** (no -brk / -rec probe dir exists)")
                 + ". Complete campaigns: " + "; ".join(
                     f"{m} {n}: re-runs {len(v['broken_reply_reruns'])}, recoveries {len(v['recovered'])}, final "
                     f"record = first-attempt record for {v['same_as_first']} of {v['slots']} slots"
                     for (m, n), v in sorted(summary["results_provenance"].items())) + ".")
    retried = [e for e in wave_ledger if (e.get("attempts") or 0) > 1]
    lines.append("- ATP retries inside a campaign (slots with more than one attempt dir; the dataset row is ATP's "
                 f"graded attempt, the cost counts every attempt): {len(retried)} (" + ", ".join(
                     f"{m} {sum(e['model'] == m for e in retried)}" for m in MODELS) + ").")
    broken = [e for e in wave_ledger if e.get("ended_by_broken_reply")]
    lines.append(f"- Broken-ended graded slots: {len(broken)}" + (": " + "; ".join(
        f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in broken) + "."
        if broken else "."))
    timeouts = [e for e in wave_ledger if e.get("ended_by_agent_timeout") and e["status"] in ("pass", "fail")]
    lines.append(f"- Graded slots ended by the agent timeout: {len(timeouts)}" + (": " + "; ".join(
        f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in timeouts) + "."
        if timeouts else "."))
    review = [e for e in wave_ledger if not e.get("status_override") and (
        e["state"] == "final_infra" or (e["state"] == "infra" and e.get("atp_status") == "needs_review"))]
    lines.append("- Slots ATP left ungraded and does not retry (needs_review: e.g. the verifier killed at its 1,800 s "
                 f"timeout; final infra slots of complete campaigns): {len(review)}" + (": " + "; ".join(
                     f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ("
                     + (f"verifier: {e['verifier_error'][:90]}" if e.get("verifier_error") else reason_text(e.get("reason")))
                     + ")" for e in review) + ". They stay ungraded (status infra)." if review else "."))
    overridden = [e for e in wave_ledger if e.get("status_override")]
    lines.append("- Counted as a GRADED FAIL although ATP left them needs_review (GRADED_FAIL_OVERRIDES): "
                 f"{len(overridden)}" + (": " + "; ".join(
                     f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} (f2p {e.get('f2p')}; "
                     f"{e['status_override']['reason']}; {e['status_override']['source']})" for e in overridden) + "."
                     if overridden else "."))
    refusals = checks.get("budget_refusals") or []
    lines.append(f"- Hub budget refusals (4xx with a budget message) in ungraded attempts: {len(refusals)}.")
    lines.append("")

    c = summary["cost"]
    keys = sorted({key for m in MODELS + ("pilots",) for key in c[m]["by_key"]})
    lines += ["## Cost and hub keys (journal)", "",
              "| model | status | all attempts | graded rollouts (final attempt) | ungraded attempts | "
              + " | ".join(keys) + " | prompt / cached / completion tokens | hub replies 200 | non-200 replies |",
              "|---|---|---|---|---|" + "---|" * len(keys) + "---|---|---|"]
    for model in MODELS + ("pilots",):
        x = c[model]
        errors = ", ".join(f"{s}: {n}" for s, n in sorted(x["hub_errors"].items())) or "0"
        lines.append(f"| {model} | {tag.get(model, 'complete (3 one-rollout pilots)')} | {money(x['all_attempts'])} | "
                     f"{money(x['graded_final_attempts'])} | {money(x['ungraded_attempts'])} | "
                     + " | ".join(money(x["by_key"].get(key, 0.0)) for key in keys)
                     + f" | {x['prompt_tokens']:,} / {x['cached_tokens']:,} / {x['completion_tokens']:,} | "
                     f"{x['hub_calls_ok']:,} | {errors} |")
    wave_total = sum(c[m]["all_attempts"] for m in MODELS)
    key_totals = {key: sum(c[m]["by_key"].get(key, 0.0) for m in MODELS) for key in keys}
    lines += [f"| all wave legs ({'mixes complete and PARTIAL' if not complete else 'complete'}) | | {money(wave_total)} | "
              f"{money(sum(c[m]['graded_final_attempts'] for m in MODELS))} | "
              f"{money(sum(c[m]['ungraded_attempts'] for m in MODELS))} | "
              + " | ".join(money(key_totals[key]) for key in keys) + " | | | |", ""]
    lines.append("- Pilots (NVAPI_KEY_NEW_2): " + ", ".join(
        f"{m} {money(v)}" for m, v in (c["pilots"]["per_pilot"] or {}).items()) + f" = {money(c['pilots']['all_attempts'])}.")
    lines.append("- Wave legs per key: " + "; ".join(
        f"{key} {money(key_totals[key])} (" + ", ".join(f"{m} {money(c[m]['by_key'].get(key, 0.0))}" for m in MODELS)
        + ")" for key in keys) + ".")
    last, first = summary.get("new3_last_reading"), summary.get("new3_at_first_leg")
    if last and first:
        mine3 = key_totals.get("NVAPI_KEY_NEW_3", 0.0)
        lines.append(f"- NVAPI_KEY_NEW_3 (budget-log.jsonl, key_budget.py): {money(first['remaining'])} left at the "
                     f"wave-leg starts (lowest start reading, {first['at']}), {money(last['remaining'])} at its last "
                     f"reading ({last['at']}): {money(first['remaining'] - last['remaining'])} spent in between. This "
                     f"campaign's journals account for {money(mine3)}; the rest is the r6 RL run's rubric judge and any "
                     "other user of the key. LiteLLM's spend figure lags, and replicas differ by about $20.")
    n2 = summary.get("new2_readings") or {}
    if n2.get("last"):
        mine2 = key_totals.get("NVAPI_KEY_NEW_2", 0.0) + c["pilots"]["by_key"].get("NVAPI_KEY_NEW_2", 0.0)
        text = (f"- NVAPI_KEY_NEW_2 (budget-log.jsonl, key_budget.py): {money(n2['first']['spend'])} spent / "
                f"{money(n2['first']['remaining'])} left at the setup check ({n2['first']['at']})")
        if n2.get("last_before_reset") and n2.get("first_after_reset"):
            text += (f"; {money(n2['last_before_reset']['spend'])} spent at {n2['last_before_reset']['at']}. The hub "
                     f"then reported the spend as reset: {money(n2['first_after_reset']['spend'])} spent / "
                     f"{money(n2['first_after_reset']['remaining'])} left at {n2['first_after_reset']['at']}. This "
                     "agent does not know why")
        text += (f". **Last reading: {money(n2['last']['spend'])} spent / {money(n2['last']['remaining'])} left "
                 f"({n2['last']['at']}, tag {n2['last'].get('tag')}).**")
        if n2.get("first_after_reset"):
            since = n2["last"]["spend"] - n2["first_after_reset"]["spend"]
            text += (f" From the reset on, NEW_2's spend grew by {money(since)}. This campaign's NEW_2 wave legs "
                     f"(from 10-31 18:29) account for {money(key_totals.get('NVAPI_KEY_NEW_2', 0.0))}. The rest "
                     f"({money(since - key_totals.get('NVAPI_KEY_NEW_2', 0.0))}) is the r6 RL rubric judge, which "
                     "uses NEW_2 since 10-31 18:15, and any other user of the key.")
        text += f" All of this campaign's NEW_2 cost, pilots included: {money(mine2)}."
        lines.append(text)
    lines.append("")

    lines += ["## Checks", "",
              f"- ATP state file: {states_meta['file'] if states_meta else 'none'}"
              + (f" (created {states_meta['created_at']}, sha256 {states_meta['sha256'][:16]}..., probes unchanged by "
                 f"the check: {states_meta['probes_unchanged']}, {states_meta['files_checked']:,} files compared, "
                 f"validity caches it would have written: {states_meta['would_write']})" if states_meta else "")]
    for key, value in sorted(manifest["checks"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("- Independent checks: final/verify_collection.py -> analysis/collect-verify.json.")
    lines += ["", "## Files", "", "| file | rows | bytes | sha256 |", "|---|---|---|---|"]
    for name, meta in manifest["files"].items():
        rows = (counts.get(name.split(".")[0]) if name.endswith(".jsonl.gz") else
                manifest["slots_ledger_rows"] if name == "slots.jsonl" else
                sum(counts.values()) if name == "index.jsonl" else "")
        lines.append(f"| dataset/{name} | {rows} | {meta['bytes']:,} | `{meta['sha256'][:16]}...` |")
    lines.append(f"| per_task.csv | {n_table} | | |")
    lines.append("| analysis/collect-summary.json | | | |")
    lines.append(f"| analysis/collect-kimi-vs-1m.csv | {len(rows_1m)} | | |")
    return lines


'''

if which == "collect":
    # ---------------------------------------------------------------- docstring
    sub('''slots ATP left needs_review (the verifier killed at its 1,800 s timeout), which stay ungraded.
''', '''slots ATP left needs_review (the verifier killed at its 1,800 s timeout), which stay ungraded.

FINAL COLLECTION (2026-11-02; final/collect_final_20261102.py). After the OPERATOR_NOTE entry headed "2026-11-02 ~02:00
PST (main) -- USER DECISION, CANCEL GLM" (hengyuf, verbatim: "cancel the DeepSWE teacher rollout for GLM."): the 5 Kimi
and 5 DeepSeek campaigns are complete (results files); the 5 GLM campaigns were cancelled for good (main's STOP files in
babysit/glm-*, legs cancelled at 01:46:45 PST) and are read from their probe dirs (--allow-incomplete) with campaign
state "cancelled_by_user" (USER_CANCELLED; check user_cancel_stop_missing). GLM is PARTIAL (model_status
"partial_cancelled_by_user"): its graded slots count as they stand, and its started-ungraded slots are
  cancelled_in_flight  in flight at the user's cancel: started.json, no end marker, ATP "active"; cut off, not graded
  interrupted          ended only by ATP's interrupted.json (its leg was stopped: main's GLM pause of 2026-11-01 05:15
                       PST or wave-2 stop of 15:35 PST); ATP retryable, never retried before the cancel
  cancelled            failure.json CancelledError (the 2026-10-31 NEW_3 budget-floor stop), never retried
  infra                ended on an infra error (ATP retryable or needs_review), never retried
pass@1 raw counts infra as a fail and leaves the stopped ones (cancelled / interrupted / cancelled_in_flight) out. The
uncovered-task verdict is yes / no / undetermined (undetermined only through ungraded slots of an unfinished campaign).
No broken-reply re-run (brk) or recovery (rec) campaign ran: Kimi and DeepSeek had no broken-ended slot and no budget
refusal; GLM's one broken-ended slot lost its planned re-run with the cancel.
''')
    sub('''  _broken_final, _runaway, _agent_timeout, _compacted.
''', '''  _broken_final, _runaway, _agent_timeout, _compacted (2026-11-02: + _interrupted, _cancelled_in_flight after
  _cancelled, and _model_status = complete / partial_cancelled_by_user / partial).
''')
    # ---------------------------------------------------------------- imports, constants
    sub('''from concurrent.futures import ProcessPoolExecutor  # noqa: E402
''', '''from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
''')
    sub('''              "status_overrides_unmatched")
''', '''              "status_overrides_unmatched", "user_cancel_stop_missing", "active_outside_cancel",
              "in_flight_started_after_cancel")
''')
    sub('''OVERRIDE_PROBLEMS = {}   # key -> why an override did not match (check status_overrides_unmatched)
''', '''OVERRIDE_PROBLEMS = {}   # key -> why an override did not match (check status_overrides_unmatched)
# OPERATOR_NOTE entry headed "2026-11-02 ~02:00 PST (main) -- USER DECISION, CANCEL GLM" (items 1 and 3): the model's
# wave campaigns without a results file are "cancelled_by_user" (main's babysit/<model>-<campaign>/STOP must exist);
# the model is reported PARTIAL and kept out of any "complete" claim. The leg states are from sacct (2026-11-02).
USER_CANCELLED = {
    "glm": {"decision": "cancel the DeepSWE teacher rollout for GLM.",
            "source": "OPERATOR_NOTE entry headed 2026-11-02 ~02:00 PST (main), USER DECISION, CANCEL GLM",
            "cancelled_at": "2026-11-02T01:46:45-0800",
            "how": ("Main wrote babysit/glm-<campaign>/STOP for all 5 GLM campaigns at 2026-11-02 01:46:40 PST and "
                    "cancelled their legs (sacct: tdw-glm-w1-g0 21154933, w1-g1 21154935, w1-g2 21154938, w2-g0 "
                    "21167343 and w2-g1 21167279 CANCELLED at 01:46:45 PST). No GLM leg, broken-reply re-run (brk) or "
                    "recovery (rec) leg runs again.")}}
STOPPED_STATES = ("cancelled", "interrupted", "cancelled_in_flight")  # stopped before the slot ended on its own
INFRA_STATES = ("infra", "final_infra")
NEW2_RESET_AFTER = "2026-10-31T18:00:00-0700"  # NVAPI_KEY_NEW_2's spend read as reset at the 18:11 check
''')
    # ---------------------------------------------------------------- own_view: interrupted, start / last change / sandbox
    sub('''def own_view(slot: dict) -> dict:
''', '''def parse_stamp(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def stamp_of(path: Path):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def last_change(directory: Path):
    """The latest mtime of a file directly in an attempt dir (where ATP writes its logs and markers)."""
    stamps = []
    for path in directory.iterdir():
        try:
            if path.is_file():
                stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(stamps))) if stamps else None


def own_view(slot: dict) -> dict:
''')
    sub('''    if failure.get("error_type") == "CancelledError":
        state = "cancelled"
    elif ended:
        state = "infra"
''', '''    interrupted = read_json(latest / "interrupted.json", {}) or {}
    if failure.get("error_type") == "CancelledError":
        state = "cancelled"
    elif ended == ["interrupted.json"]:  # ATP's recovery after the attempt's leg was stopped: no grade, no failure
        state = "interrupted"
    elif ended:
        state = "infra"
''')
    sub('''              "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(stamps))) if stamps else None}
''', '''              "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(stamps))) if stamps else None,
              "interrupted_reason": interrupted.get("reason"), "started_at": stamp_of(latest / "started.json"),
              "last_file_change": last_change(latest)}
''')
    sub('''    return {"state": state, "attempts": len(attempts), "attempt_dir": str(latest), "reason": reason,
            "budget_refusal": refused, "cost": cost}
''', '''    sandbox = None
    if state == "active":  # no end marker: the sandbox it ran in (descriptor only; the file holds no credential)
        sandbox = ((read_json(latest / "agent-sandbox.json", {}) or {}).get("descriptor") or {}).get("sandbox_id")
    return {"state": state, "attempts": len(attempts), "attempt_dir": str(latest), "reason": reason,
            "budget_refusal": refused, "cost": cost, "sandbox_id": sandbox}
''')
    # ---------------------------------------------------------------- summarize
    replace_function("summarize", "def main():", SUMMARIZE)
    # ---------------------------------------------------------------- main: campaign states, in-flight slots
    sub('''                views = list(pool.map(own_view, slots, chunksize=8))
                campaign_state[(model, name)] = "stopped" if exists else "not_started"
                for slot, view in zip(slots, views):
                    atp = states.get((model, name, slot["task_id"], slot["repeat"]))
''', '''                views = list(pool.map(own_view, slots, chunksize=8))
                cancelled_by_user = model in USER_CANCELLED
                campaign_state[(model, name)] = ("not_started" if not exists else
                                                 "cancelled_by_user" if cancelled_by_user else "stopped")
                if exists and cancelled_by_user and not (HERE / "babysit" / f"{model}-{name}" / "STOP").exists():
                    checks["user_cancel_stop_missing"].append(f"{model}-{name}")
                for slot, view in zip(slots, views):
                    atp = states.get((model, name, slot["task_id"], slot["repeat"]))
                    if view["state"] == "active" and cancelled_by_user:  # cut off by the user's cancel, never graded
                        view = dict(view, state="cancelled_in_flight")
                        started = (view.get("reason") or {}).get("started_at")
                        if not started or parse_stamp(started) >= parse_stamp(USER_CANCELLED[model]["cancelled_at"]):
                            checks["in_flight_started_after_cancel"].append(view["attempt_dir"])
                    elif view["state"] == "active":
                        checks["active_outside_cancel"].append(view["attempt_dir"])
''')
    sub('''                        elif atp["status"] == "pending" and view["state"] != "not_started":
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP pending vs view {view['state']}")
''', '''                        elif atp["status"] == "pending" and view["state"] != "not_started":
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP pending vs view {view['state']}")
                        elif (view["state"] == "cancelled_in_flight") != (atp["status"] == "active"):
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP {atp['status']} vs view "
                                f"{view['state']}")
''')
    sub('''                    "reason": view.get("reason"), "budget_refusal": view.get("budget_refusal"), "in_dataset": False,
''', '''                    "reason": view.get("reason"), "budget_refusal": view.get("budget_refusal"), "in_dataset": False,
                    "sandbox_id": view.get("sandbox_id"),
''')
    # ---------------------------------------------------------------- model status (before the checks / manifest)
    sub('''    # 4. checks (slot counts per model per task, dataset vs ledger)
''', '''    model_status = {}
    for model in MODELS:
        mine = [campaign_state[(model, name)] for name in CAMPAIGNS]
        model_status[model] = ("complete" if all(s == "complete" for s in mine) else
                               "partial_cancelled_by_user" if "cancelled_by_user" in mine else
                               "not_started" if all(s == "not_started" for s in mine) else "partial")

    # 4. checks (slot counts per model per task, dataset vs ledger)
''')
    sub('''                "collection": "complete" if complete else "PARTIAL",
''', '''                "collection": "complete" if complete else "PARTIAL", "model_status": model_status,
                "user_cancelled": {m: USER_CANCELLED[m] for m in MODELS if model_status[m] == "partial_cancelled_by_user"},
''')
    # ---------------------------------------------------------------- per_task.csv
    sub('''        header += [f"{model}_{c}" for c in ("planned", "graded", "pass", "complete", "k5", "cancelled", "infra",
                                             "not_started", "broken_first", "broken_final", "runaway", "agent_timeout",
                                             "compacted")]
''', '''        header += [f"{model}_{c}" for c in ("planned", "graded", "pass", "complete", "k5", "cancelled", "interrupted",
                                             "cancelled_in_flight", "infra", "not_started", "broken_first",
                                             "broken_final", "runaway", "agent_timeout", "compacted", "model_status")]
''')
    sub('''                         f"{model}_cancelled": sum(e["state"] == "cancelled" for e in slots),
''', '''                         f"{model}_cancelled": sum(e["state"] == "cancelled" for e in slots),
                         f"{model}_interrupted": sum(e["state"] == "interrupted" for e in slots),
                         f"{model}_cancelled_in_flight": sum(e["state"] == "cancelled_in_flight" for e in slots),
                         f"{model}_model_status": model_status[model],
''')
    # ---------------------------------------------------------------- summary: model status, cancel facts
    sub('''               "kimi_vs_1m": {}, "cost": {}, "checks": check_summary, "complete": complete}
''', '''               "kimi_vs_1m": {}, "cost": {}, "checks": check_summary, "complete": complete,
               "model_status": model_status, "user_cancelled": {}}
    for model in MODELS:
        if model_status[model] == "partial_cancelled_by_user":
            summary["user_cancelled"][model] = dict(USER_CANCELLED[model], stop_files={
                name: stamp_of(HERE / "babysit" / f"{model}-{name}" / "STOP") for name in CAMPAIGNS})
    summary["rerun_dirs"] = sorted(f"{p.parent.name}/{p.name}" for model in MODELS
                                   if (ROOT / "probes" / model).is_dir()
                                   for p in (ROOT / "probes" / model).iterdir() if p.is_dir()
                                   and re.search(r"-(brk|rec)\\d*$", p.name))
    summary["results_provenance"] = {}
    for (model, name), data in results.items():
        if name == "pilot":
            continue
        firsts = {(t, r["slot"]): r.get("attempt_dir") for t, recs in (data.get("first_attempt_records") or {}).items()
                  for r in recs}
        finals = {(t, r["slot"]): r.get("attempt_dir") for t, recs in data["records"].items() for r in recs}
        summary["results_provenance"][(model, name)] = {
            "slots": len(finals), "same_as_first": sum(firsts.get(k) == v for k, v in finals.items()),
            "broken_reply_reruns": data.get("broken_reply_reruns") or {},
            "recovered": data.get("recovered_budget_refusals") or {},
            "stats_final": data.get("stats_final"), "hub_keys_by_leg": data.get("hub_keys_by_leg")}
''')
    # ---------------------------------------------------------------- uncovered verdicts
    sub('''            graded = sum(e["status"] in ("pass", "fail") for e in slots)
            passes = sum(e["status"] == "pass" for e in slots)
            left = REPEATS - graded
            verdicts[task_id][model] = {"pass": passes, "graded": graded,
                                        "verdict": "yes" if passes >= 3 else "open" if passes + left >= 3 else "no"}
    per_teacher = {model: {v: [t for t in uncovered if verdicts[t][model]["verdict"] == v] for v in ("yes", "open", "no")}
                   for model in MODELS}
    any_yes = [t for t in uncovered if any(verdicts[t][m]["verdict"] == "yes" for m in MODELS)]
    any_open = [t for t in uncovered if t not in any_yes and any(verdicts[t][m]["verdict"] == "open" for m in MODELS)]
    summary["uncovered"] = {"definition": "wave-1 tasks (no accepted synthetic task) with base_k10 <= 2; a teacher "
                                          "reaches >= 3/5 = at least 3 passing slots among its 5",
                            "tasks": len(uncovered), "per_teacher": {m: {v: len(ids) for v, ids in d.items()}
                                                                     for m, d in per_teacher.items()},
                            "per_teacher_yes": {m: d["yes"] for m, d in per_teacher.items()},
                            "any_teacher_yes": any_yes, "any_teacher_open_only": any_open, "verdicts": verdicts}
''', '''            graded = sum(e["status"] in ("pass", "fail") for e in slots)
            passes = sum(e["status"] == "pass" for e in slots)
            # only an unfinished campaign's ungraded slots leave the verdict open (a complete campaign's final infra
            # slot is final); GLM's are cut off by the user's cancel, so "undetermined"
            left = sum(e["status"] not in ("pass", "fail") and e["campaign_state"] != "complete" for e in slots)
            verdicts[task_id][model] = {"pass": passes, "graded": graded, "ungraded_unfinished": left,
                                        "verdict": "yes" if passes >= 3 else "undetermined" if passes + left >= 3
                                        else "no"}
    per_teacher = {model: {v: [t for t in uncovered if verdicts[t][model]["verdict"] == v]
                           for v in ("yes", "no", "undetermined")} for model in MODELS}
    any_yes = [t for t in uncovered if any(verdicts[t][m]["verdict"] == "yes" for m in MODELS)]
    any_open = [t for t in uncovered if t not in any_yes
                and any(verdicts[t][m]["verdict"] == "undetermined" for m in MODELS)]
    complete_teachers = [m for m in MODELS if model_status[m] == "complete"]
    summary["uncovered"] = {"definition": "wave-1 tasks (no accepted synthetic task) with base_k10 <= 2; a teacher "
                                          "reaches >= 3/5 = at least 3 passing slots among its 5; yes = >= 3 passes; "
                                          "no = < 3 even if every ungraded slot of an unfinished campaign had passed; "
                                          "undetermined = neither (GLM: cancelled by the user)",
                            "tasks": len(uncovered), "task_ids": uncovered,
                            "per_teacher": {m: {v: len(ids) for v, ids in d.items()} for m, d in per_teacher.items()},
                            "per_teacher_yes": {m: d["yes"] for m, d in per_teacher.items()},
                            "per_teacher_undetermined": {m: d["undetermined"] for m, d in per_teacher.items()},
                            "complete_teachers": complete_teachers,
                            "complete_teachers_yes": [t for t in uncovered if any(
                                verdicts[t][m]["verdict"] == "yes" for m in complete_teachers)],
                            "any_teacher_yes": any_yes, "any_teacher_undetermined_only": any_open,
                            "no_teacher_yes": [t for t in uncovered if t not in any_yes], "verdicts": verdicts}
''')
    # ---------------------------------------------------------------- Kimi vs 1M: wave split, pass@5, uncovered
    sub('''        "kimi1m_wave1_avg": pct(sum(k1m[t]["correct"] for t in waves["wave1"] if t in k1m),
                                sum(k1m[t]["scored"] for t in waves["wave1"] if t in k1m))}
''', '''        "kimi1m_wave1_avg": pct(sum(k1m[t]["correct"] for t in waves["wave1"] if t in k1m),
                                sum(k1m[t]["scored"] for t in waves["wave1"] if t in k1m))}
    for wave in (1, 2):
        mine = [r for r in paired if r["wave"] == wave]
        summary["kimi_vs_1m"][f"wave{wave}_256k"] = pct(sum(r["kimi256k_pass"] for r in mine),
                                                         sum(r["kimi256k_graded"] for r in mine))
        summary["kimi_vs_1m"][f"wave{wave}_1m_same_tasks"] = pct(sum(r["kimi1m_correct"] for r in mine),
                                                                  sum(r["kimi1m_scored"] for r in mine))
    summary["kimi_vs_1m"].update({
        "kimi1m_wave2_avg": pct(sum(k1m[t]["correct"] for t in waves["wave2"] if t in k1m),
                                sum(k1m[t]["scored"] for t in waves["wave2"] if t in k1m)),
        "kimi1m_correct_total": sum(v["correct"] for v in k1m.values()),
        "kimi1m_scored_total": sum(v["scored"] for v in k1m.values()),
        "all5_256k": sum(r["kimi256k_pass"] == REPEATS for r in full),
        "all5_1m": sum(r["kimi1m_correct"] == REPEATS for r in full),
        "uncovered_yes_256k": [t for t in uncovered if verdicts[t]["kimi"]["verdict"] == "yes"],
        "uncovered_yes_1m": [t for t in uncovered if t in k1m and k1m[t]["correct"] >= 3]})
''')
    # ---------------------------------------------------------------- NEW_2 readings, legs per key
    sub('''    summary["new3_at_first_leg"] = ({k: min(first_leg, key=lambda b: b["remaining"]).get(k)
                                     for k in ("at", "remaining", "spend")} if first_leg else None)
''', '''    summary["new3_at_first_leg"] = ({k: min(first_leg, key=lambda b: b["remaining"]).get(k)
                                     for k in ("at", "remaining", "spend")} if first_leg else None)
    fields = ("at", "remaining", "spend", "max_budget", "tag")
    new2 = [b for b in budget_log if b.get("key") == "NVAPI_KEY_NEW_2" and b.get("remaining") is not None
            and b.get("spend") is not None and b.get("at")]
    before = [b for b in new2 if parse_stamp(b["at"]) < parse_stamp(NEW2_RESET_AFTER)]
    after = [b for b in new2 if parse_stamp(b["at"]) >= parse_stamp(NEW2_RESET_AFTER)]
    summary["new2_readings"] = {
        "first": {k: new2[0].get(k) for k in fields} if new2 else None,
        "last_before_reset": {k: before[-1].get(k) for k in fields} if before else None,
        "first_after_reset": {k: after[0].get(k) for k in fields} if after else None,
        "last": {k: new2[-1].get(k) for k in fields} if new2 else None,
        "max_spend_after_reset": max((b["spend"] for b in after), default=None)}
    legs = []
    for model in MODELS:
        for name in CAMPAIGNS + ("pilot",):
            for start, job, key in _module("costs").legs_of(model, name):
                legs.append({"model": model, "campaign": name, "job": job, "key": key, "start": start,
                             "at": time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(start))})
    summary["legs"] = [{k: v for k, v in leg.items() if k != "start"} for leg in sorted(legs, key=lambda x: x["start"])]
    summary["legs_by_key"] = {}
    for key in sorted({leg["key"] for leg in legs}):
        mine = sorted((leg for leg in legs if leg["key"] == key), key=lambda x: x["start"])
        summary["legs_by_key"][key] = {"legs": len(mine), "pilot_legs": sum(x["campaign"] == "pilot" for x in mine),
                                       "wave_legs": sum(x["campaign"] != "pilot" for x in mine),
                                       "first": mine[0]["at"], "last": mine[-1]["at"]}
''')
    sub('''    print(json.dumps({"rows": counts, "ledger": len(ledger), "complete": complete, "checks": check_summary,
''', '''    print(json.dumps({"rows": counts, "ledger": len(ledger), "complete": complete, "model_status": model_status,
                      "checks": check_summary,
''')
    # results_provenance has tuple keys: make it JSON-safe for collect-summary.json
    sub('''    (ROOT / "analysis" / "collect-summary.json").write_text(json.dumps(
        summary | {"manifest": manifest, "checks_detail": checks}, indent=1, ensure_ascii=False) + "\\n")
''', '''    (ROOT / "analysis" / "collect-summary.json").write_text(json.dumps(
        summary | {"manifest": manifest, "checks_detail": checks,
                   "results_provenance": {f"{m}-{n}": v for (m, n), v in summary["results_provenance"].items()}},
        indent=1, ensure_ascii=False) + "\\n")
''')
    # ---------------------------------------------------------------- the report
    replace_function("render_report", '\n\nif __name__ == "__main__":', RENDER.rstrip("\n"))
elif which == "verify":
    sub('''ledger side are cancelled / infra / active / final_infra (a complete campaign's final infra slot is ATP needs_review or
retryable too) plus the overrides.
''', '''ledger side are cancelled / infra / active / final_infra (a complete campaign's final infra slot is ATP needs_review or
retryable too) plus the overrides.
Final collection (2026-11-02, final/collect_final_20261102.py; GLM cancelled by the user): the ledger states
"interrupted" (ATP retryable) and "cancelled_in_flight" (ATP "active"; progress.py's "active") count as started-ungraded
too; progress.py's ended_ungraded = cancelled + interrupted + infra. 5. complete campaigns vs their results files
(pass / fail / infra per campaign, every final record = its first-attempt record, no re-run or recovery). 6. a model
cancelled by the user (MANIFEST model_status partial_cancelled_by_user) has main's STOP file in every
babysit/<model>-<campaign>/ and no results file.
''')
    sub('''        ungraded_started = sum(e["state"] in ("cancelled", "infra", "active", "final_infra")
                               for e in wave if e["model"] == model) + overrides[model]
''', '''        ungraded_started = sum(e["state"] in ("cancelled", "infra", "active", "final_infra", "interrupted",
                                              "cancelled_in_flight")
                               for e in wave if e["model"] == model) + overrides[model]
''')
    sub('''        if last.get("ended_ungraded") is not None and last["ended_ungraded"] != sum(
                e["state"] in ("cancelled", "infra") for e in wave):
            problems.append(f"progress.py ended_ungraded {last['ended_ungraded']} vs ledger "
                            f"{sum(e['state'] in ('cancelled', 'infra') for e in wave)}")
''', '''        ended_states = ("cancelled", "infra", "interrupted")
        if last.get("ended_ungraded") is not None and last["ended_ungraded"] != sum(
                e["state"] in ended_states for e in wave):
            problems.append(f"progress.py ended_ungraded {last['ended_ungraded']} vs ledger "
                            f"{sum(e['state'] in ended_states for e in wave)}")
        in_flight = sum(e["state"] in ("cancelled_in_flight", "active") for e in wave)
        if last.get("active") is not None and last["active"] != in_flight:
            problems.append(f"progress.py active {last['active']} vs ledger in flight (cancelled_in_flight / active) "
                            f"{in_flight}")
''')
    sub('''    out["progress_last_row"] = {k: last.get(k) for k in ("time", "graded", "graded_total", "pass", "ended_ungraded")
                                if k in last} if last else None
''', '''    out["progress_last_row"] = {k: last.get(k) for k in ("time", "graded", "graded_total", "pass", "ended_ungraded",
                                                         "active", "infra_final") if k in last} if last else None
''')
    sub('''    out["cost"] = cost
''', '''    out["cost"] = cost

    # 5. complete campaigns vs their results files; 6. models cancelled by the user
    status = manifest.get("model_status") or {}
    out["model_status"] = status
    provenance = {}
    for model in MODELS:
        for name in ("w1-g0", "w1-g1", "w1-g2", "w2-g0", "w2-g1"):
            result = read_json(TD / "results" / f"{model}-{name}.json")
            mine = [e for e in wave if e["model"] == model and e["campaign"] == name]
            stop = TD / "babysit" / f"{model}-{name}" / "STOP"
            if result is None:
                if status.get(model) == "complete":
                    problems.append(f"{model}-{name}: no results file but model_status complete")
                if status.get(model) == "partial_cancelled_by_user" and not stop.exists():
                    problems.append(f"{model}-{name}: cancelled by the user but no STOP file")
                provenance[f"{model}-{name}"] = {"results": False, "stop_file": stop.exists(),
                                                 "states": dict(collections.Counter(e["state"] for e in mine))}
                continue
            final = {"pass": 0, "fail": 0, "infra": 0}
            same = slots = 0
            firsts = {(t, r["slot"]): r.get("attempt_dir") for t, recs in (result.get("first_attempt_records") or {}).items()
                      for r in recs}
            for task_id, records in result["records"].items():
                for record in records:
                    final[record["status"] if record["status"] in final else "infra"] += 1
                    slots += 1
                    same += firsts.get((task_id, record["slot"])) == record.get("attempt_dir")
            mine_overrides = sum(1 for o in manifest.get("status_overrides") or [] if o["model"] == model
                                 and o["campaign"] == name)
            ledger_counts = {"pass": sum(e["status"] == "pass" for e in mine),
                             "fail": sum(e["status"] == "fail" for e in mine) - mine_overrides,
                             "infra": sum(e["state"] == "final_infra" for e in mine) + mine_overrides}
            if ledger_counts != final:
                problems.append(f"{model}-{name}: ledger {ledger_counts} (overrides as infra) vs results {final}")
            stats = result.get("stats_final") or {}
            if {k: stats.get(k) for k in final} != final:
                problems.append(f"{model}-{name}: results stats_final {stats} vs records {final}")
            if same != slots:
                problems.append(f"{model}-{name}: {slots - same} final records differ from their first-attempt record")
            if result.get("broken_reply_reruns") or result.get("recovered_budget_refusals"):
                problems.append(f"{model}-{name}: re-runs / recoveries recorded; check their provenance")
            provenance[f"{model}-{name}"] = {"results": True, "final": final, "slots": slots,
                                             "final_equals_first": same, "stop_file": stop.exists(),
                                             "broken_reply_reruns": len(result.get("broken_reply_reruns") or {}),
                                             "recovered_budget_refusals": len(result.get("recovered_budget_refusals") or {})}
    rerun_dirs = sorted(f"{p.parent.name}/{p.name}" for model in MODELS for p in (TD / "probes" / model).iterdir()
                        if p.is_dir() and ("-brk" in p.name or "-rec" in p.name))
    if rerun_dirs:
        problems.append(f"re-run / recovery probe dirs exist: {rerun_dirs}")
    out["campaign_provenance"] = provenance
    out["rerun_dirs"] = rerun_dirs
''')
else:
    raise SystemExit(f"unknown target {which}")
dst.write_text(text)
print("ok", which, dst)
