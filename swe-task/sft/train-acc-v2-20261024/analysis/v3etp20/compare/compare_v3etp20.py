#!/usr/bin/env python3
"""train-acc-v2 (2026-10-31): one comparison table of all checkpoints on the 50 training tasks x 5 (SFT-v3
empty-think-prefix 5 / 10 ep added), assembled from the existing analysis outputs (read-only on them):

  accuracy      TA/results/<model>-group-*.json (pass / fail / infra; patch_chars) -> pass@1 = passes / graded (infra
                excluded, as aggregate.py) with a 95% task-bootstrap CI (10,000 resamples of the 50 tasks, seed 20261031),
                pass@5 (>= 1 pass among a task's graded rollouts), tasks 5/5, resolved / rollouts with a patch
  empty patches analysis/empty-patch/table.json (DeepSWE classifier on each slot's final attempt; of all 250 slots)
  bad thinking  analysis/think/split_table.json + calls-<model>.jsonl (missing </think> per model call of the final
                attempts, from the counting parser's log lines; n/a for base / raw 5 ep, served without it),
                analysis/features/*.jsonl (teacher-rollout extractor: action turns without reasoning, rollouts with every
                action reasoned), analysis/v3etp/thinkx_ta.json (the DeepSWE sftv3-etp thinkx.py run on the TA
                classifier rows: reasoning with tool-call markup / special tokens, per turn and rollout)
  timeouts      table.json deadline_kill_3h (session not finished, not sandbox-lost N4, and agent_error TimeoutError or
                session wall >= 10,500 s; the same slots as ATP's generation-validity budget_timeout flag), runaway thinking =
                calls cut at the length limit while still thinking (split_table unclosed_no_tool_length), those whose
                last 20,000 chars compress below 0.15 (notool_kinds text_cut_by_length_loop_tail), and rollouts that end
                on such a call (table.json E3a_runaway_cap)
  time          analysis/cost/cost.json (session wall minutes: median / mean / p90)
  infra         results status "infra" (ATP state not "completed"), with the per-slot cause found on 2026-10-31
                (causes of the context-overflow and closed slots corrected 2026-10-31 from the router clamp records)

    python3 -I -B compare_v3etp.py -> comparison_v3etp.{json,csv,txt}

2026-11-04: copy as analysis/v3etp20/compare/compare_v3etp20.py, + v3etp20 (SFT-v3 empty-think-prefix 20 ep, iter284;
250/250 graded, 0 infra). Inputs are this directory's copies, which hold every model and reproduce the earlier rows:
../empty-patch/table.json, ../think/{split_table,notool_kinds}.json, ./thinkx_ta.json, ../cost/cost.json; per-call rows
and extractor features of the earlier models from analysis/think and analysis/features (read-only), v3etp20's from
../think and ../features. Same seed (20261031), so the earlier columns reproduce comparison_v3etp.json.
    python3 -I -B compare_v3etp20.py -> comparison_v3etp20.{json,csv,txt}
"""
import collections
import csv
import json
from pathlib import Path
import random

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
HERE = TA / "analysis/v3etp20/compare"
V20 = TA / "analysis/v3etp20"
LABEL = {"sc14": "base sc14", "sftv2": "raw v2 5 ep", "sftv2e10": "raw v2 10 ep", "sftv2e20": "raw v2 20 ep",
         "etp5": "v2 etp 5 ep", "etp10": "v2 etp 10 ep", "etp20": "v2 etp 20 ep",
         "v3etp5": "v3 etp 5 ep", "v3etp10": "v3 etp 10 ep", "v3etp20": "v3 etp 20 ep"}
MODELS = tuple(LABEL)
COUNTING = {"sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"}  # counting lenient-think parser
OWN = ("v3etp20",)  # per-call rows / features computed in analysis/v3etp20


def calls_path(model):
    return (V20 / "think" if model in OWN else TA / "analysis/think") / f"calls-{model}.jsonl"
FEATURES = {"sc14": TR / "features/sc14.jsonl", "sftv2": TR / "features/sftv2.jsonl"}
# infra slots (ATP state != completed), cause and whether the failure is model-side (2026-10-31 inspection;
# REPORT.md "Infra slots" for the earlier runs)
INFRA = {
    "sftv2": [("syn-transitions-cand-c03-c910069f r0", "fatal", "context overflow: the next prompt was 269,887 tokens (> the 262,144 window by itself; previous 240,324) -> HTTP 400, router clamp not applicable", True)],
    "etp5": [("syn-yq-cand-c02-r1b0-0af03743 r0", "needs_review", "sandbox lost at 2.3 h (SandboxPtyError, then every command timed out)", False)],
    "etp10": [("syn-jest-cand-c05-9e1caaee r3", "fatal", "context overflow: the next prompt was 269,992 tokens (> the 262,144 window by itself; previous 240,743) -> HTTP 400, router clamp not applicable", True)],
    "v3etp10": [
        ("syn-wasmtime-cand-c04-r1b1-bbb510ca r0", "fatal", "context overflow: one cargo-test output took the prompt from 231,188 to 277,222 tokens (> the 262,144 window by itself) -> HTTP 400 at 02:12:51, router clamp not applicable (remaining -15,078); 0 compactions", True),
        ("syn-wasmtime-cand-c04-r1b1-bbb510ca r1", "fatal", "context overflow: one cargo-test output took the prompt from 238,510 to 289,891 tokens (> the 262,144 window by itself) -> HTTP 400 at 02:01:43, router clamp not applicable (remaining -27,747); 0 compactions", True),
        ("syn-gomacro-cand-c02-r1b0-e8804dd9 r4", "needs_review", "self-kill: `go run .` started the gomacro REPL, then `pkill -f gomacro` killed OpenCode (exit -1) at 2 min", True),
        ("syn-wasmtime-cand-c03-353ec720 r1", "needs_review", "sandbox lost at ~2.2 h (SandboxPtyError; patch collection timed out)", False),
        ("syn-ibis-cand-c03-r1b4-3d3b86b5 r3", "retryable", "sandbox OOM-killed (137) at 16 min; never retried; campaign closed by main, results file written 07:30:28", False)],
}


def load_results(model):
    per_task = collections.defaultdict(list)
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            per_task[task_id] += [(r["status"], bool(r.get("patch_chars")), r.get("agent_error")) for r in records]
    return per_task


def accuracy(per_task, seed=20261031, n=10000):
    tasks = sorted(per_task)
    cells = {t: (sum(s == "pass" for s, _, _ in per_task[t]), sum(s in ("pass", "fail") for s, _, _ in per_task[t]))
             for t in tasks}
    passes, graded = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        boots.append(100.0 * sum(cells[t][0] for t in sample) / sum(cells[t][1] for t in sample))
    boots.sort()
    patched = sum(p for t in tasks for s, p, _ in per_task[t] if s in ("pass", "fail"))
    infra = sum(s == "infra" for t in tasks for s, _, _ in per_task[t])
    return {"passes": passes, "graded": graded, "infra": infra, "pass_at_1": round(100.0 * passes / graded, 1),
            "pass_at_1_ci95": [round(boots[int(0.025 * n)], 1), round(boots[int(0.975 * n) - 1], 1)],
            "pass_at_5": sum(1 for t in tasks if cells[t][0]), "tasks": len(tasks),
            "tasks_5_of_5": sum(1 for t in tasks if cells[t][0] == len(per_task[t]) == 5),
            "patched_graded": patched, "resolved_per_patched_pct": round(100.0 * passes / patched, 1) if patched else None,
            "pass_at_1_infra_as_fail": round(100.0 * passes / (graded + infra), 1)}


def missing_close(model):
    """recovered + unclosed-no-tool per call (final attempts), and rollouts with >= 1 (union)."""
    calls = 0
    n = collections.Counter()
    rollouts = collections.defaultdict(set)
    for line in open(calls_path(model)):
        r = json.loads(line)
        calls += 1
        key = (r["task_id"], r["slot"])
        kind = r.get("log_kind")
        if kind in ("RECOVERED", "UNCLOSED_NO_TOOL"):
            n["missing"] += 1
            rollouts["missing"].add(key)
        if r.get("empty_output"):
            n["empty_output"] += 1
            rollouts["empty_output"].add(key)
    return {"calls": calls, "missing_calls": n["missing"], "missing_rollouts": len(rollouts["missing"]),
            "missing_pct_of_calls": round(100.0 * n["missing"] / calls, 3),
            "empty_output_calls": n["empty_output"], "empty_output_rollouts": len(rollouts["empty_output"])}


def cap_calls(model):
    """calls (final attempts) that generated the full 131,072-token output cap (finish "length"), and their rollouts;
    the other length cuts were stopped earlier by the router's context clamp (max_tokens <= 262,144 - prompt)."""
    calls, rollouts = 0, set()
    for line in open(calls_path(model)):
        r = json.loads(line)
        if r["finish"] == "length" and (r["completion_tokens"] or 0) >= 131072:
            calls += 1
            rollouts.add((r["task_id"], r["slot"]))
    return calls, len(rollouts)


def reasoning(model):
    path = FEATURES.get(model, (V20 / "features" if model in OWN else TA / "analysis/features") / f"{model}.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    ex = [r for r in rows if r.get("export") and r.get("turns")]
    actions = sum(r["action_turns"] for r in ex)
    unreasoned = sum(r["action_turns"] - r["action_with_reasoning"] for r in ex)
    return {"action_turns": actions, "unreasoned_action_turns": unreasoned,
            "unreasoned_pct": round(100.0 * unreasoned / actions, 3) if actions else None,
            "rollouts_with_unreasoned": sum(1 for r in ex if r["action_turns"] > r["action_with_reasoning"]),
            "rollouts_all_reasoned": sum(1 for r in ex if r.get("all_actions_reasoned")), "rollouts_with_export": len(ex)}


def main():
    table = json.loads((V20 / "empty-patch/table.json").read_text())
    split = json.loads((V20 / "think/split_table.json").read_text())
    kinds = json.loads((V20 / "think/notool_kinds.json").read_text())
    thinkx = json.loads((HERE / "thinkx_ta.json").read_text())
    cost = json.loads((V20 / "cost/cost.json").read_text())["models"]
    out = {}
    for m in MODELS:
        acc = accuracy(load_results(m))
        t, s, k, x, c = table[m], split[m], kinds[m], thinkx[m], cost[m]
        mc = missing_close(m)
        rs = reasoning(m)
        model_side = sum(1 for *_, ms in INFRA.get(m, []) if ms)
        out[m] = {
            "label": LABEL[m], **acc,
            "pass_at_1_model_side_infra_as_fail": round(100.0 * acc["passes"] / (acc["graded"] + model_side), 1),
            "empty_patch": t["empty_patch"], "empty_patch_pct_of_250": t["empty_patch_pct"],
            "empty_patch_graded": sum(1 for _ in [0]) and None,
            "deadline_kill_3h": t["deadline_kill_3h"], "deadline_kill_3h_empty": t["deadline_kill_3h_empty"],
            "edited_uncommitted": t["E1_edited_uncommitted"], "toolcall_in_think_headless": t["E3b_toolcall_in_think"],
            "reasoning_only_last_turn": t["E3c_reasoning_only"], "malformed_toolcall": t["E3d_malformed_toolcall"],
            "runaway_to_length_limit_final": t["E3a_runaway_cap"], "context_overflow_400": t["E3a_context_overflow_400"],
            "no_edit": t["E2_no_edit"], "sandbox_lost": t["sandbox_lost"], "not_finished_other": t["not_finished_other"],
            "stopped_after_compaction_continue": t["stopped_after_compaction_continue"],
            "one_token_stops": t["one_token_stops"],
            "missing_close": mc if m in COUNTING else None,
            "missing_close_recovered": s["recovered"]["calls"] if s.get("recovered") else None,
            "missing_close_unclosed_no_tool": s["unclosed_no_tool"]["calls"],
            "missing_close_unclosed_eos": s["unclosed_no_tool_eos"]["calls"],
            "missing_close_unclosed_length": s["unclosed_no_tool_length"]["calls"],
            "missing_close_lower_bound_uncounted": (s["recovered_bare_lower_bound"]["calls"] + s["unclosed_no_tool"]["calls"])
            if m not in COUNTING else None,
            "model_calls_final_attempts": s["calls"],
            "empty_output_calls": s["empty_output"]["calls"], "empty_output_rollouts": s["empty_output"]["rollouts"],
            **{f"reasoning_{key}": v for key, v in rs.items()},
            "markup_in_reasoning_turns": x.get("tc_markup_turns", 0), "markup_in_reasoning_rollouts": x.get("tc_markup_rollouts", 0),
            "markup_in_reasoning_headless_turns": x.get("tc_headless_turns", 0),
            "special_tokens_in_reasoning_turns": x.get("special_turns", 0),
            "special_tokens_in_reasoning_rollouts": x.get("special_rollouts", 0),
            "special_tokens": x.get("special_tokens"),
            "length_cut_thinking_calls": s["unclosed_no_tool_length"]["calls"],
            "length_cut_thinking_rollouts": s["unclosed_no_tool_length"]["rollouts"],
            "full_cap_131k_calls": cap_calls(m)[0], "full_cap_131k_rollouts": cap_calls(m)[1],
            "length_cut_loop_tail_calls": k["text_cut_by_length_loop_tail"]["calls"],
            "length_cut_loop_tail_rollouts": k["text_cut_by_length_loop_tail"]["rollouts"],
            "wall_min_median": c["wall_min_median"], "wall_min_mean": c["wall_min_mean"], "wall_min_p90": c["wall_min_p90"],
            "rollouts_over_2h": c["rollouts_over_2h"],
            "infra_slots": [{"slot": a, "atp_state": b, "cause": d, "model_side": e} for a, b, d, e in INFRA.get(m, [])],
        }
        out[m].pop("empty_patch_graded")
        g = load_results(m)
        out[m]["empty_patch_among_graded"] = sum(1 for v in g.values() for st, p, _ in v if st in ("pass", "fail") and not p)
    tmp = HERE / ".comparison_v3etp20.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(HERE / "comparison_v3etp20.json")
    rows = [
        ("pass@1 (95% task-bootstrap CI)", lambda o: f"{o['pass_at_1']}% ({o['passes']}/{o['graded']}) [{o['pass_at_1_ci95'][0]}, {o['pass_at_1_ci95'][1]}]"),
        ("pass@5", lambda o: f"{2 * o['pass_at_5']}% ({o['pass_at_5']}/50)"),
        ("tasks 5/5", lambda o: o["tasks_5_of_5"]),
        ("resolved / graded rollouts with a patch", lambda o: f"{o['resolved_per_patched_pct']}% ({o['passes']}/{o['patched_graded']})"),
        ("empty patch, of 250 (of graded)", lambda o: f"{o['empty_patch']} ({o['empty_patch_pct_of_250']}%) ({o['empty_patch_among_graded']}/{o['graded']})"),
        ("- 3 h deadline kill", lambda o: o["deadline_kill_3h_empty"]),
        ("- edited, uncommitted", lambda o: o["edited_uncommitted"]),
        ("- tool call inside think (headless)", lambda o: o["toolcall_in_think_headless"]),
        ("- reasoning-only last turn", lambda o: o["reasoning_only_last_turn"]),
        ("- malformed tool call", lambda o: o["malformed_toolcall"]),
        ("- runaway to the length limit", lambda o: o["runaway_to_length_limit_final"]),
        ("- context overflow (HTTP 400)", lambda o: o["context_overflow_400"]),
        ("- no edit", lambda o: o["no_edit"]),
        ("- sandbox lost", lambda o: o["sandbox_lost"]),
        ("- other not finished (self-kill)", lambda o: o["not_finished_other"]),
        ("(overlap) stopped after compaction continue", lambda o: o["stopped_after_compaction_continue"]),
        ("missing </think>: calls (% of calls); rollouts", lambda o: "n/a (>= {})".format(o["missing_close_lower_bound_uncounted"]) if o["missing_close"] is None else
         f"{o['missing_close']['missing_calls']} ({o['missing_close']['missing_pct_of_calls']}%); {o['missing_close']['missing_rollouts']}"),
        ("- recovered call in unclosed think / unclosed no call (EOS, length)", lambda o: f"{'n/a' if o['missing_close_recovered'] is None else o['missing_close_recovered']} / {o['missing_close_unclosed_no_tool']} ({o['missing_close_unclosed_eos']}, {o['missing_close_unclosed_length']})"),
        ("model calls (final attempts)", lambda o: o["model_calls_final_attempts"]),
        ("empty outputs (EOS right after <think>): calls; rollouts", lambda o: f"{o['empty_output_calls']}; {o['empty_output_rollouts']}"),
        ("action turns without reasoning", lambda o: f"{o['reasoning_unreasoned_action_turns']}/{o['reasoning_action_turns']} = {o['reasoning_unreasoned_pct']}%"),
        ("rollouts with every action reasoned", lambda o: f"{o['reasoning_rollouts_all_reasoned']}/{o['reasoning_rollouts_with_export']}"),
        ("reasoning with tool markup: turns / rollouts (headless turns)", lambda o: f"{o['markup_in_reasoning_turns']} / {o['markup_in_reasoning_rollouts']} ({o['markup_in_reasoning_headless_turns']})"),
        ("reasoning with special tokens: turns / rollouts", lambda o: f"{o['special_tokens_in_reasoning_turns']} / {o['special_tokens_in_reasoning_rollouts']}"),
        ("3 h deadline kills (rollouts; empty)", lambda o: f"{o['deadline_kill_3h']}; {o['deadline_kill_3h_empty']}"),
        ("thinking cut at the length limit: calls; rollouts", lambda o: f"{o['length_cut_thinking_calls']}; {o['length_cut_thinking_rollouts']}"),
        ("- ending in a repetition loop: calls; rollouts", lambda o: f"{o['length_cut_loop_tail_calls']}; {o['length_cut_loop_tail_rollouts']}"),
        ("- calls that reached the full 131,072-token cap: calls; rollouts", lambda o: f"{o['full_cap_131k_calls']}; {o['full_cap_131k_rollouts']}"),
        ("- rollouts that end on one", lambda o: o["runaway_to_length_limit_final"]),
        ("infra slots (excluded)", lambda o: len(o["infra_slots"])),
        ("pass@1 with model-side infra as fail / all infra as fail", lambda o: f"{o['pass_at_1_model_side_infra_as_fail']}% / {o['pass_at_1_infra_as_fail']}%"),
        ("rollout wall time: median / mean / p90 (min)", lambda o: f"{o['wall_min_median']} / {o['wall_min_mean']} / {o['wall_min_p90']}"),
        ("rollouts over 2 h", lambda o: o["rollouts_over_2h"]),
    ]
    lines = ["| | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    with open(HERE / ".comparison_v3etp20.csv.tmp", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + [LABEL[m] for m in MODELS])
        for title, fn in rows:
            cells = [str(fn(out[m])) for m in MODELS]
            writer.writerow([title] + cells)
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
    (HERE / ".comparison_v3etp20.csv.tmp").replace(HERE / "comparison_v3etp20.csv")
    tmp = HERE / ".comparison_v3etp20.txt.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(HERE / "comparison_v3etp20.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
