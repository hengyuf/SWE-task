#!/usr/bin/env python3
"""Session-ending lens, step 3: tables (reads out/derived.jsonl.gz and out/sft_sessions.jsonl.gz only).

usage: python3 -I -B tables.py   -> out/tables.json, out/tables.txt
(2026-11-04 copy in train-acc analysis/v3etp20/degenerate/endings: train-acc models sc14, etp10, etp20, v3etp5, v3etp10,
v3etp20 only; the T7 / T10 SFT and narration tables are not built here.)

Ending class of a rollout (first match; mutually exclusive):
  resolved / failed_patch                      reward >= 1 / reward < 1 with a non-empty patch
  empty patch:
    deadline_kill        derive.py deadline_kill (3 h agent cap)
    self_kill            derive.py self_kill (the model's own kill / pkill / killall ended OpenCode)
    infra                classifier N4 (sandbox lost) / other unfinished sessions
    context_overflow     finished, the final request was rejected (vLLM HTTP 400 context overflow; error, or finish 'length'
                         with 0 output tokens in train-acc exports)
    runaway_end          final call cut by length (classifier E3a)
    format_end           tool call in think / reasoning-only / malformed visible call (E3b / E3c / E3d)
    uncommitted:*        edited, finished, no commit (E1), split by the final visible text (endlib.final_flags):
       asks_or_defers      defers the commit or asks the user (incl. "Shall I commit?")
       announced_action    no done-claim, and the text ends with an announced next step ("... Let me run X:") -- the call
                           never came (the reasoning is used when there is no visible text)
       claims_done         claims the work is done / gives a summary, does not defer or ask
       other_text          anything else
    no_edit              E2
"""
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
# 2026-11-04 copy (train-acc v3etp20 analysis): train-acc only, + v3etp20
DS_MODELS = {
    "ta": ["sc14", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"],
}
MAIN = {"ta": ["sc14", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]}


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ending(d):
    if d["outcome"] != "empty":
        return d["outcome"]
    code, sub = d.get("code"), d.get("subcode") or ""
    if d.get("deadline_kill"):
        return "deadline_kill"
    if d.get("self_kill"):
        return "self_kill"
    if code and code.startswith("N"):
        return "infra"
    fin = d.get("final") or {}
    if fin.get("err") or (sub == "E3a_cap_cut" and not fin.get("out")):
        # the last request was rejected (vLLM HTTP 400: prompt + max_tokens over the window; train-acc exports record
        # it as finish 'length' with 0 tokens) -- the model's own huge tool output overflowed the context
        return "context_overflow"
    if sub == "E3a_cap_cut":
        return "runaway_end"
    if code == "E3_PREMATURE_END":
        return "format_end"
    if code == "E1_UNCOMMITTED":
        fl = fin.get("flags") or {}
        if fl.get("defers_commit") or fl.get("asks_user"):
            return "uncommitted:asks_or_defers"
        if (fl.get("intent_end") or fl.get("intent_end_r")) and not fl.get("claims_done"):
            return "uncommitted:announced_action"
        if fl.get("claims_done") or fl.get("claims_committed"):
            return "uncommitted:claims_done"
        return "uncommitted:other_text"
    if code == "E2_NO_EDIT":
        return "no_edit"
    return "other:" + str(code)


END_ORDER = ["resolved", "failed_patch", "deadline_kill", "self_kill", "infra", "context_overflow", "runaway_end", "format_end",
             "uncommitted:asks_or_defers", "uncommitted:claims_done", "uncommitted:announced_action",
             "uncommitted:other_text", "no_edit"]


def load():
    D = defaultdict(list)
    with gzip.open(OUT / "derived.jsonl.gz", "rt") as h:
        for line in h:
            d = json.loads(line)
            d["ending"] = ending(d)
            D[(d["dataset"], d["model"])].append(d)
    return D


def t_endings(D, lines, res):
    res["endings"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T1 [{ds}] ending class per model (all rollouts; empty-patch classes partition the empty patches)")
        hdr = "class".ljust(30) + "".join(m.rjust(11) for m in models)
        lines.append(hdr)
        res["endings"][ds] = {}
        cnt = {m: Counter(d["ending"] for d in D[(ds, m)]) for m in models}
        for m in models:
            res["endings"][ds][m] = dict(cnt[m])
        keys = END_ORDER + sorted({k for m in models for k in cnt[m]} - set(END_ORDER))
        for k in keys:
            lines.append(k.ljust(30) + "".join(str(cnt[m].get(k, 0)).rjust(11) for m in models))
        lines.append("n".ljust(30) + "".join(str(len(D[(ds, m)])).rjust(11) for m in models))


def t_uncommitted(D, lines, res):
    """The commit-instruction loss: compaction, summaries that drop the instruction, the system prompt's no-commit rule."""
    res["uncommitted"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T2 [{ds}] finished sessions that edited: compaction, the commit instruction in the last summary, and uncommitted endings")
        res["uncommitted"][ds] = {}
        for m in models:
            R = [d for d in D[(ds, m)] if d.get("finished") and d.get("has_export") and d.get("edited")]
            comp = [d for d in R if (d.get("comp_n") or 0) > 0]
            nocomp = [d for d in R if not (d.get("comp_n") or 0)]
            e1 = lambda xs: sum(1 for d in xs if d.get("code") == "E1_UNCOMMITTED")
            with_instr = [d for d in comp if d.get("last_summary_instr")]
            without = [d for d in comp if not d.get("last_summary_instr")]
            E1 = [d for d in R if d.get("code") == "E1_UNCOMMITTED"]
            rule = sum(1 for d in E1 if ((d.get("final") or {}).get("flags") or {}).get("nocommit_rule"))
            false_claim = [d for d in R if d.get("code") == "E1_UNCOMMITTED" and ((d.get("final") or {}).get("flags") or {}).get("claims_committed")
                           and not ((d.get("final") or {}).get("flags") or {}).get("defers_commit")]
            row = {"edited_finished": len(R), "compacted": len(comp), "e1_compacted": e1(comp), "e1_not_compacted": e1(nocomp),
                   "not_compacted": len(nocomp), "last_summary_has_instr": len(with_instr), "e1_with_instr": e1(with_instr),
                   "e1_without_instr": e1(without), "e1": len(E1), "e1_cites_nocommit_rule": rule,
                   "summaries": sum(d.get("comp_n") or 0 for d in R), "summaries_with_instr": sum(d.get("summ_instr_n") or 0 for d in R),
                   "summaries_mention_commit": sum(d.get("summ_commit_n") or 0 for d in R),
                   "false_commit_claims": len(false_claim)}
            res["uncommitted"][ds][m] = row
            lines.append(f"{m:10s} edited&finished {len(R)} | compacted {pct(len(comp), len(R))}: uncommitted {pct(e1(comp), len(comp))} "
                         f"vs not compacted {pct(e1(nocomp), len(nocomp))} | summaries keeping the commit instruction "
                         f"{pct(row['summaries_with_instr'], row['summaries'])} (mention 'commit' {pct(row['summaries_mention_commit'], row['summaries'])}) | "
                         f"compacted sessions whose LAST summary keeps it: uncommitted {pct(e1(with_instr), len(with_instr))} vs dropped "
                         f"{pct(e1(without), len(without))} | E1 citing the no-commit rule {pct(rule, len(E1))} | false 'committed' claims {len(false_claim)}")


def t_deadline(D, lines, res):
    res["deadline"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T3 [{ds}] deadline-kill (3 h) sessions: what they were doing")
        res["deadline"][ds] = {}
        for m in models:
            K = [d for d in D[(ds, m)] if d.get("deadline_kill") and d.get("has_export")]
            n = len(K)
            if not n:
                lines.append(f"{m:10s} 0")
                res["deadline"][ds][m] = {"n": 0}
                continue
            wall = sum((d.get("wall_s") or 0) for d in K)
            ms = sum(d.get("model_s") or 0 for d in K)
            rs = sum(d.get("runaway_s") or 0 for d in K)
            ts = sum(d.get("tool_s") or 0 for d in K)
            ss = sum(d.get("summary_s") or 0 for d in K)
            infl = Counter((d.get("inflight") or {}).get("kind") for d in K)
            long_model = sum(1 for d in K if (d.get("inflight") or {}).get("kind") == "model" and ((d.get("inflight") or {}).get("s") or 0) >= 600)
            tool_cat = Counter((d.get("inflight") or {}).get("cat") for d in K if (d.get("inflight") or {}).get("kind") == "tool")
            act = Counter(d.get("activity") for d in K)
            with_run = sum(1 for d in K if (d.get("runaway_calls") or 0) > 0)
            with_loop = sum(1 for d in K if (d.get("loop_calls") or 0) > 0)
            commit_try = sum(1 for d in K if (d.get("commit_attempts") or 0) > 0)
            since = [d.get("since_last_edit_s") for d in K if d.get("since_last_edit_s") is not None]
            no_edit_30 = sum(1 for x in since if x >= 1800)
            broken = sum(1 for d in K if (d.get("comp_broken") or 0) > 0)
            row = {"n": n, "median_calls": med([d.get("n_calls") for d in K]), "median_compactions": med([d.get("comp_n") for d in K]),
                   "with_broken_summary": broken, "share_model_time": round(ms / wall, 3) if wall else None,
                   "share_runaway_time": round(rs / wall, 3) if wall else None, "share_tool_time": round(ts / wall, 3) if wall else None,
                   "share_summary_time": round(ss / wall, 3) if wall else None, "with_runaway_call": with_run, "with_loop_call": with_loop,
                   "inflight": dict(infl), "inflight_model_ge_600s": long_model, "inflight_tool_cat": dict(tool_cat),
                   "activity": dict(act), "median_min_since_last_edit": round(med(since) / 60, 1) if since else None,
                   "no_edit_last_30min": no_edit_30, "with_commit_attempt": commit_try,
                   "median_calls_last30": med([d.get("l30_calls") for d in K]),
                   "with_edit_last30": sum(1 for d in K if (d.get("l30_cats") or {}).get("edit")),
                   "inflight_tool_ge_600s": sum(1 for d in K if (d.get("inflight") or {}).get("kind") == "tool" and ((d.get("inflight") or {}).get("s") or 0) >= 600)}
            res["deadline"][ds][m] = row
            lines.append(f"{m:10s} n={n} calls med {row['median_calls']} compactions med {row['median_compactions']} broken-summary {broken} | "
                         f"time: model {row['share_model_time']} (runaway {row['share_runaway_time']}) tools {row['share_tool_time']} summaries {row['share_summary_time']} | "
                         f"runaway call {with_run} (loop {with_loop}) | in flight {dict(infl)} model>=10min {long_model} tool {dict(tool_cat)} | "
                         f"tail activity {dict(act.most_common())} | since last edit med {row['median_min_since_last_edit']} min, >=30 min {no_edit_30} | commit attempted {commit_try} | "
                         f"last 30 min: median calls {row['median_calls_last30']}, sessions with an edit {row['with_edit_last30']} | tool running >=600 s at the kill {row['inflight_tool_ge_600s']}")


def t_tail(D, lines, res):
    res["tail"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T4 [{ds}] last-20-call activity (primary label) by outcome")
        res["tail"][ds] = {}
        for m in models:
            res["tail"][ds][m] = {}
            for oc in ("empty", "failed_patch", "resolved"):
                R = [d for d in D[(ds, m)] if d["outcome"] == oc and d.get("has_export")]
                act = Counter(d.get("activity") for d in R)
                res["tail"][ds][m][oc] = {"n": len(R), **dict(act)}
                lines.append(f"{m:10s} {oc:12s} n={len(R):3d} " + " ".join(f"{k}={v}" for k, v in act.most_common()))


def t_flags(D, lines, res):
    res["flags"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T5 [{ds}] session-level flags over all rollouts with an export")
        res["flags"][ds] = {}
        for m in models:
            R = [d for d in D[(ds, m)] if d.get("has_export")]
            F = [d for d in R if d.get("finished")]
            n = len(R)
            row = {
                "rollouts": n, "finished": len(F),
                "kill_in_session": sum(1 for d in R if (d.get("kill_calls") or 0) > 0),
                "kill_calls": sum(d.get("kill_calls") or 0 for d in R),
                "broad_kill_in_session": sum(1 for d in R if (d.get("broad_kill_calls") or 0) > 0),
                "bash_timeouts_in_session": sum(1 for d in R if (d.get("tmo_calls") or 0) > 0),
                "bash_timeouts": sum(d.get("tmo_calls") or 0 for d in R),
                "bash_calls": sum(d.get("bash_calls") or 0 for d in R),
                "compacted": sum(1 for d in R if (d.get("comp_n") or 0) > 0),
                "summaries": sum(d.get("comp_n") or 0 for d in R),
                "broken_summaries": sum(d.get("comp_broken") or 0 for d in R),
                "broken_toolcall": sum((d.get("comp_kinds") or {}).get("toolcall", 0) for d in R),
                "summary_length_cut": sum((d.get("comp_kinds") or {}).get("summary_length_cut", 0) for d in R),
                "rollouts_with_broken_summary": sum(1 for d in R if (d.get("comp_broken") or 0) > 0),
                "stop_first_call_after_compaction": sum(1 for d in R if (d.get("after_comp") or {}).get("stop")),
                "false_commit_claim": sum(1 for d in F if d["outcome"] == "empty" and ((d.get("final") or {}).get("flags") or {}).get("claims_committed")
                                          and not ((d.get("final") or {}).get("flags") or {}).get("defers_commit") and not d.get("n_commits")),
                "announced_action_stop": sum(1 for d in F if d["ending"] == "uncommitted:announced_action"),
                "retest_tail": sum(1 for d in R if d.get("activity") == "retest"),
                "runaway_in_session": sum(1 for d in R if (d.get("runaway_calls") or 0) > 0),
                "summary_as_final_answer": sum(1 for d in F if ((d.get("final") or {}).get("flags") or {}).get("anchored_summary") and not d.get("n_commits")),
                "final_loop_escape": sum(1 for d in F if d.get("final_loop_escape")),
                "runaway_cap_calls": sum(d.get("runaway_cap_calls") or 0 for d in R), "runaway_cap_loops": sum(d.get("runaway_cap_loops") or 0 for d in R),
                "runaway_ctx_calls": sum(d.get("runaway_ctx_calls") or 0 for d in R), "runaway_ctx_loops": sum(d.get("runaway_ctx_loops") or 0 for d in R),
                "rollouts_runaway_cap": sum(1 for d in R if d.get("runaway_cap_calls")),
                "rollouts_runaway_ctx_only": sum(1 for d in R if d.get("runaway_ctx_calls") and not d.get("runaway_cap_calls")),
                "tail_edit_err_ge3": sum(1 for d in R if (d.get("tail_edit_err") or 0) >= 3),
                "tail_bash_timeouts_ge2": sum(1 for d in R if (d.get("tail_tmo") or 0) >= 2),
                "committed_sessions": sum(1 for d in R if d.get("first_commit_calls_before_end") is not None),
                "first_commit_within_last5": sum(1 for d in R if d.get("first_commit_calls_before_end") is not None and d["first_commit_calls_before_end"] <= 5),
                "post_comp_reads": sum(d.get("post_comp_reads") or 0 for d in R), "post_comp_rereads": sum(d.get("post_comp_rereads") or 0 for d in R),
                "runaway_hours": round(sum(d.get("runaway_s") or 0 for d in R) / 3600, 1),
                "session_hours": round(sum(d.get("wall_s") or 0 for d in R) / 3600, 1),
            }
            res["flags"][ds][m] = row
            lines.append(f"{m:10s} n={n} | kill cmd in session {row['kill_in_session']} (broad {row['broad_kill_in_session']}, calls {row['kill_calls']}) | "
                         f"bash timeouts: {row['bash_timeouts_in_session']} rollouts, {row['bash_timeouts']}/{row['bash_calls']} bash calls | "
                         f"summaries {row['summaries']} broken {row['broken_summaries']} (tool call {row['broken_toolcall']}, length-cut {row['summary_length_cut']}) in {row['rollouts_with_broken_summary']} rollouts | "
                         f"stop as first call after compaction {row['stop_first_call_after_compaction']} | false 'committed' claim {row['false_commit_claim']} | "
                         f"announced-action stop {row['announced_action_stop']} | summary-template final answer {row['summary_as_final_answer']} | retest tail {row['retest_tail']} | "
                         f"loop-escape final {row['final_loop_escape']} | length-cut calls at 131k {row['runaway_cap_calls']} (loop {row['runaway_cap_loops']}) / context window {row['runaway_ctx_calls']} "
                         f"(loop {row['runaway_ctx_loops']}); rollouts cap {row['rollouts_runaway_cap']} + ctx-only {row['rollouts_runaway_ctx_only']}; runaway {row['runaway_hours']} h of {row['session_hours']} h | "
                         f"tail edit errors>=3 {row['tail_edit_err_ge3']} | tail bash timeouts>=2 {row['tail_bash_timeouts_ge2']} | first commit within the last 5 calls {pct(row['first_commit_within_last5'], row['committed_sessions'])} | "
                         f"re-reads in the 30 calls after a compaction {pct(row['post_comp_rereads'], row['post_comp_reads'])}")


def t_same_task(D, lines, res):
    """For each v3 ending class: the same class's rate on the same tasks for the reference models."""
    res["same_task"] = {}
    classes = ["deadline_kill", "runaway_end", "format_end", "uncommitted", "self_kill", "empty"]

    def cls_of(d, c):
        if c == "uncommitted":
            return d["ending"].startswith("uncommitted")
        if c == "empty":
            return d["outcome"] == "empty"
        return d["ending"] == c

    for ds in ("ta",):  # 2026-11-04 copy: train-acc only, + v3etp20
        refs = ["sc14", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]
        res["same_task"][ds] = {}
        lines.append(f"\n## T6 [{ds}] same-task comparison: tasks where the v3 model had >= 1 rollout of the class; the class rate of each model on those tasks (all tasks in brackets)")
        bytask = {m: defaultdict(list) for m in refs}
        for m in refs:
            for d in D[(ds, m)]:
                bytask[m][d["task_id"]].append(d)
        for v3 in ("v3etp5", "v3etp10", "v3etp20"):
            for c in classes:
                tasks = sorted(t for t, xs in bytask[v3].items() if any(cls_of(d, c) for d in xs))
                nroll = sum(1 for t in tasks for d in bytask[v3][t] if cls_of(d, c))
                row = {"v3_rollouts": nroll, "tasks": len(tasks)}
                parts = []
                for m in refs:
                    on = [d for t in tasks for d in bytask[m].get(t, [])]
                    allr = [d for xs in bytask[m].values() for d in xs]
                    k_on = sum(1 for d in on if cls_of(d, c))
                    k_all = sum(1 for d in allr if cls_of(d, c))
                    tasks_hit = sum(1 for t in tasks if any(cls_of(d, c) for d in bytask[m].get(t, [])))
                    row[m] = {"on_tasks": [k_on, len(on)], "all": [k_all, len(allr)], "tasks_with_class": tasks_hit}
                    parts.append(f"{m} {k_on}/{len(on)} [{k_all}/{len(allr)}] tasks {tasks_hit}/{len(tasks)}")
                res["same_task"][ds][f"{v3}:{c}"] = row
                lines.append(f"{v3:8s} {c:14s} v3 {nroll} rollouts on {len(tasks)} tasks | " + " | ".join(parts))


def t_sft(lines, res):
    rows = [json.loads(l) for l in gzip.open(OUT / "sft_sessions.jsonl.gz", "rt")]
    res["sft"] = {}
    lines.append("\n## T7 SFT corpora: session-level patterns per conversation (v3 by source; v2)")
    groups = {"v3:all": [r for r in rows if r["corpus"] == "v3"], "v2:all": [r for r in rows if r["corpus"] == "v2"]}
    for src in ("hengyuf", "baihe", "v4", "v5"):
        groups[f"v3:{src}"] = [r for r in rows if r["corpus"] == "v3" and r["src"] == src]
    for g, R in groups.items():
        n = len(R)
        if not n:
            continue
        ff = lambda k: sum(1 for r in R if (r.get("final_flags") or {}).get(k))
        tac = Counter(r.get("turns_after_last_commit") if r.get("turns_after_last_commit") is not None and r["turns_after_last_commit"] < 4 else ("none" if r.get("turns_after_last_commit") is None else ">=4") for r in R)
        cats_tail = Counter()
        for r in R:
            cats_tail.update(r.get("cats_tail20") or {})
        tot = sum(cats_tail.values())
        row = {
            "conversations": n, "commit_instruction": sum(r["commit_instruction"] for r in R),
            "with_successful_commit": sum(1 for r in R if r["commits_ok"] > 0), "commit_attempts>0": sum(1 for r in R if r["commit_attempts"] > 0),
            "final_has_tool_call": sum(r["final_has_call"] for r in R), "final_claims_done": ff("claims_done"),
            "final_claims_committed": ff("claims_committed"), "final_defers_or_asks": sum(1 for r in R if (r.get("final_flags") or {}).get("defers_commit") or (r.get("final_flags") or {}).get("asks_user")),
            "final_announced_action": sum(1 for r in R if (r.get("final_flags") or {}).get("intent_end") and not (r.get("final_flags") or {}).get("claims_done")),
            "final_starts_done": sum(1 for r in R if r.get("final_starts_done")),
            "turns_after_last_commit": dict(sorted(tac.items(), key=lambda kv: str(kv[0]))), "edits_after_last_commit": sum(1 for r in R if (r.get("edits_after_last_commit") or 0) > 0),
            "kill": sum(1 for r in R if r["kill_n"] > 0), "broad_kill": sum(1 for r in R if r["broad_kill_n"] > 0),
            "sleep": sum(1 for r in R if r["sleep_n"] > 0), "background": sum(1 for r in R if r["bg_n"] > 0),
            "tool_timeouts_conv": sum(1 for r in R if r["tmo_calls"] > 0), "tool_timeouts": sum(r["tmo_calls"] for r in R),
            "bash_calls": sum(sum(v for k, v in (r.get("cats") or {}).items() if k not in ("edit", "read", "todo", "subagent", "web")) for r in R),
            "retest_tail(>=8 tests & <=2 edits or same test x4)": sum(1 for r in R if (r["tail_test_calls"] >= 8 and r["tail_edits"] <= 2) or (r["tail_test_maxrep"] >= 4 and r["tail_edits"] <= 2)),
            "compaction_text_seen": sum(r["compaction_hint"] for r in R), "nocommit_rule_in_assistant_text": sum(1 for r in R if r.get("nocommit_rule_turns")),
            "commit_licensed_by_user_in_reasoning": sum(1 for r in R if r.get("commit_override_turns")),
            "first_commit_within_last5": sum(1 for r in R if r.get("first_commit_turn") is not None and r["n_turns"] - 1 - r["first_commit_turn"] <= 5),
            "commits_per_conversation": dict(sorted(Counter(min(r["commits_ok"], 3) for r in R).items())),
            "turns_median": med([r["n_turns"] for r in R]), "turns_p90": sorted(r["n_turns"] for r in R)[int(0.9 * n)], "turns_max": max(r["n_turns"] for r in R),
            "tail20_category_share": {k: round(v / tot, 3) for k, v in sorted(cats_tail.items(), key=lambda kv: (-kv[1], kv[0]))},
        }
        res["sft"][g] = row
        lines.append(f"{g:12s} n={n} commit-instr {row['commit_instruction']} | successful commit {row['with_successful_commit']} | final: tool call {row['final_has_tool_call']}, "
                     f"claims done {row['final_claims_done']}, claims committed {row['final_claims_committed']}, defers/asks {row['final_defers_or_asks']}, "
                     f"announced action {row['final_announced_action']}, starts 'Done' {row['final_starts_done']} | turns after last commit {row['turns_after_last_commit']} | "
                     f"kill {row['kill']} (broad {row['broad_kill']}), sleep {row['sleep']}, bg {row['background']}, tool timeouts {row['tool_timeouts_conv']} conv / {row['tool_timeouts']} | "
                     f"retest tail {row['retest_tail(>=8 tests & <=2 edits or same test x4)']} | compaction text {row['compaction_text_seen']} | no-commit-rule talk {row['nocommit_rule_in_assistant_text']} | "
                     f"reasoning licenses the commit by the user's request {row['commit_licensed_by_user_in_reasoning']} | first commit within the last 5 turns {row['first_commit_within_last5']} | commits/conv {row['commits_per_conversation']} | "
                     f"turns med {row['turns_median']} p90 {row['turns_p90']} max {row['turns_max']}")
        lines.append(f"{'':12s} tail-20 category share {row['tail20_category_share']}")


def t_broken_strata(D, lines, res):
    """Resolve rate by compaction count, with vs without a broken (tool-call or runaway) summary."""
    res["broken_strata"] = {}
    groups = {"ta": {"v3": ("v3etp5", "v3etp10"), "v3etp20": ("v3etp20",), "etp10+20": ("etp10", "etp20"), "base": ("sc14",)}}  # 2026-11-04 copy
    for ds, gs in groups.items():
        lines.append(f"\n## T12 [{ds}] resolved / rollouts by compaction count (1, 2, 3, 4+): all summaries fine vs >= 1 tool-call or runaway summary")
        res["broken_strata"][ds] = {}
        for g, models in gs.items():
            st = defaultdict(lambda: [0, 0, 0, 0])
            for m in models:
                for d in D[(ds, m)]:
                    n = d.get("comp_n") or 0
                    if not d.get("has_export") or not n:
                        continue
                    k = min(n, 4)
                    kinds = d.get("comp_kinds") or {}
                    brk = (kinds.get("toolcall", 0) + kinds.get("summary_length_cut", 0)) > 0
                    st[k][2 if brk else 0] += 1
                    st[k][3 if brk else 1] += d["outcome"] == "resolved"
            res["broken_strata"][ds][g] = {str(k): {"fine": [v[1], v[0]], "broken": [v[3], v[2]]} for k, v in sorted(st.items())}
            lines.append(f"{g:9s} " + " | ".join(f"{k}{'+' if k == 4 else ''}: fine {v[1]}/{v[0]} broken {v[3]}/{v[2]}" for k, v in sorted(st.items())))


def t_instr_chain(D, lines, res):
    """How the commit instruction is first lost in compacted sessions, whether a later summary restores it, and the ending."""
    res["instr_chain"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T11 [{ds}] compacted sessions: first summary without the commit instruction, by kind; recovery; uncommitted endings")
        res["instr_chain"][ds] = {}
        for m in models:
            R = [d for d in D[(ds, m)] if d.get("instr_first_loss")]
            k = Counter(d["instr_first_loss"] for d in R)
            lost = [d for d in R if d["instr_first_loss"] != "never"]
            rec = Counter(d.get("instr_recovered") for d in lost)
            e1 = sum(1 for d in lost if d.get("code") == "E1_UNCOMMITTED")
            e1_never = sum(1 for d in R if d["instr_first_loss"] == "never" and d.get("code") == "E1_UNCOMMITTED")
            row = {"compacted": len(R), "first_loss": dict(k), "recovered_later": rec.get(True, 0), "not_recovered": rec.get(False, 0),
                   "no_later_summary": rec.get(None, 0), "uncommitted_after_loss": e1, "uncommitted_never_lost": e1_never}
            res["instr_chain"][ds][m] = row
            lines.append(f"{m:10s} compacted {len(R)} | never lost {k.get('never', 0)} | first lost via tool-call summary {k.get('toolcall', 0)}, "
                         f"runaway summary {k.get('runaway', 0)}, empty {k.get('empty', 0)}, omitted by a proper summary {k.get('omitted', 0)} | "
                         f"restored by a later summary {rec.get(True, 0)} / not {rec.get(False, 0)} / no later summary {rec.get(None, 0)} | "
                         f"uncommitted: {e1}/{len(lost)} after a loss vs {e1_never}/{k.get('never', 0)} never lost")


def t_summaries(D, lines, res):
    """Compaction summary health and the outcome of the rollouts it touched."""
    res["summaries"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T8 [{ds}] compaction summaries: tool call instead of a summary / runaway (length-cut) summary; outcomes of the affected rollouts")
        res["summaries"][ds] = {}
        for m in models:
            R = [d for d in D[(ds, m)] if d.get("has_export")]
            n_s = sum(d.get("comp_n") or 0 for d in R)
            tc = sum((d.get("comp_kinds") or {}).get("toolcall", 0) for d in R)
            lc = sum((d.get("comp_kinds") or {}).get("summary_length_cut", 0) for d in R)
            ro_tc = [d for d in R if (d.get("comp_kinds") or {}).get("toolcall")]
            ro_lc = [d for d in R if (d.get("comp_kinds") or {}).get("summary_length_cut")]
            ro_comp = [d for d in R if (d.get("comp_n") or 0) > 0]
            oc = lambda xs: dict(Counter(d["outcome"] for d in xs))
            rate = lambda xs: (sum(1 for d in xs if d["outcome"] == "resolved"), len(xs))
            row = {"summaries": n_s, "toolcall": tc, "runaway_length_cut": lc, "rollouts_compacted": len(ro_comp),
                   "rollouts_toolcall": len(ro_tc), "rollouts_runaway": len(ro_lc), "outcomes_runaway": oc(ro_lc),
                   "outcomes_toolcall": oc(ro_tc), "resolved_compacted": rate(ro_comp), "resolved_runaway": rate(ro_lc),
                   "resolved_toolcall": rate(ro_tc)}
            res["summaries"][ds][m] = row
            lines.append(f"{m:10s} summaries {n_s}: tool call {pct(tc, n_s)}, runaway {pct(lc, n_s)} | rollouts: tool-call summary {len(ro_tc)} "
                         f"(resolved {rate(ro_tc)[0]}), runaway summary {len(ro_lc)} (resolved {rate(ro_lc)[0]}; {oc(ro_lc)}) | resolved among all compacted {rate(ro_comp)[0]}/{len(ro_comp)}")


def t_tailmix(D, lines, res):
    res["tailmix"] = {}
    for ds, models in DS_MODELS.items():
        lines.append(f"\n## T9 [{ds}] category share of the tool calls in the last 20 calls (finished sessions with a commit vs empty patch)")
        res["tailmix"][ds] = {}
        for m in models:
            for grp, sel in (("committed", lambda d: d.get("finished") and d.get("n_commits")), ("empty", lambda d: d["outcome"] == "empty")):
                c = Counter()
                R = [d for d in D[(ds, m)] if d.get("has_export") and sel(d)]
                for d in R:
                    c.update(d.get("tail_cats") or {})
                tot = sum(c.values()) or 1
                share = {k: round(v / tot, 3) for k, v in c.most_common()}
                res["tailmix"][ds].setdefault(m, {})[grp] = {"n": len(R), "share": share}
                lines.append(f"{m:10s} {grp:9s} n={len(R):3d} " + " ".join(f"{k} {v:.2f}" for k, v in list(share.items())[:9]))


def main():
    D = load()
    lines, res = [], {}
    t_endings(D, lines, res)
    t_uncommitted(D, lines, res)
    t_deadline(D, lines, res)
    t_tail(D, lines, res)
    t_flags(D, lines, res)
    t_same_task(D, lines, res)
    t_summaries(D, lines, res)
    t_instr_chain(D, lines, res)
    t_broken_strata(D, lines, res)
    t_tailmix(D, lines, res)
    if (OUT / "sft_sessions.jsonl.gz").exists():
        t_sft(lines, res)
    if (OUT / "narration.json").exists():
        nar = json.loads((OUT / "narration.json").read_text())
        res["narration"] = nar
        lines.append("\n## T10 narrated action turns (visible text before the tool call; out/narration.json)")
        for k, v in nar["models"].items():
            lines.append(f"{k:22s} {pct(v.get('narrated', 0), v.get('action_turns', 0))}")
        for k, v in nar["sft"].items():
            lines.append(f"SFT {k:18s} {pct(v['narrated'], v['action_turns'])}; ending with ':' {pct(v['colon'], v['action_turns'])}")
    (OUT / "tables.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "tables.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
