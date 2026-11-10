#!/usr/bin/env python3
"""verify-1: aggregate out/export-<model>.jsonl.gz (scan_export.py) into the claims about empty patches, endings,
compaction breakage, reasoning per action, markup, deadline kills, wall time. Stdlib only.

    python3 -I -B an_export.py -> out/an_export.{json,txt}
"""
import collections
import gzip
import json
import math
import random
import statistics
from pathlib import Path

OUT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/v3etp20/verify-1/out")
MODELS = ["sc14", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def load(m):
    with gzip.open(OUT / f"export-{m}.jsonl.gz", "rt") as h:
        return [json.loads(l) for l in h]


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    lo, hi = max(0, k - n2), min(k, n1)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def tc_summary(s):
    return bool(s and (s["tools"] > 0 or s["markup"]))


def wall(r):
    if r.get("resp_created_at") and r.get("first_t"):
        return r["resp_created_at"] - r["first_t"] / 1000.0
    return None


def ending(r):
    """my reimplementation of the classifier's primary-cause logic, for empty-patch rollouts."""
    if not r.get("finished"):
        if r["stdout_len"] == 0 and not r.get("export_found"):
            return "sandbox_lost"
        w = wall(r)
        if r.get("agent_error") == "TimeoutError" or (w or 0) >= 10500:
            return "deadline_kill"
        return "not_finished_other"
    if r.get("commits_ok"):
        return "commit_off_head"
    last = r.get("last_turn") or {}
    if last.get("finish") == "length":
        return "runaway" if (last.get("out") or 0) > 0 else "context_overflow"
    if last.get("think_only") or last.get("unclosed"):
        return "headless" if (last.get("r_markup") or last.get("r_special")) else "reasoning_only"
    if last.get("v_markup") or last.get("v_special"):
        return "malformed"
    if r.get("edits_ok"):
        return "uncommitted"
    return "no_edit_or_other"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    res, lines = {}, []
    data = {m: load(m) for m in MODELS}
    for m in MODELS:
        rows = data[m]
        ex = [r for r in rows if r.get("export")]
        o = {"slots": len(rows), "with_export": len(ex)}
        # reasoning per action
        o["action_turns"] = sum(r["action_turns"] for r in ex)
        o["unreasoned_actions"] = sum(r["unreasoned_actions"] for r in ex)
        o["rollouts_with_unreasoned"] = sum(r["unreasoned_actions"] > 0 for r in ex)
        o["rollouts_all_reasoned"] = sum(r["action_turns"] > 0 and r["unreasoned_actions"] == 0 for r in ex)
        o["unreasoned_detail"] = [(r["task_id"][4:30], r["slot"], r["status"], r["unreasoned_detail"]) for r in ex if r["unreasoned_actions"]]
        o["markup_turns"] = sum(r["markup_turns"] for r in ex)
        o["markup_rollouts"] = sum(r["markup_turns"] > 0 for r in ex)
        o["markup_headless"] = sum(r["markup_headless_turns"] for r in ex)
        o["special_turns"] = sum(r["special_turns"] for r in ex)
        o["special_rollouts"] = sum(r["special_turns"] > 0 for r in ex)
        o["special_tokens"] = sorted({s for r in ex for s in r["special_tokens"]})
        o["turns"] = sum(r["n_turns"] for r in ex)
        o["visible_action_pct"] = round(100 * sum(r["visible_action_turns"] for r in ex) / max(1, o["action_turns"]), 1)
        o["commit_attempt_rollouts"] = sum(r["commit_attempts"] > 0 for r in ex)
        o["commit_ok_rollouts"] = sum(r["commits_ok"] > 0 for r in ex)
        o["invalid_tool_rollouts"] = sum(r["invalid_tool_calls"] > 0 for r in ex)
        o["invalid_tool_calls"] = sum(r["invalid_tool_calls"] for r in ex)
        # compaction
        comp_rows = [r for r in ex if r["n_comp"]]
        o["compacted"] = len(comp_rows)
        summ = [(r, c) for r in ex for c in r["comps"] if c.get("summary")]
        o["summaries"] = len(summ)
        o["summaries_missing"] = sum(1 for r in ex for c in r["comps"] if not c.get("summary"))
        o["tc_summaries"] = sum(tc_summary(c["summary"]) for _, c in summ)
        o["tc_summaries_tools_parts"] = sum(c["summary"]["tools"] > 0 for _, c in summ)
        o["tc_summaries_markup"] = sum(c["summary"]["markup"] for _, c in summ)
        o["length_cut_summaries"] = sum(c["summary"]["finish"] == "length" for _, c in summ)
        o["no_goal_summaries"] = sum(not c["summary"]["goal"] for _, c in summ)
        o["broken_no_goal_or_markup"] = sum((not c["summary"]["goal"]) or c["summary"]["markup"] or c["summary"]["tools"] > 0 for _, c in summ)
        o["broken_rollouts"] = sum(any((not c["summary"]["goal"]) or c["summary"]["markup"] or c["summary"]["tools"] > 0
                                       for c in r["comps"] if c.get("summary")) for r in ex)
        by_ord = collections.defaultdict(lambda: [0, 0])
        for r, c in summ:
            k = str(min(c["ord"], 3))
            by_ord[k][0] += tc_summary(c["summary"])
            by_ord[k][1] += 1
        o["by_ord_tc"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_ord.items())}
        o["tc_rollouts"] = sum(any(tc_summary(c.get("summary")) for c in r["comps"]) for r in ex)
        for key in ("my_instr", "their_instr", "mentions_commit"):
            o[f"instr_kept_{key}"] = f"{sum(c['summary'][key] for _, c in summ)}/{len(summ)} = {100 * sum(c['summary'][key] for _, c in summ) / max(1, len(summ)):.1f}%"
        # first loss of the instruction per compacted session (their regex and mine)
        for key in ("their_instr", "my_instr"):
            lost = lost_tc = e1_lost = e1_kept = 0
            for r in comp_rows:
                first = None
                for c in r["comps"]:
                    s = c.get("summary")
                    if s is None:
                        continue
                    if not s[key]:
                        first = "toolcall" if tc_summary(s) else "other"
                        break
                is_lost = first is not None
                lost += is_lost
                lost_tc += first == "toolcall"
                e1 = r["patch_chars"] in (0, None) and ending(r) == "uncommitted"
                e1_lost += is_lost and e1
                e1_kept += (not is_lost) and e1
            o[f"lost_{key}"] = {"compacted": len(comp_rows), "lost": lost, "lost_via_toolcall": lost_tc,
                                "uncommitted_after_loss": e1_lost, "uncommitted_never_lost": e1_kept}
        # empty patches and endings
        empty = [r for r in rows if not r.get("patch_chars")]
        o["empty"] = len(empty)
        o["empty_graded"] = sum(r["status"] in ("pass", "fail") for r in empty)
        o["endings"] = dict(collections.Counter(ending(r) if r.get("export") or not r.get("finished") else "no_export" for r in empty))
        o["stopped_after_compaction_continue"] = sum(1 for r in empty if r.get("n_comp") and r.get("finished"))
        o["empty_compacted"] = f"{sum(1 for r in empty if r.get('n_comp'))}/{len(comp_rows)}"
        o["empty_not_compacted"] = f"{sum(1 for r in empty if not r.get('n_comp'))}/{len(rows) - len(comp_rows)}"
        bro = [r for r in empty if any(tc_summary(c.get("summary")) or (c.get("summary") or {}).get("finish") == "length" for c in r.get("comps") or [])]
        o["empty_after_broken_summary"] = len(bro)
        o["empty_after_broken_by_ending"] = dict(collections.Counter(ending(r) for r in bro))
        o["empty_endings_x_tc"] = dict(collections.Counter((ending(r), bool(any(tc_summary(c.get("summary")) for c in r.get("comps") or []))) for r in empty).items()) if False else \
            {f"{a}|{b}": v for (a, b), v in collections.Counter((ending(r), any(tc_summary(c.get("summary")) for c in r.get("comps") or [])) for r in empty).items()}
        # deadline kills
        dk = [r for r in rows if not r.get("finished") and not (r["stdout_len"] == 0 and not r.get("export_found"))
              and (r.get("agent_error") == "TimeoutError" or (wall(r) or 0) >= 10500)]
        o["deadline_kills"] = len(dk)
        o["deadline_kills_timeouterror"] = sum(r.get("agent_error") == "TimeoutError" for r in dk)
        o["deadline_kills_empty"] = sum(not r.get("patch_chars") for r in dk)
        o["budget_timeout_flags"] = sum(bool(r.get("budget_timeout")) for r in rows)
        o["deadline_eq_budget_timeout"] = sorted((r["task_id"], r["slot"]) for r in dk) == sorted((r["task_id"], r["slot"]) for r in rows if r.get("budget_timeout"))
        o["deadline_detail"] = [(r["task_id"][4:34], r["slot"], r.get("agent_error"), round(wall(r) or -1), r["n_comp"],
                                 sum(1 for c in r["comps"] if tc_summary(c.get("summary"))), r["n_turns"]) for r in dk]
        o["finished_over_10500"] = [(r["task_id"][4:34], r["slot"], round(wall(r))) for r in rows if r.get("finished") and (wall(r) or 0) >= 10500]
        o["not_finished"] = sum(not r.get("finished") for r in rows)
        o["not_finished_detail"] = [(r["task_id"][4:34], r["slot"], r.get("agent_error"), round(wall(r) or -1), r["stdout_len"], r.get("export_found")) for r in rows if not r.get("finished")]
        walls = [wall(r) / 60 for r in rows if wall(r) is not None]
        o["wall_n"] = len(walls)
        o["wall_median_mean_p90"] = [round(statistics.median(walls), 1), round(statistics.fmean(walls), 1), round(q(walls, 0.9), 1)]
        o["over_2h"] = sum(w > 120 for w in walls)
        # resolution by summary health among compacted rollouts
        broken_tc = [r for r in comp_rows if any(tc_summary(c.get("summary")) for c in r["comps"])]
        fine = [r for r in comp_rows if not any(tc_summary(c.get("summary")) or (c.get("summary") or {}).get("finish") == "length" or not (c.get("summary") or {}).get("goal", True) for c in r["comps"])]
        o["resolved_tc_rollouts"] = f"{sum(r['status'] == 'pass' for r in broken_tc)}/{len(broken_tc)}"
        o["resolved_fine_rollouts"] = f"{sum(r['status'] == 'pass' for r in fine)}/{len(fine)}"
        fine2 = [r for r in comp_rows if not any(tc_summary(c.get("summary")) for c in r["comps"])]
        o["resolved_not_tc_rollouts"] = f"{sum(r['status'] == 'pass' for r in fine2)}/{len(fine2)}"
        # per task tool-call summary rollouts
        o["per_task_tc"] = dict(collections.Counter(r["task_id"] for r in broken_tc))
        res[m] = o
    # tests
    t = res["v3etp20"]
    tests = {}
    for ref in ("v3etp10", "v3etp5", "etp20", "sc14"):
        r = res[ref]
        a, n, b, mm = t["tc_rollouts"], t["with_export"], r["tc_rollouts"], r["with_export"]
        tests[ref] = {"tc_rollouts": [f"{a}/{n}", f"{b}/{mm}", fisher(a, n - a, b, mm - b)],
                      "tc_summaries": [f"{t['tc_summaries']}/{t['summaries']}", f"{r['tc_summaries']}/{r['summaries']}",
                                       fisher(t["tc_summaries"], t["summaries"] - t["tc_summaries"], r["tc_summaries"], r["summaries"] - r["tc_summaries"])]}
        fa, fn = map(int, t["by_ord_tc"]["1"].split("/"))
        fb, fm = map(int, r["by_ord_tc"]["1"].split("/"))
        tests[ref]["first"] = [f"{fa}/{fn}", f"{fb}/{fm}", fisher(fa, fn - fa, fb, fm - fb)]
        la, lc = t["lost_their_instr"]["lost"], t["lost_their_instr"]["compacted"]
        lb, ld = r["lost_their_instr"]["lost"], r["lost_their_instr"]["compacted"]
        tests[ref]["lost"] = [f"{la}/{lc}", f"{lb}/{ld}", fisher(la, lc - la, lb, ld - lb)]
    res["tests"] = tests
    (OUT / "an_export.json").write_text(json.dumps(res, indent=1, default=str))
    for m in MODELS:
        o = res[m]
        lines.append(f"== {m}")
        for k, v in o.items():
            if k in ("unreasoned_detail", "per_task_tc", "deadline_detail", "not_finished_detail") and len(str(v)) > 600:
                v = str(v)[:600] + "..."
            lines.append(f"  {k}: {v}")
    lines.append(f"tests: {json.dumps(tests)}")
    (OUT / "an_export.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
