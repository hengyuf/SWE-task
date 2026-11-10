#!/usr/bin/env python3
"""verify-1: DeepSWE cross-check of the compaction finding, my own session parsing (the same per-session logic as
scan_export.py) on the export.json of every DeepSWE v3etp10 / v3etp20 rollout listed in the DeepSWE classifier rows
(read-only), with the classifier row's raw fields for finished / stderr / patch bytes / wall. Stdlib only.

    python3 -I -B ds_check.py -> out/ds_check.txt
"""
import collections
import importlib.util
import json
import math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis")
RUNS = {"v3etp10": A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
        "v3etp20": A / "sftv3-etp20-20261104/classifier/v3etp20/all.jsonl"}
spec = importlib.util.spec_from_file_location("scan_export", HERE / "scan_export.py")
SE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SE)


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def session(row):
    """parse the export with scan_export.scan's logic by faking an attempt dir view."""
    out = {"task_id": row["task_id"], "ri": row["ri"], "reward": row.get("reward"), "empty": (row.get("model_patch_bytes") or 0) == 0,
           "finished": row.get("opencode_finished"), "export_found": row.get("opencode_export_found"),
           "stdout_len": row.get("stdout_len"), "stderr": (row.get("stderr") or "")[-60:], "wall_s": row.get("wall_s"),
           "code": row.get("code"), "subcode": row.get("subcode")}
    p = row.get("opencode_results_fpath")
    exp = SE.read_json(p) if p and row.get("opencode_export_found") else None
    if not exp:
        out["export"] = False
        return out
    out["export"] = True
    # reuse the message walk of scan_export.scan on a synthetic slot: write nothing, call the inner logic
    msgs = exp.get("messages") or []
    comps, turns, pending = [], [], None
    edits_ok = commits_ok = 0
    for idx, m in enumerate(msgs):
        info = m.get("info") or {}
        parts = m.get("parts") or []
        if info.get("role") == "user":
            if idx and any(pp.get("type") == "compaction" for pp in parts):
                pending = {"ord": len(comps) + 1, "summary": None}
                comps.append(pending)
            continue
        if info.get("role") != "assistant":
            continue
        text = "".join(pp.get("text") or "" for pp in parts if pp.get("type") == "text" and not pp.get("synthetic") and not pp.get("ignored"))
        tools = [pp for pp in parts if pp.get("type") == "tool"]
        if info.get("mode") == "compaction" or info.get("summary") is True:
            inner, after, unclosed = SE.split_think(text)
            body = after if inner is not None else text
            s = {"tools": len(tools), "markup": bool(SE.MARKUP.search(text)), "finish": info.get("finish"),
                 "goal": "## Goal" in body, "their_instr": bool(SE.THEIR_INSTR.search(body)), "my_instr": bool(SE.MY_INSTR.search(body))}
            if pending is not None and pending["summary"] is None:
                pending["summary"] = s
            else:
                comps.append({"ord": len(comps) + 1, "summary": s, "orphan": True})
            continue
        inner, visible, unclosed = SE.split_think(text)
        reasoning = ("".join(pp.get("text") or "" for pp in parts if pp.get("type") == "reasoning") + (inner or "")).strip()
        for t in tools:
            st = t.get("state") or {}
            inp = st.get("input") if isinstance(st.get("input"), dict) else {}
            if t.get("tool") in SE.EDIT_TOOLS and st.get("status") == "completed":
                edits_ok += 1
            if t.get("tool") == "bash":
                cmd = inp.get("command") if isinstance(inp.get("command"), str) else ""
                outp = st.get("output") if isinstance(st.get("output"), str) else ((st.get("metadata") or {}).get("output") or "")
                if SE.GITCOMMIT.search(cmd) and SE.COMMIT_OK.search(outp or ""):
                    commits_ok += 1
        turns.append({"n_tools": len(tools), "finish": info.get("finish"), "out": (info.get("tokens") or {}).get("output"),
                      "unclosed": unclosed, "think_only": inner is not None and not (visible or "").strip() and not tools,
                      "r_markup": bool(SE.MARKUP.search(reasoning)), "r_special": bool(SE.SPECIAL.search(reasoning)),
                      "v_markup": bool(SE.MARKUP.search(visible or "")), "v_special": bool(SE.SPECIAL.search(visible or ""))})
    out.update({"comps": comps, "last": turns[-1] if turns else None, "edits_ok": edits_ok, "commits_ok": commits_ok})
    return out


def ending(r):
    if not r["empty"]:
        return "patch"
    if not r["finished"]:
        if (r["stdout_len"] or 0) == 0 and not r["export_found"]:
            return "sandbox_lost"
        if r["stderr"].strip().endswith("signal: killed"):
            return "deadline_kill"
        return "not_finished_other"
    if r.get("commits_ok"):
        return "commit_off_head"
    last = r.get("last") or {}
    if last.get("finish") == "length":
        return "runaway" if (last.get("out") or 0) > 0 else "context_overflow"
    if last.get("think_only") or last.get("unclosed"):
        return "format"
    if last.get("v_markup") or last.get("v_special"):
        return "format"
    if r.get("edits_ok"):
        return "uncommitted"
    return "no_edit"


def main():
    lines = []
    res = {}
    for m, path in RUNS.items():
        rows = [json.loads(l) for l in open(path)]
        with Pool(10) as pool:
            ss = pool.map(session, rows, chunksize=4)
        tc = lambda s: bool(s and (s["tools"] > 0 or s["markup"]))
        summ = [c["summary"] for r in ss for c in r.get("comps") or [] if c.get("summary")]
        first = [c["summary"] for r in ss for c in (r.get("comps") or [])[:1] if c.get("summary")]
        compacted = [r for r in ss if r.get("comps")]
        lost = 0
        unc_after_loss = unc_never = 0
        for r in compacted:
            is_lost = any(not c["summary"]["their_instr"] for c in r["comps"] if c.get("summary"))
            lost += is_lost
            e = ending(r) == "uncommitted"
            unc_after_loss += is_lost and e
            unc_never += (not is_lost) and e
        ends = collections.Counter(ending(r) for r in ss)
        dk = [r for r in ss if ending(r) == "deadline_kill"]
        dk_broken = sum(any(tc(c.get("summary")) or (c.get("summary") or {}).get("finish") == "length" for c in r.get("comps") or []) for r in dk)
        dk_broken_goal = sum(any(tc(c.get("summary")) or (c.get("summary") or {}).get("finish") == "length" or not (c.get("summary") or {}).get("goal", True)
                                 for c in r.get("comps") or []) for r in dk)
        o = {"rollouts": len(ss), "resolved": sum((r["reward"] or 0) >= 1 for r in ss), "empty": sum(r["empty"] for r in ss),
             "summaries": len(summ), "tc_summaries": sum(tc(s) for s in summ), "first": f"{sum(tc(s) for s in first)}/{len(first)}",
             "tc_rollouts": sum(any(tc(c.get("summary")) for c in r.get("comps") or []) for r in ss),
             "compacted": len(compacted), "lost": lost, "unc_after_loss": unc_after_loss, "unc_never_lost": unc_never,
             "endings_of_empty": dict(ends), "deadline_kills": len(dk), "deadline_with_broken(tc or length-cut)": dk_broken,
             "deadline_with_broken(+no goal)": dk_broken_goal,
             "classifier_codes_empty": dict(collections.Counter(r["code"] for r in ss if r["empty"]))}
        res[m] = o
        lines.append(f"{m}: {json.dumps(o)}")
    a, b = res["v3etp20"], res["v3etp10"]
    lines.append(f"Fisher tc summaries {a['tc_summaries']}/{a['summaries']} vs {b['tc_summaries']}/{b['summaries']}: "
                 f"{fisher(a['tc_summaries'], a['summaries'] - a['tc_summaries'], b['tc_summaries'], b['summaries'] - b['tc_summaries']):.3g}")
    ua, ub = a["endings_of_empty"].get("uncommitted", 0), b["endings_of_empty"].get("uncommitted", 0)
    lines.append(f"Fisher uncommitted {ua}/565 vs {ub}/565: {fisher(ua, 565 - ua, ub, 565 - ub):.3g}")
    (OUT / "ds_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
