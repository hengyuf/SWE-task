#!/usr/bin/env python3
"""verify-2: own DeepSWE compaction / ending scan from the RAW run dirs (shards/shard_*/rollouts.jsonl -> each rollout's
OpenCode export.json). Read-only. For each run: rollouts (dedup by (task index, rollout index), last row wins), resolved,
empty patches, compactions (user message with a 'compaction' part), summaries (assistant message with summary == True or
mode == 'compaction'), tool-call summaries (tool markup in the summary text or a tool part), commit instruction kept
(own regex on the summary text after </think>), and the ending of each empty-patch rollout (own simple rules).

    python3 -I -B ds_compaction.py
"""
import json
import math
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

R = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs")
RUNS = {"v3etp20": R / "v3etp20-i284-r5-lt",
        "v3etp10": R / "sft-super-kimi-v3-emptythink-wsd-10ep-iter142-rep5-out131k-lenient-think"}
RE_TOOLMARK = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")
RE_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work)|all changes)"
    r"|commit\b[^.\n]{0,60}\b(?:when|once|after)\b[^.\n]{0,40}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and)\s+commit\b"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`))", re.I)


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def rows_of(shard_file):
    out = []
    with open(shard_file) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            resp = d.get("response") or {}
            out.append({"ti": d.get("_ng_task_index"), "ri": d.get("_ng_rollout_index"), "task": d.get("task_id"),
                        "reward": d.get("reward"), "patch_bytes": d.get("model_patch_bytes") or 0,
                        "finished": d.get("opencode_finished"), "export_found": d.get("opencode_export_found"),
                        "export": d.get("opencode_results_fpath"), "stderr": (d.get("opencode_run_stderr") or "")[-80:],
                        "resp_created": resp.get("created_at"), "verifier_error": (d.get("verifier_error") or "")[:80]})
    return out


def scan(row):
    r = dict(row)
    r["comps"] = []
    r["n_comp_req"] = 0
    if not (row["export_found"] and row["export"] and Path(row["export"]).exists()):
        r["no_export"] = True
        return r
    d = json.loads(Path(row["export"]).read_text())
    msgs = d.get("messages") or []
    first_t = None
    last_asst = None
    edited = False
    for m in msgs:
        info = m.get("info") or {}
        parts = m.get("parts") or []
        t = (info.get("time") or {}).get("created")
        if first_t is None and t:
            first_t = t
        if info.get("role") == "user":
            if any(p.get("type") == "compaction" for p in parts):
                r["n_comp_req"] += 1
            continue
        if info.get("role") != "assistant":
            continue
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
        if info.get("summary") is True or info.get("mode") == "compaction":
            body = text.split("</think>", 1)[1] if "</think>" in text else ("" if text.strip().startswith("<think>") else text)
            r["comps"].append({"tool": bool(RE_TOOLMARK.search(text)) or any(p.get("type") == "tool" for p in parts),
                               "len_cut": info.get("finish") == "length", "instr": bool(RE_INSTR.search(body)),
                               "goal": "## Goal" in body})
            continue
        for p in parts:
            if p.get("type") == "tool" and p.get("tool") in ("edit", "write", "patch", "multiedit") and \
                    ((p.get("state") or {}).get("status") == "completed"):
                edited = True
        tools = [p for p in parts if p.get("type") == "tool"]
        last_asst = {"finish": info.get("finish"), "tools": len(tools), "markup": bool(RE_TOOLMARK.search(text)),
                     "unclosed": text.strip().startswith("<think>") and "</think>" not in text,
                     "visible": len(text.split("</think>", 1)[1].strip()) if "</think>" in text else len(text.strip())}
    r["edited"] = edited
    r["last"] = last_asst
    r["wall_s"] = (row["resp_created"] - first_t / 1000) if (row["resp_created"] and first_t) else None
    return r


def ending(r):
    if not r["finished"]:
        if r.get("no_export"):
            return "sandbox_lost/no_export"
        if r["stderr"].strip().endswith("signal: killed") or (r.get("wall_s") or 0) >= 10500:
            return "deadline_kill"
        return "not_finished_other"
    la = r.get("last") or {}
    if la.get("finish") == "length":
        return "runaway"
    if la.get("unclosed") or la.get("markup"):
        return "format"
    if la.get("tools"):
        return "ends_on_tool_call"
    if la.get("visible", 0) == 0:
        return "format"
    return "text_end_edited" if r.get("edited") else "text_end_no_edit"


def main():
    out = {}
    lines = []
    for name, run in RUNS.items():
        rows = []
        for sf in sorted(run.glob("shards/shard_*/rollouts.jsonl")):
            rows += rows_of(sf)
        dedup = {}
        for x in rows:
            dedup[(x["ti"], x["ri"])] = x
        rows = list(dedup.values())
        with Pool(12) as pool:
            res = pool.map(scan, rows, chunksize=4)
        out[name] = res
        n = len(res)
        resolved = sum((x["reward"] or 0) == 1 for x in res)
        empty = [x for x in res if x["patch_bytes"] == 0]
        comp = [x for x in res if x["comps"]]
        summ = [c for x in res for c in x["comps"]]
        tc = sum(c["tool"] for c in summ)
        first = [x["comps"][0] for x in comp]
        lost = [x for x in comp if any(not c["instr"] for c in x["comps"])]
        ends = Counter(ending(x) for x in empty)
        broken = lambda x: any(c["tool"] or c["len_cut"] for c in x["comps"])
        unc = [x for x in empty if ending(x) == "text_end_edited"]
        dk = [x for x in empty if ending(x) == "deadline_kill"]
        lines.append(f"{name}: rollouts {n}; resolved {resolved} ({100 * resolved / n:.2f}%); empty patches {len(empty)}; "
                     f"compacted sessions {len(comp)} (compaction requests {sum(x['n_comp_req'] for x in res)}); summaries {len(summ)}; "
                     f"tool-call summaries {tc} ({100 * tc / max(1, len(summ)):.1f}%); first compaction tool call {sum(c['tool'] for c in first)}/{len(first)}; "
                     f"sessions with a summary lacking the commit instruction {len(lost)}/{len(comp)}; empty-patch endings {dict(ends)}; "
                     f"uncommitted (finished, text ending, edited) {len(unc)}, after a loss {sum(x in lost for x in unc)}; "
                     f"deadline kills (empty) {len(dk)}, with a broken summary {sum(broken(x) for x in dk)}")
        out[name + "_summ"] = (len(summ), tc, len(first), sum(c["tool"] for c in first), len(comp), len(lost), len(unc))
    a, b = out["v3etp20_summ"], out["v3etp10_summ"]
    lines.append(f"Fisher tool-call summaries v3etp20 {a[1]}/{a[0]} vs v3etp10 {b[1]}/{b[0]}: p={fisher(a[1], a[0] - a[1], b[1], b[0] - b[1]):.2g}")
    lines.append(f"Fisher uncommitted v3etp20 {a[6]}/565 vs v3etp10 {b[6]}/565: p={fisher(a[6], 565 - a[6], b[6], 565 - b[6]):.2g}")
    Path(__file__).resolve().with_name("ds_compaction.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
