#!/usr/bin/env python3
"""reconcile (2026-11-04): DeepSWE v3etp20 vs v3etp10 from the RAW runs (shards/shard_*/rollouts.jsonl -> each
rollout's OpenCode export.json), with the DeepSWE classifier codes (all.jsonl) only as ending labels. Read-only; stdlib.

    /usr/bin/python3 -I -B r_deepswe.py -> out/deepswe.txt

  resolved@1, the gap to base (base-step0-rep5-out131k metrics.json), empty patches by ending (classifier code:
  E1 uncommitted, N1 3 h deadline kill, E3a runaway / context overflow, E3b-d format, N4 sandbox lost, N5 self-kill,
  E4 other), compaction summaries (assistant message with info.summary or mode 'compaction'): tool call = tool-call
  markup in the text or a tool part; length-cut = finish 'length'; no-goal = no '## Goal' in the body (text after
  </think>); commit instruction = the endings lens's RE_COMMIT_INSTR on the body.
  Deadline kills (N1) with >= 1 tool-call or length-cut summary, and with >= 1 'broken' summary in the endings lens's
  sense (tool call, empty / think-only, or no 'Goal' in the body's first 400 chars), over all N1 kills and over the
  empty-patch ones.
"""
import collections
import json
import re
from multiprocessing import Pool
from pathlib import Path

E = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1")
RUNS = {"v3etp10": (E / "runs/sft-super-kimi-v3-emptythink-wsd-10ep-iter142-rep5-out131k-lenient-think",
                    E / "analysis/sftv3-etp-20261030/classifier/v3etp10/all.jsonl"),
        "v3etp20": (E / "runs/v3etp20-i284-r5-lt", E / "analysis/sftv3-etp20-20261104/classifier/v3etp20/all.jsonl")}
BASE = E / "runs/base-step0-rep5-out131k/metrics.json"
OUT = Path(__file__).resolve().parent / "out"
RE_TOOLCALL = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")
RE_COMMIT_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work))"
    r"|commit\b[^.\n]{0,40}\b(?:when|once|after)\b[^.\n]{0,30}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and|,)\s+commit\b"
    r"|commit(?:ting)? (?:it |them |everything |all )?at (?:the )?end"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`)"
    r"|new branch from main and commit|\bcommit\s*[.;]?\s*(?:\n|$)|^\s*[-*\d.)]+\s*commit\b)", re.I | re.M)


def summaries(path):
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    out = []
    for m in d.get("messages") or []:
        info = m.get("info") or {}
        if info.get("role") != "assistant" or not (info.get("summary") is True or info.get("mode") == "compaction"):
            continue
        parts = m.get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        s = text.lstrip()
        if s.startswith("<think>"):
            body = s.split("</think>", 1)[1] if "</think>" in s else ""
            think_only = "</think>" in s and not body.strip()
        else:
            body, think_only = text, False
        tc = bool(RE_TOOLCALL.search(text)) or any(p.get("type") == "tool" for p in parts)
        out.append({"tc": tc, "len_cut": info.get("finish") == "length", "goal": "## Goal" in body,
                    "lens_goal": "## Goal" in body or "Goal" in body[:400], "empty": think_only or not body,
                    "instr": bool(RE_COMMIT_INSTR.search(body))})
    return out


def load(name):
    run, cls = RUNS[name]
    rows = {}
    for sf in sorted(run.glob("shards/shard_*/rollouts.jsonl")):
        for line in open(sf):
            if line.strip():
                d = json.loads(line)
                rows[(d.get("_ng_task_index"), d.get("_ng_rollout_index"))] = {
                    "ti": d.get("_ng_task_index"), "ri": d.get("_ng_rollout_index"), "reward": d.get("reward"),
                    "patch": d.get("model_patch_bytes") or 0, "export": d.get("opencode_results_fpath") if d.get("opencode_export_found") else None}
    codes = {}
    for line in open(cls):
        c = json.loads(line)
        codes[(c["ti"], c["ri"])] = (c["code"], c.get("subcode"), (c.get("f") or {}).get("last_out_tokens"))
    keys = sorted(rows)
    with Pool(12) as pool:
        S = pool.map(summaries, [rows[k]["export"] for k in keys], chunksize=4)
    for k, s in zip(keys, S):
        rows[k]["summ"] = s
        rows[k]["code"], rows[k]["sub"], rows[k]["last_out"] = codes[k]
    return rows


def ending(r):
    code, sub = r["code"], r["sub"] or ""
    if r["reward"] == 1:
        return "resolved"
    if r["patch"]:
        return "failed_patch" if code != "N1_TIMEOUT_3H" else "failed_patch (deadline kill with a patch)"
    if code == "N1_TIMEOUT_3H":
        return "deadline_kill"
    if code == "N4_SANDBOX_LOST":
        return "infra (sandbox lost)"
    if code == "N5_SELF_KILL":
        return "self_kill"
    if sub == "E3a_cap_cut":
        return "runaway_end" if r["last_out"] else "context_overflow"
    if code == "E3_PREMATURE_END":
        return "format_end"
    if code == "E1_UNCOMMITTED":
        return "uncommitted"
    return "other:" + str(code)


def main():
    L = []
    base = json.loads(BASE.read_text())
    D = {m: load(m) for m in RUNS}
    for m, rows in D.items():
        n = len(rows)
        res = sum(1 for r in rows.values() if r["reward"] == 1)
        L.append(f"{m}: rollouts {n}; resolved {res} ({100 * res / n:.2f}%); empty patches {sum(1 for r in rows.values() if not r['patch'])}")
    b = round(base["resolved@1"] * 565 / 100)
    r20 = sum(1 for r in D["v3etp20"].values() if r["reward"] == 1)
    L.append(f"base resolved@1 {base['resolved@1']} ({b}/565); v3etp20 - base = {r20 - b}/565 = {100 * (r20 - b) / 565:+.2f} pp")
    end = {m: collections.Counter(ending(r) for r in rows.values()) for m, rows in D.items()}
    L.append("\nendings (empty patches partitioned; all rollouts):")
    for k in sorted(set(end["v3etp10"]) | set(end["v3etp20"])):
        L.append(f"  {k:42s} v3etp10 {end['v3etp10'][k]:4d}  v3etp20 {end['v3etp20'][k]:4d}  diff {end['v3etp20'][k] - end['v3etp10'][k]:+d}")
    for m, rows in D.items():
        allS = [s for r in rows.values() for s in (r["summ"] or [])]
        comp = [r for r in rows.values() if r["summ"]]
        L.append(f"\n{m}: summaries {len(allS)}, tool call {sum(s['tc'] for s in allS)} ({100 * sum(s['tc'] for s in allS) / len(allS):.1f}%), "
                 f"length-cut {sum(s['len_cut'] for s in allS)}; compacted sessions {len(comp)}; first summary a tool call "
                 f"{sum(1 for r in comp if r['summ'][0]['tc'])}; sessions with a summary lacking the commit instruction "
                 f"{sum(1 for r in comp if any(not s['instr'] for s in r['summ']))}")
        unc = [r for r in rows.values() if ending(r) == "uncommitted"]
        L.append(f"  uncommitted {len(unc)}; after a summary lacking the commit instruction "
                 f"{sum(1 for r in unc if r['summ'] and any(not s['instr'] for s in r['summ']))}")
        n1 = [r for r in rows.values() if r["code"] == "N1_TIMEOUT_3H"]
        for lab, sel in (("all N1 kills", n1), ("N1 kills with an empty patch", [r for r in n1 if not r["patch"]])):
            tcl = sum(1 for r in sel if r["summ"] and any(s["tc"] or s["len_cut"] for s in r["summ"]))
            tco = sum(1 for r in sel if r["summ"] and any(s["tc"] for s in r["summ"]))
            lens = sum(1 for r in sel if r["summ"] and any(s["tc"] or s["empty"] or not s["lens_goal"] for s in r["summ"]))
            ex = sum(1 for r in sel if r["summ"] is not None)
            L.append(f"  {lab}: {len(sel)} (with an export {ex}); with a tool-call or length-cut summary {tcl}; with a tool-call summary {tco}; "
                     f"broken in the endings lens's sense (tool call, empty, or no 'Goal' in the first 400 chars) {lens}; "
                     f"not compacted {sum(1 for r in sel if not r['summ'])}")
        fmt = [r for r in rows.values() if ending(r) == "format_end"]
        L.append(f"  format endings {len(fmt)} by subcode {dict(collections.Counter(r['sub'] for r in fmt))}; with a tool-call or length-cut summary "
                 f"{sum(1 for r in fmt if r['summ'] and any(s['tc'] or s['len_cut'] for s in r['summ']))}; compacted {sum(1 for r in fmt if r['summ'])}")
    (OUT / "deepswe.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
