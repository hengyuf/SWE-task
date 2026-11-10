#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): compaction breakage on the 50 training tasks x 5, per model and by compaction
ordinal, from the session-ending lens rows (endings/out/sessions/ta__<model>.jsonl.gz = extract_endings.py copy; the
sc14 / etp10 / etp20 / v3etp5 / v3etp10 files are identical to the published degenerate-behaviors-20261031 ones) and
the derived rows (endings/out/derived.jsonl.gz). Stdlib only; read-only outside this directory.

    python3 -I -B compaction_v3etp20.py -> out/compaction_v3etp20.{json,txt}

Definitions (the endings lens's, derive.py):
  summary          a compaction whose summary message exists ("si"); its ordinal = 1st, 2nd, ... compaction of the session
  tool-call summary  the summary holds tool-call markup or tool parts (b2: a tool call written instead of the summary)
  length-cut summary the summary call finished on "length" (b5: a loop inside the summary call)
  instruction kept   the summary text still carries the task's commit instruction (endlib.RE_COMMIT_INSTR)
  first loss         the first summary of the session without the instruction, by kind (derive.py instr_first_loss)
Tests: two-sided Fisher exact (hypergeometric, computed here) on rollouts with >= 1 tool-call summary and on
summaries, v3etp20 vs v3etp10 / v3etp5 / etp20 / sc14.
Also: per empty-patch rollout of v3etp20 its ending class (tables.py ending()), compaction count, broken summaries and
the first-loss kind -> which empty patches a broken compaction precedes.
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
END = HERE / "endings"
sys.path.insert(0, str(END))
import tables as ET  # noqa: E402  (endings/tables.py copy: ending())

MODELS = ["sc14", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]
REFS = ["v3etp10", "v3etp5", "etp20", "sc14"]


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]] (sum of probabilities <= the observed one)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    lo, hi = max(0, k - n2), min(k, n1)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def sessions(model):
    with gzip.open(END / f"out/sessions/ta__{model}.jsonl.gz", "rt") as handle:
        return [json.loads(line) for line in handle]


def main():
    out, lines = {"models": {}, "tests": {}}, []
    for m in MODELS:
        rows = sessions(m)
        by_ord = collections.defaultdict(collections.Counter)
        roll = collections.Counter()
        for r in rows:
            s = r.get("s") or {}
            summ = [c for c in (s.get("comps") or []) if "si" in c]
            if not s:
                continue
            roll["with_export"] += 1
            roll["compacted"] += bool(s.get("comps"))
            tc_any = False
            for i, c in enumerate(summ):
                o = str(min(i + 1, 3)) + ("+" if i >= 2 else "")
                tc = bool(c.get("s_toolmarkup") or c.get("s_tools"))
                tc_any |= tc
                for key in (o, "all"):
                    by_ord[key]["summaries"] += 1
                    by_ord[key]["toolcall"] += tc
                    by_ord[key]["length_cut"] += c.get("s_fin") == "length"
                    by_ord[key]["instr_kept"] += bool(c.get("s_commit_instr"))
            roll["with_toolcall_summary"] += tc_any
        out["models"][m] = {"rollouts": roll, "by_ordinal": {k: dict(v) for k, v in sorted(by_ord.items())}}
        a = by_ord["all"]
        lines.append(f"{m:8s} rollouts {roll['with_export']} compacted {roll['compacted']} | summaries {a['summaries']}: "
                     f"tool call {a['toolcall']} ({100 * a['toolcall'] / max(1, a['summaries']):.1f}%), length-cut {a['length_cut']}, "
                     f"commit instruction kept {a['instr_kept']} ({100 * a['instr_kept'] / max(1, a['summaries']):.1f}%) | "
                     f"rollouts with a tool-call summary {roll['with_toolcall_summary']} | by ordinal (tool call / summaries): "
                     + ", ".join(f"{k}: {v['toolcall']}/{v['summaries']}" for k, v in sorted(by_ord.items()) if k != "all"))
    t = out["models"]["v3etp20"]
    for ref in REFS:
        r = out["models"][ref]
        a, n = t["rollouts"]["with_toolcall_summary"], t["rollouts"]["with_export"]
        b, m = r["rollouts"]["with_toolcall_summary"], r["rollouts"]["with_export"]
        sa, sn = t["by_ordinal"]["all"]["toolcall"], t["by_ordinal"]["all"]["summaries"]
        sb, sm = r["by_ordinal"]["all"]["toolcall"], r["by_ordinal"]["all"]["summaries"]
        out["tests"][f"v3etp20 vs {ref}"] = {
            "rollouts": [f"{a}/{n}", f"{b}/{m}", round(fisher_two_sided(a, n - a, b, m - b), 6)],
            "summaries": [f"{sa}/{sn}", f"{sb}/{sm}", round(fisher_two_sided(sa, sn - sa, sb, sm - sb), 6)]}
        lines.append(f"Fisher v3etp20 vs {ref}: rollouts with a tool-call summary {a}/{n} vs {b}/{m} p={out['tests'][f'v3etp20 vs {ref}']['rollouts'][2]}; "
                     f"tool-call summaries {sa}/{sn} vs {sb}/{sm} p={out['tests'][f'v3etp20 vs {ref}']['summaries'][2]}")
        # first compaction of a session: tool call instead of the summary (added 2026-11-04, same Fisher test)
        fa, fn = t["by_ordinal"]["1"]["toolcall"], t["by_ordinal"]["1"]["summaries"]
        fb, fm = r["by_ordinal"]["1"]["toolcall"], r["by_ordinal"]["1"]["summaries"]
        p1 = round(fisher_two_sided(fa, fn - fa, fb, fm - fb), 6)
        out["tests"][f"v3etp20 vs {ref}"]["first_summary"] = [f"{fa}/{fn}", f"{fb}/{fm}", p1]
        lines.append(f"    first summary of a session is a tool call: {fa}/{fn} vs {fb}/{fm} p={p1}")
    # compacted sessions whose commit instruction left the context (derive.py instr_first_loss != never), and the
    # uncommitted endings (E1) after such a loss (= endings tables.py T11); Fisher v3etp20 vs each reference
    loss = {}
    with gzip.open(END / "out/derived.jsonl.gz", "rt") as handle:
        for line in handle:
            x = json.loads(line)
            if not x.get("instr_first_loss"):
                continue
            c = loss.setdefault(x["model"], collections.Counter())
            c["compacted"] += 1
            lost = x["instr_first_loss"] != "never"
            c["lost"] += lost
            c["lost_via_toolcall"] += x["instr_first_loss"] == "toolcall"
            c["e1_after_loss"] += lost and x.get("code") == "E1_UNCOMMITTED"
            c["e1_never_lost"] += (not lost) and x.get("code") == "E1_UNCOMMITTED"
    out["instruction_loss"] = {m: dict(loss[m]) for m in MODELS}
    for m in MODELS:
        c = loss[m]
        lines.append(f"{m:8s} compacted {c['compacted']}: commit instruction lost {c['lost']} (via a tool-call summary {c['lost_via_toolcall']}); "
                     f"uncommitted {c['e1_after_loss']}/{c['lost']} after a loss vs {c['e1_never_lost']}/{c['compacted'] - c['lost']} never lost")
    for ref in REFS:
        a, n, b, m_ = loss["v3etp20"]["lost"], loss["v3etp20"]["compacted"], loss[ref]["lost"], loss[ref]["compacted"]
        p = round(fisher_two_sided(a, n - a, b, m_ - b), 6)
        out["tests"][f"v3etp20 vs {ref}"]["instruction_lost"] = [f"{a}/{n}", f"{b}/{m_}", p]
        lines.append(f"Fisher v3etp20 vs {ref}: compacted sessions that lost the commit instruction {a}/{n} vs {b}/{m_} p={p}")
    # empty patches of every model: ending class x compaction state (and the total preceded by a broken summary)
    for m in MODELS:
        rows = []
        with gzip.open(END / "out/derived.jsonl.gz", "rt") as handle:
            for line in handle:
                d = json.loads(line)
                if d["model"] == m and d["outcome"] == "empty":
                    d["ending"] = ET.ending(d)
                    rows.append(d)
        cls = collections.Counter()
        for d in rows:
            kinds = d.get("comp_kinds") or {}
            broken = (kinds.get("toolcall", 0) + kinds.get("summary_length_cut", 0)) > 0
            lost = d.get("instr_first_loss") not in (None, "never")
            state = "not compacted" if not d.get("comp_n") else ("broken summary" if broken else ("instruction lost (proper summary)" if lost else "compacted, summaries fine"))
            cls[(d["ending"].split(":")[0], state)] += 1
        out[f"empty_{m}"] = {f"{a} | {b}": v for (a, b), v in sorted(cls.items())}
        nb = sum(v for (a, b), v in cls.items() if b == "broken summary")
        lines.append(f"{m} empty patches {len(rows)} (after a broken summary {nb}) by ending x compaction state: "
                     + "; ".join(f"{a} | {b}: {v}" for (a, b), v in sorted(cls.items())))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/compaction_v3etp20.json").write_text(json.dumps(out, indent=1, default=dict) + "\n")
    (HERE / "out/compaction_v3etp20.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
