#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): DeepSWE cross-check of the train-acc compaction finding. The endings-lens tables
(T1 ending classes, T3 deadline kills, T8 compaction summaries, T11 commit-instruction loss) for DeepSWE v3etp10 and
v3etp20, computed with the endings-lens code (extract_endings.py / derive.py copies here; tables.py functions from
../endings/tables.py with DS_MODELS set to these two runs). DeepSWE v3etp10's session file is byte-identical to the
published degenerate-behaviors-20261031 one, so its rows reproduce that report (T8: 45/593 tool-call summaries).
Also the tool-call summary rate by compaction ordinal (1st, 2nd, 3rd+). Read-only outside this directory.

    python3 -I -B derive.py && python3 -I -B ds_crosscheck.py -> out/ds_crosscheck.txt
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "endings"))
import tables as ET  # noqa: E402


def main():
    ET.OUT = HERE / "out"
    ET.DS_MODELS = {"deepswe": ["v3etp10", "v3etp20"]}
    D = ET.load()
    lines, res = [], {}
    ET.t_endings(D, lines, res)
    ET.t_deadline(D, lines, res)
    ET.t_summaries(D, lines, res)
    ET.t_instr_chain(D, lines, res)
    lines.append("\n## tool-call summaries by compaction ordinal (DeepSWE)")
    for m in ("v3etp10", "v3etp20"):
        c = collections.defaultdict(collections.Counter)
        with gzip.open(HERE / f"out/sessions/deepswe__{m}.jsonl.gz", "rt") as handle:
            for line in handle:
                s = json.loads(line).get("s") or {}
                summ = [x for x in (s.get("comps") or []) if "si" in x]
                for i, x in enumerate(summ):
                    o = "1" if i == 0 else ("2" if i == 1 else "3+")
                    c[o]["n"] += 1
                    c[o]["toolcall"] += bool(x.get("s_toolmarkup") or x.get("s_tools"))
        lines.append(f"{m:8s} " + ", ".join(f"{o}: {c[o]['toolcall']}/{c[o]['n']}" for o in ("1", "2", "3+")))
        res.setdefault("by_ordinal", {})[m] = {o: dict(v) for o, v in c.items()}
    # two-sided Fisher exact, v3etp20 vs v3etp10 (the same hypergeometric test as ../compaction_v3etp20.py)
    sys.path.insert(0, str(HERE.parent))
    from compaction_v3etp20 import fisher_two_sided as fisher  # noqa: E402
    S = res["summaries"]["deepswe"]
    I = res["instr_chain"]["deepswe"]
    O = res["by_ordinal"]
    E = res["endings"]["deepswe"]
    unc = {m: sum(v for k, v in E[m].items() if k.startswith("uncommitted")) for m in ("v3etp10", "v3etp20")}
    tests = {
        "toolcall_summaries": (S["v3etp20"]["toolcall"], S["v3etp20"]["summaries"], S["v3etp10"]["toolcall"], S["v3etp10"]["summaries"]),
        "rollouts_with_toolcall_summary": (S["v3etp20"]["rollouts_toolcall"], 565, S["v3etp10"]["rollouts_toolcall"], 565),
        "first_summary_toolcall": (O["v3etp20"]["1"]["toolcall"], O["v3etp20"]["1"]["n"], O["v3etp10"]["1"]["toolcall"], O["v3etp10"]["1"]["n"]),
        "compacted_sessions_instruction_lost": (I["v3etp20"]["compacted"] - I["v3etp20"]["first_loss"].get("never", 0), I["v3etp20"]["compacted"],
                                                I["v3etp10"]["compacted"] - I["v3etp10"]["first_loss"].get("never", 0), I["v3etp10"]["compacted"]),
        "uncommitted_endings": (unc["v3etp20"], 565, unc["v3etp10"], 565),
    }
    lines.append("\n## Fisher exact, DeepSWE v3etp20 vs v3etp10")
    res["fisher_v3etp20_vs_v3etp10"] = {}
    for k, (a, n, b, m_) in tests.items():
        p = fisher(a, n - a, b, m_ - b)
        res["fisher_v3etp20_vs_v3etp10"][k] = [f"{a}/{n}", f"{b}/{m_}", p]
        lines.append(f"{k}: {a}/{n} vs {b}/{m_} p={p:.3g}")
    (HERE / "out/ds_crosscheck.json").write_text(json.dumps(res, indent=1, default=dict) + "\n")
    (HERE / "out/ds_crosscheck.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
