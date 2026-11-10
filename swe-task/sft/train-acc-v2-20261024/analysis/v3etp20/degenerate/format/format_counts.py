#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): FORMAT-lens behaviors (b8 headless call in think, b9 argument run-on, b10 malformed
visible call, b12 other chat formats, b19 schema errors, ...) for the train-acc models, with the detector and behavior
predicates of degenerate-behaviors-20261031/format (copied here: fmt_detect.py, fmt_aggregate.py; BEHAVIORS / stats /
fisher are imported from the copy, unchanged). Read-only on everything outside this directory.

    .../envs/vllm/bin/python -I -B format_counts.py   (needs scipy, as fmt_aggregate.py)  -> out/format_counts.{json,txt}

Inputs: out/ta-<model>.rollouts.jsonl (fmt_detect.py on the classifier rows; sc14 / etp20 / v3etp5 / v3etp10 are
byte-identical to the published files). Cells: units / rollouts with >= 1 / units per 10k calls or tool calls.
Tests: two-sided Fisher exact on rollouts with >= 1, v3etp20 vs v3etp10, v3etp5, etp20 and sc14.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fmt_aggregate as FA  # noqa: E402  (the copy in this directory)

MODELS = ["ta-sc14", "ta-etp20", "ta-v3etp5", "ta-v3etp10", "ta-v3etp20"]
REFS = ["ta-v3etp10", "ta-v3etp5", "ta-etp20", "ta-sc14"]


def main():
    data = {n: FA.load(n) for n in MODELS}
    out = {"denominators": {n: {"rollouts": len(r), "calls": sum(x["n_calls"] for x in r),
                                "tool_calls": sum(x["n_tool_parts"] for x in r)} for n, r in data.items()},
           "behaviors": {}}
    lines = ["| behavior | " + " | ".join(n[3:] for n in MODELS) + " | v3etp20 vs v3etp10 / v3etp5 / etp20 / sc14 (Fisher p, rollouts) |",
             "|---|" + "---|" * (len(MODELS) + 1)]
    for b, (unit, fn) in FA.BEHAVIORS.items():
        row = {n: FA.stats(rows, unit, fn) for n, rows in data.items()}
        tests = {}
        for ref in REFS:
            p = FA.fisher(row["ta-v3etp20"], row[ref])
            tests[ref] = round(p, 4) if p is not None else None
        out["behaviors"][b] = {"unit": unit, **row, "fisher_v3etp20_vs": tests}
        cells = [f"{row[n]['units']} / {row[n]['rollouts']} / {row[n]['per10k']}" for n in MODELS]
        lines.append(f"| {b} | " + " | ".join(cells) + " | " + " / ".join(str(tests[r]) for r in REFS) + " |")
    (HERE / "out/format_counts.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "out/format_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
