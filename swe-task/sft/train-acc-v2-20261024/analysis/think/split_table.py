#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): per-model missing-</think> table from calls-<model>.jsonl (think_analysis.py output).
Logged categories for the counted fleets (10/20 ep); for 5 ep / base the unclosed-no-tool category uses the signature
validated on 10/20 ep (no tool call and no content <=> logged UNCLOSED_NO_TOOL: 94/94, 0 false positives), split by
finish reason (stop = EOS inside the think block; length = cut by the per-call cap / context while still thinking).
Recovered (missing </think> before a tool call) is logged-only; for 5 ep / base only its exact subset with no text
before the tool call (reasoning None + tool call: precision 100% by construction, recall 90% / 67% on 10 / 20 ep) is
counted, as a lower bound.
2026-10-27: + etp5 / etp10 (empty-think-prefix SFT 5 / 10 ep; counting parser, so logged like 10 / 20 ep).
    python3 -I -B split_table.py -> split_table.{json,md}
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = {"sc14": "base", "sftv2": "5 ep", "sftv2e10": "10 ep", "sftv2e20": "20 ep",
         "etp5": "etp 5 ep",
         "etp10": "etp 10 ep",
         "etp20": "etp 20 ep",
         "v3etp5": "v3 etp 5 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 5 ep (iter71), counting parser
         "v3etp10": "v3 etp 10 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 10 ep (iter142), counting parser
         }  # the one model registry (one line per model, column order)
MODELS = tuple(LABEL)
LOGGED = tuple(m for m in MODELS if m not in ("sc14", "sftv2"))  # all later fleets were served with the counting parser
SLOTS = 250


def main():
    out = {}
    for m in MODELS:
        calls = 0
        n = collections.Counter()
        rollouts = collections.defaultdict(set)
        logged = m in LOGGED
        for line in open(HERE / f"calls-{m}.jsonl"):
            r = json.loads(line)
            calls += 1
            key = (r["task_id"], r["slot"])
            unclosed = (r["log_kind"] == "UNCLOSED_NO_TOOL") if logged else (r["tool_calls"] == 0 and not r["content_len"])
            cats = {
                "recovered": r["log_kind"] == "RECOVERED" if logged else None,
                "recovered_bare_lower_bound": r["reasoning_len"] is None and r["tool_calls"] > 0,
                "unclosed_no_tool": unclosed,
                "unclosed_no_tool_eos": unclosed and r["finish"] == "stop",
                "unclosed_no_tool_length": unclosed and r["finish"] == "length",
                "empty_output": r["empty_output"],
                "tool_call_in_closed_think": r["tool_in_reasoning"],
                "one_token_stop": r["one_token_stop"],
            }
            for cat, hit in cats.items():
                if hit:
                    n[cat] += 1
                    rollouts[cat].add(key)
        out[m] = {"calls": calls, **{cat: {"calls": n[cat], "per_call_pct": round(100 * n[cat] / calls, 3),
                                          "rollouts": len(rollouts[cat]),
                                          "rollouts_pct": round(100 * len(rollouts[cat]) / SLOTS, 1)}
                                    for cat in ("recovered", "recovered_bare_lower_bound", "unclosed_no_tool",
                                                "unclosed_no_tool_eos", "unclosed_no_tool_length", "empty_output",
                                                "tool_call_in_closed_think", "one_token_stop")}}
        if not logged:
            out[m]["recovered"] = None
    (HERE / "split_table.json").write_text(json.dumps(out, indent=1) + "\n")
    cats = ["recovered", "recovered_bare_lower_bound", "unclosed_no_tool", "unclosed_no_tool_eos", "unclosed_no_tool_length",
            "empty_output", "tool_call_in_closed_think", "one_token_stop"]
    lines = ["| category (calls, % of calls; rollouts with >= 1, % of 250) | " + " | ".join(LABEL[m] for m in MODELS) + " |",
             "|---|" + "---|" * len(MODELS), "| model calls (final attempts) | " + " | ".join(str(out[m]["calls"]) for m in MODELS) + " |"]
    for cat in cats:
        cells = []
        for m in MODELS:
            v = out[m].get(cat)
            cells.append("n/a" if v is None else f"{v['calls']} ({v['per_call_pct']}%); {v['rollouts']} ({v['rollouts_pct']}%)")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    (HERE / "split_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
