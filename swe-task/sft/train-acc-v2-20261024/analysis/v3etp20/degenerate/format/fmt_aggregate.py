#!/usr/bin/env python3
"""Aggregate fmt_detect.py outputs into per-behavior tables with denominators and Fisher tests (read-only on inputs).

usage: fmt_aggregate.py  -> out/format_table.{json,txt}
Inputs: out/<NAME>.rollouts.jsonl (fmt_detect.py), out/spill_detail.json, out/patch_markup.json, out/fence_stats.json,
out/sft_format.json, and the train-acc counting-parser call logs (TA/analysis/think/calls-<model>.jsonl).

Per behavior and model: units (calls or tool calls) / denominator, rate per 10k units, rollouts with >= 1 / rollouts.
Tests (two-sided Fisher exact on rollouts with >= 1 vs without):
  DeepSWE: v3etp5 vs etp5 (same epochs), vs etp10 (~same steps), vs base-pool (base-orig + base-rerun, 1,130 rollouts);
           v3etp10 vs etp10 (same epochs), vs etp20 (~same steps), vs base-pool
  train-acc: the same pairs with sc14 as base (250 slots each)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
DS = ["ds-base-orig", "ds-base-rerun", "ds-v1raw5", "ds-v2raw5", "ds-v2raw10", "ds-v2raw20", "ds-etp5", "ds-etp10",
      "ds-etp20", "ds-v3etp5", "ds-v3etp10"]
TAM = ["ta-sc14", "ta-sftv2", "ta-sftv2e10", "ta-sftv2e20", "ta-etp5", "ta-etp10", "ta-etp20", "ta-v3etp5", "ta-v3etp10"]
LABEL = {"ds-base-orig": "base-orig", "ds-base-rerun": "base-rerun", "ds-v1raw5": "v1raw5", "ds-v2raw5": "v2raw5",
         "ds-v2raw10": "v2raw10", "ds-v2raw20": "v2raw20", "ds-etp5": "etp5", "ds-etp10": "etp10", "ds-etp20": "etp20",
         "ds-v3etp5": "v3etp5", "ds-v3etp10": "v3etp10", "ta-sc14": "base", "ta-sftv2": "v2raw5",
         "ta-sftv2e10": "v2raw10", "ta-sftv2e20": "v2raw20", "ta-etp5": "etp5", "ta-etp10": "etp10", "ta-etp20": "etp20",
         "ta-v3etp5": "v3etp5", "ta-v3etp10": "v3etp10"}

# behavior -> (unit, predicate over a rollout record returning the unit count)
def sub_sum(r, flag, pred):
    return sum(v for k, v in (r["sub"].get(flag) or {}).items() if pred(k))


BEHAVIORS = {
    "B01 tool call written inside the think block (reasoning ends in call markup; no parsed call)":
        ("calls", lambda r: r["calls"].get("R_CALL_TAIL", 0)),
    "B02 tool-call markup quoted inside the reasoning (not a trailing call)":
        ("calls", lambda r: r["calls"].get("R_CALL_MENTION", 0)),
    "B03 another chat format's syntax (<|open|>/<|close|>/<|sep|>, harmony, DSML, <invoke>) in reasoning / text / args":
        ("calls", lambda r: r["calls"].get("FOREIGN", 0) + r["tools"].get("FOREIGN_ARG", 0)),
    "B04 <|name|> special-token strings in the reasoning":
        ("calls", lambda r: r["calls"].get("R_SPECIAL", 0)),
    "B05 malformed visible tool call (markup in the visible text, not parsed)":
        ("calls", lambda r: r["calls"].get("V_CALL", 0)),
    "B06 doubled / misplaced think tag (</think> or <think> in the visible text, <think> in the reasoning)":
        ("calls", lambda r: r["calls"].get("THINK_TAG", 0)),
    "B07 empty output / <= 1 output token (EOS right after <think>)":
        ("calls", lambda r: max(r["calls"].get("EMPTY", 0), r["calls"].get("LE1", 0))),
    "B08 finish tool-calls without a tool part / cut inside a call by the 131k cap":
        ("calls", lambda r: r["calls"].get("TC_NO_TOOLS", 0) + r["calls"].get("CUT_IN_CALL", 0)),
    "B09 hallucinated tool name (OpenCode 'invalid' tool)":
        ("tools", lambda r: r["tools"].get("T_INVALID", 0)),
    "B10 schema error (missing required key / wrong type)":
        ("tools", lambda r: r["tools"].get("T_SCHEMA", 0)),
    "B10a ... missing key, input without foreign keys and without a run-on":
        ("tools", lambda r: sub_sum(r, "T_SCHEMA", lambda k: ":missing:" in k and ":with_foreign_keys" not in k
                                    and ":with_spill" not in k)),
    "B10c ... missing key in a run-on call (foreign keys or spill in the same input)":
        ("tools", lambda r: sub_sum(r, "T_SCHEMA", lambda k: ":missing:" in k and (":with_foreign_keys" in k
                                                                                    or ":with_spill" in k))),
    "B10b ... wrong type":
        ("tools", lambda r: sub_sum(r, "T_SCHEMA", lambda k: ":type:" in k)),
    "B11 unknown parameter name (excluding base's bash.description habit)":
        ("tools", lambda r: sub_sum(r, "T_UNKNOWN_KEY", lambda k: not k.startswith("base_habit"))),
    "B11b ... standalone (no run-on in the same input)":
        ("tools", lambda r: sub_sum(r, "T_UNKNOWN_KEY", lambda k: not k.startswith("base_habit") and ":with_spill" not in k)),
    "B11c ... merged from a run-on (spill in the same input)":
        ("tools", lambda r: sub_sum(r, "T_UNKNOWN_KEY", lambda k: not k.startswith("base_habit") and ":with_spill" in k)),
    "B11a ... base habit bash.description":
        ("tools", lambda r: sub_sum(r, "T_UNKNOWN_KEY", lambda k: k.startswith("base_habit"))),
    "B12 parameter run-on: next reasoning / call absorbed into an argument (ARG_SPILL)":
        ("tools", lambda r: r["tools"].get("ARG_SPILL", 0)),
    "B12a ... written into a file (completed edit / write)":
        ("tools", lambda r: r["tools"].get("ARG_SPILL_FILE", 0)),
    "B13 degenerate repetition inside a tool argument":
        ("tools", lambda r: r["tools"].get("T_ARG_LOOP", 0)),
    "B14 oversized argument (>= 65,536 chars)":
        ("tools", lambda r: r["tools"].get("T_BIG_ARG", 0)),
    "B14a ... arguments >= 32,768 chars":
        ("tools", lambda r: sum(1 for _, s in r.get("arg_sizes_ge16k", []) if s >= 32768)),
    "B15 duplicated parallel call (identical tool + input twice in one call)":
        ("calls", lambda r: r["calls"].get("DUP_PARALLEL", 0)),
    "B16 >= 10 tool calls in one model call":
        ("calls", lambda r: r["calls"].get("MANY_PARALLEL", 0)),
    "B17 no-op edit (oldString == newString)":
        ("tools", lambda r: r["tools"].get("T_NOOP_EDIT", 0)),
    "B18 bad path argument (newline / markup / padding in filePath, path, workdir)":
        ("tools", lambda r: r["tools"].get("T_PATH_BAD", 0)),
    "B19 read-tool line-number prefixes copied into edit / write":
        ("tools", lambda r: r["tools"].get("T_LINENO", 0)),
    "B20 template placeholder as tool / parameter name (parameter_name, function_name, *_placeholder, command)":
        ("tools", lambda r: sub_sum(r, "T_UNKNOWN_KEY", lambda k: k.startswith("placeholder"))
         + sub_sum(r, "T_INVALID", lambda k: k == "placeholder")),
    "B21 blank must-be-non-empty argument (command, filePath, pattern, ...)":
        ("tools", lambda r: r["tools"].get("T_EMPTY_REQ", 0)),
    "B22 self-written <tool_response> / <|im_start|>,<|im_end|> in reasoning or text":
        ("calls", lambda r: r["calls"].get("TOOL_RESP", 0) + r["calls"].get("ROLE", 0)),
}

PAIRS = {"ds": [("ds-v3etp5", "ds-etp5"), ("ds-v3etp5", "ds-etp10"), ("ds-v3etp5", "BASE"),
                ("ds-v3etp10", "ds-etp10"), ("ds-v3etp10", "ds-etp20"), ("ds-v3etp10", "BASE")],
         "ta": [("ta-v3etp5", "ta-etp5"), ("ta-v3etp5", "ta-etp10"), ("ta-v3etp5", "BASE"),
                ("ta-v3etp10", "ta-etp10"), ("ta-v3etp10", "ta-etp20"), ("ta-v3etp10", "BASE")]}


def load(name):
    return [json.loads(line) for line in open(OUT / f"{name}.rollouts.jsonl")]


def stats(rows, unit, fn):
    units = sum(fn(r) for r in rows)
    roll = sum(1 for r in rows if fn(r) > 0)
    den = sum(r["n_calls"] if unit == "calls" else r["n_tool_parts"] for r in rows)
    return {"units": units, "den": den, "per10k": round(1e4 * units / den, 3) if den else None, "rollouts": roll,
            "n": len(rows)}


def fisher(a, b):
    table = [[a["rollouts"], a["n"] - a["rollouts"]], [b["rollouts"], b["n"] - b["rollouts"]]]
    if table[0][0] == table[1][0] == 0:
        return None
    return fisher_exact(table)[1]


def counting_parser():
    """train-acc counting lenient-think parser: per model calls with a missing </think> (rescued tool call,
    unclosed no tool: EOS / length) and tool call inside a closed think; rollouts = (task, slot)."""
    res = {}
    for m in ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10"]:
        c, rolls = Counter(), defaultdict(set)
        for line in open(TA / f"analysis/think/calls-{m}.jsonl"):
            r = json.loads(line)
            c["calls"] += 1
            k = r.get("log_kind")
            key = (r.get("task_id"), r.get("slot"))
            if k:
                kk = k + (":" + str(r.get("finish")) if k == "UNCLOSED_NO_TOOL" else "")
                c[kk] += 1
                rolls[kk].add(key)
            if r.get("empty_output"):
                c["empty_output"] += 1
                rolls["empty_output"].add(key)
        res[f"ta-{m}"] = {**c, **{f"rollouts:{k}": len(v) for k, v in rolls.items()},
                          "counting_parser": m not in ("sc14", "sftv2")}
    return res


def main():
    data = {n: load(n) for n in DS + TAM}
    out = {"behaviors": {}, "tests": {}, "denominators": {}}
    for n, rows in data.items():
        out["denominators"][n] = {"rollouts": len(rows), "calls": sum(r["n_calls"] for r in rows),
                                  "tool_calls": sum(r["n_tool_parts"] for r in rows),
                                  "no_export": sum(1 for r in rows if r.get("no_export"))}
    base_pool = data["ds-base-orig"] + data["ds-base-rerun"]
    lines = []
    for b, (unit, fn) in BEHAVIORS.items():
        out["behaviors"][b] = {"unit": unit}
        for n, rows in data.items():
            out["behaviors"][b][n] = stats(rows, unit, fn)
        out["behaviors"][b]["ds-base-pool"] = stats(base_pool, unit, fn)
        tests = {}
        for kind, pairs in PAIRS.items():
            for a, ref in pairs:
                ra = out["behaviors"][b][a]
                rb = out["behaviors"][b]["ds-base-pool" if (ref == "BASE" and kind == "ds") else ("ta-sc14" if ref == "BASE" else ref)]
                p = fisher(ra, rb)
                tests[f"{a} vs {ref if ref != 'BASE' else ('base-pool' if kind == 'ds' else 'sc14')}"] = {
                    "a": f"{ra['rollouts']}/{ra['n']}", "b": f"{rb['rollouts']}/{rb['n']}",
                    "rate_ratio_per_unit": round(ra["per10k"] / rb["per10k"], 2) if rb["per10k"] else None,
                    "fisher_p_rollouts": round(p, 4) if p is not None else None}
        out["tests"][b] = tests
    out["counting_parser_ta"] = counting_parser()
    for extra in ("spill_detail", "patch_markup", "fence_stats", "sft_format"):
        p = OUT / f"{extra}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if extra == "spill_detail":
                d = {k: v["counts"] for k, v in d.items()}
            if extra == "patch_markup":
                d = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} | {"rows": v["rows"]} for k, v in d.items()}
            out[extra] = d
    # text table
    cols = DS + ["ds-base-pool"]
    for title, names in (("DeepSWE (113 tasks x 5; base-pool = base-orig + base-rerun)", DS),
                         ("train-acc (50 v2 training tasks x 5)", TAM)):
        lines.append(f"== {title}")
        lines.append("denominators: " + " | ".join(
            f"{LABEL[n]} {out['denominators'][n]['rollouts']} rollouts, {out['denominators'][n]['calls']:,} calls, "
            f"{out['denominators'][n]['tool_calls']:,} tool calls" for n in names))
        hdr = f"{'behavior':<44}" + "".join(f"{LABEL[n]:>17}" for n in names)
        lines.append(hdr)
        for b in BEHAVIORS:
            row = f"{b[:44]:<44}"
            for n in names:
                s = out["behaviors"][b][n]
                row += f"{s['units']:>7}/{s['rollouts']:>3} {s['per10k']:>5.2f}" if s["per10k"] is not None else f"{'-':>17}"
            lines.append(row)
        lines.append("  cells: units / rollouts with >= 1, then units per 10k calls (calls) or per 10k tool calls (tools)")
        lines.append("")
    lines.append("== Fisher exact (rollouts with >= 1), two-sided")
    for b, tests in out["tests"].items():
        sig = {k: v for k, v in tests.items() if v["fisher_p_rollouts"] is not None and v["fisher_p_rollouts"] < 0.05}
        lines.append(f"{b[:60]:<60} " + "; ".join(f"{k}: {v['a']} vs {v['b']} p={v['fisher_p_rollouts']}" for k, v in tests.items()))
    lines.append("")
    lines.append("== train-acc counting-parser (calls; rollouts)")
    for n, c in out["counting_parser_ta"].items():
        if not c.get("counting_parser"):
            lines.append(f"{LABEL[n]:<10} calls {c['calls']:,} | served with the plain lenient parser: no rescue / unclosed log (n/a)"
                         f" | empty output {c.get('empty_output', 0)}")
            continue
        lines.append(f"{LABEL[n]:<10} calls {c['calls']:,} | rescued tool call from unclosed think {c.get('RECOVERED', 0)}; "
                     f"{c.get('rollouts:RECOVERED', 0)} | tool call in closed think {c.get('TOOL_IN_CLOSED_THINK', 0)}; "
                     f"{c.get('rollouts:TOOL_IN_CLOSED_THINK', 0)} | unclosed, no tool, EOS {c.get('UNCLOSED_NO_TOOL:stop', 0)}; "
                     f"{c.get('rollouts:UNCLOSED_NO_TOOL:stop', 0)} | unclosed, cut by length {c.get('UNCLOSED_NO_TOOL:length', 0)}; "
                     f"{c.get('rollouts:UNCLOSED_NO_TOOL:length', 0)} | empty output {c.get('empty_output', 0)}")
    (OUT / "format_table.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (OUT / "format_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
