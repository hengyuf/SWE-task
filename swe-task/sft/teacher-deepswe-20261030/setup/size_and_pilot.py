#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): how big are DeepSWE rollouts compared with the synthetic tasks whose per-slot hub
costs we know, and which task is a typical-size pilot? Read-only on every source; writes analysis/size-and-pilot.json.

    /usr/bin/python3 -I -B setup/size_and_pilot.py [--workers 8]

Inputs:
  DeepSWE  the 5 Kimi-K3 slots per DeepSWE task in data/targets/sc14-out131k-kimi/records (Baihe's 1M-context Kimi
           rollouts, Responses-API output items; no token counts) and the 5 sc14 slots (base student) per task
  sel50    the 250 earlier 1M Kimi-K3 rollouts of the 50 synthetic sel50 tasks (teacher-rollouts-v2 features/kimi.jsonl:
           attempt dirs + hub_tokens = OpenCode export token sums of the main session) -> terminal.json output items
Proxy (computed the same way on both sets from the output items): an assistant turn starts at the first assistant item
(reasoning / message / function_call) after a tool output or at the start; at each turn start the prompt so far is
(system + tools, a constant) + the characters of every earlier item; proxy_chars = sum over turns of the earlier-item
characters, turns = number of turns. On sel50 the exact main-session prompt tokens (input + cache_read) are fitted as
a * turns + b * proxy_chars (least squares, no intercept); the fit predicts DeepSWE prompt tokens per rollout.
size factor f = mean predicted DeepSWE prompt tokens per rollout / mean sel50 prompt tokens per rollout (Kimi, 1M).
Pilot: a wave-1 Python or Go task whose predicted Kimi prompt tokens and turns are near the median of the 113 tasks,
with some but not all of the 5 Kimi slots passing and no infra/grader slot in the targets.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

TD = Path(__file__).resolve().parents[1]
RECORDS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi/records")
TR2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
ASSISTANT = {"reasoning", "message", "function_call"}


def item_chars(item: dict) -> int:
    kind = item.get("type")
    if kind == "reasoning":
        return sum(len(part.get("text", "")) for part in item.get("summary") or []) + \
            sum(len(part.get("text", "")) for part in item.get("content") or [] if isinstance(part, dict))
    if kind == "message":
        return sum(len(part.get("text", "")) for part in item.get("content") or [] if isinstance(part, dict))
    if kind == "function_call":
        return len(item.get("arguments") or "") + len(item.get("name") or "")
    if kind == "function_call_output":
        return len(str(item.get("output") or ""))
    return len(json.dumps(item))


def proxy(output: list) -> dict:
    turns = 0
    total = 0
    cumulative = 0
    in_turn = False
    calls = 0
    for item in output:
        kind = item.get("type")
        if kind in ASSISTANT and not in_turn:
            turns += 1
            total += cumulative
            in_turn = True
        elif kind not in ASSISTANT:
            in_turn = False
        calls += kind == "function_call"
        cumulative += item_chars(item)
    return {"turns": turns, "proxy_chars": total, "final_chars": cumulative, "function_calls": calls}


def read_task(path: str) -> dict:
    out = {"task_id": Path(path).stem, "kimi": [], "sc14": []}
    with open(path) as stream:
        for line in stream:
            record = json.loads(line)
            row = proxy(record.get("output") or [])
            row.update(slot=record["slot"], status=record["status"])
            out[record["model"]].append(row)
    return out


def read_sel50(row: dict) -> dict:
    terminal = json.loads((Path(row["attempt_dir"]) / "terminal.json").read_text())
    out = proxy(((terminal.get("response") or {}).get("output")) or [])
    tokens = row.get("hub_tokens") or {}
    out.update(task_id=row["task_id"], status=row["status"],
               prompt_tokens=(tokens.get("input") or 0) + (tokens.get("cache_read") or 0),
               export_turns=row.get("turns"))
    return out


def lstsq2(xs, ys):
    """Least squares y = a*x0 + b*x1 (no intercept) via the 2x2 normal equations."""
    s00 = sum(x[0] * x[0] for x in xs)
    s01 = sum(x[0] * x[1] for x in xs)
    s11 = sum(x[1] * x[1] for x in xs)
    t0 = sum(x[0] * y for x, y in zip(xs, ys))
    t1 = sum(x[1] * y for x, y in zip(xs, ys))
    det = s00 * s11 - s01 * s01
    return (t0 * s11 - t1 * s01) / det, (s00 * t1 - s01 * t0) / det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    waves = json.loads((TD / "tasks" / "waves.json").read_text())
    tasks = {t["task_id"]: t for t in waves["tasks"]}
    base = json.loads((TD / "tasks" / "base_k10.json").read_text())["tasks"]
    sel50_rows = [json.loads(line) for line in (TR2 / "features" / "kimi.jsonl").read_text().splitlines() if line.strip()]
    sel50_rows = [row for row in sel50_rows if row.get("attempt_dir") and row["status"] in ("pass", "fail")]
    with ProcessPoolExecutor(args.workers) as pool:
        deepswe = list(pool.map(read_task, [str(RECORDS / f"{t}.jsonl") for t in tasks]))
        sel50 = list(pool.map(read_sel50, sel50_rows))
    fit_rows = [r for r in sel50 if r["prompt_tokens"] > 0 and r["turns"] > 0]
    a, b = lstsq2([(r["turns"], r["proxy_chars"]) for r in fit_rows], [r["prompt_tokens"] for r in fit_rows])
    residual = [r["prompt_tokens"] - (a * r["turns"] + b * r["proxy_chars"]) for r in fit_rows]
    mean_y = statistics.mean(r["prompt_tokens"] for r in fit_rows)
    ss_tot = sum((r["prompt_tokens"] - mean_y) ** 2 for r in fit_rows)
    r2 = 1 - sum(e * e for e in residual) / ss_tot
    per_task = []
    for entry in deepswe:
        kimi = [r for r in entry["kimi"] if r["status"] in ("pass", "fail")]
        predicted = [a * r["turns"] + b * r["proxy_chars"] for r in kimi]
        task = tasks[entry["task_id"]]
        per_task.append({
            "task_id": entry["task_id"], "language": task["language"], "wave": task["wave"],
            "kimi_graded": len(kimi), "kimi_pass": sum(r["status"] == "pass" for r in kimi),
            "kimi_other": sum(r["status"] not in ("pass", "fail") for r in entry["kimi"]),
            "kimi_turns_mean": round(statistics.mean(r["turns"] for r in kimi), 1) if kimi else None,
            "kimi_pred_prompt_tokens_mean": round(statistics.mean(predicted)) if predicted else None,
            "sc14_turns_mean": round(statistics.mean(r["turns"] for r in entry["sc14"]), 1) if entry["sc14"] else None,
            "base_k10": base[entry["task_id"]]["base_k10"]})
    deepswe_pred = [p for t in per_task for p in [t["kimi_pred_prompt_tokens_mean"]] if p is not None]
    deepswe_rollout_pred = []
    deepswe_turns = []
    for entry in deepswe:
        for r in entry["kimi"]:
            if r["status"] in ("pass", "fail"):
                deepswe_rollout_pred.append(a * r["turns"] + b * r["proxy_chars"])
                deepswe_turns.append(r["turns"])
    sel50_prompt = [r["prompt_tokens"] for r in fit_rows]
    sel50_pred = [a * r["turns"] + b * r["proxy_chars"] for r in fit_rows]
    factor = statistics.mean(deepswe_rollout_pred) / statistics.mean(sel50_prompt)
    median_pred = statistics.median(deepswe_pred)
    median_turns = statistics.median(t["kimi_turns_mean"] for t in per_task if t["kimi_turns_mean"])
    candidates = [t for t in per_task if t["wave"] == 1 and t["language"] in ("python", "go") and t["kimi_other"] == 0
                  and t["kimi_graded"] == 5 and 1 <= t["kimi_pass"] <= 4]
    for t in candidates:
        t["distance_to_median"] = round(abs(t["kimi_pred_prompt_tokens_mean"] / median_pred - 1)
                                        + abs(t["kimi_turns_mean"] / median_turns - 1), 4)
    candidates.sort(key=lambda t: t["distance_to_median"])
    report = {
        "fit": {"model": "main-session prompt tokens ~ a*turns + b*proxy_chars (Kimi-K3, 1M, sel50)",
                "a_tokens_per_turn": round(a, 2), "b_tokens_per_char": round(b, 5), "rollouts": len(fit_rows),
                "r2": round(r2, 4),
                "sel50_prompt_tokens_mean_actual": round(statistics.mean(sel50_prompt)),
                "sel50_prompt_tokens_mean_fitted": round(statistics.mean(sel50_pred))},
        "sel50": {"turns_mean": round(statistics.mean(r["turns"] for r in fit_rows), 1),
                  "turns_median": statistics.median(r["turns"] for r in fit_rows)},
        "deepswe": {"kimi_rollouts": len(deepswe_rollout_pred),
                    "turns_mean": round(statistics.mean(deepswe_turns), 1),
                    "turns_median": statistics.median(deepswe_turns),
                    "pred_prompt_tokens_mean": round(statistics.mean(deepswe_rollout_pred)),
                    "pred_prompt_tokens_median": round(statistics.median(deepswe_rollout_pred)),
                    "task_median_pred_prompt_tokens": round(median_pred), "task_median_turns": median_turns,
                    "by_wave": {str(w): {"tasks": sum(1 for t in per_task if t["wave"] == w),
                                         "pred_prompt_tokens_mean": round(statistics.mean(
                                             t["kimi_pred_prompt_tokens_mean"] for t in per_task
                                             if t["wave"] == w and t["kimi_pred_prompt_tokens_mean"]))}
                                for w in (1, 2)}},
        "size_factor_deepswe_over_sel50": round(factor, 3),
        "size_factor_by_wave": {str(w): round(statistics.mean(
            a * r["turns"] + b * r["proxy_chars"] for e in deepswe if tasks[e["task_id"]]["wave"] == w
            for r in e["kimi"] if r["status"] in ("pass", "fail")) / statistics.mean(sel50_prompt), 3) for w in (1, 2)},
        "pilot_candidates": candidates[:8],
        "pilot": candidates[0]["task_id"] if candidates else None,
        "per_task": per_task}
    out = TD / "analysis" / "size-and-pilot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=1))


if __name__ == "__main__":
    main()
