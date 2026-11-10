#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): the phase report (read-only on inputs; writes analysis/<out>.json and .md).

    /usr/bin/python3 -I -B phase_report.py [--tasksets sel50] [--out phase1-report]

Inputs (made first): features/{sc14,kimi,kimi3}.jsonl (features.py; sc14 / kimi = teacher-rollouts-v2's files on the
50 tasks, copied unchanged), analysis/window-kimi3.jsonl + window-kimi.jsonl (window.py), costs/kimi3-<campaign>.json
(costs.py), costs/kimi-r0.json (teacher-rollouts-v2 kimi_costs.py), results/kimi-<campaign>.json, legs.jsonl.
(a) reasoning before actions: pooled % of action turns with reasoning, 95% CI by task bootstrap (2000 resamples, as
    compare.py) and the PAIRED task-bootstrap CI of kimi3 - kimi (same tasks resampled for both); % rollouts with every
    action reasoned; longest unreasoned action run (median / mean / p90 / max); reasoning tokens (sc14 tokenizer) per
    reasoning turn and per action turn incl. zeros; visible text on action turns (with / without reasoning), bare actions
(b) the 262,144 window: window.py's summary for kimi3 and kimi (compactions, hub errors, length cuts, early stops,
    commits after compaction, pass@1 compacted vs not)
(c) cost: LiteLLM-reported usd per slot (all attempts) and the projection for the remaining tasks
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SELECTION = S / "sft/corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"
LABEL = {"sc14": "base Nemotron sc14", "kimi": "Kimi-K3 previous (1M, no sentence)", "kimi3": "Kimi-K3 v3 (256k + sentence)"}


def load(name):
    path = HERE / "features" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def by_task(rows, num, den):
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("export") and r.get("turns"):
            out[r["task_id"]][0] += num(r)
            out[r["task_id"]][1] += den(r)
    return out


def boot(rows, num, den, n=2000, seed=20261025):
    values = list(by_task(rows, num, den).values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        d = sum(v[1] for v in sample)
        if d:
            stats.append(100.0 * sum(v[0] for v in sample) / d)
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 1), round(stats[int(0.975 * len(stats)) - 1], 1)] if stats else None


def paired_boot(a_rows, b_rows, num, den, n=2000, seed=20261026):
    """95% CI of pooled-ratio(a) - pooled-ratio(b), resampling the shared tasks."""
    a, b = by_task(a_rows, num, den), by_task(b_rows, num, den)
    tasks = sorted(set(a) & set(b))
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        da, db = sum(a[t][1] for t in sample), sum(b[t][1] for t in sample)
        if da and db:
            stats.append(100.0 * sum(a[t][0] for t in sample) / da - 100.0 * sum(b[t][0] for t in sample) / db)
    stats.sort()
    point = 100.0 * sum(a[t][0] for t in tasks) / sum(a[t][1] for t in tasks) - \
        100.0 * sum(b[t][0] for t in tasks) / sum(b[t][1] for t in tasks)
    return {"diff": round(point, 1), "ci95": [round(stats[int(0.025 * len(stats))], 1),
                                              round(stats[int(0.975 * len(stats)) - 1], 1)], "tasks": len(tasks)}


def reasoning(rows):
    ex = [r for r in rows if r.get("export") and r.get("turns")]
    graded = [r for r in rows if r.get("status") in ("pass", "fail")]
    A = sum(r["action_turns"] for r in ex)
    T = sum(r["turns"] for r in ex)
    runs = [r["longest_unreasoned_action_run"] for r in ex]
    rtok = [x for r in ex for x in r["reasoning_tokens"]]
    unreasoned_visible = sum(r["action_without_reasoning"] - r["action_bare"] for r in ex)
    return {
        "rollouts": len(rows), "graded": len(graded), "infra": len(rows) - len(graded), "with_export": len(ex),
        "pass_at_1": pct(sum(r["status"] == "pass" for r in graded), len(graded)),
        "pass": sum(r["status"] == "pass" for r in graded),
        "turns_per_rollout_median": statistics.median([r["turns"] for r in ex]) if ex else None,
        "action_turns": A,
        "pct_action_turns_with_reasoning": pct(sum(r["action_with_reasoning"] for r in ex), A),
        "ci95_pct_action_turns_with_reasoning": boot(rows, lambda r: r["action_with_reasoning"], lambda r: r["action_turns"]),
        "rollout_mean_pct_action_reasoned": round(statistics.fmean(
            [100 * r["action_with_reasoning"] / r["action_turns"] for r in ex if r["action_turns"]]), 1) if ex else None,
        "pct_rollouts_all_actions_reasoned": pct(sum(r["all_actions_reasoned"] for r in ex), len(ex)),
        "longest_unreasoned_run_median": statistics.median(runs) if runs else None,
        "longest_unreasoned_run_mean": round(statistics.fmean(runs), 2) if runs else None,
        "longest_unreasoned_run_p90": quantile(runs, 0.9), "longest_unreasoned_run_max": max(runs, default=None),
        "pct_first_action_reasoned": pct(sum(r["first_action_reasoning"] for r in ex), len(ex)),
        "reasoning_tokens_per_reasoning_turn_median": statistics.median(rtok) if rtok else None,
        "reasoning_tokens_per_reasoning_turn_mean": round(statistics.fmean(rtok), 1) if rtok else None,
        "reasoning_tokens_per_turn_incl_zero": round(sum(rtok) / T, 1) if T else None,
        "reasoning_tokens_per_rollout_median": statistics.median([r["reasoning_tokens_total"] for r in ex]) if ex else None,
        "pct_action_turns_with_visible_text": pct(sum(r["action_with_visible"] for r in ex), A),
        "ci95_pct_action_turns_with_visible_text": boot(rows, lambda r: r["action_with_visible"], lambda r: r["action_turns"]),
        "pct_action_turns_visible_without_reasoning": pct(unreasoned_visible, A),
        "pct_action_turns_bare": pct(sum(r["action_bare"] for r in ex), A),
        "visible_chars_median": statistics.median([x for r in ex for x in r["visible_chars"]]) if ex else None,
        "pct_rollouts_git_commit": pct(sum(bool(r["git_commit_calls"]) for r in ex), len(ex)),
        "pct_rollouts_compacted": pct(sum(r["compactions"] > 0 for r in ex), len(ex)),
        "compactions_per_rollout_mean": round(statistics.fmean([r["compactions"] for r in ex]), 2) if ex else None,
        "agent_wall_min_median": round(statistics.median([r["agent_wall_s"] for r in ex if r.get("agent_wall_s")]) / 60, 1)
        if any(r.get("agent_wall_s") for r in ex) else None,
    }


def degenerate(rows):
    """Degenerate hub replies (window.py: [PAD] / chat-template special tokens in the text, or no tool call and no text)
    per reply and per rollout, overall and split by compaction; pass@1 raw and excluding degenerate endings."""
    def block(rs):
        replies = sum((r.get("hub") or {}).get("replies", 0) for r in rs)
        bad = sum((r.get("hub") or {}).get("degenerate_replies", 0) for r in rs)
        ended = [r for r in rs if (r.get("hub") or {}).get("last_reply_degenerate")]
        graded = [r for r in rs if r.get("status") in ("pass", "fail")]
        clean = [r for r in graded if not (r.get("hub") or {}).get("last_reply_degenerate")]
        return {"rollouts": len(rs), "hub_replies": replies, "degenerate_replies": bad,
                "degenerate_per_1000_replies": round(1000 * bad / replies, 2) if replies else None,
                "pct_rollouts_with_degenerate_reply": pct(sum(bool((r.get("hub") or {}).get("degenerate_replies")) for r in rs), len(rs)),
                "rollouts_ended_by_degenerate_reply": len(ended),
                "pct_rollouts_ended_by_degenerate_reply": pct(len(ended), len(rs)),
                "pass_at_1_raw": pct(sum(r["status"] == "pass" for r in graded), len(graded)), "graded": len(graded),
                "pass_at_1_excluding_degenerate_end": pct(sum(r["status"] == "pass" for r in clean), len(clean)),
                "graded_excluding_degenerate_end": len(clean),
                "degenerate_endings_that_failed": sum(r.get("status") == "fail" for r in ended)}
    comp = [r for r in rows if ((r.get("session") or {}).get("compactions") or 0) > 0]
    nocomp = [r for r in rows if (r.get("session") or {}).get("export") and not (r.get("session") or {}).get("compactions")]
    return {"all": block(rows), "compacted": block(comp), "not_compacted": block(nocomp)}


def costs(prefix_paths):
    slots = []
    for path in prefix_paths:
        for item in json.loads(path.read_text()):
            if "slots" in item:
                slots += [{"cost": s["total"]["cost"], "attempts": s["attempts"], "prompt": s["total"]["prompt"],
                           "cached": s["total"]["cached"], "completion": s["total"]["completion"],
                           "calls": s["total"]["calls"], "task_id": s["task_id"]} for s in item["slots"]]
            else:
                slots.append({"cost": item["cost"], "prompt": item["prompt"], "cached": item["cached"],
                              "completion": item["completion"], "calls": item["calls"], "task_id": item["task_id"]})
    if not slots:
        return None
    total = [s["cost"] for s in slots]
    prompt = sum(s["prompt"] for s in slots)
    return {"slots": len(slots), "usd_total": round(sum(total), 2), "usd_per_slot_mean": round(statistics.fmean(total), 3),
            "usd_per_slot_median": round(statistics.median(total), 3),
            "prompt_Mtok_per_slot": round(prompt / len(slots) / 1e6, 2),
            "cached_pct": round(100 * sum(s["cached"] for s in slots) / max(1, prompt), 1),
            "completion_ktok_per_slot": round(sum(s["completion"] for s in slots) / len(slots) / 1e3, 1),
            "calls_per_slot": round(sum(s["calls"] for s in slots) / len(slots), 1),
            "by_task": {t: round(sum(s["cost"] for s in slots if s["task_id"] == t), 3) for t in {s["task_id"] for s in slots}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasksets", nargs="+", default=["sel50"])
    parser.add_argument("--out", default="phase1-report")
    args = parser.parse_args()
    results = [json.loads(p.read_text()) for ts in args.tasksets for p in sorted(HERE.glob(f"results/kimi-{ts}-g*.json"))]
    tasks = sorted({t for r in results for t in r["tasks"]})
    data = {name: [r for r in load(name) if r["task_id"] in set(tasks)] for name in ("sc14", "kimi", "kimi3")}
    report = {"tasksets": args.tasksets, "campaigns": [f"{r['taskset']}-g{r['group']}" for r in results],
              "tasks": len(tasks), "sentence": json.loads((HERE / "prompt/PROMPT.json").read_text())["sentence"]}
    report["reasoning"] = {name: reasoning(rows) for name, rows in data.items() if rows}
    # robustness: kimi3 rollouts without any degenerate hub reply (window.py), so [PAD] runs counted as "reasoning" and
    # sessions cut short by a broken reply do not move the behaviour metrics
    wpath = HERE / "analysis" / "window-kimi3.jsonl"
    if wpath.exists() and data["kimi3"]:
        clean_slots = {(w["task_id"], w["slot"]) for w in map(json.loads, wpath.read_text().splitlines())
                       if not (w.get("hub") or {}).get("degenerate_replies")}
        clean = [r for r in data["kimi3"] if (r["task_id"], r["slot"]) in clean_slots]
        report["reasoning_kimi3_without_degenerate_rollouts"] = reasoning(clean)
        report["paired_kimi3clean_minus_kimi"] = {
            "pct_action_turns_with_reasoning": paired_boot(clean, data["kimi"], lambda r: r["action_with_reasoning"],
                                                           lambda r: r["action_turns"])}
    if data["kimi"] and data["kimi3"]:
        report["paired_kimi3_minus_kimi"] = {
            "pct_action_turns_with_reasoning": paired_boot(data["kimi3"], data["kimi"], lambda r: r["action_with_reasoning"],
                                                           lambda r: r["action_turns"]),
            "pct_action_turns_with_visible_text": paired_boot(data["kimi3"], data["kimi"], lambda r: r["action_with_visible"],
                                                              lambda r: r["action_turns"])}
    window = json.loads((HERE / "analysis/window-summary.json").read_text()) if (HERE / "analysis/window-summary.json").exists() else {}
    report["window"] = window
    report["degenerate"] = {}
    for name in ("kimi3", "kimi"):
        path = HERE / "analysis" / f"window-{name}.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            report["degenerate"][name] = degenerate([r for r in rows if r["task_id"] in set(tasks)])
    report["cost"] = {"kimi3": costs(sorted(p for ts in args.tasksets for p in HERE.glob(f"costs/kimi3-{ts}-g*.json"))),
                      "kimi_previous_r0": costs([HERE / "costs/kimi-r0.json"])}
    legs = [json.loads(line) for line in (HERE / "legs.jsonl").read_text().splitlines() if line.strip()]
    names = set(report["campaigns"])
    report["hub_keys"] = dict(collections.Counter(leg["hub_key"] for leg in legs
                                                  if leg.get("event") == "start" and leg["campaign"] in names))
    report["legs"] = [{k: leg.get(k) for k in ("campaign", "job", "event", "hub_key", "at")} for leg in legs
                      if leg["campaign"] in names]
    report["sandbox_credentials"] = sorted({r.get("sandbox_credential") for r in results})
    # projection: the remaining v2 tasks, scaled by the previous Kimi corpus' mean trajectory length per task
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    length = collections.defaultdict(list)
    for row in rows:
        length[row["task_id"]].append(float(row["token_length"]))
    mean_len = {t: statistics.fmean(v) for t, v in length.items()}
    done = set(tasks)
    rest = sorted(set(mean_len) - done)
    kimi3 = report["cost"]["kimi3"]
    if kimi3:
        per = kimi3["usd_per_slot_mean"]
        size = statistics.fmean([mean_len[t] for t in rest]) / statistics.fmean([mean_len[t] for t in done if t in mean_len])
        hengyuf_rest = [t for t in rest if any(r["task_id"] == t and r["source"] == "hengyuf" for r in rows)]
        baihe_rest = [t for t in rest if t not in set(hengyuf_rest)]
        report["projection"] = {
            "remaining_tasks": len(rest), "hengyuf": len(hengyuf_rest), "baihe": len(baihe_rest),
            "rollouts": 5 * len(rest), "usd_per_rollout_measured": per,
            "size_factor_previous_kimi_token_length": round(size, 3),
            "usd_total_unscaled": round(5 * len(rest) * per), "usd_total_size_scaled": round(5 * len(rest) * per * size),
            "usd_hengyuf_rest47": round(5 * len(hengyuf_rest) * per), "usd_baihe97": round(5 * len(baihe_rest) * per)}
    out = HERE / "analysis" / f"{args.out}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    lines = [f"# {args.out} ({', '.join(args.tasksets)}; {len(tasks)} tasks)", ""]
    keys = [k for k in next(iter(report["reasoning"].values()))]
    lines += ["| metric | " + " | ".join(LABEL[s] for s in report["reasoning"]) + " |",
              "|---|" + "---|" * len(report["reasoning"])]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(report["reasoning"][s].get(k)) for s in report["reasoning"]) + " |")
    lines += ["", "paired kimi3 - kimi: " + json.dumps(report.get("paired_kimi3_minus_kimi")), ""]
    if report.get("reasoning_kimi3_without_degenerate_rollouts"):
        lines += ["kimi3 without degenerate rollouts: " + json.dumps(report["reasoning_kimi3_without_degenerate_rollouts"]),
                  "paired kimi3(clean) - kimi: " + json.dumps(report.get("paired_kimi3clean_minus_kimi")), ""]
    if window:
        wk = sorted({k for v in window.values() for k in v})
        lines += ["| window metric | " + " | ".join(window) + " |", "|---|" + "---|" * len(window)]
        for k in wk:
            lines.append(f"| {k} | " + " | ".join(str(window[s].get(k)) for s in window) + " |")
    for name, blocks in report["degenerate"].items():
        lines += ["", f"degenerate hub replies ({name}):"]
        for part, values in blocks.items():
            lines.append(f"  {part}: " + json.dumps(values))
    lines += ["", "cost: " + json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != "by_task"} for k, v in report["cost"].items()}),
              "hub keys (leg starts): " + json.dumps(report["hub_keys"]),
              "projection: " + json.dumps(report.get("projection"))]
    (HERE / "analysis" / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
