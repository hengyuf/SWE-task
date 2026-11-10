#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): hub cost projection per model and how far NVAPI_KEY_NEW_2 gets in the wave order.
Read-only on every source; writes analysis/cost-projection.json and analysis/kimi-price-fit.json.

    /usr/bin/python3 -I -B setup/cost_projection.py [--new2-remaining 4789.59] [--workers 8]

Inputs (measured, read-only):
  kimi   kimi-rollouts-v4 (256k, stock prompt, the same harness as here) final report: $4.08 per final slot (re-runs and
         ATP retries included), $3.82 per graded rollout, first attempts mean $3.74; 66 broken-reply re-runs per 970
         slots; analysis/degenerate-v4-final-attempts.jsonl per-attempt LiteLLM cost + tokens (-> a Kimi price fit,
         used by progress.py to price export tokens)
  glm / deepseek  teacher-rollouts-v2 (1M window, the same harness): costs/*.json per-slot journal-priced cost (all
         replies, prices.json) and every final attempt's export.json (OpenCode's per-call token counts of the main
         session) -> per-call prompt sequences
  size   analysis/size-and-pilot.json: DeepSWE / sel50 Kimi prompt-token factor f (overall and per wave)
256k simulation (glm / deepseek; the 1M sequences replayed with OpenCode's compaction at 262,144 - 20,000 = 242,144
prompt tokens): when a call's prompt would pass the point, one compaction call is inserted (prompt = the previous call's
prompt + its output, uncached part = its new tokens; output = SUMMARY tokens) and the context restarts at
BASE + SUMMARY tokens (system + tools + the first user message + the summary), uncached once; every later call keeps
its own new (uncached) tokens and output. g = simulated cost / 1M cost of the same rollouts (main session only).
This ignores behaviour changes after a compaction (KR4: compacted Kimi rollouts did ~40 more turns) and the hub's
cache-hit rate (KR4 Kimi: 96.6% at 1M on 10-17..21 vs 88.6% at 256k on 10-28/29, +29% $ per rollout), so the
projection gives a central case and a high case (x KIMI_256K_OVER_1M).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

TD = Path(__file__).resolve().parents[1]
TR2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
KR4 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028")
COMPACT_AT = 262144 - 20000
SUMMARY = 4000      # tokens of an OpenCode compaction summary (assumption)
BASE = 15000        # system + tools + first user message (KR4 sel50 first prompts ~12-16k)
KIMI = {"per_final_slot": 4.08, "per_graded_rollout": 3.82, "first_attempt_mean": 3.74, "rerun_mean": 4.25,
        "reruns_per_slot": 66 / 970, "retry_cost_share": 46.04 / 3959.12,
        "source": "kimi-rollouts-v4 analysis/FINAL-REPORT.md section 4 (LiteLLM usage.cost, NEW_2)"}
KIMI_256K_OVER_1M = 3.64 / 2.82  # KR4 sel50 $ per rollout at 256k (10-28/29) / earlier 1M Kimi (10-17..21)
SLOTS = {"1": 79 * 5, "2": 34 * 5}
CAP = {"kimi": 8, "glm": 8, "deepseek": 5}            # submit.sh defaults (rollouts in flight per campaign)
PRIOR_HOURS = {"kimi": 1.1, "glm": 0.8, "deepseek": 0.6}  # KR4 / TR2 agent wall medians + grading (progress.py)
WAVE1_HOURS = {m: round(SLOTS["1"] / (3 * CAP[m] / PRIOR_HOURS[m]), 1) for m in CAP}
FINISH_ORDER = sorted(CAP, key=lambda m: WAVE1_HOURS[m])
START_FLOOR, GUARD_FLOOR = 400.0, 150.0


def prices():
    fits = json.loads((TD / "prices.json").read_text())
    return {alias: fits[alias]["per_token"] for alias in ("glm", "deepseek")}


def kimi_price_fit() -> dict:
    """cost = a*uncached + b*cached + c*completion (least squares over KR4's per-attempt LiteLLM costs)."""
    rows = []
    for line in (KR4 / "analysis" / "degenerate-v4-final-attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("cost") and row.get("prompt_tokens"):
            rows.append((row["prompt_tokens"] - row["cached_tokens"], row["cached_tokens"], row["completion_tokens"],
                         row["cost"]))
    # 3x3 normal equations
    m = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    v = [sum(r[i] * r[3] for r in rows) for i in range(3)]

    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d = det(m)
    coef = []
    for k in range(3):
        mk = [row[:] for row in m]
        for i in range(3):
            mk[i][k] = v[i]
        coef.append(det(mk) / d)
    errors = [abs(sum(c * x for c, x in zip(coef, r[:3])) - r[3]) / r[3] for r in rows]
    return {"per_token": coef, "usd_per_M_uncached_prompt": round(coef[0] * 1e6, 4),
            "usd_per_M_cached_prompt": round(coef[1] * 1e6, 4), "usd_per_M_completion": round(coef[2] * 1e6, 4),
            "attempts": len(rows), "median_rel_error": round(statistics.median(errors), 4),
            "p90_rel_error": round(sorted(errors)[int(0.9 * len(errors))], 4),
            "source": str(KR4 / "analysis" / "degenerate-v4-final-attempts.jsonl")}


def calls_of(attempt: str) -> list:
    """[(prompt, cached, output)] per assistant message of the export (main session), in order."""
    try:
        export = json.loads((Path(attempt) / "export.json").read_text())
    except (OSError, ValueError):
        return []
    out = []
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        tokens = info.get("tokens") or {}
        if info.get("role") != "assistant" or not tokens:
            continue
        cache = tokens.get("cache") or {}
        prompt = (tokens.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
        out.append((prompt, cache.get("read") or 0, (tokens.get("output") or 0) + (tokens.get("reasoning") or 0)))
    return out


def price(calls, per_token):
    a, b, c = per_token
    return sum(a * (p - r) + b * r + c * o for p, r, o in calls)


def simulate(calls):
    """The same calls at 256k: [(prompt, cached, output)] with OpenCode compactions inserted; and the count."""
    out = []
    offset = 0
    compactions = 0
    previous = None
    restarted = False
    for prompt, cached, output in calls:
        effective = prompt - offset
        if previous is not None and effective > COMPACT_AT:
            p_prev, o_prev = previous
            out.append((p_prev + o_prev, p_prev, SUMMARY))  # the compaction call: history cached, summary output
            compactions += 1
            offset = prompt - (BASE + SUMMARY)
            effective = BASE + SUMMARY
            restarted = True
        new = prompt - cached
        if restarted:  # first call after a compaction: everything past the system prompt is new
            uncached = max(new, effective - BASE)
            restarted = False
        else:
            uncached = min(new, effective)
        out.append((effective, effective - uncached, output))
        previous = (effective, output)
    return out, compactions


def slot_rows(model: str) -> list:
    rows = []
    for path in sorted((TR2 / "results").glob(f"{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for record in records:
                if record["status"] in ("pass", "fail") and record.get("attempt_dir"):
                    rows.append({"task_id": task_id, "slot": record["slot"], "attempt_dir": record["attempt_dir"],
                                 "campaign": path.stem})
    return rows


def journal_costs(model: str) -> dict:
    """{attempt_dir: journal-priced cost of that (final) attempt} from teacher-rollouts-v2 costs/*.json."""
    out = {}
    for path in sorted((TR2 / "costs").glob(f"{model}-group-*.json")):
        for campaign in json.loads(path.read_text()):
            for slot in campaign["slots"]:
                final = slot.get("final") or {}
                if final.get("attempt"):  # costs/ spells the path via the users/ symlink: key by realpath
                    out[os.path.realpath(final["attempt"])] = {"final_cost": final.get("cost"), "slot_cost": (slot.get("total") or {}).get("cost")}
    return out


def analyse(args):
    model, row, per_token = args
    calls = calls_of(row["attempt_dir"])
    if not calls:
        return None
    simulated, compactions = simulate(calls)
    return {"model": model, "task_id": row["task_id"], "attempt_dir": row["attempt_dir"], "calls": len(calls),
            "prompt_max": max(p for p, _, _ in calls), "over_compact_point": sum(p > COMPACT_AT for p, _, _ in calls),
            "export_cost_1m": price(calls, per_token), "export_cost_256k": price(simulated, per_token),
            "compactions_256k": compactions, "prompt_tokens": sum(p for p, _, _ in calls),
            "prompt_tokens_256k": sum(p for p, _, _ in simulated)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new2-remaining", type=float, default=4789.59)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fit = kimi_price_fit()
    (TD / "analysis" / "kimi-price-fit.json").write_text(json.dumps(fit, indent=1) + "\n")
    size = json.loads((TD / "analysis" / "size-and-pilot.json").read_text())
    factor = {"all": size["size_factor_deepswe_over_sel50"], **size["size_factor_by_wave"]}
    per_token = prices()
    models = {}
    for model in ("glm", "deepseek"):
        rows = slot_rows(model)
        journal = journal_costs(model)
        with ProcessPoolExecutor(args.workers) as pool:
            results = [r for r in pool.map(analyse, [(model, row, per_token[model]) for row in rows]) if r]
        for r in results:
            r.update(journal.get(os.path.realpath(r["attempt_dir"])) or {})
        paired = [r for r in results if r.get("final_cost")]
        g = sum(r["export_cost_256k"] for r in results) / sum(r["export_cost_1m"] for r in results)
        slot_costs = [r["slot_cost"] for r in results if r.get("slot_cost")]
        models[model] = {
            "rollouts": len(results), "slots_with_journal_cost": len(paired),
            "tr2_usd_per_slot_mean": round(statistics.mean(slot_costs), 3),
            "tr2_usd_per_slot_median": round(statistics.median(slot_costs), 3),
            "export_vs_journal_cost_ratio": round(sum(r["export_cost_1m"] for r in paired) /
                                                  sum(r["final_cost"] for r in paired), 4),
            "rollouts_past_242144": sum(r["over_compact_point"] > 0 for r in results),
            "pct_rollouts_past_242144": round(100 * sum(r["over_compact_point"] > 0 for r in results) / len(results), 1),
            "pct_calls_past_242144": round(100 * sum(r["over_compact_point"] for r in results) /
                                           sum(r["calls"] for r in results), 1),
            "prompt_max_median": statistics.median(r["prompt_max"] for r in results),
            "prompt_per_call_mean": round(sum(r["prompt_tokens"] for r in results) / sum(r["calls"] for r in results)),
            "simulated_compactions_mean": round(statistics.mean(r["compactions_256k"] for r in results), 3),
            "g_256k_over_1m": round(g, 3)}
    out = {"inputs": {"kimi": KIMI, "kimi_256k_over_1m": round(KIMI_256K_OVER_1M, 3), "size_factor": factor,
                      "new2_remaining": args.new2_remaining, "start_floor": START_FLOOR, "guard_floor": GUARD_FLOOR,
                      "wave1_hours_prior": WAVE1_HOURS, "wave1_finish_order": FINISH_ORDER, "cap": CAP,
                      "assumptions": {"summary_tokens": SUMMARY, "base_tokens": BASE, "compact_at": COMPACT_AT}},
           "tr2_256k_simulation": models}
    # $ per final slot (re-runs and ATP retries included) under four cases:
    #   stated     main's per-slot numbers, no size scaling: kimi 3.8 (KR4, 256k), glm 4.2 / deepseek 2.0 (TR2, 1M)
    #   central    x f (DeepSWE / sel50 size, per wave); glm / deepseek also x g (256k compaction simulation) and
    #              + ATP retries (KR4 share); kimi = KR4 $4.08 per final slot x f
    #   high       x f; glm / deepseek x KIMI_256K_OVER_1M instead of g (hub cache-hit drop as Kimi's) + retries
    #              + broken-reply re-runs at Kimi v4's rate; kimi = KR4 $4.08 x f
    #   worst      no size scaling (f = 1); glm / deepseek x KIMI_256K_OVER_1M + retries + re-runs; kimi = $4.08
    base = {"kimi": KIMI["per_final_slot"], "glm": models["glm"]["tr2_usd_per_slot_mean"],
            "deepseek": models["deepseek"]["tr2_usd_per_slot_mean"]}
    stated = {"kimi": 3.8, "glm": 4.2, "deepseek": 2.0}
    overhead = {"central": 1 + KIMI["retry_cost_share"],
                "high": (1 + KIMI["retry_cost_share"]) * (1 + KIMI["reruns_per_slot"])}
    unit = {}
    for wave in ("1", "2"):
        f = factor[wave]
        unit[wave] = {}
        for model in ("kimi", "glm", "deepseek"):
            if model == "kimi":
                cases = {"stated": stated[model], "central": base[model] * f, "high": base[model] * f,
                         "worst": base[model]}
            else:
                g = models[model]["g_256k_over_1m"]
                cases = {"stated": stated[model], "central": base[model] * f * g * overhead["central"],
                         "high": base[model] * f * KIMI_256K_OVER_1M * overhead["high"],
                         "worst": base[model] * KIMI_256K_OVER_1M * overhead["high"]}
            unit[wave][model] = cases
    totals = {}
    for case in ("stated", "central", "high", "worst"):
        rows = []
        for wave in ("1", "2"):
            for model in ("kimi", "glm", "deepseek"):
                rows.append({"wave": wave, "model": model, "slots": SLOTS[wave],
                             "usd_per_slot": round(unit[wave][model][case], 3),
                             "usd": round(unit[wave][model][case] * SLOTS[wave], 2)})
        wave1 = sum(r["usd"] for r in rows if r["wave"] == "1")
        wave2 = sum(r["usd"] for r in rows if r["wave"] == "2")
        usable = args.new2_remaining - GUARD_FLOOR
        # Order: the 9 wave-1 campaigns run together; each model's wave 2 follows its own wave 1, in the order the
        # models finish wave 1 (395 slots / (3 x cap) x prior hours per rollout). Legs stop at the guard floor.
        left = usable - wave1
        covered = []
        for model in FINISH_ORDER:
            cost = next(r["usd"] for r in rows if r["wave"] == "2" and r["model"] == model)
            covered.append({"model": model, "wave2_cost": cost,
                            "fraction_covered": round(max(0.0, min(1.0, left / cost)), 3) if cost else 1.0})
            left -= cost
        totals[case] = {"per_wave_model": rows, "wave1_usd": round(wave1, 2), "wave2_usd": round(wave2, 2),
                        "total_usd": round(wave1 + wave2, 2), "new2_usable_to_guard_floor": round(usable, 2),
                        "wave1_fits": wave1 <= usable, "fraction_of_wave1_covered": round(min(1.0, usable / wave1), 3),
                        "wave2_after_wave1": covered, "new2_left_after_all": round(args.new2_remaining - wave1 - wave2, 2),
                        "slots_covered_estimate": round(min(1.0, usable / wave1) * 3 * SLOTS["1"] + sum(
                            c["fraction_covered"] * SLOTS["2"] for c in covered))}
    out["projection"] = totals
    out["unit_cost_per_final_slot"] = {w: {m: {c: round(v, 3) for c, v in cases.items()} for m, cases in d.items()}
                                       for w, d in unit.items()}
    (TD / "analysis" / "cost-projection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"kimi_price_fit": {k: fit[k] for k in ("usd_per_M_uncached_prompt", "usd_per_M_cached_prompt",
                                                              "usd_per_M_completion", "attempts", "median_rel_error",
                                                              "p90_rel_error")},
                      "tr2_256k_simulation": models, "unit": out["unit_cost_per_final_slot"],
                      "totals": {c: {k: v for k, v in t.items() if k != "per_wave_model"} for c, t in totals.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
