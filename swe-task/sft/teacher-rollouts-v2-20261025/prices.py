#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): per-token hub prices of glm-5.3 and deepseek-v4-pro, fitted from calls whose LiteLLM
cost is known (x-litellm-response-cost header or usage.cost), as cost = a*uncached_prompt + b*cached_prompt + c*completion.

Data: hub_smoke.json / price_calib.json (this directory: header costs of our own tiny calls) and, for glm, the r4 RL
rubric judge's audit (attempt_usage[].usage + cost_usd = the same header, same model). ATP's glm/deepseek rollouts are
non-streaming, and their recorded bodies carry no cost (kimi's streamed bodies do: usage.cost), so rollout costs are
priced from their recorded usage with these fits (costs.py). Read-only; writes prices.json here.

    /usr/bin/python3 -I -B prices.py
"""
import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
AUDIT = ("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/"
         "results/scale-rebench-super-sc-formal-20261024-r4/rubric-judge/audit-*.jsonl")
IDS = {"glm": "nvidia/zai-org/glm-5.3", "deepseek": "nvidia/deepseek-ai/deepseek-v4-pro"}


def row(usage, cost):
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return [prompt - cached, cached, usage.get("completion_tokens") or 0], float(cost)


def solve(points):
    """Least squares (normal equations, 3x3) with a relative weighting (each point weighted by 1/cost)."""
    import math
    a = [[0.0] * 3 for _ in range(3)]
    y = [0.0] * 3
    for x, cost in points:
        w = 1.0 / max(cost, 1e-9) ** 2
        for i in range(3):
            y[i] += w * x[i] * cost
            for j in range(3):
                a[i][j] += w * x[i] * x[j]
    # Gaussian elimination
    m = [a[i] + [y[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-30:
            return None
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [m[r][k] - f * m[col][k] for k in range(4)]
    coef = [m[i][3] / m[i][i] for i in range(3)]
    errors = [abs(sum(c * v for c, v in zip(coef, x)) - cost) / cost for x, cost in points if cost > 0]
    return {"usd_per_M_uncached_prompt": round(coef[0] * 1e6, 4), "usd_per_M_cached_prompt": round(coef[1] * 1e6, 4),
            "usd_per_M_completion": round(coef[2] * 1e6, 4), "points": len(points),
            "max_rel_error": round(max(errors), 5) if errors else None,
            "median_rel_error": round(sorted(errors)[len(errors) // 2], 6) if errors else None,
            "per_token": coef, "nan": any(math.isnan(c) for c in coef)}


def own_calls():
    points = {"glm": [], "deepseek": []}
    for name in ("hub_smoke.json", "price_calib.json"):
        data = json.loads((HERE / name).read_text()) if (HERE / name).exists() else {}
        for alias, entry in (data.get("models") or {}).items():
            for call in (entry.get("calls") or {}).values():
                cost = (call.get("headers") or {}).get("x-litellm-response-cost")
                if call.get("status") == 200 and cost is not None and call.get("usage"):
                    points.setdefault(alias, []).append(row(call["usage"], cost))
    return points


def judge_calls(limit=20000):
    points = []
    for path in sorted(glob.glob(AUDIT)):
        with open(path) as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("model") != IDS["glm"]:
                    continue
                for attempt in record.get("attempt_usage") or []:
                    if attempt.get("usage") and attempt.get("cost_usd") is not None and attempt["cost_usd"] > 0:
                        points.append(row(attempt["usage"], attempt["cost_usd"]))
                if len(points) >= limit:
                    return points
    return points


def main():
    own = own_calls()
    judge = judge_calls()
    fits = {"glm": solve(own["glm"] + judge), "glm_own_calls_only": solve(own["glm"]) if len(own["glm"]) >= 3 else None,
            "glm_judge_only": solve(judge), "deepseek": solve(own["deepseek"]) if len(own["deepseek"]) >= 3 else None}
    (HERE / "prices.json").write_text(json.dumps(fits, indent=1) + "\n")
    for name, fit in fits.items():
        print(name, json.dumps({k: v for k, v in (fit or {}).items() if k != "per_token"}))


if __name__ == "__main__":
    main()
