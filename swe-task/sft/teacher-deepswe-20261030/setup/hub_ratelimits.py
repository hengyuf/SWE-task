#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): hub rate-limit headroom per model at this campaign's concurrency (read-only;
writes analysis/hub-ratelimits.json).

    /usr/bin/python3 -I -B setup/hub_ratelimits.py [--workers 8]

1. The hub's x-ratelimit-* reply headers in a sample of journals (every 6th attempt, <= 60 per model): kimi from
   kimi-rollouts-v4 (256k, ~112 Kimi rollouts in flight at peak), glm / deepseek from teacher-rollouts-v2 (1M, 48 GLM
   and 16 DeepSeek in flight): the limits (per model deployment: "model_saturation_check" / "priority_model" are
   hub-wide, shared with every other user of that model) and the lowest remaining values seen, plus reply statuses.
2. Tokens per minute of one running rollout (main session: OpenCode export token sums / agent wall time; TR2
   features/*.jsonl for glm / deepseek / kimi at 1M) x the in-flight counts of this campaign (3 wave-1 campaigns x 8 = 24
   per model) vs the tightest token limit.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

TD = Path(__file__).resolve().parents[1]
TR2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
KR4 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028")


def scan(path: str) -> tuple:
    statuses = collections.Counter()
    limits, lowest = {}, {}
    with open(path, "rb") as stream:
        for raw in stream:
            if b'"event":"response"' not in raw[:3000] and b'"event": "response"' not in raw[:3000]:
                continue
            event = json.loads(raw)
            statuses[str(event.get("status"))] += 1
            for key, value in (event.get("headers") or {}).items():
                if not key.startswith("x-ratelimit-"):
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if "-limit-" in key:
                    limits[key] = value
                if "-remaining-" in key:
                    lowest[key] = min(lowest.get(key, value), value)
    return dict(statuses), limits, lowest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    roots = {"kimi": KR4 / "probes/kimi", "glm": TR2 / "probes/glm", "deepseek": TR2 / "probes/deepseek"}
    report = {"headers": {}, "throughput": {}}
    for model, root in roots.items():
        paths = [str(p) for p in sorted(root.glob("*/models/*/attempts/*/attempt-*/http.jsonl"))[::6][:60]]
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(scan, paths))
        statuses = collections.Counter()
        limits, lowest = {}, {}
        for found, limit, low in results:
            statuses.update(found)
            limits.update(limit)
            for key, value in low.items():
                lowest[key] = min(lowest.get(key, value), value)
        report["headers"][model] = {"journals": len(paths), "statuses": dict(statuses), "limits": limits,
                                    "lowest_remaining": lowest,
                                    "source": "kimi-rollouts-v4" if model == "kimi" else "teacher-rollouts-v2"}
    for model in ("kimi", "glm", "deepseek"):
        rows = [json.loads(line) for line in (TR2 / "features" / f"{model}.jsonl").read_text().splitlines() if line.strip()]
        rates = []
        for row in rows:
            tokens = row.get("hub_tokens") or {}
            wall = row.get("agent_wall_s")
            total = (tokens.get("input") or 0) + (tokens.get("cache_read") or 0) + (tokens.get("output") or 0)
            if wall and total:
                rates.append(total / (wall / 60))
        token_limits = [v for k, v in report["headers"][model]["limits"].items() if k.endswith("limit-tokens")]
        tightest = min(token_limits) if token_limits else None
        per_rollout = statistics.median(rates)
        report["throughput"][model] = {
            "rollouts": len(rates), "tokens_per_min_per_running_rollout_median": round(per_rollout),
            "tokens_per_min_per_running_rollout_mean": round(statistics.mean(rates)),
            "tightest_token_limit_per_min": tightest,
            "at_24_in_flight_tokens_per_min": round(24 * per_rollout),
            "share_of_tightest_at_24": round(24 * per_rollout / tightest, 2) if tightest else None,
            "in_flight_for_75pct_of_tightest": int(0.75 * tightest / per_rollout) if tightest else None,
            "note": "1M-context rollouts of the 50 sel50 tasks (TR2 features: export tokens, agent wall)"}
    (TD / "analysis" / "hub-ratelimits.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
