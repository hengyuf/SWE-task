#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28) copy of kimi-rollouts-v4's ratelimits.py (= v3's), code unchanged (reads this directory's probes/kimi/*). v3 docstring follows:
kimi-rollouts-v3 (2026-10-26): our hub load and the hub's rate-limit headroom for Kimi-K3 (read-only).

    /usr/bin/python3 -I -B ratelimits.py [--campaigns pilot sel50-g0 ...] [--out analysis/ratelimits-<tag>.json]

Every "response" event of every attempt's http.jsonl in the campaigns: timestamp, status, usage (prompt incl. cached +
completion tokens) and the x-ratelimit-* reply headers (limit / remaining for requests and tokens; Kimi-K3 replies carry
no model_saturation_check-* / priority_model-* headers). Per minute: our calls and tokens (all campaigns together), our
rollouts with a call in that minute, and the lowest remaining-tokens / remaining-requests the hub reported. Summary:
our tokens/min (median, p90, max over active minutes), tokens/min per active rollout, the hub limits, the minimum
remaining seen (as a fraction of the limit in the same reply: the hub answers from several limit tiers), and 429 counts.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def events(path):
    with open(path, "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                yield json.loads(raw)
            except ValueError:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    probes = [HERE / "probes" / "kimi" / name for name in args.campaigns] if args.campaigns else \
        sorted((HERE / "probes" / "kimi").iterdir())
    minutes = collections.defaultdict(lambda: {"calls": 0, "tokens": 0, "rollouts": set(), "min_rem_tokens": None,
                                               "min_rem_requests": None, "s429": 0})
    limits = collections.Counter()
    statuses = collections.Counter()
    for probe in probes:
        for http in probe.glob("models/*/attempts/*/attempt-*/http.jsonl"):
            rollout = str(http.parent)
            for event in events(http):
                minute = int(event.get("timestamp", 0) // 60)
                slot = minutes[minute]
                status = event.get("status")
                statuses[str(status)] += 1
                headers = event.get("headers") or {}
                for name in ("x-ratelimit-limit-tokens", "x-ratelimit-limit-requests"):
                    if name in headers:
                        limits[f"{name}={headers[name]}"] += 1
                for name, key in (("x-ratelimit-remaining-tokens", "min_rem_tokens"),
                                  ("x-ratelimit-remaining-requests", "min_rem_requests")):
                    try:  # remaining as a fraction of the limit in the same reply (the hub has several limit tiers)
                        value = float(headers[name]) / float(headers[name.replace("remaining", "limit")])
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        continue
                    slot[key] = value if slot[key] is None else min(slot[key], value)
                if status == 429:
                    slot["s429"] += 1
                if status != 200:
                    continue
                try:
                    usage = json.loads(event.get("body") or "{}").get("usage") or {}
                except ValueError:
                    continue
                slot["calls"] += 1
                slot["tokens"] += (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                slot["rollouts"].add(rollout)
    active = [m for m in minutes.values() if m["calls"]]
    tokens = [m["tokens"] for m in active]
    per_rollout = [m["tokens"] / len(m["rollouts"]) for m in active if m["rollouts"]]
    concurrent = [len(m["rollouts"]) for m in active]
    rem_tokens = [m["min_rem_tokens"] for m in minutes.values() if m["min_rem_tokens"] is not None]
    rem_requests = [m["min_rem_requests"] for m in minutes.values() if m["min_rem_requests"] is not None]
    min_limit = min((float(k.split("=")[1]) for k in limits if k.startswith("x-ratelimit-limit-tokens")), default=None)
    summary = {
        "campaigns": [p.name for p in probes], "replies": sum(statuses.values()), "statuses": dict(statuses),
        "limits_seen": dict(limits), "active_minutes": len(active),
        "our_tokens_per_min": {"median": quantile(tokens, 0.5), "p90": quantile(tokens, 0.9), "max": max(tokens, default=None)},
        "our_calls_per_min": {"median": quantile([m["calls"] for m in active], 0.5),
                              "max": max((m["calls"] for m in active), default=None)},
        "rollouts_calling_per_min": {"median": quantile(concurrent, 0.5), "max": max(concurrent, default=None)},
        "tokens_per_min_per_calling_rollout": {"median": round(statistics.median(per_rollout)) if per_rollout else None,
                                               "mean": round(statistics.fmean(per_rollout)) if per_rollout else None},
        "hub_min_remaining_tokens_fraction": round(min(rem_tokens), 4) if rem_tokens else None,
        "hub_remaining_tokens_fraction_p10": round(quantile(rem_tokens, 0.1), 4) if rem_tokens else None,
        "hub_min_remaining_requests_fraction": round(min(rem_requests), 4) if rem_requests else None,
        "our_tokens_per_min_p90_vs_lower_limit": round(quantile(tokens, 0.9) / min_limit, 3) if tokens and min_limit else None,
        "minutes_with_429": sum(1 for m in minutes.values() if m["s429"]),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
