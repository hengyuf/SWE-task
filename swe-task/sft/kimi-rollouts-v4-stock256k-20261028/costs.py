#!/usr/bin/env python3
"""kimi-rollouts-v4 copy of kimi-rollouts-v3's costs.py (teacher-rollouts-v2's, legs_of adapted to this directory; v4: kr4-legs.jsonl): hub usage and cost of probe campaigns, from ATP's per-attempt HTTP journals
(http.jsonl "response" events = every upstream reply, main session and subagents, retries included). Read-only.

    /usr/bin/python3 -I -B costs.py PROBE_DIR [PROBE_DIR ...] [--out costs/<name>.json]

Cost per call: usage.cost when the body carries it (kimi's streamed replies), else the call's usage priced with
prices.json (LiteLLM's own per-token prices, fitted exactly from x-litellm-response-cost headers: prices.py), i.e.
uncached prompt x a + cached prompt x b + completion x c. Per attempt and per slot (all attempts of the slot = what
the slot cost; the final attempt = the rollout analysed), plus the time window of the calls (for the key-spend check).
By key (2026-10-26): every reply is attributed to the leg of its campaign that was running at the reply's timestamp
(legs.jsonl start records: the latest leg start <= the timestamp; the legs of one campaign never overlap, and a leg's
probe worker binds one hub key for all its calls), hence to that leg's key NAME: "by_key" / "by_job" per attempt, slot
and campaign (calls, ok, prompt, cached, completion, cost). ATP's journal keeps no key-identifying header (its
credential_id is the slot label "existing" for every key), so the leg timing is the attribution.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import bisect  # noqa: E402
from datetime import datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "legs.jsonl"
ALIASES = {"nvidia/zai-org/glm-5.3": "glm", "nvidia/deepseek-ai/deepseek-v4-pro": "deepseek",
           "nvidia/moonshotai/kimi-k3": "kimi"}
SPLIT_FIELDS = ("calls", "ok", "prompt", "cached", "completion", "cost")


def prices():
    fits = json.loads((HERE / "prices.json").read_text())
    return {alias: fits[alias]["per_token"] for alias in ("glm", "deepseek") if fits.get(alias)}


def legs_of(model: str, campaign_name: str) -> list:
    """[(start epoch, job, key NAME)] of one campaign's legs, oldest first. kimi-rollouts-v4: the campaign's own
    probes/<model>/<name>/kr4-legs.jsonl (one line per ATP worker start, written by this directory's probe_worker.py
    wrapper, recovery campaigns included), else legs.jsonl start records (this directory's driver schema)."""
    out = []
    own = HERE / "probes" / model / campaign_name / "kr4-legs.jsonl"
    try:
        lines = own.read_text().splitlines() if own.exists() else LEDGER.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if own.exists() or (entry.get("event") == "start" and entry.get("campaign") == campaign_name):
            out.append((datetime.strptime(entry["at"], "%Y-%m-%dT%H:%M:%S%z").timestamp(), str(entry.get("job")),
                        entry.get("hub_key") or entry.get("key")))
    return sorted(out)


def leg_at(legs: list, stamp) -> tuple:
    """(job, key NAME) of the leg running at epoch `stamp`; ("?", "unknown") without legs or a timestamp."""
    if not legs or not isinstance(stamp, (int, float)):
        return "?", "unknown"
    index = bisect.bisect_right([start for start, _, _ in legs], stamp) - 1
    return legs[max(index, 0)][1:]  # a reply stamped before the first leg start (clock skew) -> the first leg


def add_split(split: dict, name: str, **values) -> None:
    entry = split.setdefault(name, dict.fromkeys(SPLIT_FIELDS, 0))
    for key, value in values.items():
        entry[key] += value


def merge_split(target: dict, source: dict) -> None:
    for name, values in (source or {}).items():
        add_split(target, name, **values)


def attempt_usage(attempt: Path, price: dict, alias: str, legs: list = ()) -> dict:
    out = {"attempt": str(attempt), "calls": 0, "ok": 0, "errors": {}, "prompt": 0, "cached": 0, "completion": 0,
           "reasoning_reported": 0, "cost_reported": 0.0, "cost_priced": 0.0, "cost": 0.0, "first": None, "last": None,
           "max_prompt": 0, "by_key": {}, "by_job": {}}
    path = attempt / "http.jsonl"
    if not path.exists():
        return out
    with open(path, errors="replace") as stream:
        for line in stream:
            head = line[:1500]
            if '"event":"response"' not in head:
                if '"event":"request"' in head:
                    try:
                        stamp = json.loads(line).get("timestamp")
                    except ValueError:
                        stamp = None
                    if stamp:
                        out["first"] = stamp if out["first"] is None else min(out["first"], stamp)
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            out["calls"] += 1
            stamp = event.get("timestamp")
            if stamp:
                out["last"] = stamp if out["last"] is None else max(out["last"], stamp)
            job, key = leg_at(legs, stamp)
            status = event.get("status")
            if status != 200:
                out["errors"][str(status)] = out["errors"].get(str(status), 0) + 1
                add_split(out["by_key"], key, calls=1)
                add_split(out["by_job"], job, calls=1)
                continue
            try:
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                out["errors"]["bad_body"] = out["errors"].get("bad_body", 0) + 1
                add_split(out["by_key"], key, calls=1)
                add_split(out["by_job"], job, calls=1)
                continue
            usage = body.get("usage") or {}
            prompt = usage.get("prompt_tokens") or 0
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            out["ok"] += 1
            out["prompt"] += prompt
            out["cached"] += cached
            out["completion"] += completion
            out["max_prompt"] = max(out["max_prompt"], prompt)
            out["reasoning_reported"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            model = ALIASES.get(body.get("model"), alias)
            coef = price.get(model)
            priced = (coef[0] * (prompt - cached) + coef[1] * cached + coef[2] * completion) if coef else 0.0
            out["cost_priced"] += priced
            reported = usage.get("cost")
            if isinstance(reported, (int, float)):
                out["cost_reported"] += reported
                out["cost"] += reported
            else:
                out["cost"] += priced
            cost = reported if isinstance(reported, (int, float)) else priced
            for split, name in ((out["by_key"], key), (out["by_job"], job)):
                add_split(split, name, calls=1, ok=1, prompt=prompt, cached=cached, completion=completion, cost=cost)
    return out


def campaign(probe: Path, price: dict, workers: int = 1) -> dict:
    probe = Path(probe)
    manifest = json.loads((probe / "manifest.json").read_text())
    tasks = {t["_ng_task_index"]: t["task_id"] for t in manifest["tasks"]}
    slots = []
    by_key, by_job = {}, {}
    for alias in manifest["settings"]["models"]:
        legs = legs_of(alias, probe.name)  # probes/<model>/<campaign name>
        slot_dirs = sorted((probe / "models" / alias / "attempts").glob("*-*"))
        per_slot = [sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
                    for slot_dir in slot_dirs]
        flat = [attempt for attempts in per_slot for attempt in attempts]
        if workers > 1 and len(flat) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(min(workers, len(flat))) as pool:
                flat_usages = list(pool.map(attempt_usage, flat, [price] * len(flat), [alias] * len(flat),
                                            [legs] * len(flat)))
        else:
            flat_usages = [attempt_usage(attempt, price, alias, legs) for attempt in flat]
        usage_iter = iter(flat_usages)
        for slot_dir, attempts in zip(slot_dirs, per_slot):
            index, repeat = (int(x) for x in slot_dir.name.split("-"))
            usages = [next(usage_iter) for _ in attempts]
            total = {key: sum(u[key] for u in usages) for key in ("calls", "ok", "prompt", "cached", "completion",
                                                                  "reasoning_reported", "cost", "cost_priced", "cost_reported")}
            total["by_key"], total["by_job"] = {}, {}
            for usage in usages:
                merge_split(total["by_key"], usage["by_key"])
                merge_split(total["by_job"], usage["by_job"])
            merge_split(by_key, total["by_key"])
            merge_split(by_job, total["by_job"])
            firsts = [u["first"] for u in usages if u["first"]]
            lasts = [u["last"] for u in usages if u["last"]]
            slots.append({"model": alias, "task_id": tasks.get(index), "task_index": index, "repeat": repeat,
                          "attempts": len(attempts), "final": usages[-1] if usages else None, "total": total,
                          "first": min(firsts) if firsts else None, "last": max(lasts) if lasts else None})
    costs = [s["total"]["cost"] for s in slots]
    return {"probe": str(probe), "slots": slots, "n_slots": len(slots), "cost_total": sum(costs),
            "cost_per_slot_mean": statistics.fmean(costs) if costs else None,
            "first": min((s["first"] for s in slots if s["first"]), default=None),
            "last": max((s["last"] for s in slots if s["last"]), default=None),
            "by_key": by_key, "by_job": by_job}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", nargs="+")
    parser.add_argument("--out")
    parser.add_argument("--workers", type=int, default=1, help="attempt journals parsed in parallel")
    parser.add_argument("--quiet", action="store_true", help="campaign lines only (no per-slot lines)")
    args = parser.parse_args()
    price = prices()
    results = [campaign(Path(p), price, args.workers) for p in args.probes]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n")
    for result in results:
        print(result["probe"], f"slots {result['n_slots']} total ${result['cost_total']:.2f} by key " +
              json.dumps({key: round(values["cost"], 2) for key, values in sorted(result["by_key"].items())}))
        if args.quiet:
            continue
        for s in result["slots"]:
            t = s["total"]
            print(f"  {s['model']} {s['task_id']} r{s['repeat']} attempts {s['attempts']} calls {t['calls']} ok {t['ok']} "
                  f"prompt {t['prompt']/1e6:.2f}M cached {t['cached']/1e6:.2f}M ({100*t['cached']/max(1,t['prompt']):.0f}%) "
                  f"completion {t['completion']/1e3:.1f}k reasoning(reported) {t['reasoning_reported']/1e3:.1f}k "
                  f"cost ${t['cost']:.3f}" + (f" (reported ${t['cost_reported']:.3f})" if t['cost_reported'] else ""))


if __name__ == "__main__":
    main()
