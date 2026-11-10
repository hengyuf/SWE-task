#!/usr/bin/env python3
"""Summarize GLM-5.3 local-endpoint E2E campaigns vs the hosted-GLM records of the same tasks (stdlib only).

    python3 e2e/summarize.py e2e/root2            # every results/glmlocal-*.json under the root
    python3 e2e/summarize.py e2e/root2 --live     # also slots still running (reads attempt dirs, no results file yet)

Per task: local outcome(s) vs the hub's 5 records (TR2, 1M context); totals; per-rollout LLM call counts, generated
tokens and wall time from the attempt journals (http.jsonl).
"""
import argparse
import glob
import json
import os
import statistics


def journal_stats(attempt_dir):
    path = os.path.join(attempt_dir or "", "http.jsonl")
    if not os.path.isfile(path):
        return None
    calls = gen = 0
    first = last = None
    errors = 0
    with open(path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            ts = ev.get("timestamp")
            if isinstance(ts, (int, float)):
                first = ts if first is None else min(first, ts)
                last = ts if last is None else max(last, ts)
            if ev.get("event") == "response":
                if ev.get("status") not in (None, 200):
                    errors += 1
                calls += 1
                body = ev.get("body")
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except ValueError:
                        body = None
                if isinstance(body, dict):
                    gen += (body.get("usage") or {}).get("completion_tokens") or 0
    return {"calls": calls, "gen_tokens": gen, "wall_min": round((last - first) / 60, 1) if first and last else None,
            "non200": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, "results", "glmlocal-*.json")))
    tot_local = {"pass": 0, "graded": 0}
    tot_hub = {"pass": 0, "graded": 0}
    rows = []
    for path in files:
        data = json.load(open(path))
        hub = data.get("hub_glm_tr2") or {}
        for task_id, records in sorted(data["records"].items()):
            loc = [r.get("status") for r in records]
            hub_s = [r.get("status") for r in hub.get(task_id, [])]
            st = [journal_stats(r.get("attempt_dir")) for r in records]
            rows.append((os.path.basename(path), task_id, loc, hub_s, st))
            for s in loc:
                if s in ("pass", "fail"):
                    tot_local["graded"] += 1
                    tot_local["pass"] += s == "pass"
            for s in hub_s:
                if s in ("pass", "fail"):
                    tot_hub["graded"] += 1
                    tot_hub["pass"] += s == "pass"
    for name, task_id, loc, hub_s, st in rows:
        extra = "; ".join(f"{s['calls']} calls, {s['gen_tokens']} gen tok, {s['wall_min']} min" for s in st if s)
        print(f"{name:28s} {task_id:48s} local={','.join(map(str, loc)):10s} hub={''.join('P' if s == 'pass' else ('F' if s == 'fail' else '?') for s in hub_s):6s} {extra}")
    if rows:
        def rate(d):
            return f"{d['pass']}/{d['graded']} = {100.0 * d['pass'] / d['graded']:.1f}%" if d["graded"] else "n/a"
        print(f"\nlocal GLM-5.3 (dfw H100): {rate(tot_local)}   hosted GLM-5.3 (same tasks, 5 each): {rate(tot_hub)}")
    if args.live:
        live_graded = {}
        print("\nlive slots:")
        for campaign in sorted(glob.glob(os.path.join(args.root, "probes", "glmlocal", "*"))):
            for slot in sorted(glob.glob(os.path.join(campaign, "models", "glmlocal", "attempts", "*"))):
                attempts = sorted(glob.glob(os.path.join(slot, "attempt-*")))
                if not attempts:
                    continue
                a = attempts[-1]
                s = journal_stats(a) or {}
                done = os.path.isfile(os.path.join(a, "terminal.json"))
                term = ""
                if done:
                    try:
                        t = json.load(open(os.path.join(a, "terminal.json")))
                        term = f"reward={t.get('reward')} f2p={t.get('f2p')} p2p={t.get('p2p')}"
                        if t.get("reward") is not None:
                            live_graded[os.path.basename(campaign)] = live_graded.get(os.path.basename(campaign), []) + [t.get("reward")]
                    except ValueError:
                        term = "?"
                print(f"  {os.path.basename(campaign):12s} {os.path.basename(slot):6s} {os.path.basename(a)} "
                      f"calls={s.get('calls')} gen={s.get('gen_tokens')} wall={s.get('wall_min')}min "
                      f"non200={s.get('non200')} {'DONE ' + term if done else 'running'}")
        for camp, rewards in sorted(live_graded.items()):
            passed = sum(1 for r in rewards if r and r >= 1.0)
            print(f"  {camp}: graded {len(rewards)}, pass {passed} ({100.0 * passed / len(rewards):.1f}%)")


if __name__ == "__main__":
    main()
