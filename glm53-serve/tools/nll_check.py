#!/usr/bin/env python3
"""Teacher-forced NLL of real hosted-GLM-5.3 generations under a serving config (numerical-quality gate).

For selected calls k of recorded sessions (tools/replay/data/*.jsonl.gz), the recorded prompt of call k contains the
re-rendered assistant turn of call k-1 (what the hosted GLM-5.3 generated: reasoning + content + tool calls).
We send prompt_k as token ids to /v1/completions with prompt_logprobs=0 and average -logprob over exactly those
generated tokens. A serving stack that corrupts attention/KV/routing raises this NLL (vllm #63170 saw +16% for the
PP top-k bug). Compare configs on the SAME selection (same --seed / --n).

    python3 tools/nll_check.py --base-url http://IP:8000/v1 --out runs/nll/<label>.json [--n 12] [--min-gen 300]
    python3 tools/nll_check.py --compare runs/nll/a.json runs/nll/b.json
Stdlib only.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "replay"))
from replay_load import load_replay  # noqa: E402

DEFAULT_SET = os.path.join(HERE, "replay", "data", "deepswe_glm_64.jsonl.gz")


def select(sessions, n, seed, min_len, max_len, min_gen):
    picks = []
    for s in sessions:
        ids_prev = None
        for k, c in enumerate(s["calls"]):
            ids = (ids_prev[:c["shared_prefix_len"]] if ids_prev is not None else []) + list(c["delta"])
            if (ids_prev is not None and c["shared_prefix_len"] == len(ids_prev) and c.get("splice_len")
                    and c["splice_len"] >= min_gen and min_len <= len(ids) <= max_len):
                picks.append((s.get("session_id") or s.get("id") or s["_file"], k, len(ids_prev), c["splice_len"], ids))
            ids_prev = ids
    rng = random.Random(seed)
    rng.shuffle(picks)
    out, seen = [], set()
    for p in picks:  # at most one call per session
        if p[0] in seen:
            continue
        seen.add(p[0])
        out.append(p)
        if len(out) >= n:
            break
    return out


def query(base_url, model, ids, timeout):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
    req = urllib.request.Request(base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--replay", default=DEFAULT_SET)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=12000)
    ap.add_argument("--max-len", type=int, default=70000)
    ap.add_argument("--min-gen", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        runs = [json.load(open(p)) for p in a.compare]
        keys = set.intersection(*[set(r["per_call"]) for r in runs])
        print(f"{len(keys)} common calls")
        for p, r in zip(a.compare, runs):
            tot = sum(r["per_call"][k]["nll_sum"] for k in keys)
            cnt = sum(r["per_call"][k]["n"] for k in keys)
            print(f"{os.path.basename(p):40s} mean NLL {tot / cnt:.4f} nats/token over {cnt} generated tokens "
                  f"(ppl {math.exp(tot / cnt):.3f})")
        return
    _, sessions = load_replay([a.replay])
    picks = select(sessions, a.n, a.seed, a.min_len, a.max_len, a.min_gen)
    print(f"{len(picks)} calls selected (prompt {a.min_len}-{a.max_len} tokens, >= {a.min_gen} generated tokens)")
    per_call = {}
    for sid, k, start, glen, ids in picks:
        t0 = time.time()
        res = query(a.base_url, a.model, ids, a.timeout)
        plp = res["choices"][0].get("prompt_logprobs")
        nll, n, worst = 0.0, 0, []
        for pos in range(start, start + glen):
            entry = plp[pos]
            tok = str(ids[pos])
            if not entry or tok not in entry:
                continue
            lp = entry[tok]["logprob"] if isinstance(entry[tok], dict) else entry[tok]
            nll -= lp
            n += 1
        key = f"{sid}#{k}"
        per_call[key] = {"nll_sum": nll, "n": n, "mean": nll / max(1, n), "prompt_len": len(ids), "gen_len": glen,
                         "seconds": round(time.time() - t0, 1)}
        print(f"  {key[:60]:60s} prompt {len(ids):6d} gen {glen:5d} mean NLL {nll / max(1, n):.4f} ({time.time() - t0:.0f} s)",
              flush=True)
    tot = sum(v["nll_sum"] for v in per_call.values())
    cnt = sum(v["n"] for v in per_call.values())
    summary = {"base_url": a.base_url, "replay": a.replay, "seed": a.seed, "n_calls": len(per_call), "tokens": cnt,
               "mean_nll": tot / max(1, cnt), "per_call": per_call}
    print(f"mean NLL {summary['mean_nll']:.4f} nats/token over {cnt} generated tokens")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
