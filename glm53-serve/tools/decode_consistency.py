#!/usr/bin/env python3
"""Decode-path vs prefill-path consistency check for a vLLM endpoint (numerical-quality gate for the decode path).

For each selected real agent prompt (tools/replay/data/*.jsonl.gz), generate --gen tokens (temperature 1.0,
top_p 0.95, logprobs=0 -> logprob of each sampled token computed in the DECODE path), then re-score prompt + the
generated tokens with prompt_logprobs (PREFILL path) on the same server. A healthy server gives the same logprobs on
both paths up to small numeric noise; a broken decode path (attention/KV/indexer/DCP/MTP bug) shows a systematic gap.
Also reports the NLL of the sampled tokens and a crude repetition score.

    python3 tools/decode_consistency.py --base-url http://IP:8000/v1 --n 6 --gen 768 --out runs/consistency/X.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "replay"))
from replay_load import load_replay  # noqa: E402

DEFAULT_SET = os.path.join(HERE, "replay", "data", "deepswe_glm_64.jsonl.gz")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base_url, body, timeout):
    req = urllib.request.Request(base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def pick_prompts(sessions, n, seed, min_len, max_len):
    cands = []
    for s in sessions:
        ids_prev = None
        for k, c in enumerate(s["calls"]):
            ids = (ids_prev[:c["shared_prefix_len"]] if ids_prev is not None else []) + list(c["delta"])
            if min_len <= len(ids) <= max_len:
                cands.append((s.get("session_id") or s["_file"], k, ids))
            ids_prev = ids
    rng = random.Random(seed)
    rng.shuffle(cands)
    out, seen = [], set()
    for c in cands:
        if c[0] in seen:
            continue
        seen.add(c[0])
        out.append(c)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--replay", default=DEFAULT_SET)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--gen", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=15000)
    ap.add_argument("--max-len", type=int, default=60000)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out")
    a = ap.parse_args()
    _, sessions = load_replay([a.replay])
    picks = pick_prompts(sessions, a.n, a.seed, a.min_len, a.max_len)
    rows = []
    for sid, k, ids in picks:
        t0 = time.time()
        g = post(a.base_url, {"model": a.model, "prompt": ids, "max_tokens": a.gen, "min_tokens": a.gen,
                              "ignore_eos": True, "temperature": 1.0, "top_p": 0.95, "logprobs": 0,
                              "return_token_ids": True, "seed": a.seed}, a.timeout)
        ch = g["choices"][0]
        out_ids = ch.get("token_ids") or []
        lp = ch.get("logprobs") or {}
        dec_lps = lp.get("token_logprobs") or []
        if not out_ids or len(dec_lps) != len(out_ids):
            print(f"  {sid}#{k}: missing token ids/logprobs (ids {len(out_ids)}, lps {len(dec_lps)})")
            continue
        s = post(a.base_url, {"model": a.model, "prompt": ids + out_ids, "max_tokens": 1, "temperature": 0.0,
                              "prompt_logprobs": 0}, a.timeout)
        plp = s["choices"][0]["prompt_logprobs"]
        pre_lps = []
        for j, tok in enumerate(out_ids):
            e = plp[len(ids) + j]
            v = e.get(str(tok)) if e else None
            pre_lps.append((v.get("logprob") if isinstance(v, dict) else v) if v is not None else None)
        pairs = [(d, p) for d, p in zip(dec_lps, pre_lps) if d is not None and p is not None]
        diffs = [d - p for d, p in pairs]
        text = ch.get("text") or ""
        words = text.split()
        rep = 1.0 - len(set(zip(words, words[1:], words[2:]))) / max(1, len(words) - 2)
        row = {"call": f"{sid}#{k}", "prompt_len": len(ids), "gen": len(out_ids),
               "nll_decode": -statistics.fmean(d for d, _ in pairs), "nll_prefill": -statistics.fmean(p for _, p in pairs),
               "mean_diff": statistics.fmean(diffs), "mean_abs_diff": statistics.fmean(abs(x) for x in diffs),
               "frac_abs_diff_gt_0.5": sum(abs(x) > 0.5 for x in diffs) / len(diffs), "trigram_repeat": rep,
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  {row['call'][:52]:52s} P={len(ids):6d} G={len(out_ids):4d} NLL dec {row['nll_decode']:.3f} "
              f"pre {row['nll_prefill']:.3f} |d| {row['mean_abs_diff']:.3f} >0.5 {100 * row['frac_abs_diff_gt_0.5']:.1f}% "
              f"rep {rep:.2f} ({row['seconds']} s)", flush=True)
    if rows:
        agg = {k: statistics.fmean(r[k] for r in rows) for k in ("nll_decode", "nll_prefill", "mean_diff",
                                                                    "mean_abs_diff", "frac_abs_diff_gt_0.5", "trigram_repeat")}
        print("MEAN " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump({"base_url": a.base_url, "rows": rows, "mean": agg}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
