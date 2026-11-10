#!/usr/bin/env python3
"""Decode-path vs prefill-path parity for a vLLM endpoint (stdlib only).

For each prompt: greedy-generate --gen tokens (decode path, logprobs=1, return_token_ids), then re-score
prompt + output in one prefill (prompt_logprobs=1). A faithful decode path picks the prefill argmax at (almost) every
step and reports the same logprob. Prompts: two synthetic code prompts (~1.2k and ~6k tokens; < 2048 tokens means the
sparse-attention top-k keeps every token) and --n-real real agent prompts from the replay set (15-60k tokens).

    python3 tools/greedy_parity.py --base-url http://IP:8000/v1 --label X [--out runs/parity/X.json]
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "replay"))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CODE = "/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/env/vllm-0.33.0/lib/python3.13/site-packages/vllm/v1/worker/gpu/model_runner.py"


def post(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def one(url, model, prompt, gen, temperature):
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "min_tokens": gen, "ignore_eos": True,
            "temperature": temperature, "logprobs": 1, "return_token_ids": True, "seed": 7}
    if temperature > 0:
        body["top_p"] = 0.95
    g = post(url, body)
    ch = g["choices"][0]
    pids = ch.get("prompt_token_ids") or (prompt if isinstance(prompt, list) else None)
    oids = ch["token_ids"]
    dl = ch["logprobs"]["token_logprobs"]
    s = post(url, {"model": model, "prompt": pids + oids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1})
    plp = s["choices"][0]["prompt_logprobs"]
    agree, diffs, first_div = 0, [], None
    for j, tok in enumerate(oids):
        e = plp[len(pids) + j] or {}
        best = max(e.items(), key=lambda kv: kv[1]["logprob"])[0] if e else None
        if best == str(tok):
            agree += 1
        elif first_div is None:
            first_div = j
        v = e.get(str(tok))
        if v is not None and dl[j] is not None:
            diffs.append(dl[j] - v["logprob"])
    return {"prompt_len": len(pids), "gen": len(oids), "argmax_agree": agree / len(oids),
            "first_divergence": first_div, "mean_diff": statistics.fmean(diffs),
            "mean_abs_diff": statistics.fmean(abs(x) for x in diffs),
            "p90_abs_diff": sorted(abs(x) for x in diffs)[int(0.9 * (len(diffs) - 1))],
            "text_head": (ch.get("text") or "")[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--label", default="")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--concurrent", type=int, default=1,
                    help="run this many prompts' greedy generations at once (ragged MTP verify batches)")
    ap.add_argument("--temps", default="0.0,1.0")
    ap.add_argument("--out")
    a = ap.parse_args()
    src = open(CODE).read()
    prompts = [("code-1.2k", src[:5400] + "\n\n# Explain the code above in detail:\n"),
               ("code-6k", src[:27000] + "\n\n# Explain the code above in detail:\n")]
    if a.n_real:
        from decode_consistency import pick_prompts  # noqa: E402
        from replay_load import load_replay  # noqa: E402
        _, sessions = load_replay([os.path.join(HERE, "replay", "data", "deepswe_glm_64.jsonl.gz")])
        for sid, k, ids in pick_prompts(sessions, a.n_real, 0, 15000, 60000):
            prompts.append((f"real-{len(ids) // 1000}k", ids))
    rows = []
    temps = [float(t) for t in a.temps.split(",")]
    jobs = [(name, prompt, temp) for name, prompt in prompts for temp in temps]

    def run(job):
        name, prompt, temp = job
        t0 = time.time()
        r = one(a.base_url, a.model, prompt, a.gen, temp)
        r.update({"prompt": name, "temperature": temp, "seconds": round(time.time() - t0, 1)})
        return r

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, a.concurrent)) as ex:
        for r in ex.map(run, jobs):
            rows.append(r)
            print(f"[{a.label}] {r['prompt']:10s} T={r['temperature']} P={r['prompt_len']:6d} agree {100 * r['argmax_agree']:5.1f}% "
                  f"first-div {r['first_divergence']} mean d {r['mean_diff']:+.4f} |d| {r['mean_abs_diff']:.4f} "
                  f"p90|d| {r['p90_abs_diff']:.3f}", flush=True)
    for t in temps:
        sel = [r for r in rows if r["temperature"] == t]
        print(f"[{a.label}] T={t}: agree {100 * statistics.fmean(r['argmax_agree'] for r in sel):.2f}% "
              f"|d| {statistics.fmean(r['mean_abs_diff'] for r in sel):.4f} "
              f"mean d {statistics.fmean(r['mean_diff'] for r in sel):+.4f} (n={len(sel)}, concurrent={a.concurrent})")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"label": a.label, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
