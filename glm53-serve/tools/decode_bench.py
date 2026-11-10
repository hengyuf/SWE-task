#!/usr/bin/env python3
"""Fixed-shape load test for a vLLM OpenAI endpoint (stdlib only, python>=3.8).

Every request gets a UNIQUE random prompt of ~--ctx tokens (no prefix-cache hits unless --shared-prefix) and is
forced to generate exactly --gen tokens (ignore_eos + min_tokens), streamed, so per-request decode speed is
measured at a known context length. Closed loop: --conc workers each issue --per-worker requests back to back.

  python3 decode_bench.py --url http://IP:8000/v1 --ctx 32000 --gen 1024 --conc 16 --per-worker 2

Reports TTFT, per-request decode tok/s (tokens after the first / time after the first token), aggregate output
and prompt tok/s, and writes per-request rows to --out (JSONL) if given.
"""
import argparse
import asyncio
import json
import random
import statistics
import string
import sys
import time
import urllib.parse

WORDS = None


CORPUS = None


def load_corpus(root, max_bytes=200_000_000):
    """Concatenate .py files under root (natural code text gives realistic MTP acceptance)."""
    import os
    parts, total = [], 0
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if f.endswith(".py"):
                try:
                    t = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                parts.append(f"# file: {f}\n{t}\n")
                total += len(t)
                if total > max_bytes:
                    return "".join(parts)
    return "".join(parts)


def make_prompt(n_tokens, rng, shared_prefix=""):
    if CORPUS:
        n_chars = int(n_tokens * 4.9)  # ~4.9 chars/token for python source with the GLM tokenizer
        start = rng.randrange(0, max(1, len(CORPUS) - n_chars))
        body = CORPUS[start:start + n_chars]
        return shared_prefix + body + "\n\n# Continue writing more code in the same style:\n"
    # fallback: random words (~1.8 GLM tokens per 3-7 letter random word; low MTP acceptance)
    n_words = int(n_tokens / 1.8)
    words = [''.join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 7))) for _ in range(n_words)]
    return shared_prefix + " ".join(words) + "\nContinue the list above with more random words."


async def one_request(url, model, prompt, gen, timeout, extra):
    u = urllib.parse.urlparse(url)
    host, port = u.hostname, u.port or 80
    path = u.path.rstrip("/") + "/completions"
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "min_tokens": gen, "ignore_eos": True,
            "temperature": 1.0, "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    data = json.dumps(body).encode()
    t0 = time.perf_counter()
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    req = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode() + data
    writer.write(req)
    await writer.drain()
    status = await asyncio.wait_for(reader.readline(), timeout)
    code = int(status.split()[1]) if len(status.split()) > 1 else 0
    while True:  # headers
        line = await asyncio.wait_for(reader.readline(), timeout)
        if line in (b"\r\n", b"\n", b""):
            break
    t_first = None
    n_chunks = 0
    usage = None
    err = None
    buf = b""
    while True:
        chunk = await asyncio.wait_for(reader.read(65536), timeout)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if "error" in ev:
                err = str(ev["error"])[:300]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    n_chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    writer.close()
    t_end = time.perf_counter()
    if code != 200 and err is None:
        err = f"HTTP {code}"
    out_tokens = (usage or {}).get("completion_tokens", 0)
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    decode_s = (t_end - t_first) if t_first else None
    dec_tps = ((out_tokens - 1) / decode_s) if (decode_s and out_tokens > 1) else None
    return {"t0": t0, "ttft": ttft, "e2e": t_end - t0, "decode_tps": dec_tps, "out_tokens": out_tokens,
            "prompt_tokens": prompt_tokens, "chunks": n_chunks, "error": err}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--ctx", type=int, default=8000, help="approx prompt tokens (random words)")
    ap.add_argument("--corpus", default="/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/env/vllm-0.32.0/lib/python3.13/site-packages/vllm",
                    help="dir of .py files used as prompt text ('' = random words)")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--per-worker", type=int, default=1)
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between worker starts")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    global CORPUS
    if args.corpus:
        CORPUS = load_corpus(args.corpus)
    shared = make_prompt(args.shared_prefix_tokens, random.Random(12345)) + "\n" if args.shared_prefix_tokens else ""
    rows = []

    async def worker(wid):
        await asyncio.sleep(wid * args.stagger)
        for i in range(args.per_worker):
            prompt = make_prompt(args.ctx, rng, shared)
            try:
                r = await one_request(args.url, args.model, prompt, args.gen, args.timeout, {})
            except Exception as e:  # noqa
                r = {"error": f"{type(e).__name__}: {e}"[:300], "ttft": None, "decode_tps": None, "out_tokens": 0,
                     "prompt_tokens": 0, "e2e": None, "t0": time.perf_counter()}
            r.update({"worker": wid, "i": i})
            rows.append(r)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(args.conc)))
    wall = time.perf_counter() - t0
    ok = [r for r in rows if not r.get("error")]
    out_tok = sum(r["out_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    summ = {
        "label": args.label, "ctx_target": args.ctx, "gen": args.gen, "conc": args.conc, "requests": len(rows),
        "ok": len(ok), "errors": len(rows) - len(ok), "wall_s": round(wall, 1),
        "prompt_tokens_med": statistics.median([r["prompt_tokens"] for r in ok]) if ok else None,
        "agg_out_tps": round(out_tok / wall, 1), "agg_in_tps": round(in_tok / wall, 1),
        "ttft_p50": round(pct([r["ttft"] for r in ok], 50), 2), "ttft_p90": round(pct([r["ttft"] for r in ok], 90), 2),
        "dec_tps_p50": round(pct([r["decode_tps"] for r in ok], 50), 1),
        "dec_tps_p10": round(pct([r["decode_tps"] for r in ok], 10), 1),
        "e2e_p50": round(pct([r["e2e"] for r in ok], 50), 1),
        "first_error": next((r["error"] for r in rows if r.get("error")), None),
    }
    print(json.dumps(summ))
    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps({**r, "label": args.label}) + "\n")
            f.write(json.dumps({"summary": summ}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
