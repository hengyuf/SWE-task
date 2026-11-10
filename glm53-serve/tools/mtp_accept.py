#!/usr/bin/env python3
"""MTP draft acceptance of a vLLM endpoint under one bench_endpoint scenario (stdlib only).

    python3 tools/mtp_accept.py --url http://IP:8000/v1 --scenario tools [-c 8] [-n 32] [--max-tokens 1024]

Reads the server's spec-decode counters (/metrics) before and after running hsg-ref/serve_scripts/bench_endpoint.py
and prints drafts, draft tokens, accepted tokens, the acceptance rate and the mean acceptance length
(1 + accepted / drafts), plus bench_endpoint's correctness lines.
"""
import argparse
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "hsg-ref", "serve_scripts", "bench_endpoint.py")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LINE = re.compile(r"^(vllm:spec_decode_[a-z_]+?)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def counters(url):
    base = url.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    with OPENER.open(base + "/metrics", timeout=30) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if match and "per_pos" not in match.group(1):
            out[match.group(1)] = out.get(match.group(1), 0.0) + float(match.group(2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--scenario", default="tools")
    parser.add_argument("-c", type=int, default=8)
    parser.add_argument("-n", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--label", default="")
    a = parser.parse_args()
    before = counters(a.url)
    run = subprocess.run([sys.executable, BENCH, "--url", a.url, "--model", "glm-5.3", "--scenario", a.scenario,
                          "-c", str(a.c), "-n", str(a.n), "--max-tokens", str(a.max_tokens)],
                         capture_output=True, text=True)
    after = counters(a.url)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    keep = [l for l in run.stdout.splitlines() if any(k in l for k in ("ok / failed", "CORRECT", "finish_reasons",
                                                                     "output tok/s"))]
    print(f"[{a.label}] scenario {a.scenario} c{a.c} n{a.n}: drafts {drafts:.0f}, draft tokens {draft_tokens:.0f}, "
          f"accepted {accepted:.0f}, acceptance rate {accepted / draft_tokens if draft_tokens else float('nan'):.3f}, "
          f"mean acceptance length {1 + accepted / drafts if drafts else float('nan'):.2f}")
    for line in keep:
        print("   ", line.strip())
    if run.returncode:
        print(run.stderr[-2000:])


if __name__ == "__main__":
    main()
