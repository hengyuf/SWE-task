#!/usr/bin/env python3
"""Compare the shape of GLM-5.3 replies between campaigns (degeneration check), stdlib only.

    python3 e2e/reasoning_stats.py LABEL=GLOB [LABEL=GLOB ...]
    e.g. hub='/.../teacher-rollouts-v2-20261025/probes/glm/group-*/models/glm/attempts/*/attempt-*/http.jsonl'
         local='/.../glm53-serve/e2e/root2/probes/glmlocal/*/models/glmlocal/attempts/*/attempt-*/http.jsonl'

Per label: replies, completion-token percentiles and tail shares, and filler/loop markers in the reasoning text
("Hmm" per 1k words, share of replies whose reasoning has < 50% unique lines among >= 200 lines).
"""
import glob
import json
import sys


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100.0 * (len(xs) - 1)))] if xs else None


def scan(pattern, max_files=400):
    comp, hmm_rates, loopy, n = [], [], 0, 0
    for path in sorted(glob.glob(pattern))[:max_files]:
        with open(path) as f:
            for line in f:
                if '"event":"response"' not in line and '"event": "response"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                body = ev.get("body")
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except ValueError:
                        continue
                if not isinstance(body, dict):
                    continue
                usage = body.get("usage") or {}
                ch = (body.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                text = msg.get("reasoning") or msg.get("reasoning_content") or ""
                n += 1
                comp.append(usage.get("completion_tokens") or 0)
                words = text.split()
                if len(words) >= 200:
                    hmm_rates.append(1000.0 * sum(w.strip(".,—-!?").lower() == "hmm" for w in words) / len(words))
                lines = [x for x in text.split("\n") if x.strip()]
                if len(lines) >= 200 and len(set(lines)) / len(lines) < 0.5:
                    loopy += 1
    return n, comp, hmm_rates, loopy


def main():
    for arg in sys.argv[1:]:
        label, pattern = arg.split("=", 1)
        n, comp, hmm, loopy = scan(pattern)
        if not n:
            print(f"{label}: no replies"); continue
        share = lambda t: 100.0 * sum(c > t for c in comp) / n  # noqa: E731
        print(f"{label}: replies {n}; completion p50 {pct(comp, 50)} p90 {pct(comp, 90)} p99 {pct(comp, 99)} max {max(comp)}; "
              f">4k {share(4096):.1f}% >16k {share(16384):.2f}% >32k {share(32768):.2f}%; "
              f"'hmm' per 1k words (replies >=200 words) p50 {pct(hmm, 50) if hmm else 0:.1f} p90 {pct(hmm, 90) if hmm else 0:.1f}; "
              f"loopy replies {loopy}")


if __name__ == "__main__":
    main()
