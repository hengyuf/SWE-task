#!/usr/bin/env python3
"""reconcile (2026-11-04): character-level check of 'reasoning got longer and drifts from base' (the report's token
counts come from the teacher-rollout extractor's tokenizer, which /usr/bin/python3 cannot load). From each slot's final
export.json: reasoning characters of every non-summary assistant message that has >= 1 tool part and non-empty
reasoning (text inside <think>..</think>). Quantiles and the two-sample KS distance to base sc14. Read-only; stdlib.

    /usr/bin/python3 -I -B r_reasoning_chars.py -> out/reasoning_chars.txt
"""
import bisect
import json
from multiprocessing import Pool
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
MODELS = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def lens(attempt_dir):
    p = Path(attempt_dir) / "export.json" if attempt_dir else None
    if not p or not p.exists():
        return []
    out = []
    for m in json.loads(p.read_text()).get("messages") or []:
        info = m.get("info") or {}
        if info.get("role") != "assistant" or info.get("summary") is True or info.get("mode") == "compaction":
            continue
        parts = m.get("parts") or []
        if not any(x.get("type") == "tool" for x in parts):
            continue
        text = "".join(x.get("text") or "" for x in parts if x.get("type") == "text" and not x.get("synthetic") and not x.get("ignored")).lstrip()
        if not text.startswith("<think>"):
            continue
        r = text[7:].split("</think>", 1)[0]
        if r.strip():
            out.append(len(r))
    return out


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def ks(a, b):
    pts = sorted(set(a) | set(b))
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in pts)


def main():
    D = {}
    for m in MODELS:
        ads = []
        for path in sorted(TA.glob(f"results/{m}-group-*.json")):
            for recs in json.loads(path.read_text())["records"].values():
                ads += [r.get("attempt_dir") for r in recs]
        with Pool(12) as pool:
            D[m] = sorted(x for lst in pool.map(lens, ads, chunksize=4) for x in lst)
    L = []
    for m in MODELS:
        xs = D[m]
        L.append(f"{m:8s} reasoned action turns {len(xs)}; chars median {q(xs, .5)} mean {sum(xs) / len(xs):.0f} p90 {q(xs, .9)}; "
                 f"KS to sc14 {ks(xs, D['sc14']):.3f}")
    (OUT / "reasoning_chars.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
