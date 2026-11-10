#!/usr/bin/env python3
"""train-acc-v2 (2026-10-27): is the reasoning before actions substantive or filler? Per model, from the session export
(export.json) of each slot's final attempt, with the teacher-rollout extractor's turn split (TR/features.py split_turn,
imported read-only): every action turn (>= 1 tool call) with reasoning (non-empty after strip).

    python3 -I -B reasoning_quality.py [--models sc14 sftv2 sftv2e10 sftv2e20 etp5 etp10] [--procs 8]
      -> reasoning_quality.{json,md}

Per model (pooled over action turns with reasoning):
  * chars quantiles; % under 40 / 100 chars;
  * exact repeats: % whose stripped text equals an earlier reasoning of the same rollout (verbatim re-use = filler);
  * near repeats: % whose first 60 chars (lowercased, digits -> 0) equal an earlier one's in the same rollout;
  * % short (< 100 chars) that are generic: no code-ish token (path, identifier with _ . / ( ` or a digit), i.e. text
    such as "Let me check the tests." that names nothing specific;
  * loop-like long reasoning: reasoning >= 4,000 chars whose zlib ratio (compressed / raw bytes) is < 0.15 (repetition
    loops compress far better than prose or code; spot checks: "hold on — hold on — ..." / "hmm — hmm — ..." runs of
    thousands of chars that the model then escapes, "OK — I see I'm looping again"), the rollouts with >= 1, and the
    most repeated line's share of lines >= 30%;
  * top 12 short (< 100 chars) reasonings after normalization, with counts.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import zlib  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
HERE = TA / "analysis/think"
LABEL = {"sc14": "base", "sftv2": "5 ep", "sftv2e10": "10 ep", "sftv2e20": "20 ep",
         "etp5": "etp 5 ep",
         "etp10": "etp 10 ep",
         "etp20": "etp 20 ep",
         "v3etp5": "v3 etp 5 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 5 ep (iter71)
         "v3etp10": "v3 etp 10 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 10 ep (iter142)
         }  # the one model registry (one line per model, column order)
MODELS = tuple(LABEL)
SPECIFIC = re.compile(r"[/_`()\[\]{}.:=<>#]|\d|[a-z][A-Z]")  # a path, identifier, call, number, camelCase ...


def norm(text):
    return re.sub(r"\d+", "0", " ".join(text.lower().split()))


def scan(attempt_dir):
    if str(TR) not in sys.path:
        sys.path.insert(0, str(TR))
    import features as tr_features  # TR/features.py (read-only); split_turn needs no tokenizer
    try:
        export = json.loads((Path(attempt_dir) / "export.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    out = collections.Counter()
    chars, shorts, loops = [], collections.Counter(), []
    seen_exact, seen_head = set(), set()
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") != "assistant" or info.get("mode") == "compaction" or info.get("summary") is True:
            continue
        reasoning, _visible, tools, _open = tr_features.split_turn(message)
        text = reasoning.strip()
        if not tools or not text:
            continue
        out["turns"] += 1
        chars.append(len(text))
        out["exact_repeat"] += text in seen_exact
        head = norm(text)[:60]
        out["near_repeat"] += head in seen_head
        seen_exact.add(text)
        seen_head.add(head)
        if len(text) < 100:
            out["short_lt100"] += 1
            out["short_generic"] += not SPECIFIC.search(text)
            shorts[norm(text)] += 1
        out["lt40"] += len(text) < 40
        if len(text) >= 4000:
            out["long"] += 1
            raw = text.encode()
            ratio = len(zlib.compress(raw, 6)) / len(raw)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            top = collections.Counter(lines).most_common(1)[0][1] / len(lines) if lines else 0
            out["long_zlib_lt_0.15"] += ratio < 0.15
            out["long_top_line_ge_30pct"] += len(lines) >= 10 and top >= 0.3
            loops.append(round(ratio, 3))
    return {"counts": dict(out), "chars": chars, "shorts": dict(shorts), "zlib_long": loops}


def attempts(model):
    out = []
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        for records in json.loads(path.read_text())["records"].values():
            out += [r["attempt_dir"] for r in records if r.get("attempt_dir")]
    return out


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    path = HERE / "reasoning_quality.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for model in args.models:
        with ProcessPoolExecutor(args.procs) as pool:
            rows = [r for r in pool.map(scan, attempts(model), chunksize=4) if r]
        counts, shorts, chars, ratios = collections.Counter(), collections.Counter(), [], []
        for r in rows:
            counts.update(r["counts"])
            shorts.update(r["shorts"])
            chars += r["chars"]
            ratios += r["zlib_long"]
        chars.sort()
        ratios.sort()
        n = counts["turns"]
        pct = lambda k: round(100.0 * counts[k] / n, 2) if n else None  # noqa: E731
        result[model] = {
            "rollouts_with_export": len(rows), "reasoned_action_turns": n,
            "chars_p10": q(chars, .1), "chars_p25": q(chars, .25), "chars_median": q(chars, .5), "chars_p75": q(chars, .75),
            "chars_p90": q(chars, .9), "pct_lt40_chars": pct("lt40"), "pct_lt100_chars": pct("short_lt100"),
            "pct_exact_repeat_in_rollout": pct("exact_repeat"), "pct_near_repeat_in_rollout": pct("near_repeat"),
            "pct_short_generic": pct("short_generic"),
            "long_ge4000_chars": counts["long"], "pct_long": pct("long"),
            "long_loop_like_zlib_lt_0.15": counts["long_zlib_lt_0.15"],
            "pct_loop_like_of_reasoned_turns": pct("long_zlib_lt_0.15"),
            "rollouts_with_loop_like": sum(1 for r in rows if r["counts"].get("long_zlib_lt_0.15")),
            "long_top_line_ge_30pct": counts["long_top_line_ge_30pct"],
            "long_zlib_ratio_median": q(ratios, .5),
            "top_short": shorts.most_common(12),
        }
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        tmp.replace(path)
        print(model, json.dumps({k: v for k, v in result[model].items() if k != "top_short"}), flush=True)
    present = [m for m in MODELS if m in result]
    keys = [k for k in result[present[0]] if k != "top_short"]
    lines = ["| reasoned action turns | " + " | ".join(LABEL[m] for m in present) + " |", "|---|" + "---|" * len(present)]
    lines += [f"| {k} | " + " | ".join(str(result[m][k]) for m in present) + " |" for k in keys]
    for m in present:
        lines.append(f"\n{LABEL[m]} top short reasonings: {result[m]['top_short']}")
    (HERE / "reasoning_quality.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
