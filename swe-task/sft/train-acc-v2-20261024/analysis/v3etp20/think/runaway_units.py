#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): what the runaway thinking calls loop on. For every model call of the final attempts
that the counting parser logged as an unclosed think without a tool call AND that finished on "length" (= the
split_table / notool_kinds "length cut" calls), read the returned reasoning from the attempt's http.jsonl (response
body id) and classify its last 20,000 characters:
  tail zlib ratio  compressed / raw bytes (notool_kinds.py: < 0.15 = ends in a repetition loop)
  unit, class      degenerate-behaviors-20261031 text/textlib.py dominant_unit + classify_unit on the tail (hmm,
                   shouting (AH/OH/WAIT), hold on, wait, bare dash, check marks, zero-width, arrows, other symbols,
                   words, code/data); imported read-only from the copy in ../degenerate/text
  hmm share        'hmm' words (case-insensitive) x 3 chars + separators, as a share of the tail: the fraction of the
                   tail that is an 'hmm' run (reported, not used for the class)
  cap              completion_tokens >= 131,072 (full output cap; the others were clamped by the router to the
                   context window)
Models: etp20, v3etp5, v3etp10 (calls from analysis/think) and v3etp20 (calls from ./calls-v3etp20.jsonl).

    python3 -I -B runaway_units.py -> runaway_units.{json,txt}
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import zlib  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "degenerate/text"))
import textlib as T  # noqa: E402

MODELS = ["etp20", "v3etp5", "v3etp10", "v3etp20"]
HMM = re.compile(r"(?i)(?<![a-z])hmm(?![a-z])")


def calls_path(m):
    return (HERE if m == "v3etp20" else TA / "analysis/think") / f"calls-{m}.jsonl"


def attempts(m):
    out = {}
    for path in TA.glob(f"results/{m}-group-*.json"):
        for task_id, recs in json.loads(path.read_text())["records"].items():
            for r in recs:
                out[(task_id, r["slot"])] = r["attempt_dir"]
    return out


def main():
    res, lines = {}, []
    for m in MODELS:
        att = attempts(m)
        want = collections.defaultdict(dict)
        for line in open(calls_path(m)):
            r = json.loads(line)
            if r.get("log_kind") == "UNCLOSED_NO_TOOL" and r.get("finish") == "length":
                want[att[(r["task_id"], r["slot"])]][r["id"]] = r
        rows = []
        for ad, ids in want.items():
            with open(Path(ad) / "http.jsonl", "rb") as handle:
                for raw in handle:
                    if b'"event":"response"' not in raw[:2000]:
                        continue
                    event = json.loads(raw)
                    body = event.get("body")
                    body = json.loads(body) if isinstance(body, str) else body
                    if not isinstance(body, dict) or body.get("id") not in ids:
                        continue
                    r = ids[body["id"]]
                    msg = body["choices"][0]["message"]
                    text = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    tail = text[-20000:]
                    raw_tail = tail.encode("utf-8", "surrogatepass")
                    ratio = len(zlib.compress(raw_tail, 6)) / max(1, len(raw_tail))
                    u = T.dominant_unit(tail)
                    cls = T.classify_unit(u[0] if u else None)
                    hmm_share = round(4 * len(HMM.findall(tail)) / max(1, len(tail)), 3)
                    rows.append({"task_id": r["task_id"], "slot": r["slot"], "tokens": r.get("completion_tokens"),
                                 "cap": (r.get("completion_tokens") or 0) >= 131072, "chars": len(text),
                                 "tail_zlib": round(ratio, 3), "loop_tail": ratio < 0.15, "unit": (u[3] if u else None),
                                 "unit_cov": round(u[1], 2) if u else None, "class": cls, "hmm_share": hmm_share})
        res[m] = rows
        loops = [x for x in rows if x["loop_tail"]]
        lines.append(f"{m:8s} length-cut calls {len(rows)} in {len({(x['task_id'], x['slot']) for x in rows})} rollouts | full 131,072 cap "
                     f"{sum(x['cap'] for x in rows)} | tail zlib < 0.15 {len(loops)} | class of the loop tails: "
                     f"{dict(collections.Counter(x['class'] for x in loops).most_common())} | non-loop tails: "
                     + "; ".join(f"{x['task_id']} {x['slot']} zlib {x['tail_zlib']} class {x['class']} hmm share {x['hmm_share']}"
                                 for x in rows if not x["loop_tail"]))
        for x in loops:
            if x["class"] != "hmm":
                lines.append(f"    non-hmm loop: {x['task_id']} {x['slot']} tokens {x['tokens']} unit {x['unit']!r} cov {x['unit_cov']}")
    (HERE / "runaway_units.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    (HERE / "runaway_units.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
