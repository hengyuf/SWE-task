#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): what the no-tool, no-content model calls (= unclosed think without a tool call; logged
for 10/20 ep, validated signature for 5 ep / base) contain, from the raw response reasoning in http.jsonl:
empty (EOS right after <think>), headless tool call (tool-call closing tags such as "</tool_call>" / "</function>"
without the "<tool_call>" opener, so the lenient rule cannot rescue it), runaway text cut by length, other text (EOS).
2026-10-27: + etp5 / etp10 (empty-think-prefix SFT; logged like 10 / 20 ep).
2026-10-28: + etp20, and for the calls cut by length: how many end in a repetition loop (zlib ratio of the last 20,000
chars of the reasoning < 0.15: "text_cut_by_length_loop_tail") and their median reasoning chars (new keys only).
2026-11-04: + v3etp20, copy in analysis/v3etp20/think (every model recomputed; earlier columns must reproduce
analysis/think/notool_kinds.json).
    python3 -I -B notool_kinds.py -> notool_kinds.json
"""
import collections
import json
from pathlib import Path
import statistics
import zlib

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = TA / "analysis/v3etp20/think"  # 2026-11-04 copy: outputs here; calls of earlier models from ORIG (read-only)
ORIG = TA / "analysis/think"


def calls_path(m):
    return (HERE if (HERE / f"calls-{m}.jsonl").exists() else ORIG) / f"calls-{m}.jsonl"


MODELS = ("sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20")  # the one model list (v3etp5/10: 2026-10-31; v3etp20: 2026-11-04)
out = {}
for m in MODELS:
    attempt = {}
    for path in TA.glob(f"results/{m}-group-*.json"):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for rec in records:
                attempt[(task_id, rec["slot"])] = rec["attempt_dir"]
    want = collections.defaultdict(dict)
    for line in open(calls_path(m)):
        r = json.loads(line)
        if r["tool_calls"] == 0 and not r["content_len"]:
            want[attempt[(r["task_id"], r["slot"])]][r["id"]] = r
    kinds = collections.Counter()
    rollouts = collections.defaultdict(set)
    cut_chars = []
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
                text = (body["choices"][0]["message"].get("reasoning") or "")
                if not text.strip():
                    kind = "empty"
                elif ("</tool_call>" in text or "</function>" in text) and "<tool_call>" not in text:
                    kind = "headless_tool_call"
                elif r["finish"] == "length":
                    kind = "text_cut_by_length"
                    cut_chars.append(len(text))
                    tail = text[-20000:].encode()
                    if len(zlib.compress(tail, 6)) / len(tail) < 0.15:
                        kinds["text_cut_by_length_loop_tail"] += 1
                        rollouts["text_cut_by_length_loop_tail"].add((r["task_id"], r["slot"]))
                else:
                    kind = "text_eos"
                kinds[kind] += 1
                rollouts[kind].add((r["task_id"], r["slot"]))
    out[m] = {k: {"calls": kinds[k], "rollouts": len(rollouts[k])} for k in
              ("empty", "headless_tool_call", "text_eos", "text_cut_by_length", "text_cut_by_length_loop_tail")}
    out[m]["text_cut_by_length_reasoning_chars_median"] = statistics.median(cut_chars) if cut_chars else None
    print(m, out[m])
(HERE / "notool_kinds.json").write_text(json.dumps(out, indent=1) + "\n")
