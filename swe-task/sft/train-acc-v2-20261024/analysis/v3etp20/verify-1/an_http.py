#!/usr/bin/env python3
"""verify-1: aggregate out/http-<model>.jsonl.gz (scan_http.py) + the counting parser's vLLM log lines
(fleets/<fleet>/logs/vllm-*.log) into: model calls, missing </think> (join), length cuts / loops / full cap, empty
outputs, completion tokens, model time, foreign-script runs not in context, loop regions; per-slot rows for paired
tests. Stdlib only.

    python3 -I -B an_http.py MODEL [MODEL ...] -> out/an_http.json, out/an_http.txt, out/slots-<model>.json
"""
import collections
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
FLEET = {"sc14": "base"}
LOG = re.compile(r"TA_THINK_(RECOVERED|UNCLOSED_NO_TOOL|TOOL_IN_CLOSED_THINK) request_id=(\S+)(?: reasoning_chars=(\d+)| chars=(\d+))?")
COUNTS = re.compile(r"TA_THINK_COUNTS pid=(\d+) (.*)$")
CAP = 131072


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def parse_logs(model):
    marks, counts, logs_with_counts, nlogs = {}, {}, set(), 0
    for log in sorted((TA / "fleets" / FLEET.get(model, model) / "logs").glob("vllm-*.log")):
        nlogs += 1
        with open(log, "rb") as h:
            for raw in h:
                if b"TA_THINK_" not in raw:
                    continue
                line = raw.decode("utf-8", "replace")
                m = LOG.search(line)
                if m:
                    marks.setdefault(m.group(2), []).append((m.group(1), int(m.group(3) or m.group(4) or 0), log.name))
                    continue
                m = COUNTS.search(line)
                if m:
                    logs_with_counts.add(log.name)
                    counts[(log.name, m.group(1))] = {k: int(v) for k, v in (x.split("=") for x in m.group(2).split())}
    return marks, counts, logs_with_counts, nlogs


def main():
    models = sys.argv[1:]
    allres = {}
    lines = []
    for model in models:
        with gzip.open(OUT / f"http-{model}.jsonl.gz", "rt") as h:
            slots = [json.loads(l) for l in h]
        marks, counts, lwc, nlogs = parse_logs(model)
        o = {"slots": len(slots), "http_missing": sum(s["http_missing"] for s in slots)}
        calls = [(s, c) for s in slots for c in s["calls"] if "non200" not in c]
        non200 = collections.Counter(str(c["non200"]) for s in slots for c in s["calls"] if "non200" in c)
        o["model_calls"] = len(calls)
        o["non200"] = dict(non200)
        o["compaction_calls"] = sum(1 for _, c in calls if c.get("comp"))
        # duplicate 200s per call_id
        per_cid = collections.Counter((s["task_id"], s["slot"], c["call_id"]) for s, c in calls)
        o["call_ids_with_multiple_200"] = sum(1 for v in per_cid.values() if v > 1)
        ids = {}
        for s, c in calls:
            if c.get("id"):
                ids[c["id"][len("chatcmpl-"):] if c["id"].startswith("chatcmpl-") else c["id"]] = (s, c)
        # parser join
        o["log_marks"] = sum(len(v) for v in marks.values())
        o["log_marks_by_kind"] = dict(collections.Counter(k for v in marks.values() for k, _, _ in v))
        matched = [(rid, v) for rid, v in marks.items() if rid in ids]
        o["log_marks_matched_final"] = sum(len(v) for _, v in matched)
        o["log_marks_unmatched"] = [(rid, v) for rid, v in marks.items() if rid not in ids]
        miss_calls = []
        for rid, v in matched:
            s, c = ids[rid]
            for kind, chars, log in v:
                miss_calls.append({"kind": kind, "chars": chars, "finish": c["finish"], "ct": c["ct"], "r_len": c.get("r_len"),
                                   "c_len": c.get("c_len"), "tools": c["tools"], "r_markup": c.get("r_markup", False),
                                   "r_tail_zr": c.get("r_tail_zr"), "task": s["task_id"], "slot": s["slot"],
                                   "r_tail": (c.get("r_tail") or "")[-60:]})
        o["missing_calls"] = len(miss_calls)
        o["missing_rollouts"] = len({(m["task"], m["slot"]) for m in miss_calls})
        o["missing_pct"] = round(100 * len(miss_calls) / max(1, len(calls)), 3)
        o["recovered"] = sum(m["kind"] == "RECOVERED" for m in miss_calls)
        o["recovered_chars0"] = sum(m["kind"] == "RECOVERED" and m["chars"] == 0 for m in miss_calls)
        o["recovered_resp_reasoning"] = [(m["r_len"], m["c_len"], m["tools"]) for m in miss_calls if m["kind"] == "RECOVERED"]
        un = [m for m in miss_calls if m["kind"] == "UNCLOSED_NO_TOOL"]
        o["unclosed_eos"] = sum(m["finish"] != "length" for m in un)
        o["unclosed_eos_detail"] = [(m["finish"], m["r_len"], m["r_markup"], m["tools"], m["r_tail"]) for m in un if m["finish"] != "length"]
        o["unclosed_length"] = sum(m["finish"] == "length" for m in un)
        o["unclosed_length_rollouts"] = len({(m["task"], m["slot"]) for m in un if m["finish"] == "length"})
        lc = [m for m in un if m["finish"] == "length"]
        o["length_loop_tail015"] = sum((m["r_tail_zr"] or 1) < 0.15 for m in lc)
        o["length_loop_tail015_rollouts"] = len({(m["task"], m["slot"]) for m in lc if (m["r_tail_zr"] or 1) < 0.15})
        o["length_cut_not_loop"] = [(m["task"][4:30], m["slot"], m["ct"], m["r_tail_zr"]) for m in lc if (m["r_tail_zr"] or 1) >= 0.15]
        # all finish=length calls (any)
        fl = [(s, c) for s, c in calls if c["finish"] == "length"]
        o["finish_length_calls_any"] = len(fl)
        o["finish_length_comp"] = sum(1 for _, c in fl if c.get("comp"))
        o["full_cap_calls"] = sum(1 for _, c in fl if (c["ct"] or 0) >= CAP)
        o["full_cap_rollouts"] = len({(s["task_id"], s["slot"]) for s, c in fl if (c["ct"] or 0) >= CAP})
        # counters (last line per (log, pid))
        tot = collections.Counter()
        for v in counts.values():
            tot.update(v)
        o["parser_counters"] = dict(tot)
        o["parser_counters_missing_pct"] = round(100 * (tot["recovered"] + tot["unclosed_no_tool"]) / max(1, tot["calls"]), 3)
        o["logs"] = nlogs
        o["logs_with_counts"] = len(lwc)
        # empty outputs
        o["empty_outputs"] = sum(1 for _, c in calls if not c.get("r_len") and not c.get("c_len") and not c["tools"])
        # completion tokens
        cts = [c["ct"] or 0 for _, c in calls]
        o["ct_per_call_median_mean_p90"] = [statistics.median(cts), round(statistics.fmean(cts)), q(cts, 0.9)]
        per = collections.defaultdict(lambda: {"calls": 0, "ct": 0, "model_s": 0.0})
        for s, c in calls:
            k = (s["task_id"], s["slot"])
            per[k]["calls"] += 1
            per[k]["ct"] += c["ct"] or 0
            if c.get("req_ts") and c.get("ts"):
                per[k]["model_s"] += c["ts"] - c["req_ts"]
        for s in slots:
            per[(s["task_id"], s["slot"])]["status"] = s["status"]
        pv = list(per.values())
        o["calls_per_rollout_mean_median"] = [round(statistics.fmean(p["calls"] for p in pv), 1), statistics.median(p["calls"] for p in pv)]
        o["rollouts_ge500_calls"] = sum(p["calls"] >= 500 for p in pv)
        o["ct_per_rollout_mean"] = round(statistics.fmean(p["ct"] for p in pv))
        o["model_min_per_rollout_mean"] = round(statistics.fmean(p["model_s"] for p in pv) / 60, 1)
        # foreign-script runs not in context (reasoning), rollouts
        fr = collections.Counter()
        fr_any = set()
        fr_ex = []
        for s, c in calls:
            if c.get("r_foreign_spont"):
                fr_any.add((s["task_id"], s["slot"]))
                if len(fr_ex) < 6:
                    fr_ex.append(c.get("r_foreign_ex"))
        o["foreign_spont_rollouts"] = len(fr_any)
        o["foreign_any_rollouts"] = len({(s["task_id"], s["slot"]) for s, c in calls if c.get("r_foreign")})
        o["foreign_examples"] = fr_ex
        # loop regions in reasoning (2k windows zlib < 0.10)
        lc_all = [(s, c) for s, c in calls if c.get("r_loops")]
        o["loop_calls"] = len(lc_all)
        o["loop_calls_noncomp"] = sum(1 for _, c in lc_all if not c.get("comp"))
        o["loop_rollouts"] = len({(s["task_id"], s["slot"]) for s, _ in lc_all})
        hm = [(s, c) for s, c in lc_all if any(r["hmm_share"] >= 0.3 for r in c["r_loops"])]
        o["hmm_loop_calls"] = len(hm)
        o["hmm_loop_rollouts"] = len({(s["task_id"], s["slot"]) for s, _ in hm})
        o["loop_outcomes"] = dict(collections.Counter(
            "runaway" if c["finish"] == "length" else ("escaped" if any(r["escaped"] for r in c["r_loops"]) else "self_ended")
            for _, c in lc_all))
        # length cuts ending in a loop: unit of the last region
        units = collections.Counter()
        for m in lc:
            pass
        lcl = []
        for s, c in calls:
            if c["finish"] == "length" and not c.get("comp") and c.get("r_tail_zr") is not None and c["r_tail_zr"] < 0.15:
                last = (c.get("r_loops") or [{}])[-1]
                lcl.append((s["task_id"][4:30], s["slot"], c["ct"], c.get("r_len"), last.get("top"), last.get("hmm_share"), (c.get("r_tail") or "")[-50:]))
        o["length_cut_loop_units"] = lcl
        # escaped loops in action reasoning (>= 4000 chars, tail zlib < 0.15, call has a tool call)
        esc = [(s, c) for s, c in calls if c["tools"] and (c.get("r_len") or 0) >= 4000 and (c.get("r_tail_zr") or 1) < 0.15]
        o["action_reasoning_loops_turns"] = len(esc)
        o["action_reasoning_loops_rollouts"] = len({(s["task_id"], s["slot"]) for s, _ in esc})
        # largest tool-call argument
        o["max_tool_arg_len"] = max((c["argmax"] for _, c in calls), default=0)
        o["calls_arg_ge_65536"] = [(s["task_id"][4:30], s["slot"], c["argmax"], c["tools"]) for s, c in calls if c["argmax"] >= 65536]
        # runaway endings: last call of a rollout cut at length
        allres[model] = o
        (OUT / f"slots-{model}.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in per.items()}))
        lines.append(f"== {model}")
        for k, v in o.items():
            sv = json.dumps(v, default=str)
            lines.append(f"  {k}: {sv[:1500]}")
    (OUT / "an_http.json").write_text(json.dumps(allres, indent=1, default=str))
    (OUT / "an_http.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
