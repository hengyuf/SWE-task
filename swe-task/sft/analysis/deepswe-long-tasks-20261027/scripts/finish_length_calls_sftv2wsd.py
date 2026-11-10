#!/usr/bin/env python3
"""Characterise finish == "length" model calls (runaway generations) in DeepSWE eval rollouts (READ-ONLY inputs).

usage: python3 -I finish_length_calls_sftv2wsd.py --out-dir OUT [--procs 8] RUN [RUN ...]

Reads OUT/<RUN>.jsonl (written by evalrun_peakctx_sftv2wsd.py), re-opens only the exports of rollouts with
n_finish_length > 0, and for every main-session assistant message with finish == "length" records mode,
prompt (input + cache.read + cache.write), output tokens, total, and a repetition score = zlib-compressed size /
raw size of the message text (degenerate loops such as "hmm — hmm — ..." compress to < 0.05; normal prose and
code are ~0.3-0.5). Writes OUT/<RUN>.finish_length_calls.jsonl and prints counts.
"""
import argparse
import json
import os
import zlib
from collections import Counter
from multiprocessing import Pool

TARGETS = {"scriggo-method-declarations", "wasmi-trap-coredumps", "boa-hierarchical-evaluation-cancellation"}


def scan(args):
    run, task_id, ridx, path = args
    with open(path) as f:
        ex = json.load(f)
    sid = (ex.get("info") or {}).get("id")
    out = []
    for i, m in enumerate(ex.get("messages") or []):
        mi = m.get("info") or {}
        if mi.get("role") != "assistant" or mi.get("finish") != "length":
            continue
        if sid and mi.get("sessionID") not in (None, sid):
            continue
        tok = mi.get("tokens") or {}
        cache = tok.get("cache") or {}
        prompt = int(tok.get("input") or 0) + int(cache.get("read") or 0) + int(cache.get("write") or 0)
        o = int(tok.get("output") or 0)
        txt = "".join((p.get("text") or "") for p in (m.get("parts") or []) if p.get("type") in ("text", "reasoning"))
        raw = txt.encode("utf-8", "replace")
        ratio = round(len(zlib.compress(raw, 6)) / len(raw), 4) if raw else None
        tail_ratio = None
        if len(raw) > 20000:
            t = raw[-20000:]
            tail_ratio = round(len(zlib.compress(t, 6)) / len(t), 4)
        out.append({"run": run, "task_id": task_id, "rollout_index": ridx, "msg_index": i,
                    "mode": "compaction" if (mi.get("mode") == "compaction" or mi.get("summary") is True) else mi.get("mode"),
                    "prompt": prompt, "output": o, "total": prompt + o, "text_chars": len(txt),
                    "zlib_ratio": ratio, "zlib_ratio_last20k": tail_ratio,
                    "tail": txt[-160:].replace("\n", "\\n")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    jobs = []
    for run in a.runs:
        with open(os.path.join(a.out_dir, f"{run}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("n_finish_length") and r.get("export_path"):
                    jobs.append((run, r["task_id"], r["rollout_index"], r["export_path"]))
    res = []
    with Pool(a.procs) as pool:
        for calls in pool.imap_unordered(scan, jobs, chunksize=1):
            res += calls
    for run in a.runs:
        rows = sorted((c for c in res if c["run"] == run), key=lambda c: (c["task_id"], c["rollout_index"], c["msg_index"]))
        tmp = os.path.join(a.out_dir, f".{run}.finish_length_calls.jsonl.tmp")
        with open(tmp, "w") as f:
            for c in rows:
                f.write(json.dumps(c) + "\n")
        os.replace(tmp, os.path.join(a.out_dir, f"{run}.finish_length_calls.jsonl"))
        degen = [c for c in rows if c["zlib_ratio"] is not None and c["zlib_ratio"] < 0.05]
        degen_tail = [c for c in rows if (c["zlib_ratio_last20k"] if c["zlib_ratio_last20k"] is not None else c["zlib_ratio"] or 1) < 0.05]
        at_cap = [c for c in rows if c["total"] >= 262144]
        print(f"=== {run}: finish=length calls {len(rows)} in {len({(c['task_id'], c['rollout_index']) for c in rows})} rollouts; "
              f"total at 262,144 cap {len(at_cap)}; output>=131072 {sum(1 for c in rows if c['output'] >= 131072)}; "
              f"compaction-summary calls {sum(1 for c in rows if c['mode'] == 'compaction')}; "
              f"whole-text zlib<0.05 {len(degen)}; last-20k zlib<0.05 {len(degen_tail)}; "
              f"median output {sorted(c['output'] for c in rows)[len(rows) // 2] if rows else None}")
        print("   zlib ratio histogram:", dict(Counter(
            ("<0.02" if c["zlib_ratio"] < 0.02 else "<0.05" if c["zlib_ratio"] < 0.05 else "<0.1" if c["zlib_ratio"] < 0.1
             else "<0.2" if c["zlib_ratio"] < 0.2 else ">=0.2") for c in rows if c["zlib_ratio"] is not None)))
        for c in rows:
            if c["task_id"] in TARGETS:
                print(f"   {c['task_id'][:22]} r{c['rollout_index']} msg {c['msg_index']} {c['mode']} prompt {c['prompt']} "
                      f"out {c['output']} total {c['total']} zlib {c['zlib_ratio']} last20k {c['zlib_ratio_last20k']} "
                      f"tail {c['tail'][-90:]!r}")


if __name__ == "__main__":
    main()
