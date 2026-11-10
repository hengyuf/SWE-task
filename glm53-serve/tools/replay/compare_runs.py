#!/usr/bin/env python3
"""Side-by-side markdown table of replay_load.py runs (stdlib only).

  python3 compare_runs.py runs/c16 runs/c32 runs/c64 [--json out.json]

Columns: concurrency, prefix mode, calls ok/err, generated tok/s, prompt tok/s, uncached-by-structure prompt
tok/s, mean in-flight, TTFT p50/p90, per-request decode tok/s p50/p10, chunk-gap p99, e2e p50/p90, sessions
completed, model-time ratio vs the recorded hub, and (if metrics were polled) server KV usage, prefix hit rate,
waiting, MTP mean acceptance length.
"""
import json
import os
import sys


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json in args:
        args.remove(out_json)
    rows = []
    for d in args:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            print("skip %s (no summary.json)" % d, file=sys.stderr)
            continue
        s = json.load(open(p))
        srv = s.get("server_metrics_window") or {}
        tgt = srv.get("all") or (next(iter(srv.values())) if srv else {})
        rows.append({
            "run": os.path.basename(os.path.normpath(d)),
            "conc": g(s, "args", "concurrency"),
            "mode": g(s, "args", "prefix_mode"),
            "ok": g(s, "counts", "calls_ok"),
            "err": g(s, "counts", "calls_error") or 0,
            "gen_tok_s": g(s, "throughput", "generated_tok_s"),
            "prompt_tok_s": g(s, "throughput", "prompt_tok_s"),
            "new_prompt_tok_s": g(s, "throughput", "new_prompt_tok_s(vs prev call)"),
            "inflight": g(s, "throughput", "mean_inflight_requests"),
            "ttft_p50": g(s, "ttft_s", "p50"),
            "ttft_p90": g(s, "ttft_s", "p90"),
            "dec_p50": g(s, "decode_tok_s_per_request", "p50"),
            "dec_p10": g(s, "decode_tok_s_per_request", "p10"),
            "gap_p99_ms": g(s, "chunk_gap_ms", "p99"),
            "e2e_p50": g(s, "e2e_s", "p50"),
            "e2e_p90": g(s, "e2e_s", "p90"),
            "sess_done": g(s, "sessions", "by_end", "completed") or 0,
            "model_vs_hub_p50": g(s, "sessions", "completed_model_time_over_recorded_hub_latency", "p50"),
            "kv": tgt.get("kv_usage_mean_mean"),
            "hit": tgt.get("prefix_hit_rate"),
            "waiting": tgt.get("waiting_mean"),
            "mtp_len": tgt.get("spec_mean_accept_len"),
        })
    cols = [("run", 0), ("conc", 0), ("mode", 0), ("ok", 0), ("err", 0), ("gen_tok_s", 0), ("prompt_tok_s", 0),
            ("new_prompt_tok_s", 0), ("inflight", 1), ("ttft_p50", 2), ("ttft_p90", 2), ("dec_p50", 1), ("dec_p10", 1),
            ("gap_p99_ms", 0), ("e2e_p50", 1), ("e2e_p90", 1), ("sess_done", 0), ("model_vs_hub_p50", 2),
            ("kv", 3), ("hit", 3), ("waiting", 1), ("mtp_len", 2)]
    print("| " + " | ".join(c for c, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f(r[c], nd) for c, nd in cols) + " |")
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
