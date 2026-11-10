#!/usr/bin/env python3
"""Poll vLLM Prometheus /metrics endpoints into a TSV and compute window means. Stdlib only (python >= 3.8).

  # poll one or more endpoints every 5 s (Ctrl-C or --duration to stop)
  python3 scrape_metrics.py poll --url http://node1:8000/metrics --url http://node2:8000/metrics \
      --interval 5 --out run/metrics.tsv [--duration 3600]

  # window means (whole file, an explicit epoch window, or the window of a replay_load.py run)
  python3 scrape_metrics.py window run/metrics.tsv [--t0 EPOCH --t1 EPOCH] [--calls run/calls.jsonl --warmup 120]

One row per poll per target (each URL, plus 'all' = sum over URLs when >1 URL). Gauges are summed over engines
(running/waiting) or averaged/maxed (KV usage); counters are summed over engines/label sets, stored cumulatively
(c_* columns) and turned into per-interval rates. Metric names follow vLLM v1 (vllm/v1/metrics/loggers.py,
vllm/v1/spec_decode/metrics.py in vLLM 0.29): vllm:num_requests_running, vllm:num_requests_waiting(_by_reason),
vllm:kv_cache_usage_perc, vllm:prompt_tokens_total, vllm:generation_tokens_total, vllm:prompt_tokens_cached_total,
vllm:prefix_cache_{queries,hits}_total, vllm:external_prefix_cache_{queries,hits}_total, vllm:num_preemptions_total,
vllm:request_success_total, histograms vllm:{time_to_first_token,inter_token_latency,request_time_per_output_token,
e2e_request_latency,request_queue_time,request_prefill_time,request_decode_time}_seconds, and
vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total + vllm:spec_decode_num_accepted_tokens_per_pos_total.
Missing metrics leave empty cells.
"""
import argparse
import json
import re
import sys
import threading
import time
import urllib.request

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

COUNTERS = {
    "prompt": "vllm:prompt_tokens_total",
    "gen": "vllm:generation_tokens_total",
    "cached": "vllm:prompt_tokens_cached_total",
    "pc_q": "vllm:prefix_cache_queries_total",
    "pc_h": "vllm:prefix_cache_hits_total",
    "xpc_q": "vllm:external_prefix_cache_queries_total",
    "xpc_h": "vllm:external_prefix_cache_hits_total",
    "preempt": "vllm:num_preemptions_total",
    "req_ok": "vllm:request_success_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tok": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
HISTS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
MAX_POS = 8

COUNTER_COLS = (["c_" + k for k in COUNTERS] + ["c_%s_sum" % k for k in HISTS] + ["c_%s_cnt" % k for k in HISTS]
                + ["c_pos%d" % i for i in range(MAX_POS)])
RATE_COLS = ["prompt_tok_s", "gen_tok_s", "gen_tok_s_per_running", "cached_prompt_frac", "prefix_hit_rate",
             "ext_prefix_hit_rate", "preemptions", "finished_req_s"] + ["%s_mean_s" % k for k in HISTS] + [
             "spec_accept_rate", "spec_mean_accept_len", "spec_pos_accept"]
GAUGE_COLS = ["running", "waiting", "waiting_capacity", "waiting_deferred", "kv_usage_mean", "kv_usage_max",
              "n_engines"]
COLS = ["ts", "target"] + GAUGE_COLS + RATE_COLS + COUNTER_COLS


def parse_prom(text):
    """-> {metric_name: [(labels_dict, value), ...]} (skips *_created samples and NaN)."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), m.group(3)
        if name.endswith("_created"):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if v != v:  # NaN
            continue
        lab = dict(_LABEL.findall(labels)) if labels else {}
        out.setdefault(name, []).append((lab, v))
    return out


# never route cluster-internal scrapes through http_proxy/https_proxy from the environment
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def scrape(url, timeout=10.0):
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with _OPENER.open(req, timeout=timeout) as r:
        return parse_prom(r.read().decode("utf-8", "replace"))


def reduce_sample(p):
    """Reduce one parsed scrape to {gauges..., c_* cumulative counters} summed over engines/labels."""
    d = {}

    def s(name, pred=None):
        vals = [v for lab, v in p.get(name, []) if pred is None or pred(lab)]
        return sum(vals) if vals else None

    d["running"] = s("vllm:num_requests_running")
    d["waiting"] = s("vllm:num_requests_waiting")
    d["waiting_capacity"] = s("vllm:num_requests_waiting_by_reason", lambda l: l.get("reason") == "capacity")
    d["waiting_deferred"] = s("vllm:num_requests_waiting_by_reason", lambda l: l.get("reason") == "deferred")
    kv = [v for _, v in p.get("vllm:kv_cache_usage_perc", [])]
    d["kv_usage_mean"] = sum(kv) / len(kv) if kv else None
    d["kv_usage_max"] = max(kv) if kv else None
    d["n_engines"] = len(kv) if kv else None
    for k, name in COUNTERS.items():
        d["c_" + k] = s(name)
    for k, name in HISTS.items():
        d["c_%s_sum" % k] = s(name + "_sum")
        d["c_%s_cnt" % k] = s(name + "_count")
    for i in range(MAX_POS):
        d["c_pos%d" % i] = s(PER_POS, lambda l, i=i: l.get("position") == str(i))
    return d


def merge_samples(samples):
    """Sum several reduced samples (several API servers / nodes)."""
    out = {}
    for key in GAUGE_COLS + COUNTER_COLS:
        vals = [x.get(key) for x in samples if x.get(key) is not None]
        if not vals:
            out[key] = None
        elif key == "kv_usage_mean":
            out[key] = sum(vals) / len(vals)
        elif key == "kv_usage_max":
            out[key] = max(vals)
        else:
            out[key] = sum(vals)
    return out


def _delta(cur, prev, key):
    a, b = cur.get(key), prev.get(key)
    if a is None or b is None:
        return None
    return a - b


def rates(cur, prev, dt):
    """Per-interval rates between two reduced samples (prev may be None -> empty rates)."""
    r = {k: None for k in RATE_COLS}
    if prev is None or dt <= 0:
        return r
    dp, dg = _delta(cur, prev, "c_prompt"), _delta(cur, prev, "c_gen")
    if dp is not None:
        r["prompt_tok_s"] = dp / dt
    if dg is not None:
        r["gen_tok_s"] = dg / dt
        run = cur.get("running")
        if run:
            r["gen_tok_s_per_running"] = dg / dt / run
    dc = _delta(cur, prev, "c_cached")
    if dc is not None and dp:
        r["cached_prompt_frac"] = dc / dp
    q, h = _delta(cur, prev, "c_pc_q"), _delta(cur, prev, "c_pc_h")
    if q and h is not None:
        r["prefix_hit_rate"] = h / q
    q, h = _delta(cur, prev, "c_xpc_q"), _delta(cur, prev, "c_xpc_h")
    if q and h is not None:
        r["ext_prefix_hit_rate"] = h / q
    r["preemptions"] = _delta(cur, prev, "c_preempt")
    ok = _delta(cur, prev, "c_req_ok")
    if ok is not None:
        r["finished_req_s"] = ok / dt
    for k in HISTS:
        sm, cn = _delta(cur, prev, "c_%s_sum" % k), _delta(cur, prev, "c_%s_cnt" % k)
        if cn:
            r["%s_mean_s" % k] = sm / cn
    dd, dt_, da = _delta(cur, prev, "c_drafts"), _delta(cur, prev, "c_draft_tok"), _delta(cur, prev, "c_accepted")
    if dt_ and da is not None:
        r["spec_accept_rate"] = da / dt_
    if dd and da is not None:
        r["spec_mean_accept_len"] = 1.0 + da / dd
        pos = []
        for i in range(MAX_POS):
            x = _delta(cur, prev, "c_pos%d" % i)
            if x is None:
                break
            pos.append("%.3f" % (x / dd))
        r["spec_pos_accept"] = ",".join(pos) if pos else None
    return r


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return "%.6g" % v
    return str(v)


class Poller:
    """Background thread: scrape all URLs every `interval` s, append rows to `out` TSV."""

    def __init__(self, urls, out, interval=5.0, timeout=10.0, log=sys.stderr):
        self.urls = list(urls)
        self.out = out
        self.interval = interval
        self.timeout = timeout
        self.log = log
        self._stop = threading.Event()
        self._thread = None
        self.n_rows = 0
        self.n_errors = 0
        self._prev = {}

    def poll_once(self, f):
        now = time.time()
        reduced = {}
        for u in self.urls:
            try:
                reduced[u] = reduce_sample(scrape(u, self.timeout))
            except Exception as e:  # noqa: BLE001
                self.n_errors += 1
                if self.n_errors <= 5 or self.n_errors % 100 == 0:
                    print("scrape_metrics: %s: %r" % (u, e), file=self.log, flush=True)
        targets = dict(reduced)
        if len(self.urls) > 1 and len(reduced) == len(self.urls):
            targets["all"] = merge_samples(list(reduced.values()))
        for tgt, cur in targets.items():
            prev = self._prev.get(tgt)
            r = rates(cur, prev[1] if prev else None, now - prev[0] if prev else 0.0)
            row = {"ts": "%.3f" % now, "target": tgt}
            row.update(cur)
            row.update(r)
            f.write("\t".join(fmt(row.get(c)) for c in COLS) + "\n")
            self._prev[tgt] = (now, cur)
            self.n_rows += 1
        f.flush()

    def run(self, duration=None):
        t_end = time.time() + duration if duration else None
        with open(self.out, "a") as f:
            if f.tell() == 0:
                f.write("\t".join(COLS) + "\n")
            while not self._stop.is_set():
                t = time.time()
                self.poll_once(f)
                if t_end and time.time() >= t_end:
                    break
                self._stop.wait(max(0.0, self.interval - (time.time() - t)))
            self.poll_once(f)  # final sample so a window ends with fresh counters

    def start(self):
        self._thread = threading.Thread(target=self.run, name="metrics-poller", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=30.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)


# --------------------------------------------------------------------------------------------------------------
# window summaries
# --------------------------------------------------------------------------------------------------------------

def read_tsv(path):
    rows = []
    with open(path) as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if len(vals) != len(cols):
                continue
            r = {}
            for c, v in zip(cols, vals):
                if c == "target" or c == "spec_pos_accept":
                    r[c] = v
                else:
                    try:
                        r[c] = float(v) if v != "" else None
                    except ValueError:
                        r[c] = None
            rows.append(r)
    return rows


def window_summary(rows, t0=None, t1=None):
    """Per target: gauge means over rows in [t0, t1]; rates from counter deltas between the first and last row."""
    out = {}
    targets = sorted(set(r["target"] for r in rows))
    for tgt in targets:
        rs = [r for r in rows if r["target"] == tgt and (t0 is None or r["ts"] >= t0) and (t1 is None or r["ts"] <= t1)]
        if len(rs) < 2:
            out[tgt] = {"n_samples": len(rs)}
            continue
        first, last = rs[0], rs[-1]
        dt = last["ts"] - first["ts"]
        s = {"n_samples": len(rs), "window_s": round(dt, 1), "t0": first["ts"], "t1": last["ts"]}
        for g in GAUGE_COLS:
            vals = [r[g] for r in rs if r.get(g) is not None]
            if vals:
                s[g + "_mean"] = round(sum(vals) / len(vals), 4)
                if g in ("running", "waiting", "kv_usage_max"):
                    s[g + "_max"] = round(max(vals), 4)
        rr = rates(last, first, dt)
        if rr.get("gen_tok_s") is not None and s.get("running_mean"):
            rr["gen_tok_s_per_running"] = rr["gen_tok_s"] / s["running_mean"]
        else:
            rr["gen_tok_s_per_running"] = None
        for k, v in rr.items():
            if v is not None:
                s[k] = round(v, 5) if isinstance(v, float) else v
        # per-request decode speed like hsg-ref/notes/rp_window.py: sum(gen tok/s) / sum(running) over samples
        run = sum(r["running"] for r in rs[1:] if r.get("running"))
        gen = sum(r["gen_tok_s"] for r in rs[1:] if r.get("gen_tok_s") is not None)
        if run:
            s["gen_tok_s_per_running_sample_weighted"] = round(gen / run, 3)
        out[tgt] = s
    return out


def calls_window(calls_path, warmup=0.0):
    t_first, t_last = None, None
    with open(calls_path) as f:
        for line in f:
            c = json.loads(line)
            if c.get("t_send") is None:
                continue
            a, b = c["t_send"], c["t_send"] + (c.get("e2e_s") or 0.0)
            t_first = a if t_first is None else min(t_first, a)
            t_last = b if t_last is None else max(t_last, b)
    if t_first is None:
        return None, None
    return t_first + warmup, t_last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--url", action="append", required=True, help="http://host:port/metrics (repeatable)")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out", required=True)
    w = sub.add_parser("window")
    w.add_argument("tsv")
    w.add_argument("--t0", type=float, default=None)
    w.add_argument("--t1", type=float, default=None)
    w.add_argument("--calls", default=None, help="replay_load.py calls.jsonl: use its time window")
    w.add_argument("--warmup", type=float, default=0.0, help="with --calls: skip the first N seconds")
    a = ap.parse_args()
    if a.cmd == "poll":
        urls = [u if u.rstrip("/").endswith("/metrics") else u.rstrip("/") + "/metrics" for u in a.url]
        pl = Poller(urls, a.out, a.interval, a.timeout)
        try:
            pl.run(a.duration)
        except KeyboardInterrupt:
            pass
        print("wrote %d rows to %s (%d scrape errors)" % (pl.n_rows, a.out, pl.n_errors))
    else:
        t0, t1 = a.t0, a.t1
        if a.calls:
            t0, t1 = calls_window(a.calls, a.warmup)
        print(json.dumps(window_summary(read_tsv(a.tsv), t0, t1), indent=1))


if __name__ == "__main__":
    main()
