#!/usr/bin/env python3
"""End-to-end offline test: replay set -> stub server -> replay_load.py, with exact checks.

  python3 tests/offline_test.py [--replay data/xxx.jsonl.gz] [--build-python .venv/bin/python]

Runs everything in a private scratch dir (tempfile.mkdtemp, mode 0700) and removes it unless --keep.
Without --replay it first builds a small replay set (2 DeepSWE + 2 synthetic sessions, 15 calls each) with
build_replay_set.py using --build-python. The stub and the client run with THIS interpreter (use the login
node's python3.9 to prove the client is stdlib-only).

Checks:
  1 dry-run: every recorded prompt rebuilds to the build-time sha1
  2 recorded mode: every prompt the stub received == recorded prompt (sha1), min_tokens == max_tokens ==
    recorded output, ignore_eos, sampling params; output_got == target for every call; stub cache hits >= the
    previous call's prompt blocks
  3 splice and splice-exact modes: prompts == recorded prompts with the previous call's stub-generated ids
    spliced over the re-rendered assistant span (expected splice length recomputed independently; splice-exact
    only where recorded output == span + 1); cache hits cover the spliced span
  4 summary consistency: generated tok/s * window == generated tokens; per-request decode tok/s ~ stub rate;
    TTFT >= stub base; stub-reported cached fraction == client summary; /metrics window ~ client numbers
  5 --duration deadline: no call sent after the deadline, sessions end 'deadline'
  6 --max-context: no request exceeds the limit, sessions end 'context'
  7 server-side context error (stub --max-model-len): HTTP 400 recorded, sessions end 'error'
"""
import argparse
import array
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import replay_load  # noqa: E402

DEEPSWE = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/dataset/glm.jsonl.gz"
SYNTH = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/features/glm.jsonl"
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def sha1_ids(ids):
    a = array.array("I", ids)
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha1(a.tobytes()).hexdigest()


def unpack(b64):
    a = array.array("I")
    a.frombytes(base64.b64decode(b64))
    if sys.byteorder != "little":
        a.byteswap()
    return a


def start_stub(scratch, name, extra):
    log = os.path.join(scratch, name + "_stub.jsonl")
    out = open(os.path.join(scratch, name + "_stub.out"), "w")
    p = subprocess.Popen([sys.executable, "-I", os.path.join(ROOT, "stub_server.py"), "--port", "0", "--log", log]
                         + extra, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(100):
        time.sleep(0.1)
        txt = open(out.name).read()
        if "listening on" in txt:
            port = int(txt.strip().split(":")[-1])
            return p, port, log
    p.kill()
    raise RuntimeError("stub did not start: " + open(out.name).read())


def run_client(scratch, name, port, args):
    out = os.path.join(scratch, name)
    cmd = [sys.executable, "-I", os.path.join(ROOT, "replay_load.py"), "--base-url", "http://127.0.0.1:%d" % port,
           "--out", out, "--print-every", "2"] + args
    t = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
    with open(os.path.join(scratch, name + ".log"), "w") as f:
        f.write(r.stdout)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError("client failed: %s" % name)
    calls = [json.loads(l) for l in open(os.path.join(out, "calls.jsonl"))]
    summ = json.load(open(os.path.join(out, "summary.json")))
    sess = [json.loads(l) for l in open(os.path.join(out, "sessions.jsonl"))]
    print("  (%s: %d calls in %.1fs)" % (name, len(calls), time.time() - t), flush=True)
    return calls, summ, sess, out


def load_stub_log(path):
    recs = [json.loads(l) for l in open(path)]
    by = {}
    for r in recs:
        by[(r["session"], int(r["call"]))] = r
    return recs, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None)
    ap.add_argument("--build-python", default=os.path.join(ROOT, ".venv", "bin", "python"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--scratch-parent", default="/tmp")
    a = ap.parse_args()
    scratch = tempfile.mkdtemp(prefix="glm53-replay-test-", dir=a.scratch_parent)
    print("scratch:", scratch, "python:", sys.version.split()[0], flush=True)
    procs = []
    try:
        replay = a.replay
        if not replay:
            replay = os.path.join(scratch, "small.jsonl.gz")
            env = dict(os.environ, PYTHONWARNINGS="ignore", HF_HUB_OFFLINE="1", TRANSFORMERS_VERBOSITY="error")
            t = time.time()
            subprocess.run([a.build_python, "-I", os.path.join(ROOT, "build_replay_set.py"),
                            "--source", "deepswe=" + DEEPSWE, "--source", "synth=" + SYNTH, "--n-sessions", "2",
                            "--max-calls", "15", "--workers", "4", "--verify-every", "3", "--seed", "7",
                            "--out", replay], check=True, env=env, stdout=subprocess.DEVNULL)
            print("built %s in %.0fs" % (replay, time.time() - t), flush=True)
        headers, sessions = replay_load.load_replay([replay])
        by_sid = {s["session_id"]: s for s in sessions}
        n_sess = min(4, len(sessions))
        STUB_RATE = 300.0
        STUB_BASE = 0.03
        common = ["--replay", replay, "--sessions", str(n_sess), "--concurrency", "3", "--max-calls-per-session", "15",
                  "--max-output", "400", "--think-scale", "0.05", "--seed", "1"]

        print("[1] dry-run", flush=True)
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "replay_load.py"), "--replay", replay,
                            "--dry-run"], stdout=subprocess.PIPE, universal_newlines=True)
        check(r.returncode == 0 and '"sha1_mismatch": 0' in r.stdout, "dry-run rebuilds every prompt: " +
              r.stdout.strip().splitlines()[-1])

        p, port, slog = start_stub(scratch, "main", ["--decode-tok-s", str(STUB_RATE), "--prefill-tok-s", "2e6",
                                                     "--ttft-base", str(STUB_BASE), "--max-chunk", "4"])
        procs.append(p)
        murl = "http://127.0.0.1:%d/metrics" % port

        # ---------------------------------------------------------------- recorded mode
        print("[2] recorded mode", flush=True)
        calls, summ, sess, out = run_client(scratch, "recorded", port, common + [
            "--prefix-mode", "recorded", "--verify-prompts", "--metrics-url", murl, "--metrics-interval", "0.5"])
        recs, by = load_stub_log(slog)
        ok_calls = [c for c in calls if not c.get("error") and not c.get("skipped")]
        check(len(recs) == len(ok_calls) == summ["counts"].get("calls_ok"), "stub saw every call (%d)" % len(recs))
        check(summ["counts"].get("prompt_sha1_mismatch", 0) == 0 and summ["counts"].get("prompts_verified", 0) == len(ok_calls),
              "client-side sha1 verification of all %d prompts" % len(ok_calls))
        skey_of = {}
        for r_ in recs:
            skey_of.setdefault(r_["session"], None)
        mism = 0
        bad_params = 0
        cache_lb_viol = 0
        for c in ok_calls:
            s = by_sid[c["session_id"]]
            rc = s["calls"][c["call_idx"]]
            skey = [k for k in skey_of if k.endswith(":e%d:s%d" % (c["epoch"], c["session_idx"]))][0]
            st = by[(skey, c["call_idx"])]
            if st["prompt_sha1"] != rc["prompt_sha1"] or st["prompt_len"] != rc["prompt_len"]:
                mism += 1
            exp_out = min(rc["output_len"] or 1, 400)
            if not (st["max_tokens"] == st["min_tokens"] == exp_out == c["output_got"] and st["ignore_eos"] is True
                    and st["temperature"] == 1.0 and st["top_p"] == 0.95 and st["return_token_ids"] is False):
                bad_params += 1
            if c["call_idx"] > 0 and rc["shared_prefix_len"] == s["calls"][c["call_idx"] - 1]["prompt_len"]:
                lb = (min(rc["shared_prefix_len"], rc["prompt_len"] - 1) // 64) * 64
                if st["cached_tokens"] < lb:
                    cache_lb_viol += 1
        check(mism == 0, "stub-received prompts == recorded prompts (sha1+len) for all calls")
        check(bad_params == 0, "max_tokens == min_tokens == recorded output (capped 400) == tokens received; "
                               "ignore_eos, temperature 1.0, top_p 0.95")
        check(cache_lb_viol == 0, "stub cache hit >= previous prompt's full blocks on every extension call")
        tot_gen = sum(c["output_got"] for c in ok_calls)
        gen_rate_x_win = summ["throughput"]["generated_tok_s"] * summ["window"]["duration_s"]
        check(abs(gen_rate_x_win - tot_gen) <= 0.02 * tot_gen + 5,
              "generated_tok_s x window (%.0f) == generated tokens (%d)" % (gen_rate_x_win, tot_gen))
        tot_prompt = sum(c["prompt_len"] for c in ok_calls)
        prompt_rate_x_win = summ["throughput"]["prompt_tok_s"] * summ["window"]["duration_s"]
        check(abs(prompt_rate_x_win - tot_prompt) <= 0.01 * tot_prompt + 5,
              "prompt_tok_s x window (%.0f) == prompt tokens (%d)" % (prompt_rate_x_win, tot_prompt))
        d50 = summ["decode_tok_s_per_request"]["p50"]
        check(0.8 * STUB_RATE <= d50 <= 1.25 * STUB_RATE, "per-request decode tok/s p50 %.1f ~ stub %g" % (d50, STUB_RATE))
        t50 = summ["ttft_s"]["p50"]
        check(STUB_BASE <= t50 <= STUB_BASE + 0.3, "TTFT p50 %.3f s >= stub base %.2f s" % (t50, STUB_BASE))
        st_cached = sum(r_["cached_tokens"] for r_ in recs)
        st_prompt = sum(r_["prompt_len"] for r_ in recs)
        check(abs(summ["throughput"]["server_cached_prompt_frac"] - st_cached / float(st_prompt)) < 1e-3,
              "client server_cached_prompt_frac %.4f == stub %.4f" % (summ["throughput"]["server_cached_prompt_frac"],
                                                                      st_cached / float(st_prompt)))
        check(summ["sessions"]["by_end"].get("completed") == n_sess, "all %d sessions completed" % n_sess)
        mw = summ.get("server_metrics_window", {}).get(murl, {})
        check(mw.get("n_samples", 0) >= 3, "metrics.tsv polled (%s samples)" % mw.get("n_samples"))
        if mw.get("prefix_hit_rate") is not None:
            # metrics window is clipped to the client window; hit rate should match closely
            check(abs(mw["prefix_hit_rate"] - st_cached / float(st_prompt)) < 0.08,
                  "metrics prefix_hit_rate %.3f ~ stub cached frac %.3f" % (mw["prefix_hit_rate"], st_cached / float(st_prompt)))
        if mw.get("gen_tok_s") is not None:
            ratio = mw["gen_tok_s"] / max(1e-9, summ["throughput"]["generated_tok_s"])
            check(0.6 <= ratio <= 1.4, "metrics gen_tok_s %.0f ~ client generated_tok_s %.0f" % (
                mw["gen_tok_s"], summ["throughput"]["generated_tok_s"]))
        rec_hit = st_cached / float(st_prompt)

        # ---------------------------------------------------------------- splice modes
        for mode in ("splice", "splice-exact"):
            print("[3] %s mode" % mode, flush=True)
            n_before = len(load_stub_log(slog)[0])
            calls, summ, sess, out = run_client(scratch, mode, port, common + ["--prefix-mode", mode])
            recs_all, _ = load_stub_log(slog)
            recs = recs_all[n_before:]
            by = {(r_["session"], int(r_["call"])): r_ for r_ in recs}
            ok_calls = sorted([c for c in calls if not c.get("error")], key=lambda c: (c["session_idx"], c["call_idx"]))
            skeys = set(r_["session"] for r_ in recs)
            mism = spl_mism = cache_viol = n_spliced_calls = n_count_mismatch = 0
            cur = {}
            for c in ok_calls:
                s = by_sid[c["session_id"]]
                rc = s["calls"][c["call_idx"]]
                skey = [k for k in skeys if k.endswith(":e%d:s%d" % (c["epoch"], c["session_idx"]))][0]
                st = by[(skey, c["call_idx"])]
                prev = cur.get(skey)
                shared = rc["shared_prefix_len"]
                base = prev["ids"][:shared] if prev else array.array("I")
                delta = rc["delta"]
                n_exp = 0
                if prev is not None and shared == len(prev["ids"]) and rc.get("splice_len") is not None and len(delta):
                    exact = s["calls"][c["call_idx"] - 1]["output_len"] == rc["splice_len"] + 1
                    n_count_mismatch += 0 if exact else 1
                    if mode == "splice" or exact:
                        n_exp = min(rc["splice_len"] + 1, len(prev["gen"]), len(delta))
                nd = array.array("I", prev["gen"][:n_exp]) if n_exp else array.array("I")
                nd.extend(delta[n_exp:])
                exp_ids = base + nd
                if n_exp != c["spliced"]:
                    spl_mism += 1
                if sha1_ids(exp_ids) != st["prompt_sha1"] or len(exp_ids) != st["prompt_len"]:
                    mism += 1
                if n_exp:
                    n_spliced_calls += 1
                    lb = (min(len(prev["ids"]) + min(n_exp, len(prev["gen"]) - 1), len(exp_ids) - 1) // 64) * 64
                    if st["cached_tokens"] < lb:
                        cache_viol += 1
                if not st["return_token_ids"]:
                    spl_mism += 1
                cur[skey] = {"ids": exp_ids, "gen": unpack(st["gen_ids_b64"])}
            check(n_spliced_calls > 0, "%s: splice applied on %d calls (%d extension calls with a count mismatch)" % (
                mode, n_spliced_calls, n_count_mismatch))
            check(spl_mism == 0, "%s: client splice length == independently recomputed length on every call" % mode)
            check(mism == 0, "%s: stub-received prompts == recorded prompts with previous generation spliced in" % mode)
            check(cache_viol == 0, "%s: stub cache hit covers previous prompt + spliced generated tokens" % mode)
            spl_hit = sum(r_["cached_tokens"] for r_ in recs) / float(sum(r_["prompt_len"] for r_ in recs))
            check(spl_hit >= rec_hit, "%s: cached frac %.4f >= recorded-mode cached frac %.4f" % (mode, spl_hit, rec_hit))
            check(summ["counts"].get("splice_unavailable_no_token_ids", 0) == 0, "%s: token ids returned on every call" % mode)
            check(all((c.get("prompt_ids_echoed") or 0) == c["prompt_len"] for c in ok_calls),
                  "%s: first-chunk prompt_token_ids echo parsed (vLLM return_token_ids behavior)" % mode)

        # ---------------------------------------------------------------- deadline
        print("[4] --duration deadline", flush=True)
        calls, summ, sess, out = run_client(scratch, "deadline", port, [
            "--replay", replay, "--sessions", str(n_sess), "--concurrency", "3", "--max-output", "400",
            "--think-scale", "20", "--think-cap", "2", "--duration", "4", "--drain-timeout", "30", "--seed", "1"])
        args_json = json.load(open(os.path.join(out, "client_args.json")))
        t_sends = [c["t_send"] for c in calls if c.get("t_send")]
        t0 = min(t_sends)
        check(max(t_sends) - t0 <= 4.0 + 0.2, "no call sent after the 4 s deadline (last at +%.2fs)" % (max(t_sends) - t0))
        check(summ["sessions"]["by_end"].get("deadline", 0) >= 1, "sessions ended by deadline: %s" % summ["sessions"]["by_end"])
        check(summ["wall_s"] < 4 + 10, "run stopped promptly (wall %.1fs)" % summ["wall_s"])

        # ---------------------------------------------------------------- max-context
        print("[5] --max-context", flush=True)
        plens = sorted(c["prompt_len"] for s in sessions for c in s["calls"][:15])
        limit = plens[len(plens) // 2]
        calls, summ, sess, out = run_client(scratch, "maxctx", port, common + ["--max-context", str(limit)])
        sent = [c for c in calls if not c.get("skipped") and not c.get("error")]
        check(all(c["prompt_len"] + c["output_target"] <= limit for c in sent),
              "every sent call fits max-context %d" % limit)
        check(summ["counts"].get("calls_skipped_context", 0) >= 1 and summ["sessions"]["by_end"].get("context", 0) >= 1,
              "over-limit calls skipped and sessions ended 'context': %s" % summ["sessions"]["by_end"])

        # ---------------------------------------------------------------- server-side 400
        print("[6] server context error -> HTTP 400 handling", flush=True)
        p2, port2, slog2 = start_stub(scratch, "small", ["--max-model-len", str(limit), "--decode-tok-s", "1000"])
        procs.append(p2)
        calls, summ, sess, out = run_client(scratch, "err400", port2, common)
        errs = [c for c in calls if c.get("error")]
        check(len(errs) >= 1 and all("HTTP 400" in c["error"] for c in errs), "HTTP 400 recorded (%d calls)" % len(errs))
        check(summ["sessions"]["by_end"].get("error", 0) >= 1, "sessions ended 'error': %s" % summ["sessions"]["by_end"])
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(5)
            except Exception:  # noqa: BLE001
                p.kill()
        if not a.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            print("kept", scratch)
    print("\nRESULT: %s (%d failures)" % ("PASS" if not FAILS else "FAIL", len(FAILS)))
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
