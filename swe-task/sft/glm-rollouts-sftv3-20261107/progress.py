#!/usr/bin/env python3
"""glm-rollouts-sftv3 progress: rewrites PROGRESS.md (and progress.json) from the campaign dirs, the results files,
the router logs and the keeper status. Read-only on everything else.

    /usr/bin/python3 -I -B progress.py [--stdout]

Per campaign: slots (tasks x repeats), terminal attempts graded pass / fail, running, pass rate; rollouts that
compacted (an OpenCode summary request in http.jsonl); LLM calls, generated tokens and replies at >= 131,072 /
>= 32,768 completion tokens; router errors. Plus the GPU servers (keeper status) and total generation throughput.
"""
import collections
import glob
import json
import os
from pathlib import Path
import time

GR = Path(__file__).resolve().parent
CAMPAIGNS = ["pilot", "baihe-pilot"] + [f"hengyuf-g{g}" for g in range(4)] + [f"baihe-g{g}" for g in range(4)]
SLOTS = {"pilot": 1, "baihe-pilot": 1, **{f"hengyuf-g{g}": 155 for g in range(4)}, **{f"baihe-g{g}": 120 for g in range(4)}}
CACHE = GR / "progress-cache.json"


def attempt_stats(attempt: Path, cache: dict) -> dict:
    """Calls, generated tokens, long replies and compaction of one attempt (cached once terminal)."""
    key = str(attempt)
    terminal = (attempt / "terminal.json").exists()
    if terminal and key in cache:
        return cache[key]
    calls = gen = ge131k = ge32k = errors = 0
    compacted = False
    max_prompt = 0
    path = attempt / "http.jsonl"
    if path.exists():
        with open(path, "rb") as stream:
            for raw in stream:
                head = raw[:1500]
                if b'"event":"request"' in head or b'"event": "request"' in head:
                    # OpenCode 1.17.11's compaction call (core/session/compaction.ts buildPrompt):
                    # "Create a new anchored summary ..." / "Update the anchored summary below ..."
                    if b"anchored summary" in raw:
                        compacted = True
                    continue
                if b'"event":"response"' not in head and b'"event": "response"' not in head:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if event.get("status") != 200:
                    errors += 1
                    continue
                try:
                    body = json.loads(event.get("body") or "{}")
                except ValueError:
                    continue
                usage = body.get("usage") or {}
                completion = usage.get("completion_tokens") or 0
                calls += 1
                gen += completion
                max_prompt = max(max_prompt, usage.get("prompt_tokens") or 0)
                ge131k += completion >= 131072
                ge32k += completion >= 32768
    out = {"calls": calls, "gen": gen, "ge131k": ge131k, "ge32k": ge32k, "non200": errors, "compacted": compacted,
           "max_prompt": max_prompt, "terminal": terminal}
    if terminal:
        try:
            out["terminal_info"] = {k: json.loads((attempt / "terminal.json").read_text()).get(k)
                                    for k in ("reward", "f2p", "p2p", "status")}
        except (OSError, ValueError):
            pass
        try:  # sandbox-side failure that ended OpenCode before it finished (driver.py re-runs these once)
            native = json.loads((attempt / "native-result.json").read_text())
            error = native.get("opencode_run_error_type")
            out["sandbox_failed"] = bool(native.get("opencode_finished") is False and error and "Sandbox" in str(error))
        except (OSError, ValueError):
            out["sandbox_failed"] = False
        cache[key] = out
    return out


def campaign_rows(cache):
    rows = []
    recs = sorted(p.name for p in (GR / "probes" / "glmlocal").glob("*-rec*"))
    for name in CAMPAIGNS + recs:
        SLOTS.setdefault(name, 0)
        camp = GR / "probes" / "glmlocal" / name
        result = GR / "results" / f"glmlocal-{name}.json"
        row = {"campaign": name, "slots": SLOTS[name], "started": camp.exists(), "complete": result.exists()}
        if result.exists():
            data = json.loads(result.read_text())
            row.update(data.get("stats") or {})
        slots = sorted(glob.glob(str(camp / "models" / "glmlocal" / "attempts" / "*")))
        graded = passed = running = sandbox_failed = 0
        agg = collections.Counter()
        for slot in slots:
            attempts = sorted(glob.glob(slot + "/attempt-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
            if not attempts:
                continue
            stats = attempt_stats(Path(attempts[-1]), cache)
            for key in ("calls", "gen", "ge131k", "ge32k", "non200"):
                agg[key] += stats[key]
            agg["compacted"] += stats["compacted"]
            info = stats.get("terminal_info") or {}
            if stats["terminal"]:
                if stats.get("sandbox_failed"):
                    sandbox_failed += 1
                elif info.get("reward") is not None:
                    graded += 1
                    passed += (info.get("reward") or 0) >= 1.0
            else:
                running += 1
        row.update({"slot_dirs": len(slots), "graded": graded, "passed": passed, "running": running,
                    "sandbox_failed": sandbox_failed, **agg})
        rows.append(row)
    return rows


def router_totals(since_s=1800):
    now = time.time()
    gen_recent = 0
    errors = collections.Counter()
    for path in glob.glob(str(GR / "logs" / "router-gr-*.jsonl")):
        try:
            with open(path) as stream:
                stream.seek(max(0, os.path.getsize(path) - 50_000_000))
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("event") == "call":
                        if event.get("t", 0) >= now - since_s and event.get("status") == 200:
                            gen_recent += event.get("completion_tokens") or 0
                        if event.get("status") != 200:
                            errors[str(event.get("status"))] += 1
                    elif event.get("event") in ("call_failed", "backend_down"):
                        errors[event["event"]] += 1
        except OSError:
            continue
    return {"gen_tok_per_s_last_30min": round(gen_recent / since_s, 1), "router_errors": dict(errors)}


def main():
    import sys
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = campaign_rows(cache)
    tmp = CACHE.with_name(".progress-cache.tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, CACHE)
    keeper = json.loads((GR / "keeper-status.json").read_text()) if (GR / "keeper-status.json").exists() else {}
    router = router_totals()
    main_rows = [r for r in rows if r["campaign"] not in ("pilot", "baihe-pilot")]
    tot = collections.Counter()
    for r in main_rows:
        for key in ("slots", "graded", "passed", "running", "calls", "gen", "ge131k", "ge32k", "compacted", "non200",
                    "sandbox_failed"):
            tot[key] += r.get(key, 0)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    servers_left = ", ".join("%s:%sm" % (s["slot"][-7:], s["left_min"]) for s in keeper.get("running") or [])
    lines = [f"# glm-rollouts-sftv3 progress ({now})", "",
             "Self-hosted GLM-5.3 x 5 on the 220 SFT-v3 tasks (1,100 rollouts; 256k output limit, 262,144 context, "
             "OpenCode compaction at 242,144). Generated by progress.py.", "",
             (f"**Total:** graded {tot['graded']}/{tot['slots']} (+ {tot['sandbox_failed']} ended by sandbox failures, "
              f"re-run once in <campaign>-rec*), pass {tot['passed']} "
              + (f"({100 * tot['passed'] / tot['graded']:.1f}% of graded)" if tot["graded"] else "")
              + f"; running {tot['running']}; compacted rollouts {tot['compacted']}; replies >= 32k {tot['ge32k']}, "
              f">= 131k {tot['ge131k']}; LLM calls {tot['calls']:,}; generated {tot['gen']:,} tokens."), "",
             f"GPU servers (keeper): target {keeper.get('target')}, running {len(keeper.get('running') or [])} "
             f"({servers_left}), "
             f"pending {len(keeper.get('pending') or [])}; generation {router['gen_tok_per_s_last_30min']} tok/s "
             f"(last 30 min); router errors {router['router_errors'] or 0}.", "",
             "| campaign | slots | graded | pass | rate | running | sandbox-failed | compacted | calls | gen tokens | >=32k | non-200 | done |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rate = f"{100 * r['passed'] / r['graded']:.1f}%" if r.get("graded") else "-"
        lines.append(f"| {r['campaign']} | {r['slots']} | {r.get('graded', 0)} | {r.get('passed', 0)} | {rate} | "
                     f"{r.get('running', 0)} | {r.get('sandbox_failed', 0)} | {r.get('compacted', 0)} | {r.get('calls', 0):,} | {r.get('gen', 0):,} | "
                     f"{r.get('ge32k', 0)} | {r.get('non200', 0)} | {'yes' if r['complete'] else ''} |")
    text = "\n".join(lines) + "\n"
    (GR / ".PROGRESS.md.tmp").write_text(text)
    os.replace(GR / ".PROGRESS.md.tmp", GR / "PROGRESS.md")
    (GR / ".progress.json.tmp").write_text(json.dumps({"at": now, "rows": rows, "total": tot, "router": router,
                                                         "keeper": keeper}, indent=1))
    os.replace(GR / ".progress.json.tmp", GR / "progress.json")
    if "--stdout" in sys.argv:
        print(text)


if __name__ == "__main__":
    main()
