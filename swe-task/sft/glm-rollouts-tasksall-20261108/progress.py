#!/usr/bin/env python3
"""glm-rollouts-tasksall progress: rewrites PROGRESS.md (and progress.json) from the campaign dirs, the results files,
the router logs and the keeper status. Read-only on everything else. (Derived from glm-rollouts-sftv3's progress.py.)

    /usr/bin/python3 -I -B progress.py [--stdout]

Per campaign (tasks.json order) and per tier: slots (tasks x repeats), terminal attempts graded pass / fail, running,
pass rate; rollouts that compacted (an OpenCode summary request in http.jsonl); LLM calls, generated tokens and replies
at >= 131,072 / >= 32,768 completion tokens. Plus the GPU servers and campaign scheduling (keeper status) and the
generation throughput of the last 30 min (router logs).
"""
import collections
import glob
import json
import os
from pathlib import Path
import time

GR = Path(__file__).resolve().parent
PLAN = json.loads((GR / "tasks.json").read_text())
CAMPAIGNS = [c["name"] for c in PLAN["campaigns"]]
SLOTS = {c["name"]: c["slots"] for c in PLAN["campaigns"]}
TIER = {c["name"]: c["tier"] for c in PLAN["campaigns"]}
CACHE = GR / "progress-cache.json"


ZERO = {"calls": 0, "gen": 0, "ge131k": 0, "ge32k": 0, "non200": 0, "compacted": False, "max_prompt": 0}


def attempt_stats(attempt: Path, cache: dict) -> dict:
    """Calls, generated tokens, long replies and compaction of one attempt. http.jsonl is append-only, so a running
    attempt is parsed incrementally (byte offset + partial aggregates in the cache); a terminal one is finished once
    and then served from the cache. The grade comes from terminal.json (ATP writes native-result.json later)."""
    key = str(attempt)
    entry = cache.get(key)
    if entry and entry.get("done"):
        return entry
    entry = entry if entry and "offset" in entry else dict(ZERO, offset=0, ino=None)
    terminal = (attempt / "terminal.json").exists()
    path = attempt / "http.jsonl"
    try:
        st = path.stat()
        if st.st_ino != entry.get("ino") or st.st_size < entry["offset"]:
            entry.update(ZERO, offset=0, ino=st.st_ino)
        with open(path, "rb") as stream:
            stream.seek(entry["offset"])
            for raw in stream:
                if not raw.endswith(b"\n"):
                    break  # a line still being written: next run
                entry["offset"] += len(raw)
                head = raw[:1500]
                if b'"event":"request"' in head or b'"event": "request"' in head:
                    # OpenCode 1.17.11's compaction call (core/session/compaction.ts buildPrompt):
                    # "Create a new anchored summary ..." / "Update the anchored summary below ..."
                    if b"anchored summary" in raw:
                        entry["compacted"] = True
                    continue
                if b'"event":"response"' not in head and b'"event": "response"' not in head:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if event.get("status") != 200:
                    entry["non200"] += 1
                    continue
                try:
                    usage = (json.loads(event.get("body") or "{}").get("usage")) or {}
                except ValueError:
                    continue
                completion = usage.get("completion_tokens") or 0
                entry["calls"] += 1
                entry["gen"] += completion
                entry["max_prompt"] = max(entry["max_prompt"], usage.get("prompt_tokens") or 0)
                entry["ge131k"] += completion >= 131072
                entry["ge32k"] += completion >= 32768
    except OSError:
        pass
    entry["terminal"] = terminal
    if terminal:
        try:
            info = json.loads((attempt / "terminal.json").read_text())
        except (OSError, ValueError):
            info = None  # being written: classify on the next run
        if info is not None:
            error = str(info.get("opencode_run_error_type") or "")
            finished = info.get("opencode_finished")
            entry["sandbox_failed"] = finished is False and "Sandbox" in error
            # an agent timeout (TimeoutError) is a graded fail for ATP; other unfinished runs end without a grade
            entry["graded"] = bool(info.get("evaluation_completed")) and (finished is not False
                                                                          or error in ("timeout", "TimeoutError"))
            entry["reward"] = info.get("reward")
            entry["done"] = True
    cache[key] = entry
    return entry


def campaign_rows(cache, keeper):
    rows = []
    parked = {p["name"] for p in ((keeper.get("campaigns") or {}).get("parked") or [])}
    recs = sorted(p.name for p in (GR / "probes" / "glmlocal").glob("*-rec*"))
    for name in CAMPAIGNS + recs:
        base = name.split("-rec")[0]
        camp = GR / "probes" / "glmlocal" / name
        result = GR / "results" / f"glmlocal-{base}.json"
        row = {"campaign": name, "tier": TIER.get(base), "slots": SLOTS.get(name, 0),
               "released": (GR / f"GO-{base}").exists(), "started": camp.exists(), "complete": result.exists(),
               "parked": base in parked}
        if result.exists() and name == base:
            try:
                row["results"] = json.loads(result.read_text()).get("stats") or {}
            except (OSError, ValueError):
                pass
        if not camp.exists():
            rows.append(row)
            continue
        slots = sorted(glob.glob(str(camp / "models" / "glmlocal" / "attempts" / "*")))
        graded = passed = running = sandbox_failed = ungraded = 0
        agg = collections.Counter()
        for slot in slots:
            attempts = sorted(glob.glob(slot + "/attempt-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
            if not attempts:
                continue
            stats = attempt_stats(Path(attempts[-1]), cache)
            for key in ("calls", "gen", "ge131k", "ge32k", "non200"):
                agg[key] += stats[key]
            agg["compacted"] += stats["compacted"]
            if not stats.get("done"):
                running += 1
            elif stats.get("sandbox_failed"):
                sandbox_failed += 1
            elif stats.get("graded"):
                graded += 1
                passed += (stats.get("reward") or 0) >= 1.0
            else:
                ungraded += 1
        row.update({"slot_dirs": len(slots), "graded": graded, "passed": passed, "running": running,
                    "sandbox_failed": sandbox_failed, "ungraded": ungraded, **agg})
        rows.append(row)
    return rows


def router_totals(since_s=1800):
    now = time.time()
    gen_recent = 0
    errors = collections.Counter()
    for path in glob.glob(str(GR / "logs" / "router-gta-*.jsonl")):
        try:
            if os.path.getmtime(path) < now - since_s:
                continue
            with open(path) as stream:
                stream.seek(max(0, os.path.getsize(path) - 20_000_000))
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("event") == "call":
                        if event.get("t", 0) >= now - since_s and event.get("status") == 200:
                            gen_recent += event.get("completion_tokens") or 0
                        if event.get("status") != 200 and event.get("t", 0) >= now - since_s:
                            errors[str(event.get("status"))] += 1
                    elif event.get("event") in ("call_failed", "backend_down") and event.get("t", now) >= now - since_s:
                        errors[event["event"]] += 1
        except OSError:
            continue
    return {"gen_tok_per_s_last_30min": round(gen_recent / since_s, 1), "router_errors_last_30min": dict(errors)}


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    import sys
    try:
        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    except (OSError, ValueError):
        cache = {}  # unreadable cache: rebuild
    try:
        keeper = json.loads((GR / "keeper-status.json").read_text()) if (GR / "keeper-status.json").exists() else {}
    except (OSError, ValueError):
        keeper = {}
    rows = campaign_rows(cache, keeper)
    write_atomic(CACHE, json.dumps(cache))
    router = router_totals()
    keys = ("slots", "graded", "passed", "running", "calls", "gen", "ge131k", "ge32k", "compacted", "non200",
            "sandbox_failed", "ungraded")
    tot = collections.Counter()
    tiers = collections.defaultdict(collections.Counter)
    for r in rows:
        for key in keys:
            tot[key] += r.get(key, 0)
            tiers[r["tier"]][key] += r.get(key, 0)
        tiers[r["tier"]]["campaigns"] += "-rec" not in r["campaign"]
        tiers[r["tier"]]["complete"] += r["complete"] and "-rec" not in r["campaign"]
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    servers_left = ", ".join("%s:%sm" % (s["slot"], s["left_min"]) for s in keeper.get("running") or [])
    sched = keeper.get("campaigns") or {}
    pct = lambda p, n: f"{100 * p / n:.1f}%" if n else "-"
    lines = [f"# glm-rollouts-tasksall progress ({now})", "",
             f"Self-hosted GLM-5.3 x 5 on {PLAN['summary']['tasks']} tasks of sft/tasks-all-20261106 without GLM-5.3 "
             f"rollouts ({PLAN['summary']['slots']:,} rollouts; 256k output limit, 262,144 context, OpenCode compaction "
             f"at 242,144), priority tiers 1 accepted/strong > 2 partial > 3 weak > 4 none/unlabeled. Generated by "
             f"progress.py.", "",
             (f"**Total:** graded {tot['graded']:,}/{tot['slots']:,} (+ {tot['sandbox_failed']} ended by sandbox "
              f"failures, re-run once in <campaign>-rec*; {tot['ungraded']} without a grade), pass {tot['passed']:,} "
              f"({pct(tot['passed'], tot['graded'])} of "
              f"graded); running {tot['running']}; compacted rollouts {tot['compacted']:,}; replies >= 32k "
              f"{tot['ge32k']}, >= 131k {tot['ge131k']}; LLM calls {tot['calls']:,}; generated {tot['gen']:,} tokens."),
             "",
             f"Campaigns: {sched.get('done', 0)}/{sched.get('campaigns', len(CAMPAIGNS))} done, released "
             f"{sched.get('released', 0)}, active {len(sched.get('active') or [])} (load {sched.get('load')}, by owner "
             f"{sched.get('owner_load')}), next {sched.get('next')}; PARKED (need a human): "
             f"{[p['name'] for p in sched.get('parked') or []] or 'none'}.", "",
             f"GPU servers (keeper): target {keeper.get('target')}, ready {keeper.get('ready_servers')}, running "
             f"{len(keeper.get('running') or [])} ({servers_left}), pending {len(keeper.get('pending') or [])}; "
             f"generation {router['gen_tok_per_s_last_30min']} tok/s (last 30 min); router errors (30 min) "
             f"{router['router_errors_last_30min'] or 0}.", "",
             "| tier | campaigns done | slots | graded | pass | rate | running | sandbox-failed | compacted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tier in sorted(t for t in tiers if t is not None):
        t = tiers[tier]
        lines.append(f"| {tier} | {t['complete']}/{t['campaigns']} | {t['slots']:,} | {t['graded']:,} | {t['passed']:,} | "
                     f"{pct(t['passed'], t['graded'])} | {t['running']} | {t['sandbox_failed']} | {t['compacted']:,} |")
    lines += ["", "| campaign | tier | slots | graded | pass | rate | running | sandbox-failed | no grade | compacted | calls | gen tokens | >=32k | non-200 | state | results (pass/fail/infra) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not (r["released"] or r["started"] or r["complete"]):
            continue
        state = ("done" if r["complete"] else "PARKED" if r["parked"] else
                 "running" if r["started"] else "released")
        res = r.get("results") or {}
        lines.append(f"| {r['campaign']} | {r['tier']} | {r['slots']} | {r.get('graded', 0)} | {r.get('passed', 0)} | "
                     f"{pct(r.get('passed', 0), r.get('graded', 0))} | {r.get('running', 0)} | {r.get('sandbox_failed', 0)} | "
                     f"{r.get('ungraded', 0)} | {r.get('compacted', 0)} | {r.get('calls', 0):,} | {r.get('gen', 0):,} | "
                     f"{r.get('ge32k', 0)} | {r.get('non200', 0)} | {state} | "
                     f"{'/'.join(str(res.get(k, '')) for k in ('pass', 'fail', 'infra')) if res else ''} |")
    text = "\n".join(lines) + "\n"
    write_atomic(GR / "PROGRESS.md", text)
    write_atomic(GR / "progress.json", json.dumps({"at": now, "rows": rows, "total": tot,
                                                    "tiers": {str(k): v for k, v in tiers.items()},
                                                    "router": router, "keeper": keeper}, indent=1))
    if "--stdout" in sys.argv:
        print(text)


if __name__ == "__main__":
    main()
