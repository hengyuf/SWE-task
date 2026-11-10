#!/usr/bin/env python3
"""Session-ending lens, step 2: derived per-rollout ending features (reads out/sessions/*.jsonl.gz only).

usage: python3 -I -B derive.py  -> out/derived.jsonl.gz (one row per rollout, all datasets / models)

Definitions (all on the root OpenCode session; a "call" = one model call = one assistant message that is not a
compaction summary; the tail window = the last 20 calls):
  outcome        resolved (reward >= 1) / failed_patch (reward < 1, non-empty patch) / empty (model_patch_bytes == 0)
  deadline_kill  DeepSWE: classifier N1_TIMEOUT_3H with wall >= 10,700 s. train-acc: not finished, not N4, and
                 (agent error TimeoutError or wall >= 10,500 s) = the train-acc REPORT's deadline-kill rule
  self_kill      not finished, wall < 10,500 s, and a kill / pkill / killall command in the last 2 calls or the
                 running tool
  runaway call   finish == 'length' (DeepSWE: the 131,072 cap or the context window; train-acc: the router's clamp);
                 loop = its last 20,000 chars zlib-compress to < 0.10 (thinkx.py)
  tail counts    endlib.cat_call categories of the tool calls in the tail window, edit errors, OpenCode bash timeouts,
                 test runs (count, distinct commands, max repeats of one normalized command, runs whose output has
                 pass markers and no fail markers), kill / sleep / background commands, no-tool calls
  tail time      first tail call created -> session end; model time = created -> first part (the server answers in
                 one piece), tool time = first part -> completed; runaway time = model time of length-cut calls
  in-flight      for unfinished sessions: the running tool (command, seconds running at the end) or the model call
                 in flight (seconds since created) or a compaction in flight
  compactions    count, broken summaries (summary has tool-call markup or tool parts, or no 'Goal' section, or is
                 empty / think-only), first call after a compaction / after the synthetic 'Continue if you have next
                 steps...' prompt
  final turn     (finished sessions) tools or not, finish, visible text flags (endlib: claims done, mentions commit,
                 claims a commit (SHA / 'committed'), defers commit, asks user, announced next action at the end,
                 status summary), reasoning-only, whether it is the first call after the continue prompt
  activity       primary label of the tail window, first match: runaway (>= 1 length-cut call) > kill (>= 1 kill command)
                 > waiting (>= 2 bash timeouts, or a tool running >= 300 s at the end, or one call's tool time >= 600 s)
                 > retest (>= 8 test runs and <= 2 edits, or one test command repeated >= 4 times with <= 2 edits)
                 > debug (test+run >= 5 and edits >= 3) > editing (edits >= 8 or edit errors >= 3) > exploring
                 (reads >= 12) > wrapup (>= 1 commit attempt or >= 4 git calls) > mixed
"""
import gzip
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import endlib as L  # noqa: E402

IN = HERE / "out/sessions"
OUT = HERE / "out/derived.jsonl.gz"
TAIL = 20


def secs(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a:
        return (b - a) / 1000.0
    return None


def derive(r):
    s = r.get("s") or None
    f = r.get("f") or {}
    wall = r.get("wall_s") or 0
    d = {k: r.get(k) for k in ("dataset", "model", "task_id", "ri", "reward", "code", "subcode", "wall_s", "shard")}
    d["key"] = f"{r['task_id']}#{r['ri']}"
    patch = (r.get("model_patch_bytes") or 0) > 0
    d["outcome"] = "resolved" if (r.get("reward") or 0) >= 1 else ("failed_patch" if patch else "empty")
    d["finished"] = bool(r.get("opencode_finished"))
    d["n_compactions"] = f.get("n_compactions")
    d["n_commits"] = f.get("n_commits") or 0
    d["commit_attempts"] = f.get("commit_attempts") or 0
    d["edited"] = bool(f.get("edited"))
    d["edits_app"] = f.get("edit_ok_app") or 0
    d["last_edit_rel_s"] = f.get("last_edit_rel_s")
    d["has_export"] = s is not None
    if r["dataset"] == "deepswe":
        d["deadline_kill"] = r.get("code") == "N1_TIMEOUT_3H" and wall >= 10700
    else:
        d["deadline_kill"] = (not d["finished"]) and r.get("code") != "N4_SANDBOX_LOST" and (
            "TimeoutError" in str(f.get("last_error_msg") or "") or wall >= 10500)
    if not s:
        d["activity"] = "no_export"
        return d
    turns = s["turns"]
    comps = [c for c in s["comps"] if "nturn" in c and "user_msg" not in c]
    d["n_calls"] = len(turns)
    end_t = s.get("last_t")
    # an unfinished session ends at the kill: the export has no timestamp for it, so use session start + wall time
    # (classifier wall_s = session start -> Gym response; 10,802-10,822 s for the 3 h kills)
    if not d["finished"] and s.get("first_t") and wall:
        end_t = max(end_t or 0, s["first_t"] + int(wall * 1000))
    d["end_from_wall"] = bool(not d["finished"] and s.get("first_t") and wall)
    # ---- runaway / loops (whole session)
    run_turns = [t for t in turns if t.get("fin") == "length"]
    d["runaway_calls"] = len(run_turns)
    d["loop_calls"] = sum(1 for t in run_turns if (t.get("z") if t.get("z") is not None else 1) < 0.10)
    d["runaway_s"] = round(sum((secs(t["tc"], t.get("tp") or t.get("td")) or 0) for t in run_turns))
    d["model_s"] = round(sum((secs(t["tc"], t.get("tp") or t.get("td")) or 0) for t in turns))
    d["tool_s"] = round(sum((secs(t.get("tp"), t.get("td")) or 0) for t in turns if t.get("tp")))
    d["summary_s"] = round(sum((secs(c.get("s_tc"), c.get("s_td")) or 0) for c in comps))
    # runaway calls split: at the 131,072-token cap vs cut earlier by the context window (prompt + output > window);
    # 0-token 'length' turns are rejected requests (context overflow), not runaways
    cap = [t for t in run_turns if (t.get("out") or 0) >= 131072]
    ctx = [t for t in run_turns if 0 < (t.get("out") or 0) < 131072]
    isloop = lambda t: (t.get("z") if t.get("z") is not None else 1) < 0.10
    d["runaway_cap_calls"], d["runaway_cap_loops"] = len(cap), sum(1 for t in cap if isloop(t))
    d["runaway_ctx_calls"], d["runaway_ctx_loops"] = len(ctx), sum(1 for t in ctx if isloop(t))
    # ---- compactions
    broken = 0
    kinds = Counter()
    for c in comps:
        if "si" not in c:
            kinds["no_summary_msg"] += 1
            continue
        if c.get("s_toolmarkup") or c.get("s_tools"):
            kinds["toolcall"] += 1
            broken += 1
        elif c.get("s_think_only") or not c.get("s_len"):
            kinds["empty_or_think_only"] += 1
            broken += 1
        elif not c.get("s_goal"):
            kinds["no_goal"] += 1
            broken += 1
        else:
            kinds["ok"] += 1
        if c.get("s_fin") == "length":
            kinds["summary_length_cut"] += 1
    d["comp_n"] = len(comps)
    d["summ_instr_n"] = sum(1 for c in comps if c.get("s_commit_instr"))
    d["summ_commit_n"] = sum(1 for c in comps if c.get("s_commit"))
    d["last_summary_instr"] = bool(comps and comps[-1].get("s_commit_instr"))
    d["last_summary_commit"] = bool(comps and comps[-1].get("s_commit"))
    d["calls_after_last_comp"] = (len(turns) - comps[-1]["nturn"]) if comps else None
    # how the task's commit instruction left the compacted context: the first summary without it, by kind
    summ = [c for c in comps if "si" in c]
    d["instr_first_loss"] = None
    if summ:
        lost = next((i for i, c in enumerate(summ) if not c.get("s_commit_instr")), None)
        if lost is None:
            d["instr_first_loss"] = "never"
        else:
            c = summ[lost]
            d["instr_first_loss"] = ("toolcall" if (c.get("s_toolmarkup") or c.get("s_tools")) else
                                     "runaway" if c.get("s_fin") == "length" else "empty" if not c.get("s_len") else "omitted")
            later = summ[lost + 1:]
            d["instr_recovered"] = any(x.get("s_commit_instr") for x in later) if later else None
    # session-wide command flags
    kc = bk = tm = bc = 0
    for t in turns:
        for c in t["calls"]:
            if c["tool"] != "bash":
                continue
            bc += 1
            tm += bool(c.get("tmo"))
            if c.get("kill"):
                kc += 1
                bk += bool(L.RE_KILL_BROAD.search(c.get("cmd") or ""))
    d["kill_calls"], d["broad_kill_calls"], d["tmo_calls"], d["bash_calls"] = kc, bk, tm, bc
    d["kill_cmds"] = [c.get("cmd")[:200] for t in turns for c in t["calls"] if c["tool"] == "bash" and c.get("kill")][:6]
    d["comp_kinds"] = dict(kinds)
    d["comp_broken"] = broken
    d["comp_overflow"] = sum(1 for c in comps if c.get("overflow"))
    # what the first call after each compaction did
    after = Counter()
    for c in comps:
        k = c["nturn"]
        if k < len(turns):
            t = turns[k]
            after["stop" if not t["calls"] else "call"] += 1
            if t.get("fin") == "length":
                after["length"] += 1
    d["after_comp"] = dict(after)
    # ---- commit timing: calls between the first successful commit and the session end
    cturns = [i for i, t in enumerate(turns) for c in t["calls"] if c["tool"] == "bash" and c.get("commit_ok")]
    d["first_commit_calls_before_end"] = (len(turns) - 1 - cturns[0]) if cturns else None
    # ---- re-reads after compaction: read-tool calls in the 30 calls after each compaction whose file was read before it
    pr = prr = 0
    for c in comps:
        k = c["nturn"]
        before = {x.get("fp") for t in turns[:k] for x in t["calls"] if x["tool"] == "read" and x.get("fp")}
        after = [x.get("fp") for t in turns[k:k + 30] for x in t["calls"] if x["tool"] == "read" and x.get("fp")]
        pr += len(after)
        prr += sum(1 for x in after if x in before)
    d["post_comp_reads"], d["post_comp_rereads"] = pr, prr
    # ---- tail window
    tail = turns[-TAIL:]
    cats = Counter()
    test_cmds = Counter()
    test_pass = test_fail = edit_err = tmo = kill = sleep = bg = 0
    max_tool_s = 0
    for t in tail:
        for c in t["calls"]:
            cat = L.cat_call(c["tool"], c.get("cmd"))
            cats[cat] += 1
            if cat == "edit" and c.get("st") == "error":
                edit_err += 1
            if c["tool"] == "bash":
                tmo += bool(c.get("tmo"))
                kill += bool(c.get("kill"))
                sleep += bool(c.get("sleep"))
                bg += bool(c.get("bg"))
                if cat == "test":
                    test_cmds[L.norm_cmd(c.get("cmd"))] += 1
                    if c.get("npass") and not c.get("nfail"):
                        test_pass += 1
                    elif c.get("nfail"):
                        test_fail += 1
        ts = secs(t.get("tp"), t.get("td"))
        if ts and ts > max_tool_s:
            max_tool_s = ts
    d["tail_n"] = len(tail)
    d["tail_cats"] = dict(cats)
    d["tail_calls"] = sum(cats.values())
    d["tail_test"] = sum(test_cmds.values())
    d["tail_test_distinct"] = len(test_cmds)
    d["tail_test_maxrep"] = max(test_cmds.values()) if test_cmds else 0
    d["tail_test_pass"] = test_pass
    d["tail_test_fail"] = test_fail
    d["tail_edit_err"] = edit_err
    d["tail_tmo"] = tmo
    d["tail_kill"] = kill
    d["tail_sleep"] = sleep
    d["tail_bg"] = bg
    d["tail_notool"] = sum(1 for t in tail if not t["calls"])
    d["tail_runaway"] = sum(1 for t in tail if t.get("fin") == "length")
    d["tail_loop"] = sum(1 for t in tail if t.get("fin") == "length" and (t.get("z") if t.get("z") is not None else 1) < 0.10)
    d["tail_max_tool_s"] = round(max_tool_s)
    if tail:
        t0 = tail[0]["tc"]
        d["tail_span_s"] = round(secs(t0, end_t) or 0)
        d["tail_model_s"] = round(sum((secs(t["tc"], t.get("tp") or t.get("td")) or 0) for t in tail))
        d["tail_tool_s"] = round(sum((secs(t.get("tp"), t.get("td")) or 0) for t in tail if t.get("tp")))
        d["tail_runaway_s"] = round(sum((secs(t["tc"], t.get("tp") or t.get("td")) or 0) for t in tail if t.get("fin") == "length"))
        d["tail_out_tok"] = sum((t.get("out") or 0) for t in tail)
    # ---- the last 30 minutes of the session (by call creation time)
    if end_t:
        w0 = end_t - 1800 * 1000
        last30 = [t for t in turns if (t.get("tc") or 0) >= w0]
        c30 = Counter(L.cat_call(c["tool"], c.get("cmd")) for t in last30 for c in t["calls"])
        d["l30_calls"] = len(last30)
        d["l30_cats"] = dict(c30)
        d["l30_runaway_s"] = round(sum((secs(max(t["tc"], w0), t.get("tp") or t.get("td")) or 0) for t in turns
                                       if t.get("fin") == "length" and (t.get("tp") or t.get("td") or 0) >= w0))
        d["l30_model_s"] = round(sum((secs(max(t["tc"], w0), t.get("tp") or t.get("td")) or 0) for t in turns
                                     if (t.get("tp") or t.get("td") or 0) >= w0))
        d["l30_comp"] = sum(1 for c in comps if (c.get("t") or 0) >= w0)
    # ---- in-flight state at the end (unfinished sessions)
    inflight = None
    if not d["finished"]:
        if s.get("running"):
            run = s["running"][-1]
            inflight = {"kind": "tool", "tool": run.get("tool"), "cmd": run.get("cmd"),
                        "s": round(secs(run.get("t0"), end_t) or 0) if run.get("t0") else None,
                        "cat": L.cat_call(run.get("tool"), run.get("cmd"))}
        elif s.get("last_is_summary") or (s["comps"] and s["comps"][-1].get("nturn") == len(turns) and "s_td" not in s["comps"][-1]):
            inflight = {"kind": "compaction"}
        elif turns and not turns[-1]["calls"] and (not turns[-1].get("td") or turns[-1].get("err") or not turns[-1].get("fin")):
            # DeepSWE: the in-flight call has no completion time; train-acc: the deadline aborts it (error, no finish)
            inflight = {"kind": "model", "s": round(secs(turns[-1]["tc"], end_t) or 0), "err": turns[-1].get("err")}
        elif f.get("last_running_tool"):
            inflight = {"kind": "tool", "tool": f.get("last_running_tool"), "cmd": f.get("last_running_cmd"),
                        "cat": L.cat_call(f.get("last_running_tool"), f.get("last_running_cmd"))}
        else:
            inflight = {"kind": "between"}
    d["inflight"] = inflight
    # ---- self kill
    lastcmds = [c.get("cmd") or "" for t in turns[-2:] for c in t["calls"] if c["tool"] == "bash"]
    if inflight and inflight.get("kind") == "tool":
        lastcmds.append(inflight.get("cmd") or "")
    d["self_kill"] = (not d["finished"]) and wall < 10500 and any(L.RE_KILL.search(L.strip_cmd(c)) for c in lastcmds)
    # ---- final turn (finished sessions; the last non-summary call)
    if turns:
        ft = turns[-1]
        vis = (ft.get("vhead") or "") + ("\n" + ft["vtail"] if ft.get("vtail") else "")
        d["final"] = {
            "tools": len(ft["calls"]), "fin": ft.get("fin"), "out": ft.get("out"), "err": ft.get("err"),
            "vl": ft.get("vl"), "rl": ft.get("rl"), "after_continue": bool(ft.get("ac")), "after_comp": bool(ft.get("acomp")),
            "claims_done": bool(L.RE_CLAIM_DONE.search(vis)), "mentions_commit": bool(L.RE_MENTIONS_COMMIT.search(vis)),
            "claims_committed": bool(L.RE_CLAIM_COMMITTED.search(vis)), "defers_commit": bool(L.RE_DEFERS_COMMIT.search(vis)),
            "asks_user": bool(L.RE_ASKS_USER.search(vis[-300:])),
            "intent_end": bool(L.RE_INTENT_END.search((vis or ft.get("rtail") or "")[-300:])),
            "status_summary": bool(L.RE_STATUS_SUMMARY.search(vis)), "markup": ft.get("markup"), "vmarkup": ft.get("vmarkup"),
            "unc": ft.get("unc"), "z": ft.get("z"), "flags": ft.get("flags") or {},
        }
        d["final_text"] = {"vhead": (ft.get("vhead") or "")[:300], "vtail": (ft.get("vtail") or "")[-300:],
                           "rtail": (ft.get("rtail") or "")[-300:]}
    # ---- loop escape then stop: the final call (finished, no tool call, not length-cut) is itself repetitive
    # (>= 4,096 output tokens and zlib < 0.2) or its text says it is breaking out of a loop
    if d["finished"] and turns and not turns[-1]["calls"] and turns[-1].get("fin") != "length":
        ft = turns[-1]
        rep = (ft.get("out") or 0) >= 4096 and (ft.get("z") if ft.get("z") is not None else 1) < 0.2
        brk = bool(L.RE_LOOP_BREAK.search(ft.get("rtail") or "")) or bool(L.RE_LOOP_BREAK.search(ft.get("vhead") or ""))
        d["final_loop_escape"] = rep or brk
    # ---- primary activity label of the tail window
    edits = cats.get("edit", 0)
    verify = cats.get("test", 0) + cats.get("run", 0)
    running_long = bool(inflight and inflight.get("kind") == "tool" and (inflight.get("s") or 0) >= 300)
    if d["tail_runaway"]:
        act = "runaway"
    elif kill:
        act = "kill"
    elif tmo >= 2 or running_long or max_tool_s >= 600:
        act = "waiting"
    elif (d["tail_test"] >= 8 and edits <= 2) or (d["tail_test_maxrep"] >= 4 and edits <= 2):
        act = "retest"
    elif verify >= 5 and edits >= 3:
        act = "debug"
    elif edits >= 8 or edit_err >= 3:
        act = "editing"
    elif cats.get("read", 0) >= 12:
        act = "exploring"
    elif cats.get("commit", 0) or cats.get("git", 0) >= 4:
        act = "wrapup"
    else:
        act = "mixed"
    d["activity"] = act
    # time since the last successful edit at the end
    if d["last_edit_rel_s"] is not None and wall:
        d["since_last_edit_s"] = round(wall - d["last_edit_rel_s"])
    return d


def main():
    n = 0
    tmp = OUT.with_name(f".derived.tmp{os.getpid()}.gz")
    with gzip.open(tmp, "wt") as handle:
        for path in sorted(IN.glob("*.jsonl.gz")):
            with gzip.open(path, "rt") as src:
                for line in src:
                    handle.write(json.dumps(derive(json.loads(line)), separators=(",", ":")) + "\n")
                    n += 1
    os.replace(tmp, OUT)
    print(n, "rollouts ->", OUT)


if __name__ == "__main__":
    main()
