#!/usr/bin/env python3
"""verify-2: per-model metrics from out/<model>.jsonl.gz (extract_calls.py) + the counting-parser log lines of
fleets/<model>/logs/vllm-*.log. Own definitions (documented inline). Writes out/metrics_<model>.json and
out/slots_<model>.jsonl (one compact row per slot) and prints a summary.

    python3 -I -B metrics.py <model> [<model> ...]
"""
import gzip
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
RE_PARSER = re.compile(r"(TA_THINK_RECOVERED|TA_THINK_UNCLOSED_NO_TOOL) request_id=([0-9a-f]+) (?:reasoning_chars|chars)=(\d+)")
CAP = 131072


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def parser_lines(model):
    out = {}
    for f in sorted((TA / "fleets" / model / "logs").glob("vllm-*.log")):
        for line in open(f, errors="replace"):
            if "TA_THINK_" in line and "request_id=" in line:
                m = RE_PARSER.search(line)
                if m:
                    out[m.group(2)] = (m.group(1), int(m.group(3)), f.name)
    return out


def load(model):
    with gzip.open(HERE / "out" / f"{model}.jsonl.gz", "rt") as fh:
        return [json.loads(x) for x in fh]


def analyze(model):
    slots = load(model)
    plog = parser_lines(model)
    matched = set()
    M = Counter()
    per_slot = []
    ct_all = []
    lat_all = []
    for s in slots:
        calls = s["calls"]
        # session of each call: main = the system prompt of the first request; a compaction belongs to the session of
        # the preceding non-compaction request (subagent sessions of the task tool have their own system prompt)
        # main session = requests whose first user message is the task prompt (first request's) or the compacted-session
        # opener "What did we do so far?" (the system prompt can change within the main session, e.g. sc14)
        # main session: requests whose first user message is the task prompt; subagent (task tool) sessions start with
        # their own prompt and have their own system prompt; compacted sessions open with "What did we do so far?" (or
        # "Continue if ..." in update-compaction requests) -> main unless their system prompt is a subagent one. The main
        # system prompt can change within a session (sc14), so unknown system prompts count as main. A compaction belongs
        # to the session of the preceding non-compaction request.
        task_fu = (calls[0].get("fu") or "") if calls else ""
        openers = ("What did we do so far", "Continue if you have next steps")
        main_h = {c.get("sys") for c in calls if not c.get("comp") and (c.get("fu") or "") == task_fu}
        sub_h = {c.get("sys") for c in calls if not c.get("comp") and (c.get("fu") or "") != task_fu
                 and not (c.get("fu") or "").startswith(openers)} - main_h
        prev = True
        for c in calls:
            fu = c.get("fu") or ""
            if c.get("comp"):
                c["_main"] = prev
                continue
            if fu == task_fu:
                prev = True
            elif fu.startswith(openers):
                prev = c.get("sys") not in sub_h
            else:
                prev = False
            c["_main"] = prev
        resp_calls = [c for c in calls if c.get("resp")]
        main = [c for c in resp_calls if not c.get("comp")]
        comps = [c for c in calls if c.get("comp") and c.get("_main")]
        sub_comps = [c for c in calls if c.get("comp") and not c.get("_main")]
        tools = set(s.get("tools") or [])
        r = {"task": s["task"], "slot": s["slot"], "rep": int(s["slot"].rsplit("-r", 1)[1]), "status": s["status"],
             "empty": not s.get("patch_chars"), "agent_error": s.get("agent_error"), "attempt": s.get("attempt"),
             "n_attempt_dirs": s.get("n_attempt_dirs"), "budget_timeout": s.get("budget_timeout")}
        term = s.get("term") or {}
        t0 = min((c["t0"] for c in calls if c.get("t0")), default=None)
        r["finished"] = term.get("opencode_finished")
        r["wall_s"] = (term["resp_created_at"] - t0) if (term.get("resp_created_at") and t0) else None
        r["wall_from_start_s"] = (term["resp_created_at"] - s["started_at"]) if (term.get("resp_created_at") and s.get("started_at")) else None
        r["stderr_tail"] = term.get("stderr_tail")
        r["n_calls"] = len(calls)
        r["n_calls_200"] = len(resp_calls)
        r["n_non200_final"] = sum(1 for c in calls if c["statuses"] and c["statuses"][-1] != 200)
        r["n_no_resp"] = sum(1 for c in calls if not c["statuses"])
        r["n_http_retries"] = sum(max(0, c.get("n_req", 0) - 1) for c in calls)
        # action turns (main-agent calls with >= 1 tool call); unreasoned = stripped reasoning empty
        acts = [c for c in main if c["resp"]["ntc"] > 0]
        r["actions_main_session"] = sum(1 for c in acts if c.get("_main"))
        r["unreasoned_main_session"] = sum(1 for c in acts if c.get("_main") and c["resp"]["rs"] == 0)
        r["n_sub_comp"] = len(sub_comps)
        r["n_sub_calls"] = sum(1 for c in calls if not c.get("_main"))
        r["actions"] = len(acts)
        r["unreasoned"] = sum(1 for c in acts if c["resp"]["rs"] == 0)
        r["unreasoned_ids"] = [c["resp"]["rid"] for c in acts if c["resp"]["rs"] == 0]
        # parser join
        pk = Counter()
        for c in resp_calls:
            rid = (c["resp"]["rid"] or "").replace("chatcmpl-", "")
            if rid in plog:
                kind, chars, _ = plog[rid]
                matched.add(rid)
                if kind == "TA_THINK_RECOVERED":
                    pk["recovered"] += 1
                    pk["recovered_bare"] += chars == 0
                else:
                    pk["unclosed_" + ("length" if c["resp"]["fin"] == "length" else "eos")] += 1
                    pk["unclosed_eos_headless"] += c["resp"]["fin"] != "length" and c["resp"]["rmark"]
                c["_plog"] = kind
        r["parser"] = dict(pk)
        # empty output / markup / special tokens (all calls with a 200 response, compaction included)
        r["empty_outputs"] = sum(1 for c in resp_calls if c["resp"]["rs"] == 0 and c["resp"]["cs"] == 0 and c["resp"]["ntc"] == 0)
        r["rmark_turns"] = sum(1 for c in resp_calls if c["resp"]["rmark"])
        r["rmark_headless"] = sum(1 for c in resp_calls if c["resp"]["rmark"] and c["resp"]["ntc"] == 0)
        r["rspec_turns"] = sum(1 for c in resp_calls if c["resp"]["rspec"])
        # length cuts
        lc = [c for c in resp_calls if c["resp"]["fin"] == "length"]
        r["length_cuts"] = len(lc)
        r["length_cuts_main"] = sum(1 for c in lc if not c.get("comp"))
        r["length_cuts_think"] = sum(1 for c in lc if c.get("_plog") == "TA_THINK_UNCLOSED_NO_TOOL")
        r["length_loop"] = sum(1 for c in lc if c.get("_plog") == "TA_THINK_UNCLOSED_NO_TOOL" and (c["resp"].get("z") or 1) < 0.15)
        r["full_cap"] = sum(1 for c in resp_calls if (c["resp"]["ct"] or 0) >= CAP)
        r["full_cap_think"] = sum(1 for c in lc if c.get("_plog") == "TA_THINK_UNCLOSED_NO_TOOL" and (c["resp"]["ct"] or 0) >= CAP)
        r["length_tails"] = [(c["resp"]["ct"], c["resp"].get("z"), (c["resp"].get("tail") or "")[-80:], bool(c.get("comp")))
                             for c in lc]
        last = resp_calls[-1] if resp_calls else None
        r["last_fin"] = last["resp"]["fin"] if last else None
        r["last_comp"] = bool(last.get("comp")) if last else None
        r["last_ct"] = last["resp"]["ct"] if last else None
        r["last_call_inflight"] = bool(calls and not calls[-1].get("resp"))
        r["last_call_inflight_comp"] = bool(calls and not calls[-1].get("resp") and calls[-1].get("comp"))
        r["last_kind"] = None
        if last:
            lr = last["resp"]
            r["last_kind"] = ("length" if lr["fin"] == "length" else "tool" if lr["ntc"] else
                              "headless" if lr["rmark"] else "malformed" if lr["cmark"] else
                              "reasoning_only" if lr["cs"] == 0 else "text")
        # tokens / time
        cts = [c["resp"]["ct"] or 0 for c in resp_calls]
        ct_all += cts
        r["ct_sum"] = sum(cts)
        lats = [(c["t_done"] - c["t_last_req"]) for c in resp_calls if c.get("t_done") and c.get("t_last_req")]
        lat_all += lats
        r["model_time_s"] = sum(lats)
        # compaction
        r["n_comp"] = len(comps)
        cs_ = []
        for i, c in enumerate(comps):
            rr = c.get("resp")
            if not rr:
                cs_.append({"ord": i + 1, "resp": False})
                continue
            cs_.append({"ord": i + 1, "resp": True, "tool": rr["ntc"] > 0 or rr["cmark"] or rr["rmark"],
                        "tool_struct": rr["ntc"] > 0, "tool_markup": rr["cmark"], "len_cut": rr["fin"] == "length",
                        "goal": rr.get("c_goal"), "instr": rr.get("c_instr"), "commit_word": rr.get("c_commit_word"),
                        "head": (rr.get("c_head") or "")[:120]})
        r["comps"] = cs_
        # post-compaction contexts: first main request after each compaction - does it still carry a commit instruction?
        post = []
        seen_comp = False
        for c in calls:
            if c.get("comp"):
                seen_comp = seen_comp or bool(c.get("_main"))
                continue
            if seen_comp and c.get("_main") and c.get("first_user_wdwdsf"):
                post.append({"instr": c.get("ctx_instr"), "orig": c.get("ctx_orig_instr")})
                seen_comp = False
        r["post_comp_ctx"] = post
        r["after_continue_last"] = bool(last and not last.get("comp") and main and main[-1] is last and False)
        # was the last main call issued right after a "Continue if you have next steps" user message?
        r["last_main_after_continue"] = bool(main and main[-1].get("after_continue"))
        # tool names not in the request's tool list (hallucinated -> OpenCode "invalid" tool)
        bad = Counter()
        for c in main:
            for n in c["resp"]["tn"]:
                if tools and n not in tools:
                    bad[n] += 1
        r["bad_tool_names"] = dict(bad)
        r["max_arg"] = max([c["resp"]["targ"] for c in main] or [0])
        r["cjk_calls"] = sum(1 for c in resp_calls if c["resp"]["cjk"])
        r["cjkg_calls"] = sum(1 for c in resp_calls if c["resp"]["cjkg"])
        r["cjkg_ex"] = [c["resp"]["cjkg_ex"] for c in resp_calls if c["resp"].get("cjkg_ex")][:3]
        r["hmm_calls"] = sum(1 for c in resp_calls if c["resp"]["hmm"] >= 10)
        r["hmm_calls_len"] = sum(1 for c in resp_calls if c["resp"]["hmm"] >= 10 and c["resp"]["fin"] == "length")
        r["events"] = s.get("events")
        per_slot.append(r)
    unmatched = {k: v for k, v in plog.items() if k not in matched}
    return per_slot, plog, unmatched, ct_all, lat_all


def summarize(model, per_slot, plog, unmatched, ct_all, lat_all):
    L = []
    n = len(per_slot)
    tot = lambda k: sum(r[k] for r in per_slot)
    roll = lambda f: sum(1 for r in per_slot if f(r))
    L.append(f"== {model}: slots {n}; model calls (200 resp) {tot('n_calls_200')}; calls incl. no-resp {tot('n_calls')}; "
             f"no-resp calls {tot('n_no_resp')}; final non-200 {tot('n_non200_final')}; http retries {tot('n_http_retries')}")
    L.append(f"slots with >1 attempt dir: {[(r['task'][4:30], r['rep'], r['attempt']) for r in per_slot if r['n_attempt_dirs'] > 1]}")
    pk = Counter()
    for r in per_slot:
        pk.update(r["parser"])
    miss_rolls = roll(lambda r: r["parser"].get("recovered", 0) + r["parser"].get("unclosed_length", 0) + r["parser"].get("unclosed_eos", 0) > 0)
    miss = pk["recovered"] + pk["unclosed_length"] + pk["unclosed_eos"]
    L.append(f"parser log lines: {len(plog)}; matched to final-attempt calls {len(plog) - len(unmatched)}; unmatched {len(unmatched)} "
             f"{list(unmatched.values())[:5]}")
    L.append(f"missing </think>: {miss} calls ({100 * miss / tot('n_calls_200'):.3f}% of {tot('n_calls_200')}); rollouts {miss_rolls}; "
             f"{dict(pk)}")
    L.append(f"main-session action turns {tot('actions_main_session')} (unreasoned {tot('unreasoned_main_session')}); subagent calls "
             f"{tot('n_sub_calls')} in {roll(lambda r: r['n_sub_calls'] > 0)} rollouts; subagent compactions {tot('n_sub_comp')}")
    L.append(f"action turns {tot('actions')}; unreasoned {tot('unreasoned')} in {roll(lambda r: r['unreasoned'] > 0)} rollouts; "
             f"every action reasoned {roll(lambda r: r['unreasoned'] == 0)}/{n}; empty outputs {tot('empty_outputs')}")
    L.append(f"reasoning with tool markup: turns {tot('rmark_turns')} rollouts {roll(lambda r: r['rmark_turns'] > 0)} headless turns "
             f"{tot('rmark_headless')}; special-token turns {tot('rspec_turns')} rollouts {roll(lambda r: r['rspec_turns'] > 0)}")
    L.append(f"length cuts (all calls) {tot('length_cuts')} in {roll(lambda r: r['length_cuts'] > 0)}; main-agent {tot('length_cuts_main')}; "
             f"thinking cuts (parser UNCLOSED + length) {tot('length_cuts_think')} in {roll(lambda r: r['length_cuts_think'] > 0)}; "
             f"ending in a loop (zlib<0.15) {tot('length_loop')} in {roll(lambda r: r['length_loop'] > 0)}; "
             f"full {CAP} cap calls {tot('full_cap')} in {roll(lambda r: r['full_cap'] > 0)} (thinking {tot('full_cap_think')})")
    ends_runaway = [r for r in per_slot if r["last_fin"] == "length" and not r["last_comp"]]
    L.append(f"rollouts whose last call is a main-agent length cut: {len(ends_runaway)} (empty patch {sum(r['empty'] for r in ends_runaway)}; "
             f"at full cap {sum((r['last_ct'] or 0) >= CAP for r in ends_runaway)})")
    L.append(f"completion tokens per call: median {statistics.median(ct_all):.0f} mean {statistics.mean(ct_all):.0f} p90 {pct(ct_all, 0.9):.0f}; "
             f"per rollout mean {statistics.mean([r['ct_sum'] for r in per_slot]) / 1000:.1f}k")
    nc = [r["n_calls_200"] for r in per_slot]
    L.append(f"model calls per rollout mean {statistics.mean(nc):.1f} median {statistics.median(nc):.0f}; rollouts >= 500 calls {sum(x >= 500 for x in nc)}")
    L.append(f"model time per rollout (sum of call latencies) mean {statistics.mean([r['model_time_s'] for r in per_slot]) / 60:.1f} min")
    w = [r["wall_s"] / 60 for r in per_slot if r["wall_s"] is not None]
    w2 = [r["wall_from_start_s"] / 60 for r in per_slot if r["wall_from_start_s"] is not None]
    L.append(f"wall (resp.created_at - first request) n={len(w)}: median {statistics.median(w):.1f} mean {statistics.mean(w):.1f} p90 {pct(w, 0.9):.1f} "
             f"(nearest-rank p90 {sorted(w)[int(0.9 * len(w))]:.1f}); > 2 h {sum(x > 120 for x in w)}")
    L.append(f"wall (resp.created_at - started_at) median {statistics.median(w2):.1f} mean {statistics.mean(w2):.1f} p90 {pct(w2, 0.9):.1f}; > 2 h {sum(x > 120 for x in w2)}")
    # deadline kills: not finished, not sandbox-lost, TimeoutError or wall >= 10,500 s
    dk = [r for r in per_slot if r["finished"] is False and (r["agent_error"] == "TimeoutError" or (r["wall_s"] or 0) >= 10500)]
    L.append(f"not finished: {roll(lambda r: r['finished'] is False)}; deadline kills {len(dk)} (TimeoutError flag {sum(r['agent_error'] == 'TimeoutError' for r in dk)}; "
             f"budget_timeout {sum(bool(r['budget_timeout']) for r in dk)}; all budget_timeout slots {roll(lambda r: bool(r['budget_timeout']))}; empty {sum(r['empty'] for r in dk)})")
    for r in dk:
        L.append(f"   DK {r['task'][4:40]} r{r['rep']} wall {r['wall_s']:.0f} calls {r['n_calls']} (200: {r['n_calls_200']}) comps {r['n_comp']} "
                 f"toolcall summaries {sum(1 for c in r['comps'] if c.get('tool'))} length cuts {r['length_cuts']} in-flight "
                 f"{'compaction' if r['last_call_inflight_comp'] else 'model' if r['last_call_inflight'] else 'no model call'} err {r['agent_error']} "
                 f"stderr {r['stderr_tail'][-40:]!r}")
    finished_late = [r for r in per_slot if r["finished"] and (r["wall_s"] or 0) >= 10500]
    L.append(f"finished on their own after 10,500 s: {[(r['task'][4:30], r['rep'], round(r['wall_s'])) for r in finished_late]}")
    # compaction
    comp_rolls = [r for r in per_slot if r["n_comp"] > 0]
    summ = [c for r in per_slot for c in r["comps"] if c["resp"]]
    tc = [c for c in summ if c["tool"]]
    first = [r["comps"][0] for r in comp_rolls if r["comps"][0]["resp"]]
    L.append(f"compacted rollouts {len(comp_rolls)}; compaction calls {tot('n_comp')}; with a response (summaries) {len(summ)}; "
             f"tool-call summaries {len(tc)} ({100 * len(tc) / max(1, len(summ)):.1f}%) [structured {sum(c['tool_struct'] for c in summ)}, "
             f"markup in content {sum(c['tool_markup'] for c in summ)}]; rollouts with >=1 {roll(lambda r: any(c.get('tool') for c in r['comps']))}; "
             f"first compaction tool call {sum(c['tool'] for c in first)}/{len(first)}; length-cut summaries {sum(c['len_cut'] for c in summ)}; "
             f"no '## Goal' {sum(not c['goal'] for c in summ)}; summary carries instr (my regex) {sum(bool(c['instr']) for c in summ)} "
             f"({100 * sum(bool(c['instr']) for c in summ) / max(1, len(summ)):.1f}%); mentions 'commit' {sum(bool(c['commit_word']) for c in summ)}")
    # commit instruction lost: some summary of the session without the instruction (my regex)
    lost = [r for r in comp_rolls if any(c["resp"] and not c["instr"] for c in r["comps"])]
    lost_tc = [r for r in lost if next(c for c in r["comps"] if c["resp"] and not c["instr"]).get("tool")]
    unc = lambda r: r["empty"] and r["finished"] and r["last_kind"] in ("text",)
    L.append(f"compacted sessions with a summary lacking the instruction (my regex): {len(lost)} (first such summary is a tool call: {len(lost_tc)}); "
             f"empty+finished+text-ending after a loss {sum(unc(r) for r in lost)}/{len(lost)} vs never lost "
             f"{sum(unc(r) for r in comp_rolls if r not in lost)}/{len(comp_rolls) - len(lost)}")
    ctx_lost = [r for r in comp_rolls if any(not p["instr"] for p in r["post_comp_ctx"])]
    L.append(f"compacted sessions where a post-compaction context lacks any commit instruction (user/assistant text, my regex): {len(ctx_lost)}; "
             f"post-compaction contexts {sum(len(r['post_comp_ctx']) for r in per_slot)}")
    # empty patches by my ending kinds
    E = [r for r in per_slot if r["empty"]]
    kinds = Counter()
    for r in E:
        if r["finished"] is False:
            k = "deadline_kill" if r in dk else "not_finished_other"
        elif r["last_kind"] == "length":
            k = "runaway(length)"
        elif r["last_kind"] in ("headless", "malformed", "reasoning_only"):
            k = "format:" + r["last_kind"]
        else:
            k = "text_end(uncommitted or no edit)"
        kinds[k] += 1
        r["_end"] = k
    L.append(f"empty patches {len(E)}: {dict(kinds)}")
    broken = lambda r: any(c.get("tool") or c.get("len_cut") for c in r["comps"])
    L.append(f"empty patches after a broken summary (tool call or length-cut): {sum(broken(r) for r in E)}; by kind "
             f"{dict(Counter(r['_end'] for r in E if broken(r)))}; not broken {sum(not broken(r) for r in E)} "
             f"{dict(Counter(r['_end'] for r in E if not broken(r)))}")
    L.append(f"empty among compacted / not compacted: {sum(r['empty'] for r in comp_rolls)}/{len(comp_rolls)}, "
             f"{sum(r['empty'] for r in per_slot if r['n_comp'] == 0)}/{n - len(comp_rolls)}")
    L.append(f"empty & finished & compacted (stopped after a compaction): {sum(1 for r in E if r['finished'] and r['n_comp'] > 0)}; "
             f"empty & last main call right after the 'Continue' message: {sum(1 for r in E if r['last_main_after_continue'])}")
    br = [r for r in comp_rolls if broken(r)]
    fine = [r for r in comp_rolls if not broken(r)]
    trc = [r for r in comp_rolls if any(c.get("tool") for c in r["comps"])]
    L.append(f"resolved: compacted with a tool-call summary {sum(r['status'] == 'pass' for r in trc)}/{len(trc)}; with a broken summary "
             f"{sum(r['status'] == 'pass' for r in br)}/{len(br)}; fine summaries {sum(r['status'] == 'pass' for r in fine)}/{len(fine)}")
    # misc
    L.append(f"rollouts with hallucinated tool names {roll(lambda r: bool(r['bad_tool_names']))}: "
             f"{Counter(k for r in per_slot for k in r['bad_tool_names'])}")
    L.append(f"max tool-call argument chars: top {sorted((r['max_arg'] for r in per_slot), reverse=True)[:5]}")
    L.append(f"rollouts with CJK in reasoning {roll(lambda r: r['cjk_calls'] > 0)}; with CJK glued to ASCII prose {roll(lambda r: r['cjkg_calls'] > 0)}")
    L.append(f"hmm runs (>=10 consecutive) calls {tot('hmm_calls')} in {roll(lambda r: r['hmm_calls'] > 0)} rollouts; of them length-cut {tot('hmm_calls_len')}")
    ev = Counter()
    for r in per_slot:
        ev.update(r["events"] or {})
    L.append(f"http.jsonl events: {dict(ev)}")
    return L


def main():
    for model in sys.argv[1:]:
        per_slot, plog, unmatched, ct_all, lat_all = analyze(model)
        L = summarize(model, per_slot, plog, unmatched, ct_all, lat_all)
        (HERE / "out" / f"metrics_{model}.txt").write_text("\n".join(L) + "\n")
        with open(HERE / "out" / f"slots_{model}.jsonl", "w") as fo:
            for r in per_slot:
                fo.write(json.dumps(r) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main()
