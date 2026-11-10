#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): one joined row per slot for v3etp20 (and the references v3etp10 / v3etp5 / etp20),
and the slot lists the report needs: empty patches by cause, 3 h deadline kills, rollouts with unreasoned action turns,
length-cut (runaway) calls, retried attempts, router client disconnects, and per-task changes v3etp10 -> v3etp20.
Stdlib only; read-only on every input.

    python3 -I -B slots_v3etp20.py -> slots_v3etp20.{jsonl,txt}

Joined per slot (task_id, slot):
  TA/results/<m>-group-*.json      status (pass / fail / infra), reward, agent_error, patch_chars, attempt_dir
  classifier all.jsonl             code / subcode, wall_s, not_finished, n_compactions, n_assistant (empty-patch dirs)
  endings derived.jsonl.gz         ending class (tables.py ending()), comp_kinds, instr_first_loss, deadline_kill
                                   (../degenerate/endings: the endings-lens copy)
  calls-<m>.jsonl                  length-cut calls (log UNCLOSED_NO_TOOL + finish length), rescued calls, EOS-in-think
  features <m>.jsonl               action turns without reasoning (teacher-rollout extractor)
  attempt dirs                     attempts per slot, started.json / terminal.json times of the final attempt
  routers/<m>-group-*/student-router-requests.jsonl   non-200 router records (client disconnects) by time
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
V20 = TA / "analysis/v3etp20"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(V20 / "degenerate/endings"))
import tables as ET  # noqa: E402  (endings-lens tables.py copy: ending())

MODELS = ["etp20", "v3etp5", "v3etp10", "v3etp20"]


def own(m):
    return m == "v3etp20"


def hm(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else None


def load(m):
    slots = {}
    for path in sorted(TA.glob(f"results/{m}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, recs in data["records"].items():
            for r in recs:
                rep = int(r["slot"].rsplit("-r", 1)[1])
                ad = Path(r["attempt_dir"]) if r.get("attempt_dir") else None
                n_att = len(list(ad.parent.glob("attempt-*"))) if ad else 0
                started = None
                if ad and (ad / "started.json").exists():
                    started = json.loads((ad / "started.json").read_text()).get("started_at")
                ended = os.path.getmtime(ad / "terminal.json") if ad and (ad / "terminal.json").exists() else None
                slots[(task_id, rep)] = {"model": m, "group": data["group"], "task_id": task_id, "rep": rep,
                                         "status": r["status"], "reward": r.get("reward"), "agent_error": r.get("agent_error"),
                                         "patch_chars": r.get("patch_chars") or 0, "attempts": n_att,
                                         "started": hm(started), "terminal": hm(ended), "started_epoch": started}
    cls = (V20 / "empty-patch" if own(m) else TA / "analysis/empty-patch") / m / "all.jsonl"
    for line in open(cls):
        c = json.loads(line)
        s = slots[(c["task_id"], c["ri"])]
        s.update({"code": c["code"], "subcode": c.get("subcode"), "wall_s": c.get("wall_s"), "not_finished": c["not_finished"],
                  "empty": c["empty_patch"], "n_compactions": c["f"].get("n_compactions") or 0,
                  "n_assistant": c["f"].get("n_assistant")})
    with gzip.open(V20 / "degenerate/endings/out/derived.jsonl.gz", "rt") as handle:
        for line in handle:
            d = json.loads(line)
            if d["model"] != m:
                continue
            s = slots[(d["task_id"], d["ri"])]
            kinds = d.get("comp_kinds") or {}
            s.update({"ending": ET.ending(d), "deadline_kill": d.get("deadline_kill"),
                      "toolcall_summaries": kinds.get("toolcall", 0), "length_cut_summaries": kinds.get("summary_length_cut", 0),
                      "instr_first_loss": d.get("instr_first_loss"), "runaway_calls": d.get("runaway_calls"),
                      "inflight": d.get("inflight"), "activity": d.get("activity")})
    calls = (V20 / "think" if own(m) else TA / "analysis/think") / f"calls-{m}.jsonl"
    for line in open(calls):
        r = json.loads(line)
        s = slots[(r["task_id"], int(r["slot"].rsplit("-r", 1)[1]))]
        s["calls"] = s.get("calls", 0) + 1
        if r.get("log_kind") == "UNCLOSED_NO_TOOL":
            key = "length_cuts" if r["finish"] == "length" else "eos_in_think"
            s[key] = s.get(key, 0) + 1
            if r["finish"] == "length":
                s["max_cut_tokens"] = max(s.get("max_cut_tokens", 0), r.get("completion_tokens") or 0)
        if r.get("log_kind") == "RECOVERED":
            s["rescued"] = s.get("rescued", 0) + 1
    feat = (V20 / "features" if own(m) else TA / "analysis/features") / f"{m}.jsonl"
    for line in open(feat):
        r = json.loads(line)
        s = slots[(r["task_id"], int(r["slot"].rsplit("-r", 1)[1]))]
        s["unreasoned_actions"] = (r.get("action_turns") or 0) - (r.get("action_with_reasoning") or 0) if r.get("export") else None
        s["action_turns"] = r.get("action_turns")
    return slots


def main():
    data = {m: load(m) for m in MODELS}
    with open(HERE / "slots_v3etp20.jsonl", "w") as handle:
        for m in MODELS:
            for k in sorted(data[m]):
                handle.write(json.dumps(data[m][k]) + "\n")
    S = data["v3etp20"]
    L = []
    emp = [s for s in S.values() if s["empty"]]
    L.append(f"== v3etp20 empty patches: {len(emp)} (of 250); by ending: {dict(collections.Counter(s['ending'] for s in emp))}")
    for s in sorted(emp, key=lambda s: (s["ending"], s["task_id"], s["rep"])):
        L.append(f"  {s['ending']:28s} {s['task_id']} r{s['rep']} code {s['code']}/{s['subcode']} wall {s['wall_s']} s calls {s.get('calls')} "
                 f"compactions {s['n_compactions']} tool-call summaries {s.get('toolcall_summaries')} length-cut summaries {s.get('length_cut_summaries')} "
                 f"instr loss {s.get('instr_first_loss')} length cuts {s.get('length_cuts', 0)}")
    dk = [s for s in S.values() if s.get("deadline_kill")]
    L.append(f"== v3etp20 3 h deadline kills: {len(dk)}")
    for s in sorted(dk, key=lambda s: s["task_id"]):
        L.append(f"  {s['task_id']} r{s['rep']} group {s['group']} started {s['started']} terminal {s['terminal']} wall {s['wall_s']} "
                 f"agent_error {s['agent_error']} calls {s.get('calls')} compactions {s['n_compactions']} tool-call summaries "
                 f"{s.get('toolcall_summaries')} length cuts {s.get('length_cuts', 0)} in flight {s.get('inflight')} tail {s.get('activity')} status {s['status']}")
    un = [s for s in S.values() if s.get("unreasoned_actions")]
    L.append(f"== v3etp20 rollouts with unreasoned action turns: {len(un)}; turns {sum(s['unreasoned_actions'] for s in un)}")
    for s in sorted(un, key=lambda s: -s["unreasoned_actions"]):
        L.append(f"  {s['task_id']} r{s['rep']} unreasoned {s['unreasoned_actions']}/{s['action_turns']} status {s['status']} ending {s.get('ending')}")
    lc = [s for s in S.values() if s.get("length_cuts")]
    L.append(f"== v3etp20 rollouts with a length cut while thinking: {len(lc)}; calls {sum(s['length_cuts'] for s in lc)}; "
             f"ending on one (E3a runaway): {sum(1 for s in lc if s.get('ending') == 'runaway_end')}; passes {sum(1 for s in lc if s['status'] == 'pass')}")
    for s in sorted(lc, key=lambda s: s["task_id"]):
        L.append(f"  {s['task_id']} r{s['rep']} cuts {s['length_cuts']} max tokens {s.get('max_cut_tokens')} status {s['status']} ending {s.get('ending')}")
    ra = [s for s in S.values() if s["attempts"] > 1]
    L.append(f"== v3etp20 slots with > 1 attempt: {len(ra)}")
    for s in ra:
        L.append(f"  {s['task_id']} r{s['rep']} group {s['group']} attempts {s['attempts']} final attempt started {s['started']} ended {s['terminal']} status {s['status']} ending {s.get('ending')}")
    # router client disconnects -> slot whose final attempt was in flight then (nearest terminal time)
    L.append("== v3etp20 router records that are not 200")
    for path in sorted(TA.glob("routers/v3etp20-group-*/student-router-requests.jsonl")):
        g = int(path.parent.name.rsplit("-", 1)[1])
        for line in open(path):
            r = json.loads(line)
            if r.get("status") == 200:
                continue
            end = r["ts"] + (r.get("duration_s") or 0)
            # nearest deadline-killed slot of the group by its 3 h deadline (started + 10,800 s)
            dks = [s for s in S.values() if s["group"] == g and s.get("deadline_kill") and s.get("started_epoch")]
            best = min(dks, key=lambda s: abs(s["started_epoch"] + 10800 - end), default=None)
            L.append(f"  group {g} {hm(r['ts'])} -> {hm(end)} status {r['status']} duration {r.get('duration_s')} s worker {r.get('workers')} | "
                     f"nearest deadline kill in the group: "
                     + (f"{best['task_id']} r{best['rep']} (deadline {hm(best['started_epoch'] + 10800)}, {end - best['started_epoch'] - 10800:+.0f} s)" if best else "none"))
    # per-task change v3etp10 -> v3etp20
    T10, T20 = collections.defaultdict(list), collections.defaultdict(list)
    for s in data["v3etp10"].values():
        T10[s["task_id"]].append(s)
    for s in S.values():
        T20[s["task_id"]].append(s)
    rows = []
    for t in sorted(T20):
        a = [s for s in T10[t] if s["status"] in ("pass", "fail")]
        b = [s for s in T20[t] if s["status"] in ("pass", "fail")]
        ra_, rb_ = sum(s["status"] == "pass" for s in a) / len(a), sum(s["status"] == "pass" for s in b) / len(b)
        rows.append((rb_ - ra_, t, f"{sum(s['status'] == 'pass' for s in a)}/{len(a)}", f"{sum(s['status'] == 'pass' for s in b)}/{len(b)}",
                     sum(s["empty"] for s in T10[t]), sum(s["empty"] for s in T20[t]),
                     sum(1 for s in T10[t] if s.get("toolcall_summaries")), sum(1 for s in T20[t] if s.get("toolcall_summaries")),
                     sum(1 for s in T10[t] if s.get("length_cuts")), sum(1 for s in T20[t] if s.get("length_cuts"))))
    rows.sort()
    L.append("== per task, v3etp10 -> v3etp20: passes, empty patches, rollouts with a tool-call summary, rollouts with a length cut")
    for d, t, pa, pb, ea, eb, ta, tb, la, lb in rows:
        if d != 0 or ea != eb or ta != tb:
            L.append(f"  {d * 100:+5.0f} pp {t:45s} passes {pa} -> {pb} | empty {ea} -> {eb} | tool-call summary {ta} -> {tb} | length cut {la} -> {lb}")
    worse = [r for r in rows if r[0] < 0]
    better = [r for r in rows if r[0] > 0]
    L.append(f"  tasks worse {len(worse)} (empty patches {sum(r[4] for r in worse)} -> {sum(r[5] for r in worse)}, tool-call-summary rollouts "
             f"{sum(r[6] for r in worse)} -> {sum(r[7] for r in worse)}), better {len(better)} (empty {sum(r[4] for r in better)} -> {sum(r[5] for r in better)}, "
             f"tool-call-summary rollouts {sum(r[6] for r in better)} -> {sum(r[7] for r in better)}), equal {len(rows) - len(worse) - len(better)}")
    (HERE / "slots_v3etp20.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
