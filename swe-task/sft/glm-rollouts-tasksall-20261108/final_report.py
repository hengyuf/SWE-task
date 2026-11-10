#!/usr/bin/env python3
"""glm-rollouts-tasksall report -> analysis/summary.json, analysis/REPORT.md, analysis/rollouts.jsonl (read-only on the
inputs; runs on partial results too: campaigns without a results file are listed as pending).

    <atp controller python> -I -B final_report.py

* Final records (results/glmlocal-<campaign>.json `records`, keyed by task id INSIDE a campaign; mapped to the global
  task key source/run/cand/task_id through the file's task_keys, since 17 task ids collide across runs).
* Per rollout (its final attempt dir): compaction (OpenCode "anchored summary" request), LLM calls, generated tokens,
  longest reply, max prompt tokens -- from progress-cache.json when present, else scanned.
* Breakdowns: tier, owner, inventory status (accepted / rejected / withdrawn / unscored), reject reason, and the build
  probe's Kimi-K3 / sc14 pass rates on the same tasks (inventory kimi_pass/kimi_graded, sc14_pass/sc14_graded).
* Dip re-runs (rerun_dip.py -> results/glmlocal-dip1.json, every round): each re-run record replaces its original slot
  (campaign, task, slot); the original attempt is kept as replaced_attempt / replaced_status.
"""
import collections
import json
from pathlib import Path
import statistics

GR = Path(__file__).resolve().parent
INVENTORY = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/tasks-all-20261106/all_tasks.jsonl")


def scan_attempt(attempt: Path) -> dict:
    calls = gen = maxrep = maxprompt = ge32k = ge131k = 0
    compacted = False
    path = attempt / "http.jsonl"
    if path.exists():
        with open(path, "rb") as stream:
            for raw in stream:
                head = raw[:1500]
                if b'"event":"request"' in head or b'"event": "request"' in head:
                    if b"anchored summary" in raw:
                        compacted = True
                    continue
                if b'"event":"response"' not in head and b'"event": "response"' not in head:
                    continue
                try:
                    event = json.loads(raw)
                    if event.get("status") != 200:
                        continue
                    usage = json.loads(event.get("body") or "{}").get("usage") or {}
                except ValueError:
                    continue
                completion = usage.get("completion_tokens") or 0
                calls += 1
                gen += completion
                maxrep = max(maxrep, completion)
                ge32k += completion >= 32768
                ge131k += completion >= 131072
                maxprompt = max(maxprompt, usage.get("prompt_tokens") or 0)
    return {"calls": calls, "gen": gen, "max_reply": maxrep, "ge32k": ge32k, "ge131k": ge131k, "max_prompt": maxprompt,
            "compacted": compacted}


def rate(statuses):
    graded = [s for s in statuses if s in ("pass", "fail")]
    return sum(s == "pass" for s in graded), len(graded)


def pct(p, n):
    return f"{p}/{n} = {100 * p / n:.1f}%" if n else "-"


def dip_reruns():
    """{(campaign, task_id, slot): mapping entry} from every results/glmlocal-dip*.json (all rounds)."""
    out = {}
    for path in sorted((GR / "results").glob("glmlocal-dip*.json")):
        for round_ in json.loads(path.read_text()).get("rounds", []):
            for campaign in round_["campaigns"].values():
                for entry in campaign["mapping"]:
                    out[(entry["campaign"], entry["task_id"], entry["slot"])] = entry
    for path in sorted((GR / "results").glob("glmlocal-dip*-parts/*.json")):  # parallel mode: one file per campaign
        for entry in json.loads(path.read_text())["mapping"]:
            out[(entry["campaign"], entry["task_id"], entry["slot"])] = entry
    return out


def main():
    plan = json.loads((GR / "tasks.json").read_text())
    replaced = dip_reruns()
    applied = []
    tasks = {t["key"]: t for t in plan["tasks"]}
    inventory = {}
    for line in INVENTORY.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            inventory[f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"] = row
    cache = json.loads((GR / "progress-cache.json").read_text()) if (GR / "progress-cache.json").exists() else {}
    rollouts, per_campaign, pending, reruns = [], {}, [], {}
    for campaign in plan["campaigns"]:
        name = campaign["name"]
        path = GR / "results" / f"glmlocal-{name}.json"
        if not path.exists():
            pending.append(name)
            continue
        data = json.loads(path.read_text())
        keys = data["task_keys"]
        for task_id, records in data["records"].items():
            for i, record in enumerate(records):
                entry = replaced.get((name, task_id, record["slot"]))
                if entry:
                    assert record.get("attempt_dir") == entry["original_attempt"], (name, task_id, record["slot"])
                    records[i] = dict(entry["rerun"], slot=record["slot"], replaced_attempt=record.get("attempt_dir"),
                                      replaced_status=record["status"], rerun_campaign=entry["rerun_campaign"])
                    applied.append({"campaign": name, "task_id": task_id, "slot": record["slot"],
                                    "original_status": record["status"], "rerun_status": entry["rerun_status"]})
        flat = [r for rs in data["records"].values() for r in rs]
        stats = {"pass": sum(r["status"] == "pass" for r in flat), "fail": sum(r["status"] == "fail" for r in flat)}
        stats["infra"] = len(flat) - stats["pass"] - stats["fail"]
        per_campaign[name] = dict(stats, tier=campaign["tier"], sandbox=campaign["sandbox"])
        reruns[name] = sum(map(len, (data.get("sandbox_failure_reruns") or {}).values()))
        for task_id, records in data["records"].items():
            key = keys[task_id]
            task, row = tasks[key], inventory[key]
            for record in records:
                attempt = Path(record.get("attempt_dir") or "")
                stats = cache.get(str(attempt)) or (scan_attempt(attempt) if attempt.is_dir() else {})
                rollouts.append({"campaign": name, "key": key, "task_id": task_id, "tier": task["tier"],
                                 "owner": task["owner"], "status_inventory": task["status"], "audit": task["audit"],
                                 "reject_reason": row.get("reject_reason"), "slot": record["slot"],
                                 "status": record["status"], "f2p": record.get("f2p"), "p2p": record.get("p2p"),
                                 "attempt": attempt.name, "attempt_dir": str(attempt),
                                 **{k: stats.get(k) for k in ("calls", "gen", "max_reply", "ge32k", "ge131k",
                                                               "max_prompt", "compacted")}})
    graded = [r for r in rollouts if r["status"] in ("pass", "fail")]
    passed = sum(r["status"] == "pass" for r in graded)

    def group(field):
        out = collections.defaultdict(list)
        for r in rollouts:
            out[r[field]].append(r["status"])
        return {str(k): dict(zip(("pass", "graded"), rate(v))) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}

    per_task = collections.defaultdict(list)
    for r in rollouts:
        per_task[r["key"]].append(r["status"])
    kimi = [(inventory[k]["kimi_pass"], inventory[k]["kimi_graded"]) for k in per_task
            if inventory[k].get("kimi_graded")]
    sc14 = [(inventory[k]["sc14_pass"], inventory[k]["sc14_graded"]) for k in per_task
            if inventory[k].get("sc14_graded")]
    same_glm = rate([s for k in per_task if inventory[k].get("kimi_graded") for s in per_task[k]])
    compacted = [r for r in rollouts if r.get("compacted")]
    summary = {
        "tasks_planned": len(tasks), "campaigns_done": len(per_campaign), "campaigns_pending": pending,
        "rollouts": len(rollouts), "graded": len(graded), "pass": passed, "infra": len(rollouts) - len(graded),
        "pass_rate": round(passed / len(graded), 4) if graded else None,
        "by_tier": group("tier"), "by_owner": group("owner"), "by_inventory_status": group("status_inventory"),
        "by_audit": group("audit"), "by_reject_reason": group("reject_reason"),
        "passes_per_task": dict(sorted(collections.Counter(sum(s == "pass" for s in v)
                                                           for v in per_task.values()).items())),
        "build_probe_same_tasks": {"tasks": len(kimi), "glm_local": same_glm,
                                   "kimi_k3": [sum(p for p, _ in kimi), sum(n for _, n in kimi)],
                                   "sc14": [sum(p for p, _ in sc14), sum(n for _, n in sc14)]},
        "compacted_rollouts": len(compacted), "compacted_pass": sum(r["status"] == "pass" for r in compacted),
        "pass_without_compaction": sum(r["status"] == "pass" and not r.get("compacted") for r in rollouts),
        "rollouts_with_reply_ge_131k": sum(1 for r in rollouts if (r.get("ge131k") or 0) > 0),
        "rollouts_with_reply_ge_32k": sum(1 for r in rollouts if (r.get("ge32k") or 0) > 0),
        "max_prompt_tokens": max((r.get("max_prompt") or 0) for r in rollouts) if rollouts else None,
        "median_calls": statistics.median([r["calls"] for r in rollouts if r.get("calls")]) if rollouts else None,
        "median_gen_tokens": statistics.median([r["gen"] for r in rollouts if r.get("gen")]) if rollouts else None,
        "attempt_index": dict(collections.Counter(r["attempt"] for r in rollouts)),
        "sandbox_failure_reruns": {k: v for k, v in reruns.items() if v}, "per_campaign": per_campaign,
        "dip_reruns_applied": {"slots": len(applied),
                               "now_pass": sum(a["rerun_status"] == "pass" for a in applied),
                               "now_fail": sum(a["rerun_status"] == "fail" for a in applied)},
        "infra_rollouts": [{k: r[k] for k in ("campaign", "key", "slot", "attempt_dir")} for r in rollouts
                           if r["status"] not in ("pass", "fail")],
    }
    (GR / "analysis").mkdir(exist_ok=True)
    (GR / "analysis" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(GR / "analysis" / "rollouts.jsonl", "w") as stream:
        for r in rollouts:
            stream.write(json.dumps(r) + "\n")
    names = {1: "accepted/strong", 2: "partial", 3: "weak", 4: "none/unlabeled"}
    lines = ["# glm-rollouts-tasksall: report", "",
             f"Self-hosted GLM-5.3 x 5 on the tasks of sft/tasks-all-20261106 without GLM-5.3 rollouts: "
             f"{len(per_campaign)}/{len(plan['campaigns'])} campaigns done; **{pct(passed, len(graded))}** of graded "
             f"rollouts pass ({len(rollouts):,} rollouts, {len(rollouts) - len(graded)} without a grade).", "",
             "| tier | pass / graded |", "|---|---|"]
    for tier, value in summary["by_tier"].items():
        lines.append(f"| {tier} {names.get(int(tier), '')} | {pct(value['pass'], value['graded'])} |")
    lines += ["", "| inventory status | pass / graded |", "|---|---|"]
    lines += [f"| {k} | {pct(v['pass'], v['graded'])} |" for k, v in summary["by_inventory_status"].items()]
    lines += ["", "| owner | pass / graded |", "|---|---|"]
    lines += [f"| {k} | {pct(v['pass'], v['graded'])} |" for k, v in summary["by_owner"].items()]
    b = summary["build_probe_same_tasks"]
    lines += ["", f"- Passes per task (of 5): {summary['passes_per_task']}.",
              f"- Same tasks, build probes (5 slots each): GLM-5.3 {pct(*b['glm_local'])} vs Kimi-K3 "
              f"{pct(*b['kimi_k3'])} vs sc14 {pct(*b['sc14'])} ({b['tasks']} tasks).",
              f"- OpenCode compaction: {len(compacted)} rollouts ({summary['compacted_pass']} pass); passes without "
              f"compaction: {summary['pass_without_compaction']}.",
              f"- Replies >= 131,072 tokens: {summary['rollouts_with_reply_ge_131k']} rollouts; >= 32k: "
              f"{summary['rollouts_with_reply_ge_32k']}; max prompt {summary['max_prompt_tokens']} tokens; median "
              f"{summary['median_calls']} LLM calls and {summary['median_gen_tokens']} generated tokens per rollout.",
              f"- Final attempt index: {summary['attempt_index']}; sandbox-failure re-runs per campaign: "
              f"{summary['sandbox_failure_reruns'] or 'none'}.",
              f"- Agent-timeout slots from the 2-server dip (2026-11-09 03:10-04:50) re-run once and applied: "
              f"{summary['dip_reruns_applied']} (results/glmlocal-dip1.json).",
              f"- Pending campaigns: {', '.join(pending) if pending else 'none'}.",
              "- Files: `analysis/summary.json`, `analysis/rollouts.jsonl` (one row per rollout, with the task key)."]
    (GR / "analysis" / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
