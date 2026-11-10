#!/usr/bin/env python3
"""glm-rollouts-sftv3 final summary -> analysis/final-summary.json + analysis/FINAL-REPORT.md (read-only on the inputs).

    /usr/bin/python3 -I -B final_report.py

* Final records (results/glmlocal-<campaign>.json `records`): pass / fail / infra per campaign and task, attempts.
* Per final rollout (its attempt dir): compaction (an OpenCode "anchored summary" request in http.jsonl), LLM calls,
  generated tokens, longest reply, max prompt tokens -- from progress-cache.json when present, else scanned.
* Comparison on the SAME tasks: Kimi-K3 v4 (stock prompt, 262,144 window) and v5 (official2) final records, and the
  hosted GLM-5.3 teacher-rollouts-v2 records (1M window) where they overlap.
* Ungraded-slot re-runs (user 2026-11-08, rerun_infra.py -> results/glmlocal-infra-rerun1.json): each re-run record
  replaces its original slot (campaign, task, slot); the original attempt is kept as "replaced_attempt".
"""
import collections
import glob
import json
import os
from pathlib import Path
import statistics

GR = Path(__file__).resolve().parent
SFT = GR.parent
CAMPAIGNS = [f"hengyuf-g{g}" for g in range(4)] + [f"baihe-g{g}" for g in range(4)]


def scan_attempt(attempt: Path) -> dict:
    calls = gen = maxrep = maxprompt = ge32k = 0
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
                maxprompt = max(maxprompt, usage.get("prompt_tokens") or 0)
    return {"calls": calls, "gen": gen, "max_reply": maxrep, "ge32k": ge32k, "max_prompt": maxprompt,
            "compacted": compacted}


def teacher_records(pattern):
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(str(pattern))):
        data = json.loads(Path(path).read_text())
        for task_id, records in (data.get("records") or {}).items():
            out[task_id] += [r.get("status") for r in records]
    return out


def rate(statuses):
    graded = [s for s in statuses if s in ("pass", "fail")]
    return (sum(s == "pass" for s in graded), len(graded))


def infra_reruns():
    """{(campaign, task_id, slot): mapping entry} from results/glmlocal-infra-rerun1.json (if present)."""
    path = GR / "results" / "glmlocal-infra-rerun1.json"
    if not path.exists():
        return {}
    out = {}
    for name, campaign in json.loads(path.read_text())["campaigns"].items():
        for entry in campaign["mapping"]:
            out[(entry["campaign"], entry["task_id"], entry["slot"])] = entry
    return out


def main():
    cache = json.loads((GR / "progress-cache.json").read_text()) if (GR / "progress-cache.json").exists() else {}
    tasks = {t["task_id"]: t for t in json.loads((GR / "tasks.json").read_text())["tasks"]}
    per_campaign, per_task, rollouts = {}, collections.defaultdict(list), []
    reruns = {}
    replaced = infra_reruns()
    applied = []
    for name in CAMPAIGNS:
        data = json.loads((GR / "results" / f"glmlocal-{name}.json").read_text())
        reruns[name] = data.get("sandbox_failure_reruns") or {}
        for task_id, records in data["records"].items():
            for i, record in enumerate(records):
                entry = replaced.get((name, task_id, record["slot"]))
                if entry:
                    assert record.get("attempt_dir") == entry["original_attempt"], (name, task_id, record["slot"])
                    records[i] = dict(entry["rerun"], slot=record["slot"], replaced_attempt=record.get("attempt_dir"),
                                      replaced_status=record["status"], rerun_campaign=entry["rerun_campaign"])
                    applied.append({"campaign": name, "task_id": task_id, "slot": record["slot"],
                                    "original_status": record["status"], "rerun_status": entry["rerun_status"],
                                    "rerun_attempt": entry["rerun_attempt"]})
        flat = [r for rs in data["records"].values() for r in rs]
        passed = sum(r["status"] == "pass" for r in flat)
        failed = sum(r["status"] == "fail" for r in flat)
        per_campaign[name] = {"pass": passed, "fail": failed, "infra": len(flat) - passed - failed,
                              "rate": round(passed / (passed + failed), 4) if passed + failed else None}
        for task_id, records in data["records"].items():
            for record in records:
                attempt = Path(record.get("attempt_dir") or "")
                stats = cache.get(str(attempt))
                if not stats:
                    stats = scan_attempt(attempt) if attempt.is_dir() else {}
                rollouts.append({"campaign": name, "task_id": task_id, "slot": record["slot"], "status": record["status"],
                                 "f2p": record.get("f2p"), "p2p": record.get("p2p"),
                                 "attempt": attempt.name, "attempt_dir": str(attempt),
                                 **{k: stats.get(k) for k in ("calls", "gen", "max_reply", "ge32k", "ge131k",
                                                               "max_prompt", "compacted")}})
                per_task[task_id].append(record["status"])
    graded = [r for r in rollouts if r["status"] in ("pass", "fail")]
    passed = sum(r["status"] == "pass" for r in graded)
    kimi = teacher_records(SFT / "kimi-rollouts-v4-stock256k-20261028/results/kimi-*.json")
    kimi.update(teacher_records(SFT / "kimi-rollouts-v5-official2-20261028/results/kimi-*.json"))
    hub_glm = teacher_records(SFT / "teacher-rollouts-v2-20261025/results/glm-*.json")
    same = [t for t in per_task if t in kimi]
    g_same = rate([s for t in same for s in per_task[t]]); k_same = rate([s for t in same for s in kimi[t]])
    overlap = [t for t in per_task if t in hub_glm]
    g_ov = rate([s for t in overlap for s in per_task[t]]); h_ov = rate([s for t in overlap for s in hub_glm[t]])
    owner = collections.defaultdict(list)
    for r in rollouts:
        owner[tasks[r["task_id"]]["source"]].append(r["status"])
    compacted = [r for r in rollouts if r.get("compacted")]
    replies = [r["max_reply"] for r in rollouts if r.get("max_reply") is not None]
    summary = {
        "rollouts": len(rollouts), "graded": len(graded), "pass": passed, "infra": len(rollouts) - len(graded),
        "pass_rate": round(passed / len(graded), 4), "per_campaign": per_campaign,
        "per_source": {s: dict(zip(("pass", "graded"), rate(v))) for s, v in owner.items()},
        "passes_per_task": dict(sorted(collections.Counter(sum(s == "pass" for s in v) for v in per_task.values()).items())),
        "tasks": len(per_task), "attempt_index": dict(collections.Counter(r["attempt"] for r in rollouts)),
        "sandbox_failure_reruns": {k: len(v) for k, v in reruns.items() if v},
        "ungraded_slot_reruns": applied,
        "compacted_rollouts": len(compacted), "compacted_pass": sum(r["status"] == "pass" for r in compacted),
        "pass_without_compaction": sum(r["status"] == "pass" and not r.get("compacted") for r in rollouts),
        "max_reply_tokens": max(replies) if replies else None,
        "rollouts_with_reply_ge_131k": sum(1 for r in rollouts if (r.get("ge131k") or 0) > 0),
        "max_prompt_tokens": max((r.get("max_prompt") or 0) for r in rollouts),
        "rollouts_with_reply_ge_32k": sum(1 for r in rollouts if (r.get("ge32k") or 0) > 0),
        "median_calls": statistics.median([r["calls"] for r in rollouts if r.get("calls")]),
        "median_gen_tokens": statistics.median([r["gen"] for r in rollouts if r.get("gen")]),
        "kimi_same_tasks": {"tasks": len(same), "glm": g_same, "kimi_k3": k_same},
        "hub_glm_overlap": {"tasks": len(overlap), "glm_local": g_ov, "glm_hub": h_ov},
        "infra_rollouts": [{k: r[k] for k in ("campaign", "task_id", "slot", "attempt_dir")} for r in rollouts
                           if r["status"] not in ("pass", "fail")],
    }
    (GR / "analysis").mkdir(exist_ok=True)
    (GR / "analysis" / "final-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(GR / "analysis" / "final-rollouts.jsonl", "w") as stream:
        for r in rollouts:
            stream.write(json.dumps(r) + "\n")
    pct = lambda p, n: f"{p}/{n} = {100 * p / n:.1f}%" if n else "-"
    lines = ["# glm-rollouts-sftv3: final report", "",
             f"Self-hosted GLM-5.3 x 5 on the 220 SFT-v3 tasks: **{pct(passed, len(graded))}** of graded rollouts pass "
             f"({len(rollouts)} rollouts, {len(rollouts) - len(graded)} ended without a grade).", "",
             "| campaign | pass | fail | infra | rate |", "|---|---|---|---|---|"]
    for name, st in per_campaign.items():
        lines.append(f"| {name} | {st['pass']} | {st['fail']} | {st['infra']} | {100 * st['rate']:.1f}% |")
    lines += ["", f"- Passes per task (of 5): {summary['passes_per_task']} (tasks with 0 passes: "
              f"{summary['passes_per_task'].get(0, 0)}).",
              f"- By task source: " + "; ".join(f"{s} {pct(*rate(v))}" for s, v in sorted(owner.items())) + ".",
              f"- Same tasks, Kimi-K3 (v4/v5 final records): GLM {pct(*g_same)} vs Kimi {pct(*k_same)} on {len(same)} tasks.",
              f"- Hosted GLM-5.3 (teacher-rollouts-v2, 1M window) on the {len(overlap)} overlapping tasks: local "
              f"{pct(*g_ov)} vs hosted {pct(*h_ov)}.",
              f"- OpenCode compaction: {len(compacted)} rollouts ({summary['compacted_pass']} of them pass); passes "
              f"without compaction: {summary['pass_without_compaction']}.",
              f"- Rollouts with a reply >= 131,072 tokens (above the old cap; allowed by the 256k limit): "
              f"{summary['rollouts_with_reply_ge_131k']}; with a reply >= 32k: {summary['rollouts_with_reply_ge_32k']}; "
              f"max prompt {summary['max_prompt_tokens']:,} tokens. Median LLM calls per rollout {summary['median_calls']}, "
              f"median generated tokens {summary['median_gen_tokens']:,}.",
              f"- Final attempt index: {summary['attempt_index']} (attempt-1+ = ATP retried an interrupted attempt).",
              f"- Ungraded slots re-run (user 2026-11-08, `results/glmlocal-infra-rerun1.json`): "
              + ("; ".join(f"{a['campaign']} {a['task_id']} {a['slot']}: {a['original_status']} -> {a['rerun_status']}"
                           for a in applied) or "none yet") + ".",
              f"- Records: `results/glmlocal-<campaign>.json`; flat list `analysis/final-rollouts.jsonl`; "
              f"summary `analysis/final-summary.json`."]
    (GR / "analysis" / "FINAL-REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
