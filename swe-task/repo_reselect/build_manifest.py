#!/usr/bin/env python3
"""Manifest for the host-repo reselection (2026-10-22): per DeepSWE target of the official1 runs, every artifact path a
selector needs, the target's failure patterns, what the current host repo achieved, and the candidate pool = the next 4
eligible top100.jsonl entries in rank order after the current host (skipping builds_on_target / same_owner_as_target,
the current host and repos abandoned for answer leaks). Read-only on everything else.
    /usr/bin/python3 -I -B repo_reselect/build_manifest.py   (from S)"""
import collections, csv, json
from pathlib import Path

S = Path(__file__).resolve().parent.parent
RUNS = S / "synth/runs"
SKIP = {"builds_on_target", "same_owner_as_target"}
# Delivery patterns (no commit / empty patch / timeout): judge_similarity.md never counts them, even where audits mark
# them reproduced (review 2026-10-22).
JUDGE_EXCLUDED = {
    ("anko-default-function-arguments", "No verified committed feature before timeout"),
    ("dasel-html-document-format", "Repeated planning never reaches committed implementation"),
    ("dynamodb-toolbox-lazy-recursive-schemas", "Delaying the required commit until cleanup is complete"),
    ("scriggo-method-declarations", "No committed deliverable before timeout"),
    ("sql-formatter-bigquery-pipe-formatting", "Repeated exploration never becomes an implementation"),
    ("sql-formatter-bigquery-pipe-formatting", "Losing the commit requirement after successful working-tree checks"),
    ("wasmi-trap-coredumps", "Work does not converge to the required committed artifact"),
    ("ytt-jsonpath-query-api", "Forgets explicit commit authorization and leaves no submitted patch"),
}
top = collections.defaultdict(list)
for line in (S / "top100.jsonl").open():
    if line.strip():
        row = json.loads(line); top[row["task_id"]].append(row)
stats = {r["task_id"]: r for r in csv.DictReader((S / "data/targets/sc14-out131k-kimi/per_task.csv").open())}

builds = {}
for run in sorted(RUNS.glob("official1-*")):
    for b in (run / "build").glob("*/*"):
        if b.is_dir():
            builds[b.parent.name] = (run, b)
hosts = {t: b.name.replace("__", "/", 1) for t, (_, b) in builds.items()}
host_of = collections.defaultdict(list)
for t, repo in hosts.items():
    host_of[repo.lower()].append(t)

def rel(p):
    return str(Path(p).resolve()).replace(str(S.resolve()), "S") if p else None

tasks = []
for target in sorted(builds):
    run, build = builds[target]
    used = hosts[target]
    abandoned = []
    for a in sorted(run.glob("abandoned-*")):
        note = (a / "ABANDONED.txt").read_text().strip() if (a / "ABANDONED.txt").exists() else ""
        repo = a.name[len("abandoned-"):].split("-")[0].replace("__", "/", 1)
        abandoned.append({"repo": repo, "dir": rel(a), "note": note})
    abandoned_repos = {a["repo"].lower() for a in abandoned}
    ranked = sorted(top[target], key=lambda r: r["rank"])
    current = next(r for r in ranked if r["repo"].lower() == used.lower())
    pool = [r for r in ranked if not SKIP & set(r["flags"]) and r["repo"].lower() != used.lower()
            and r["repo"].lower() not in abandoned_repos][:4]
    skipped_flagged = [{"rank": r["rank"], "repo": r["repo"], "flags": r["flags"]} for r in ranked
                       if SKIP & set(r["flags"]) and r["rank"] <= pool[-1]["rank"]]
    analysis = run / "analysis" / target
    result = json.loads((analysis / "result.json").read_text())
    patterns = [{"name": p.get("name"), "category": p.get("category"), "description": p.get("description"),
                 "student_slots": p.get("student_slots"), "n_student_slots": len(p.get("student_slots") or []),
                 "teacher_contrast": p.get("teacher_contrast"),
                 "judge_excluded": (target, p.get("name")) in JUDGE_EXCLUDED}
                for p in result.get("patterns", [])]
    missing_excluded = [n for (t_, n) in JUDGE_EXCLUDED if t_ == target and n not in {p["name"] for p in patterns}]
    assert not missing_excluded, (target, missing_excluded)
    # current host outcome
    scored = [json.loads(m.read_text()) for m in build.glob("round-*/cand-*/metrics.json")]
    accepted = [json.loads(l)["task_id"] for l in (run / "accepted/tasks.jsonl").open() if l.strip()] \
        if (run / "accepted/tasks.jsonl").exists() else []
    audits = [json.loads(f.read_text()) for f in sorted((run / "audit/failure_match").glob("*.json"))]
    verdicts = collections.Counter(a.get("verdict") for a in audits)
    repro = {p["name"]: collections.Counter() for p in patterns}
    for a in audits:
        slots = a.get("student_rollouts") or []
        passed_all = bool(slots) and all(s.get("status") == "pass" for s in slots)
        for p in a.get("patterns", []):
            if p.get("name") in repro:
                repro[p["name"]]["student_passed_all" if passed_all else p.get("reproduced")] += 1
    def gate(m):
        if m.get("accept"):
            return "accepted"
        if "inconclusive" in (m.get("reason") or ""):
            return "inconclusive"
        if (m.get("student_rate") or 0) > 0.2:
            return "student_too_strong"
        if (m.get("teacher_clean_rate") or 0) < 0.6:
            return "teacher_too_weak"
        return "similarity_below_6" if m.get("similarity") is not None else "other"
    toml = (S / "synth/cache/deep-swe/tasks" / target / "task.toml").read_text()
    upstream = next((l.split("=", 1)[1].strip().strip('"') for l in toml.splitlines()
                     if l.strip().startswith("repository_url")), None)
    grade = json.loads((S / "synth/cache/deep-swe/tasks" / target / "tests/config.json").read_text()).get("grade", {})
    def cand(r):
        return {k: r[k] for k in ("rank", "repo", "score", "sub", "why", "tree_path", "language", "license", "flags")} | {
            "already_host_for": host_of.get(r["repo"].lower(), [])}
    st = stats.get(target, {})
    tasks.append({
        "deepswe_task": target,
        "run": run.name,
        "target_pass_counts": {k: st.get(k) for k in ("sc14_pass", "sc14_fail", "sc14_infra", "kimi_pass", "kimi_fail")},
        "upstream_repo": upstream,
        "grade_format": {k: grade.get(k) for k in ("format", "tool_label")},
        "artifacts": {
            "deepswe_task_dir": rel(S / "synth/cache/deep-swe/tasks" / target),
            "target_rollouts_rendered": rel(analysis / "inputs/rollouts"),
            "target_rollouts_raw_records": rel(S / "data/targets/sc14-out131k-kimi/records" / f"{target}.jsonl"),
            "failure_analysis_md": rel(analysis / "failure_analysis.md"),
            "failure_analysis_json": rel(analysis / "result.json"),
            "failure_analysis_inputs": rel(analysis / "inputs"),
            "failure_analysis_evidence": rel(analysis / "evidence"),
            "current_host_build_dir": rel(build),
            "current_host_understanding": rel(build / "understanding"),
            "current_host_plan": rel(build / "plan"),
            "accepted_tasks_jsonl": rel(run / "accepted/tasks.jsonl"),
            "audit_dir": rel(run / "audit/failure_match"),
            "run_watch_log": rel(run / "watch.md"),
        },
        "target_failure_patterns": patterns,
        "transferable_signature": result.get("transferable_signature"),
        "current_host": cand(current) | {
            "candidates_probed": len(scored), "similarity_judged": sum(m.get("similarity") is not None for m in scored),
            "accepted": len(accepted),
            "best_similarity": max([m.get("similarity") for m in scored if m.get("similarity") is not None] or [None]),
            "gate_outcomes": dict(collections.Counter(gate(m) for m in scored)),
            "toolchain_and_test_command": rel(build / "env/result.json"),
            "audit_verdicts_all_audited_tasks": dict(verdicts),
            "pattern_reproduction_all_audited_tasks": {k: dict(v) for k, v in repro.items()},
        },
        "abandoned_hosts": abandoned,
        "candidate_pool": [cand(r) for r in pool],
        "flagged_entries_skipped": skipped_flagged,
        "output": {"json": f"S/repo_reselect/out/{target}.json", "md": f"S/repo_reselect/out/{target}.md"},
    })
for t in tasks:  # optional artifacts: no accepted task yet (aiomonitor, etree, promreload), no evidence dir (wasmi)
    for k in ("accepted_tasks_jsonl", "failure_analysis_evidence"):
        if not (S / t["artifacts"][k][2:]).exists():
            t["artifacts"][k] = None
missing = [(t["deepswe_task"], k, v) for t in tasks for k, v in t["artifacts"].items()
           if v and not (S / v[2:]).exists()]
manifest = {"created": "2026-10-22", "S": str(S), "note": "S/ means " + str(S),
            "all_current_hosts": hosts, "tasks": tasks, "missing_artifacts": missing}
(S / "repo_reselect/manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(len(tasks), "tasks; missing artifacts:", missing)
for t in tasks:
    print(f'{t["deepswe_task"]:42} host {t["current_host"]["repo"]:38} acc {t["current_host"]["accepted"]} pool '
          + ", ".join(f'{c["rank"]}:{c["repo"]}' for c in t["candidate_pool"]))
