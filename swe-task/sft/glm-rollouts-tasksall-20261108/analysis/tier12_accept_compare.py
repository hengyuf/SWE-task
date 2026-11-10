#!/usr/bin/env python3
"""User question 2026-11-09 (issue #1): for every tier-1 and tier-2 task, compare GLM-5.3 with the Nemotron student
(sc14) and say how many tasks are still accepted under the original acceptance rule with GLM-5.3 as the teacher.

    <atp controller python> -I -B analysis/tier12_accept_compare.py   -> analysis/tier12-accept-compare.{md,json}

Tiers as in tasks.json: 1 = accepted_or_strong, 2 = audit partial (inventory sft/tasks-all-20261106).
GLM-5.3 rollouts (5 per task, final records with every re-run applied):
  * this campaign (analysis/rollouts.jsonl from final_report.py),
  * glm-rollouts-sftv3-20261107 (analysis/final-rollouts.jsonl, the 220 SFT-v3 tasks; same self-hosted settings),
  * teacher-rollouts-v2-20261025 (hosted GLM-5.3, 1M context / 131k output) for the 2 official1-scriggo gomacro tasks.
Student = sc14 build-probe rollouts (inventory student_rate / sc14_pass, sc14_graded); Kimi-K3 = the original teacher
(teacher_clean_rate, kimi_pass, kimi_graded).
Rule (synth/task_builder.py score()): >= 3 valid rollouts per model; no answer retrieval by the teacher; student rate
<= 0.2 and clean teacher rate >= 0.6; then failure similarity >= 6. With GLM as the teacher: GLM's raw pass rate (its
passes were not audited for cheating, so it is an upper bound on its clean rate); the similarity score is the
original one (it judges the student's failures only), and a task whose rates were out of range with Kimi never got
one ("similarity not scored").
"""
import collections
import glob
import json
from pathlib import Path

GR = Path(__file__).resolve().parents[1]
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
INVENTORY = S / "sft/tasks-all-20261106/all_tasks.jsonl"
SFTV3 = S / "sft/glm-rollouts-sftv3-20261107"
HOSTED = S / "sft/teacher-rollouts-v2-20261025"


def key_of(row):
    return f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"


def tier_of(row):
    if row["accepted_or_strong"]:
        return 1
    return {"partial": 2, "weak": 3}.get(row["audit"], 4)


def glm_records(rows_by_key):
    """{task key: (source, [statuses])}"""
    out = {}
    for line in (GR / "analysis/rollouts.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["key"], ("self-hosted (this campaign)", []))[1].append(r["status"])
    v3 = {(r["run"], r["task_id"]): key_of(r) for r in rows_by_key.values() if r["sft_v3"] > 0}
    plan = {t["task_id"]: t["run"] for t in json.loads((SFTV3 / "tasks.json").read_text())["tasks"]}
    for line in (SFTV3 / "analysis/final-rollouts.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = v3[(plan[r["task_id"]], r["task_id"])]
        out.setdefault(key, ("self-hosted (SFT-v3 run, 2026-11-07)", []))[1].append(r["status"])
    runs = {t["task_id"]: t["run"] for t in json.loads((S / "sft/train-acc-v2-20261024/tasks.json").read_text())["tasks"]}
    by_run_id = {(r["run"], r["task_id"]): k for k, r in rows_by_key.items()}
    for path in sorted(glob.glob(str(HOSTED / "results/glm-*.json"))):
        for task_id, records in json.loads(Path(path).read_text())["records"].items():
            key = by_run_id[(runs[task_id], task_id)]
            if key not in out:  # only tasks without self-hosted rollouts (the 2 gomacro tasks)
                out.setdefault(f"hosted::{key}", ("hosted (teacher-rollouts-v2)", []))[1].extend(r["status"] for r in records)
    return {k.removeprefix("hosted::"): v for k, v in out.items()}


def main():
    rows = {key_of(r): r for r in (json.loads(l) for l in INVENTORY.read_text().splitlines() if l.strip())}
    glm = glm_records(rows)
    tasks = []
    for key, row in rows.items():
        tier = tier_of(row)
        if tier not in (1, 2):
            continue
        source, statuses = glm.get(key, (None, []))
        g_pass, g_graded = sum(s == "pass" for s in statuses), sum(s in ("pass", "fail") for s in statuses)
        s_graded = row.get("sc14_graded") or 0
        s_rate = row.get("student_rate")
        if s_rate is None and s_graded:
            s_rate = row["sc14_pass"] / s_graded
        g_rate = g_pass / g_graded if g_graded else None
        leaky = "answer_retrieval" in (row.get("flags") or []) or row.get("reject_reason") == "leaky"
        if g_graded < 3 or s_graded < 3 or s_rate is None:
            verdict = "inconclusive (< 3 valid rollouts of a model)"
        elif leaky:
            verdict = "rejected: answer retrieval (leaky task)"
        elif s_rate > 0.2:
            verdict = "rejected: student rate > 0.2"
        elif g_rate < 0.6:
            verdict = "rejected: GLM rate < 0.6"
        elif row.get("similarity") is None:
            verdict = "rates in range, similarity never scored"
        elif row["similarity"] >= 6:
            verdict = "accepted"
        else:
            verdict = "rejected: similarity < 6"
        tasks.append({"key": key, "tier": tier, "status": row["status"], "audit": row["audit"],
                      "originally_accepted": row["status"] == "accepted", "glm_source": source,
                      "glm_pass": g_pass, "glm_graded": g_graded, "glm_rate": g_rate,
                      "sc14_pass": row.get("sc14_pass"), "sc14_graded": s_graded, "student_rate": s_rate,
                      "kimi_pass": row.get("kimi_pass"), "kimi_graded": row.get("kimi_graded"),
                      "kimi_clean_rate": row.get("teacher_clean_rate"), "similarity": row.get("similarity"),
                      "reject_reason": row.get("reject_reason"), "glm_verdict": verdict})
    out = {"tasks": tasks}
    (GR / "analysis/tier12-accept-compare.json").write_text(json.dumps(out, indent=1) + "\n")

    def pct(p, n):
        return f"{p}/{n} = {100 * p / n:.1f}%" if n else "-"

    lines = ["# Tier 1 + 2: GLM-5.3 vs Nemotron student (sc14), and acceptance with GLM-5.3 as the teacher", ""]
    lines += ["## Accuracy (same tasks)", "",
              "| group | tasks | with GLM | GLM-5.3 | sc14 (Nemotron student) | Kimi-K3 (build probe) |",
              "|---|---|---|---|---|---|"]
    groups = [("tier 1, originally accepted", lambda t: t["tier"] == 1 and t["originally_accepted"]),
              ("tier 1, strong but not accepted", lambda t: t["tier"] == 1 and not t["originally_accepted"]),
              ("tier 2 (partial, none accepted)", lambda t: t["tier"] == 2),
              ("all tier 1 + 2", lambda t: True)]
    for name, f in groups:
        g = [t for t in tasks if f(t)]
        w = [t for t in g if t["glm_graded"]]
        lines.append(f"| {name} | {len(g)} | {len(w)} | {pct(sum(t['glm_pass'] for t in w), sum(t['glm_graded'] for t in w))} | "
                     f"{pct(sum(t['sc14_pass'] or 0 for t in w), sum(t['sc14_graded'] or 0 for t in w))} | "
                     f"{pct(sum(t['kimi_pass'] or 0 for t in w), sum(t['kimi_graded'] or 0 for t in w))} |")
    lines += ["", "## Acceptance with GLM-5.3 as the teacher (original rule otherwise)", "",
              "| group | tasks | accepted with GLM | rejected: GLM < 0.6 | rejected: student > 0.2 | similarity < 6 | "
              "rates OK, similarity never scored | leaky | inconclusive |", "|---|---|---|---|---|---|---|---|---|"]
    for name, f in groups:
        g = [t for t in tasks if f(t)]
        c = collections.Counter(t["glm_verdict"] for t in g)
        lines.append(f"| {name} | {len(g)} | **{c['accepted']}** | {c['rejected: GLM rate < 0.6']} | "
                     f"{c['rejected: student rate > 0.2']} | {c['rejected: similarity < 6']} | "
                     f"{c['rates in range, similarity never scored']} | {c['rejected: answer retrieval (leaky task)']} | "
                     f"{c['inconclusive (< 3 valid rollouts of a model)']} |")
    acc = [t for t in tasks if t["originally_accepted"]]
    still = [t for t in acc if t["glm_verdict"] == "accepted"]
    newly = [t for t in tasks if not t["originally_accepted"] and t["glm_verdict"] == "accepted"]
    lost = collections.Counter(t["glm_verdict"] for t in acc if t["glm_verdict"] != "accepted")
    lines += ["", f"- Originally accepted (tier 1): {len(acc)}; **still accepted with GLM-5.3 as the teacher: {len(still)}** "
              f"({100 * len(still) / len(acc):.1f}%); lost: {dict(lost)}.",
              f"- Not accepted before, accepted with GLM-5.3: {len(newly)} "
              f"({dict(collections.Counter(t['reject_reason'] for t in newly))} were their original reasons).",
              f"- GLM rollouts by source: {dict(collections.Counter(t['glm_source'] for t in tasks))}.",
              "- GLM's rate is its raw pass rate (its passes were not audited for cheating, unlike Kimi's clean rate), "
              "so 'accepted with GLM' is an upper bound on that check; similarity is the original score."]
    (GR / "analysis/tier12-accept-compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
