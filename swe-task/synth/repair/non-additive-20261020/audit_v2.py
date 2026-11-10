"""Per-task records of the 2026-10-20 test_paths_mode=non_additive repair (JS/TS: official1-vitest/testem):
runs/<run>/audit/grading_repair_v2/<task_id>.json. Read-only except for those files (atomic writes).
  ../atp/runtime/controller/bin/python3 -B audit_v2.py <jobs.json> [--write] [--notes notes.json] [--install installs.json]"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/tools")
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
import argparse, json
from pathlib import Path
import grading_repair as gr

S = gr.SYNTH
V1_TASKS = gr.REPAIR / "tasks"
GOOD = ("pass", "fail")


def read(path):
    return gr.read_json(path) if Path(path).exists() else None


def failing(x):
    return list((x or {}).get("failing_tests") or [])


def changed(before, after):
    b, a = set(before or []), set(after or [])
    return {"now_failing": sorted(a - b), "now_passing": sorted(b - a)}


def brief(result):
    if not result:
        return None
    if "error" in result:
        return {"error": result["error"][-300:], "at": result.get("at")}
    sanitize = result.get("sanitize") or {}
    return {"status": result.get("status"), "reward": result.get("reward"), "f2p": result.get("f2p"),
            "p2p": result.get("p2p"), "exit_code": result.get("exit_code"), "timed_out": result.get("timed_out"),
            "apply_failed": result.get("apply_failed"), "failing_tests": failing(result),
            "failing_tests_count": len(failing(result)), "sanitized": sanitize.get("actions") or {},
            "model_patch_applied": sanitize.get("applied"), "out_dir": result.get("out_dir"), "at": result.get("at"),
            "grader_sha256": result.get("grader_sha256"), "config_sha256": result.get("config_sha256")}


def rates(slots, key):
    """score()'s rates over the statuses in slots[*][key]; new kimi passes have no teacher audit (cheated None):
    'kimi_clean' counts them as clean, 'kimi_clean_audited_only' does not."""
    s = [x for x in slots if x["model"] == gr.STUDENT and x[key] in GOOD]
    k = [x for x in slots if x["model"] == gr.TEACHER and x[key] in GOOD]
    clean = [x for x in k if x[key] == "pass" and x["cheated"] is not True]
    audited = [x for x in clean if x["cheated"] is False]
    return {"sc14": len([x for x in s if x[key] == "pass"]) / len(s) if s else None,
            "sc14_pass": len([x for x in s if x[key] == "pass"]), "sc14_valid": len(s),
            "kimi_clean": len(clean) / len(k) if k else None, "kimi_clean_pass": len(clean), "kimi_valid": len(k),
            "kimi_clean_audited_only": len(audited) / len(k) if k else None,
            "kimi_cheated_pass": len([x for x in k if x[key] == "pass" and x["cheated"] is True]),
            "kimi_unaudited_pass": sorted(x["slot"] for x in clean if x["cheated"] is None)}


def task_record(job, plan, notes, installs):
    run, task = job["run"], job["task"]
    entry = gr.plan_entries(plan, run, [task], False)[0]
    scored, probe = Path(entry["scored_dir"]), Path(entry["probe"])
    home2 = gr.task_home(run, task)
    stage2, validate2, regrade2 = read(home2 / "stage.json"), read(home2 / "validate.json"), \
        read(home2 / "regrade.json") or {"slots": {}}
    v1_regrade = read(V1_TASKS / run / task / "regrade.json") or {"slots": {}}
    v1_audit_path = gr.RUNS / run / "audit/grading_repair" / f"{task}.json"
    v1_audit = read(v1_audit_path) or {}
    teacher = read(scored / "teacher_audit.json") or {}
    records = {r["slot"]: r for r in gr.probe_records(probe).get(task) or []}
    test_reset = set(entry["test_patch_paths"])
    globs = entry["globs"]
    mode = (stage2 or {}).get("test_paths_mode") or gr.task_mode(entry["language"])
    slots = []
    for name in sorted(records, key=lambda n: (n.split("-")[0] != gr.STUDENT, n)):
        rec = records[name]
        recorded = gr.md_header(scored / "rollouts" / f"{name}.md")[1] or rec["status"]
        verdict = teacher.get(name) if isinstance(teacher.get(name), dict) else None
        patch = rec["patch"]
        hits = gr.matching(gr.NEW.touched_paths(patch), globs)
        additive = gr.NEW.additive_paths(patch)
        kinds = gr.path_kinds(patch)
        surviving = [p for p in hits if p not in test_reset]
        kept = [p for p in surviving if p in additive]
        other = [p for p in surviving if p not in additive]
        cls = "unaffected" if not surviving else "same_as_all" if not kept else "same_as_old" if not other else "mixed"
        v1 = v1_regrade["slots"].get(name) or {}
        v2 = regrade2["slots"].get(name) or {}
        all_r, refined_r = brief(v1.get("new")), brief(v2.get("new"))
        control = brief(v2.get("old")) or brief(v1.get("old"))
        control_src = "v2" if v2.get("old") else "v1" if v1.get("old") else None
        if refined_r and refined_r.get("status") in GOOD:
            refined, source = refined_r["status"], "regraded"
        elif cls == "unaffected":
            refined, source = recorded, "unaffected: no test-path edit outside test.patch's files"
        elif cls == "same_as_old":
            refined, source = recorded, "inferred: every surviving test-path edit is kept-additive, so the graded " \
                                        "tree equals the original grading's"
        elif cls == "same_as_all":
            refined = (all_r or {}).get("status") if (all_r or {}).get("status") in GOOD else recorded
            source = "inferred: no kept-additive file, so the graded tree equals the 'all' grader's" + \
                     ("" if all_r else " (not regraded under 'all' either: recorded outcome)")
        else:
            refined, source = None, "unknown: mixed edits, not regraded"
        all_status = (all_r or {}).get("status") if (all_r or {}).get("status") in GOOD else None
        cheated = verdict.get("cheated") if verdict else None
        if rec["model"] == gr.TEACHER and recorded != "pass" and refined == "pass":
            cheated = None  # a new pass: no teacher audit
        item = {"slot": name, "model": rec["model"], "recorded": recorded, "probe_status": rec["status"],
                "recorded_f2p": rec.get("f2p"), "recorded_p2p": rec.get("p2p"),
                "recorded_failing_tests": rec.get("failing_tests") if recorded == rec["status"] else None,
                "cheated": cheated if rec["model"] == gr.TEACHER else None,
                "edited_test_files": {p: kinds.get(p, "modified") for p in hits},
                "surviving_test_files": {p: kinds.get(p, "modified") for p in surviving},
                "class": cls, "kept_additive_predicted": kept,
                "restored_or_deleted_predicted": other,
                "all": all_status, "all_detail": all_r, "control_old": (control or {}).get("status"),
                "control_detail": control, "control_source": control_src,
                "refined": refined, "refined_source": source, "refined_detail": refined_r,
                "test_paths_mode": mode}
        if refined_r and "sanitized" in refined_r:
            actual = refined_r["sanitized"].get("kept-additive", [])
            item["kept_additive"] = actual
            item["kept_additive_effective"] = [p for p in actual if p not in test_reset]
            item["kept_additive_matches_prediction"] = sorted(item["kept_additive_effective"]) == sorted(kept)
        else:
            item["kept_additive"] = None
            item["kept_additive_effective"] = kept if cls != "unaffected" else []
        if refined in GOOD and refined != recorded:
            ctrl = item["control_old"]
            item["verdict"] = "flip" if ctrl == recorded else "flaky" if ctrl in GOOD else "unconfirmed"
        elif refined in GOOD:
            item["verdict"] = "reverted_to_recorded" if all_status and all_status != recorded else "unchanged"
        else:
            item["verdict"] = "unknown"
        if refined_r:
            item["changed_tests_refined_vs_recorded"] = changed(item["recorded_failing_tests"], failing(refined_r)) \
                if item["recorded_failing_tests"] is not None else None
            item["changed_tests_refined_vs_all"] = changed(failing(all_r), failing(refined_r)) if all_r else None
        item["differs_from_recorded_and_all"] = bool(refined in GOOD and all_status and refined != recorded
                                                     and refined != all_status)
        note = notes.get(f"{task}/{name}")
        if note:
            item["explanation"] = note
        slots.append(item)
    for x in slots:
        x["status_all_for_rates"] = x["all"] if x["all"] else x["recorded"]
    before, all_rates, after = rates(slots, "recorded"), rates(slots, "status_all_for_rates"), None
    for x in slots:  # unknown (mixed, not regraded) slots: recorded here; bounds below
        x["status_after_for_rates"] = x["refined"] if x["refined"] in GOOD else x["recorded"]
        unknown_slot = x["refined"] not in GOOD and x["recorded"] in GOOD
        x["best_case"] = ("fail" if x["model"] == gr.STUDENT else "pass") if unknown_slot else x["status_after_for_rates"]
        x["worst_case"] = ("pass" if x["model"] == gr.STUDENT else "fail") if unknown_slot else x["status_after_for_rates"]
    after = rates(slots, "status_after_for_rates")
    best, worst = rates(slots, "best_case"), rates(slots, "worst_case")
    status_before = entry["status"]
    in_after = gr.in_range(after["sc14"], after["kimi_clean"])
    in_after_audited = gr.in_range(after["sc14"], after["kimi_clean_audited_only"])
    in_best, in_worst = gr.in_range(best["sc14"], best["kimi_clean"]), gr.in_range(worst["sc14"], worst["kimi_clean"])
    unknown = [x["slot"] for x in slots if x["refined"] not in GOOD and x["recorded"] in GOOD]
    if status_before == "accepted":
        recommendation = "keep" if in_worst else "withdraw" if not in_best else "unclear"
    else:
        recommendation = "newly_in_range" if in_worst else "still_rejected" if not in_best else "unclear"
    flips = [x["slot"] for x in slots if x.get("verdict") == "flip"]
    reverted = [x["slot"] for x in slots if x.get("verdict") == "reverted_to_recorded"]
    v1_flips = [f if isinstance(f, str) else f.get("slot") for f in v1_audit.get("flips", [])]
    record = {
        "run": run, "task_id": task, "cand": scored.name, "round": entry["round"],
        "status_before": status_before, "reason_before": entry.get("reason"), "similarity": entry.get("similarity"),
        "globs": globs, "test_paths_mode": mode, "test_patch_paths": entry["test_patch_paths"],
        "generated_at": gr.now(), "generated_by": "hengyuf non_additive repair agent (2026-10-20): tools/grading_repair.py "
        "stage/validate/regrade via repair/non-additive-20261020/drive.py; this file by its audit_v2.py",
        "summary": None, "recommendation": recommendation,
        "statuses_before": {x["slot"]: x["recorded"] for x in slots},
        "statuses_all": {x["slot"]: x["status_all_for_rates"] for x in slots},
        "statuses_after": {x["slot"]: x["refined"] for x in slots},
        "rates_before": before, "rates_all": all_rates, "rates_after": after,
        "in_range_before": gr.in_range(before["sc14"], before["kimi_clean"]),
        "in_range_all": gr.in_range(all_rates["sc14"], all_rates["kimi_clean"]),
        "in_range_after": in_after, "in_range_after_audited_only": in_after_audited,
        "bounds_over_unknown_slots": {"best": best, "worst": worst, "in_range_best": in_best,
                                      "in_range_worst": in_worst, "unknown_slots": unknown},
        "v1_recommendation": v1_audit.get("recommendation"), "v1_flips": v1_flips,
        "flips": flips, "reverted_v1_flips": reverted, "unknown_slots": unknown,
        "flaky_slots": [x["slot"] for x in slots if x.get("verdict") == "flaky"],
        "unconfirmed_slots": [x["slot"] for x in slots if x.get("verdict") == "unconfirmed"],
        "differs_from_recorded_and_all": [x["slot"] for x in slots if x["differs_from_recorded_and_all"]],
        "kept_additive_by_slot": {x["slot"]: x["kept_additive"] if x["kept_additive"] is not None
                                  else x["kept_additive_effective"] for x in slots if x["class"] != "unaffected"},
        "regraded_slots": [x for x in slots if x["refined_source"] == "regraded"],
        "not_regraded_slots": [{k: x[k] for k in ("slot", "model", "recorded", "all", "refined", "refined_source",
                                                   "class", "kept_additive_effective", "surviving_test_files")}
                               for x in slots if x["refined_source"] != "regraded"],
        "validation": None, "install": installs.get(f"{run}/{task}"),
        "stage": {k: (stage2 or {}).get(k) for k in ("at", "config_sha256", "grader_sha256", "new_dir", "old_dir",
                                                      "test_paths", "test_paths_mode", "source_grader", "old_sha256")},
        "sources": {"grader": {"path": str(gr.NEW_GRADER), "sha256": gr.file_sha(gr.NEW_GRADER)},
                    "first_sanitizer_sha256": sorted(gr.PRIOR_GRADERS),
                    "old_grader": {"path": str(gr.OLD_GRADER), "sha256": gr.file_sha(gr.OLD_GRADER)},
                    "plan": str(gr.PLAN), "plan_created_at": plan.get("created_at"), "probe": str(probe),
                    "scored_dir": str(scored), "v1_audit": str(v1_audit_path) if v1_audit else None,
                    "v1_regrade": str(V1_TASKS / run / task / "regrade.json"),
                    "stage": str(home2 / "stage.json"), "validate": str(home2 / "validate.json") if validate2 else None,
                    "regrade": str(home2 / "regrade.json") if (home2 / "regrade.json").exists() else None},
        "notes": list(notes.get(f"{task}/*", [])),
        "legend": {
            "recorded": "the slot's current status (rollouts/<slot>.md header, what score() used)",
            "all": "the first sanitizer's regrade (2026-10-19, every test path restored; repair/tasks/)",
            "refined": "test_paths_mode non_additive (this repair; repair/tasks-v2/): regraded, or inferred when the "
                       "graded tree provably equals one already graded (refined_source)",
            "class": "unaffected: no test-path edit outside test.patch's files; same_as_all: no kept-additive file; "
                     "same_as_old: every surviving test-path edit is kept-additive (the original grading's tree); "
                     "mixed: some kept, some restored/deleted (a tree no grader produced before: regraded)",
            "kept_additive": "the refined grader's '[verifier] test-sanitize: kept-additive' paths in the regrade "
                             "(kept_additive_effective: minus the files test.patch resets anyway); "
                             "kept_additive_predicted: the same computed offline from model.patch",
            "control_old": "DeepSWE's grader on the same scratch copy (v1 or v2 run)"},
        "rules": {
            "flip": "refined outcome differs from the recorded one AND the OLD-grader control reproduces the recorded "
                    "outcome",
            "sc14": "passes / valid (pass+fail) sc14 slots",
            "kimi_clean": "non-cheated passes / valid kimi slots; cheated passes stay excluded; a new kimi pass has no "
                          "teacher audit (counted as clean in kimi_clean, left out of kimi_clean_audited_only)",
            "accepted": "keep iff sc14 <= 0.2 and clean kimi >= 0.6 after the refined regrade, else withdraw (main "
                        "decides)",
            "rejected": "newly_in_range iff both hold after the refined regrade (then main runs the similarity judge), "
                        "else still_rejected",
            "unknown_slots": "mixed slots that were not regraded: the verdict uses bounds (best: unknown sc14 FAIL, "
                             "unknown kimi clean PASS; worst: the opposite): keep/newly_in_range iff the worst case "
                             "is in range, withdraw/still_rejected iff even the best case is out of range, else "
                             "unclear"},
    }
    if validate2:
        record["validation"] = {"at": validate2.get("at"), "checks": validate2.get("checks"),
                                "ok": validate2.get("all_ok"), "test_paths_mode": validate2.get("test_paths_mode"),
                                "grader_sha256": validate2.get("grader_sha256"),
                                "config_sha256": validate2.get("config_sha256"),
                                "reference": (validate2["results"].get("solution") or {}).get("reward"),
                                "empty": (validate2["results"].get("empty") or {}).get("reward"),
                                "tamper": (validate2["results"].get("tamper") or {}).get("reward"),
                                "results": {n: brief(r) for n, r in validate2["results"].items()},
                                "validate_json": str(home2 / "validate.json")}
    r_b, r_a = before, after
    record["summary"] = (
        (f"validation {'OK' if (record['validation'] or {}).get('ok') else 'NOT OK'} (reference "
         f"{(record['validation'] or {}).get('reference')}, empty {(record['validation'] or {}).get('empty')}, tamper "
         f"{(record['validation'] or {}).get('tamper')}); " if record["validation"] else "")
        + f"mode {mode}; {len(record['regraded_slots'])} slot(s) regraded with the refined grader; v1 flips "
          f"{v1_flips or 'none'}; refined flips vs recorded {flips or 'none'}; v1 flips reverted {reverted or 'none'}; "
          f"sc14 {r_b['sc14']} -> all {all_rates['sc14']} -> refined {r_a['sc14']}, clean kimi {r_b['kimi_clean']} -> "
          f"all {all_rates['kimi_clean']} -> refined {r_a['kimi_clean']}"
          + (f" (audited only {r_a['kimi_clean_audited_only']})" if r_a["kimi_unaudited_pass"] else "")
          + (f"; unknown {unknown}" if unknown else "") + f": {recommendation}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--notes")
    ap.add_argument("--install")
    ap.add_argument("--out", default="/tmp/hengyuf-nonadditive/audit_v2_preview.json")
    a = ap.parse_args()
    plan = gr.load_plan()
    notes = json.load(open(a.notes)) if a.notes else {}
    installs = json.load(open(a.install)) if a.install else {}
    seen, out = set(), []
    for job in json.load(open(a.jobs)):
        key = (job["run"], job["task"])
        if key in seen:
            continue
        seen.add(key)
        record = task_record(job, plan, notes, installs)
        out.append(record)
        print(f"{job['run'][10:]} {job['task']}: {record['summary']}")
        if a.write:
            path = gr.RUNS / job["run"] / "audit/grading_repair_v2" / f"{job['task']}.json"
            gr.write_json_atomic(path, record, backup=False)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
