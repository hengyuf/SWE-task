#!/usr/bin/env python3
"""teacher-deepswe collect phase (2026-11-02 PST): README.md for the FINAL collection (temp file + os.replace). Replaces
the top status block, adds the "Final collection" section before "Monitor window 16", puts it first on the See line and
updates the Files table. Numbers come from analysis/collect-summary.json, analysis/collect-verify.json,
analysis/independent-counts-20261102.json and dataset/MANIFEST.json; job facts are passed on the command line. Run once.

    /usr/bin/python3 -I -B final/readme_final_collect_20261102.py --job 21180895 --job-times "02:30:57-02:34:12" \
        --first-job 21180602 --first-job-times "02:17:47-02:21:08" --quota-user "..." --quota-project "..."
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parent.parent
MODELS = ("kimi", "glm", "deepseek")


def money(value):
    return f"${value:,.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--job-times", required=True)
    parser.add_argument("--first-job", required=True)
    parser.add_argument("--first-job-times", required=True)
    parser.add_argument("--quota-user", required=True)
    parser.add_argument("--quota-project", required=True)
    args = parser.parse_args()
    readme = (TD / "README.md").read_text()
    if "## Final collection (2026-11-02" in readme:
        raise SystemExit("README.md already has the Final collection section")
    s = json.loads((TD / "analysis" / "collect-summary.json").read_text())
    v = json.loads((TD / "analysis" / "collect-verify.json").read_text())
    ind = json.loads((TD / "analysis" / "independent-counts-20261102.json").read_text())
    man = json.loads((TD / "dataset" / "MANIFEST.json").read_text())
    assert v["ok"], v["problems"]
    assert all(n == 0 for n in man["checks"].values()), man["checks"]
    ov, bw, c, u, k = s["overview"], s["by_wave"], s["cost"], s["uncovered"], s["kimi_vs_1m"]
    for m in MODELS:  # the independent counts must agree with collect.py
        assert ind[m]["all"]["graded"] == ov[m]["graded"] and ind[m]["all"]["pass"] == ov[m]["pass"], m
        assert sorted(ind[m]["uncovered_yes"]) == sorted(u["per_teacher_yes"][m]), m
    assert sorted(ind["uncovered_any_yes"]) == sorted(u["any_teacher_yes"])
    n2 = s["new2_readings"]
    key_totals = {key: sum(c[m]["by_key"].get(key, 0.0) for m in MODELS) for key in ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")}
    g = ov["glm"]
    rows = []
    for m in MODELS:
        b = ov[m]
        status = "complete" if s["model_status"][m] == "complete" else "**PARTIAL** (cancelled by the user)"
        rows.append(f"| {m} | {status} | {b['graded']} / 565 | {b['pass']} | {b['pass_at_1_raw']} | "
                    f"{b['pass_at_1_excl_broken']} | {b['pass_at_1_excl_infra']} | {b['pass_at_1_excl_timeouts']} | "
                    f"{b['tasks_any_pass']} / {b['tasks_started']} | {b['compaction_rate']}% | "
                    f"{money(b['cost_per_graded_rollout'])} | {money(c[m]['all_attempts'])} |")
    waves = []
    for m in MODELS:
        for w in (1, 2):
            b = bw[f"{m} w{w}"]
            waves.append(f"| {m} w{w}{' (PARTIAL)' if m == 'glm' else ''} | {b['graded']} / {b['planned']} | "
                         f"{b['pass_at_1_excl_infra']} | {b['tasks_any_pass']} / {b['tasks_started']} | "
                         f"{b['base_pass_at_1_same_slots']} |")
    matched = s["matched"]
    m84 = matched["all three models have >= 1 graded slot"]
    m21 = matched["all three models have all 5 slots graded"]
    yes = {m: ", ".join(u["per_teacher_yes"][m]) or "none" for m in MODELS}
    section = f"""## Final collection (2026-11-02 PST): Kimi and DeepSeek complete, GLM PARTIAL (cancelled by the user)

This section follows OPERATOR_NOTE entry 2026-11-02 ~02:00 PST, item 3: GLM is reported as PARTIAL. Its graded slots
count as they stand, and its 5 slots in flight at the cancel are listed as cancelled (not run to the end, not
graded). GLM is left out of every "complete" claim. Full numbers and lists are in `FINAL-REPORT.md`.

**Outputs** (all from one run):
- `dataset/kimi.jsonl.gz` ({man['rows']['kimi']} rows), `glm.jsonl.gz` ({man['rows']['glm']}, graded slots only) and
  `deepseek.jsonl.gz` ({man['rows']['deepseek']}). Each row is one rollout: the Responses-API transcript in "output",
  patch, grade, re-run provenance, flags and hub cost.
- `dataset/pilots.jsonl.gz` ({man['rows']['pilots']}) and `dataset/index.jsonl` (every row without its transcript).
- `dataset/slots.jsonl` ({man['slots_ledger_rows']} lines): every planned slot (1,695) and the 3 pilots, with state and
  cost.
- `dataset/MANIFEST.json` (`complete` false, `model_status`), `per_task.csv` and `FINAL-REPORT.md`.
- `analysis/collect-summary.json`, `analysis/collect-kimi-vs-1m.csv`, `analysis/collect-verify.json` and
  `analysis/atp-slot-states.json`.

**Job.** CPU job {args.job} (cpu_interactive), {args.job_times} PST, exit 0. It was submitted through the babysit
pair with `final/submit_collect.sh` (`TDW_PARTITION=cpu_interactive,cpu_dataprocessing`) after `sbatch --test-only`
probes of the 3 CPU partitions. It ran `final/atp_states.py` (ATP's own slot states: {v['atp_states']}; the probe files
were unchanged by the check), then `collect.py --allow-incomplete --workers 16`. An earlier run, job {args.first_job}
({args.first_job_times} PST), produced the same numbers. Only the wording of the report's cost section differed, and
its outputs were overwritten. This agent touched no leg, sandbox or STOP file. Before the job, NVAPI_KEY_NEW_2 got one
key_budget.py reading (tag "collect final 2026-11-02 ..."), which logged the key name and numbers only.

**collect.py / verify_collection.py changes** (temp file + mv; neither file is run by a leg):
- The patch scripts are `final/collect_final_20261102.py` (main change), `final/collect_report_wording_20261102.py`
  (cost-section wording) and `final/verify_infra_rows_20261102.py`.
- Pre-change copies, hashes and diffs are in `provenance/pre-final-collect-20261102/` (`SHA256SUMS`,
  `POST-PATCH.SHA256SUMS`, `*.diff`). That directory also keeps the small outputs of the 10-31 partial collection.
- `USER_CANCELLED` (glm): a GLM campaign without a results file has state `cancelled_by_user`. Main's STOP file must
  exist, enforced by the check `user_cancel_stop_missing`. `model_status` is complete / partial_cancelled_by_user.
- Two new slot states:
  - `interrupted`: ATP's interrupted.json after a leg stop by main.
  - `cancelled_in_flight`: no end marker, ATP "active", started before the cancel (check
    `in_flight_started_after_cancel`). The start time, last file change and sandbox id are kept.
- pass@1 raw leaves out every slot stopped before it ended (cancelled / interrupted / cancelled_in_flight) and counts
  infra as a fail.
- Uncovered-task verdicts are yes / no / undetermined. Undetermined arises only through ungraded GLM slots.
- Also added: Kimi vs the 1M export by wave, pass@5 and 5/5; the hub legs per key and the NEW_2 readings; per_task.csv
  columns `_interrupted`, `_cancelled_in_flight` and `_model_status`; and a rewritten FINAL-REPORT.md.
- The changes were tested on a scratch root inside `final/` (two trimmed results files + GLM w2-g1) on the login node
  before the job. The scratch dir is removed.
- `final/verify_collection.py` adds: the new states in the ATP / progress.py comparisons; complete campaigns vs their
  results files (pass / fail / infra, final record = first-attempt record, no re-run or recovery); the GLM STOP
  files; and a complete campaign's final infra row.

**Checks:**
- collect.py's {len(man['checks'])} checks are all 0.
- `final/verify_collection.py` is ok, with no problems:
  - ledger: {v['ledger']['wave_slots']} wave slots (5 per model per task) + {v['ledger']['pilots']} pilots;
  - gz line counts = index = MANIFEST, and every sha256 = MANIFEST;
  - every row's status = its terminal.json reward ({v['rows']['status_checked']} rows; 1 final infra row: deepseek
    w1-g0 langchain r4);
  - graded counts = ATP completed (the kimi override left out);
  - progress.py's last row (01:57 PST: 1,392 graded, 803 passes, 20 ended ungraded, 5 active) = the ledger;
  - the 10 results files = the ledger: no brk / rec re-run, and every final record = the first-attempt record;
  - no -brk / -rec probe dir exists.
- `final/independent_counts_20261102.py` (-> `analysis/independent-counts-20261102.json`) recomputes the counts
  without collect.py: from the results files, a direct GLM probe scan and the two base per_task.csv files. It gives the
  same graded / pass counts, wave split, pass@5 and uncovered-task lists.

**Results** (graded slots; GLM PARTIAL):

| model | status | graded | pass | pass@1 raw | excl. broken-ended | excl. infra | excl. agent timeouts | pass@5: tasks >= 1 pass / graded | compaction | $ per graded rollout | hub cost, all attempts |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- Kimi counts w2-g1 13-0 (scriggo r0) as a graded fail (OPERATOR_NOTE 11-01 11:15 / 14:15). In ATP's view it is
  564 graded and pass@1 65.8, or 66.7 without its {ov['kimi']['agent_timeouts']} graded agent timeouts (all fails), as
  in PROGRESS.md.
- DeepSeek has 1 final infra slot: w1-g0 langchain-request-coalescing r4, where the verifier was killed at 1,800 s
  (needs_review).
- GLM: 9 infra endings, 6 ATP-retryable and 3 needs_review (the pwntools r0 / r2 verifier kills and the w2-g0
  fastapi-deprecation r2 sandbox 410). GLM also has 1 broken-ended graded slot, w1-g0 arktype r4 (a fail); its planned
  brk re-run was cancelled with GLM.

**Wave 1 (79 uncovered tasks) vs wave 2 (34 covered tasks):**

| model x wave | graded / planned | pass@1 excl. infra | pass@5: tasks >= 1 pass / graded | base, same slots |
|---|---|---|---|---|
{chr(10).join(waves)}

**Matched task sets:**
- On the {m84['tasks']} tasks where all three have a graded slot, pass@1 is kimi
  {m84['per_model']['kimi']['pass_at_1_excl_infra']}, glm {m84['per_model']['glm']['pass_at_1_excl_infra']} and deepseek
  {m84['per_model']['deepseek']['pass_at_1_excl_infra']}.
- On the {m21['tasks']} tasks where all three have all 5 graded: kimi {m21['per_model']['kimi']['pass_at_1_excl_infra']},
  glm {m21['per_model']['glm']['pass_at_1_excl_infra']}, deepseek {m21['per_model']['deepseek']['pass_at_1_excl_infra']}.

**Uncovered tasks with base <= 2/10 ({u['tasks']} of the 79) and a teacher at >= 3/5:**
- Kimi: {u['per_teacher']['kimi']['yes']} tasks ({yes['kimi']}).
- DeepSeek: {u['per_teacher']['deepseek']['yes']} tasks ({yes['deepseek']}).
- GLM (PARTIAL): {u['per_teacher']['glm']['yes']} yes ({yes['glm']}), {u['per_teacher']['glm']['undetermined']}
  undetermined, {u['per_teacher']['glm']['no']} no.
- Kimi or DeepSeek (both complete): {len(u['complete_teachers_yes'])}. Any teacher: {len(u['any_teacher_yes'])}.
- No teacher yet: {len(u['no_teacher_yes'])}. For {len(u['any_teacher_undetermined_only'])} of these, only GLM's
  missing slots could still have changed the answer.

**Kimi here (256k) vs Baihe's 1M-context Kimi export (context only, different pipeline and serving period):**
- avg@5: {k['kimi256k_pass_at_1']} vs {k['kimi1m_all_tasks_avg']} on all 113 tasks; wave 1 {k['wave1_256k']} vs
  {k['kimi1m_wave1_avg']}; wave 2 {k['wave2_256k']} vs {k['kimi1m_wave2_avg']}.
- pass@5: {k['complete_256k_any_pass']} vs {k['complete_1m_any_pass']} tasks. All 5 passing: {k['all5_256k']} vs
  {k['all5_1m']}.
- Per task: higher here {k['tasks_256k_better']}, lower {k['tasks_256k_worse']}, tied {k['tasks_tied']}.
- Uncovered base <= 2/10 tasks at >= 3/5: {len(k['uncovered_yes_256k'])} vs {len(k['uncovered_yes_1m'])}.

**GLM at the cancel:**
- {g['cancelled_in_flight']} attempts in flight: w1-g0 8-3, w1-g1 20-1, w1-g2 8-0, w2-g0 2-0 and w2-g1 0-3. Tasks and
  sandbox ids are in FINAL-REPORT.md; main deletes those sandboxes.
- {g['interrupted'] + g['cancelled'] + g['infra']} slots had ended ungraded earlier and were never retried:
  {g['interrupted']} interrupted by main's stops, {g['cancelled']} cancelled by the 10-31 floor stop and {g['infra']}
  infra endings.
- {g['not_started']} slots never started.
- Of GLM's {money(c['glm']['all_attempts'])} hub spend, {money(c['glm']['ungraded_attempts'])} went to attempts that
  ended ungraded or were cut off.

**Cost (journal; each hub reply is attributed to the leg running when it arrived):**
- By model, all attempts: kimi {money(c['kimi']['all_attempts'])}, glm {money(c['glm']['all_attempts'])} (PARTIAL),
  deepseek {money(c['deepseek']['all_attempts'])}; pilots {money(c['pilots']['all_attempts'])}.
- NVAPI_KEY_NEW_3 (wave-1 legs, 10-30 23:07 PDT until the 10-31 10:30-10:40 floor stop): {money(key_totals['NVAPI_KEY_NEW_3'])}.
- NVAPI_KEY_NEW_2: {money(key_totals['NVAPI_KEY_NEW_2'])} for the legs from 10-31 18:29, plus
  {money(c['pilots']['all_attempts'])} for the pilots.
- **NVAPI_KEY_NEW_2 now: {money(n2['last']['spend'])} spent / {money(n2['last']['remaining'])} left ({n2['last']['at']}).**
  Since the hub's 10-31 18:11 spend reset, NEW_2's spend grew by {money(n2['last']['spend'] - n2['first_after_reset']['spend'])}.
  This campaign accounts for {money(key_totals['NVAPI_KEY_NEW_2'])} of it. The rest is other use of the key (the r6
  judge, budget checks) and any gap between the journal and LiteLLM's figure.

**Quota (after the collection):** user {args.quota_user}; project 10581 {args.quota_project}. The dataset files take
about 0.33 GB and a handful of inodes.

"""
    start = readme.index("**Status (2026-11-02 01:56 PST)")
    see = readme.index('\nSee "Monitor window 16"', start)
    status = f"""**Status (2026-11-02, final collection): KIMI AND DEEPSEEK COMPLETE; GLM PARTIAL, CANCELLED BY THE USER; NO tdw-
LEG RUNS.** The final dataset, per_task.csv and FINAL-REPORT.md come from the final collection (CPU job
{args.job}); see "Final collection".
- **Kimi:** 565 / 565 slots graded (with the override), {ov['kimi']['pass']} passes, pass@1 {ov['kimi']['pass_at_1_excl_infra']}.
- **DeepSeek:** 564 graded + 1 final infra, {ov['deepseek']['pass']} passes, pass@1 {ov['deepseek']['pass_at_1_excl_infra']}
  (raw {ov['deepseek']['pass_at_1_raw']}).
- **GLM (PARTIAL):** {g['graded']} of 565 graded, {g['pass']} passes, pass@1 {g['pass_at_1_excl_infra']}; cancelled by
  the user (OPERATOR_NOTE entry 2026-11-02 ~02:00 PST: hengyuf, verbatim, "cancel the DeepSWE teacher rollout for
  GLM"). Main wrote STOP files in babysit/glm-* at 01:46:40 PST and cancelled the 5 GLM legs (CANCELLED 01:46:45).
  These STOP files stay for good.
- **Uncovered tasks with base <= 2/10 ({u['tasks']}) and a teacher at >= 3/5:** kimi {u['per_teacher']['kimi']['yes']},
  deepseek {u['per_teacher']['deepseek']['yes']}, glm {u['per_teacher']['glm']['yes']} (PARTIAL), any teacher
  {len(u['any_teacher_yes'])}.
- **Hub cost (journal):** {money(sum(c[m]['all_attempts'] for m in MODELS))} for the wave legs, plus
  {money(c['pilots']['all_attempts'])} for the pilots. NVAPI_KEY_NEW_2 has {money(n2['last']['remaining'])} left
  ({n2['last']['at']}). This campaign makes no more hub calls.
- Daylight saving time ended at 2026-11-01 02:00 PDT; times after that are PST.
"""
    readme = readme[:start] + status + readme[see:]
    readme = readme.replace('\nSee "Monitor window 16" (GLM cancelled), ',
                            '\nSee "Final collection" (2026-11-02), "Monitor window 16" (GLM cancelled), ', 1)
    anchor = "## Monitor window 16 (2026-11-02 00:15 - 01:56 PST; GLM cancelled by the user)"
    assert readme.count(anchor) == 1
    readme = readme.replace(anchor, section + anchor, 1)
    # Files table
    old_collect = "| `collect.py` | final dataset:"
    assert readme.count(old_collect) == 1
    i = readme.index(old_collect)
    j = readme.index("\n", i)
    readme = (readme[:i] + "| `collect.py` | final dataset: `dataset/<model>.jsonl.gz`, `dataset/pilots.jsonl.gz`, "
              "`dataset/index.jsonl`, `dataset/slots.jsonl`, `dataset/MANIFEST.json`, `per_task.csv`, `FINAL-REPORT.md`, "
              "`analysis/collect-summary.json`, `analysis/collect-kimi-vs-1m.csv`; `--allow-incomplete` reads campaigns "
              "without a results file from their probe dirs; `GRADED_FAIL_OVERRIDES` (window 12): kimi w2-g1 13-0 counts "
              "as a graded fail; final collection (2026-11-02): `USER_CANCELLED` (glm = cancelled_by_user, PARTIAL), "
              "slot states interrupted / cancelled_in_flight, yes / no / undetermined verdicts, per-key legs and NEW_2 "
              "readings |" + readme[j:])
    old_verify = "| `final/verify_collection.py` | independent checks of the collection"
    assert readme.count(old_verify) == 1
    i = readme.index(old_verify)
    j = readme.index("\n", i)
    readme = (readme[:i] + "| `final/verify_collection.py` | independent checks of the collection -> "
              "`analysis/collect-verify.json` (window 12: ATP / progress.py counts leave out MANIFEST `status_overrides`; "
              "2026-11-02: the interrupted / cancelled_in_flight states, complete campaigns vs their results files, the "
              "GLM STOP files, a complete campaign's final infra row) |" + readme[j:])
    old_unit = "| `final/unit_check.py`, `final/readme_collect.py` |"
    assert readme.count(old_unit) == 1
    i = readme.index(old_unit)
    j = readme.index("\n", i)
    readme = (readme[:j] + "\n| `final/collect_final_20261102.py`, `final/collect_report_wording_20261102.py`, "
              "`final/verify_infra_rows_20261102.py` | the final collection's patches of collect.py / verify_collection.py "
              "(anchored replacements; pre-change copies, diffs and hashes in `provenance/pre-final-collect-20261102/`) |"
              "\n| `final/independent_counts_20261102.py`, `analysis/independent-counts-20261102.json` | the final "
              "collection's independent headline counts (results files, a direct GLM probe scan, the base CSVs) |"
              "\n| `final/readme_final_collect_20261102.py` | this README's \"Final collection\" section and status block |"
              + readme[j:])
    old_prov = "| `provenance/pre-collect-override-20261101/` |"
    assert readme.count(old_prov) == 1
    i = readme.index(old_prov)
    j = readme.index("\n", i)
    readme = (readme[:j] + "\n| `provenance/pre-final-collect-20261102/` | collect.py and final/verify_collection.py "
              "before the final collection's patches (SHA256SUMS; POST-PATCH.SHA256SUMS = after; *.diff), and the small "
              "outputs of the 10-31 partial collection (`partial-collect-20261031-outputs/`) |" + readme[j:])
    tmp = TD / ".README.md.tmp-final-collect"
    tmp.write_text(readme)
    os.replace(tmp, TD / "README.md")
    print("README.md updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
