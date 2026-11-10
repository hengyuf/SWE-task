#!/usr/bin/env python3
"""teacher-deepswe collect phase (2026-10-31): add the "Collection" section, the status bullet, the new file rows and
the collection command to README.md (temp file + mv). Numbers come from analysis/collect-summary.json,
analysis/collect-verify.json and analysis/atp-slot-states.json; the text is fixed here. Run once.

    /usr/bin/python3 -I -B final/readme_collect.py
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parent.parent
MODELS = ("kimi", "glm", "deepseek")


def main() -> int:
    readme = (TD / "README.md").read_text()
    if "## Collection (2026-10-31" in readme:
        raise SystemExit("README.md already has the Collection section")
    s = json.loads((TD / "analysis" / "collect-summary.json").read_text())
    v = json.loads((TD / "analysis" / "collect-verify.json").read_text())
    atp = json.loads((TD / "analysis" / "atp-slot-states.json").read_text())
    assert v["ok"], v["problems"]
    counts = collections.Counter()
    for key, campaign in atp["campaigns"].items():
        model, name = key.split("/", 1)
        if name != "pilot":
            for slot in campaign["slots"]:
                counts[(model, slot["status"])] += 1
    ov, c, u, k = s["overview"], s["cost"], s["uncovered"], s["kimi_vs_1m"]
    rows = []
    for m in MODELS:
        b = ov[m]
        rows.append(f"| {m} | {b['graded']} / 565 | {b['pass']} | {b['pass_at_1_raw']} | {b['pass_at_1_excl_broken']} | "
                    f"{b['pass_at_1_excl_infra']} | {b['pass_at_1_excl_timeouts']} | {b['tasks_any_pass']} / "
                    f"{b['tasks_started']} | {b['compaction_rate']}% | ${b['cost_per_graded_rollout']:.2f} | "
                    f"${c[m]['all_attempts']:,.2f} |")
    yes = {m: ", ".join(u["per_teacher_yes"][m]) or "none" for m in MODELS}
    section = f"""## Collection (2026-10-31 11:09 PDT): partial dataset

The run is stopped at the NVAPI_KEY_NEW_3 floor (see "Monitor window 4"). No campaign wrote a results file, so
collect.py could not run as designed; it now reads the probe dirs of unfinished campaigns directly (read-only). Outputs:
`dataset/` (graded slots only), `per_task.csv`, `FINAL-REPORT.md` (with a PARTIAL banner), `analysis/collect-summary.json`
and `analysis/collect-kimi-vs-1m.csv`.

- **Job:** CPU job 21116042 (cpu_interactive, nemotron_rl_algo), 11:07:52-11:09:24, exit 0. It came from the babysit
  pair 21116041 / 21116042, submitted with `final/submit_collect.sh` (which refuses while any tdw- leg is queued). The
  `sbatch --test-only` probes said: cpu_interactive at once, cpu_dataprocessing ~11:45-11:50, cpu_datamover 15:23.
  The job ran `final/atp_states.py`, then `collect.py --allow-incomplete --workers 16` (`final/collect.sbatch`). An
  identical first run (job 21115944, 11:03-11:04) had older report wording; its outputs were overwritten.
- **collect.py changes** (temp file + mv; no job runs collect.py; the pre-change file and its sha256 are in
  `provenance/pre-partial-collect-20261031/`):
  - With `--allow-incomplete`, a campaign without a results file is read slot by slot from its probe dir.
  - graded = ATP's `slot_state()` "completed" (from `analysis/atp-slot-states.json`), cross-checked against
    collect.py's own view (first attempt with evaluation_completed and no cached generation failure).
  - The record is built as `rollouts.probe_records()` + `driver.compact()` would build it.
  - Ungraded slots: cancelled (by the stop), infra (with ATP's retryable / needs_review) and not_started. They are not
    dataset rows.
  - New outputs: `dataset/slots.jsonl` (all 1,695 planned slots + 3 pilots with their state and cost),
    `analysis/collect-summary.json` and `analysis/collect-kimi-vs-1m.csv`.
  - Journal cost per attempt with the hub key NAME of the leg (costs.py attribution), plus tokens and non-200 replies.
  - per_task.csv columns are now `<model>_planned/_graded/_pass/_complete/_k5` (k5 only when all 5 are graded) +
    `_cancelled/_infra/_not_started/...`.
  - FINAL-REPORT.md has completeness, wave 1 vs 2, matched task sets, uncovered tasks with base <= 2/10 vs a teacher
    >= 3/5, Kimi vs the 1M export, what finishing needs, cost and keys, and checks.
- **ATP's own slot states** (`final/atp_states.py` -> `analysis/atp-slot-states.json`):
  - Method: ATP's unchanged `deepswe_campaign.slot_state()` on every slot, with its `atomic_json` replaced by a no-op.
  - The {atp['files_checked']:,} attempt files were unchanged before vs after, and no validity cache would have been
    written.
  - completed: kimi {counts[('kimi', 'completed')]}, glm {counts[('glm', 'completed')]}, deepseek {counts[('deepseek', 'completed')]}.
  - retryable: {counts[('kimi', 'retryable')]} / {counts[('glm', 'retryable')]} / {counts[('deepseek', 'retryable')]}.
  - needs_review: {counts[('kimi', 'needs_review')]} / {counts[('glm', 'needs_review')]} / {counts[('deepseek', 'needs_review')]}.
  - pending (not started): {counts[('kimi', 'pending')]} / {counts[('glm', 'pending')]} / {counts[('deepseek', 'pending')]}.
- **Correction to window 4's "39 slots wait for a retry":** ATP would retry 36 of them: the 30 stop cancellations and 6
  infra endings. The other 3 are `needs_review` in ATP's view, so a resume would not retry them: glm w1-g1
  pwntools-tube-multiplexing r0 and r2, and deepseek w1-g0 langchain-request-coalescing r4. In all three the verifier
  was killed at 1,800 s ("Verifier did not produce /logs/verifier/reward.json").
- **Checks:** all zero, from collect.py's own checks and `final/verify_collection.py` (-> `analysis/collect-verify.json`,
  ok).
  - The ledger has 1,695 wave slots, 5 per model per task, and no duplicates.
  - Graded counts equal ATP's completed counts and progress.py's last row (573 graded, 284 passes, 39 ended ungraded).
  - Every row's status equals its terminal.json reward, and every row has a transcript.
  - gz line counts = index = MANIFEST, and sha256 = MANIFEST.
  - Journal cost is attributed to NVAPI_KEY_NEW_3 only (pilots: NEW_2 only).
  - progress.py's export estimate / journal = {v['cost']['kimi']['estimate_over_journal']} (kimi),
    {v['cost']['glm']['estimate_over_journal']} (glm), {v['cost']['deepseek']['estimate_over_journal']} (deepseek).
- **Results (graded first attempts, all wave 1; wave 2 never started):**

| model | graded | passes | pass@1 raw | excl. broken-ended | excl. infra | excl. agent timeouts | tasks >= 1 pass / graded | compaction | $ per graded rollout | hub cost, all attempts |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

  Each model graded a different part of ATP's fixed (task, repeat) admission order (kimi 43 tasks, glm 53, deepseek 79).
  On the 41 tasks all three have graded, pass@1 is kimi {s['matched']['all three models have >= 1 graded slot']['per_model']['kimi']['pass_at_1_excl_infra']},
  glm {s['matched']['all three models have >= 1 graded slot']['per_model']['glm']['pass_at_1_excl_infra']}, deepseek
  {s['matched']['all three models have >= 1 graded slot']['per_model']['deepseek']['pass_at_1_excl_infra']}.
  Kimi's 7 graded agent timeouts are all fails with an empty patch (nothing committed before the 3 h kill). Kimi's
  largest prompt (191,601 tokens) stayed below the 242,144 compaction point, so 0% compaction.
- **Uncovered tasks with base <= 2/10 ({u['tasks']} of the 79 wave-1 tasks) where a teacher already has >= 3 of 5 passing:**
  - kimi {len(u['per_teacher_yes']['kimi'])}: {yes['kimi']}.
  - glm {len(u['per_teacher_yes']['glm'])}: {yes['glm']}.
  - deepseek {len(u['per_teacher_yes']['deepseek'])}: {yes['deepseek']}.
  - Any teacher: {len(u['any_teacher_yes'])}.
  - Still open for at least one teacher (no yes yet): {len(u['any_teacher_open_only'])}.
- **Kimi vs baiheh's 1M-context Kimi export (context only):** on the same {k['slots']} slots / {k['tasks_with_kimi_graded']} tasks,
  {k['kimi256k_pass_at_1']} here vs {k['kimi1m_same_slots']} (1M); without the 7 timeouts {k['kimi256k_pass_at_1_excl_timeouts']} vs
  {k['kimi1m_same_slots_excl_timeouts']}. Per task: higher here {k['tasks_256k_better']}, lower {k['tasks_256k_worse']},
  tied {k['tasks_tied']}. The 1M export: avg@5 {k['kimi1m_all_tasks_avg']} on all 113 tasks and {k['kimi1m_wave1_avg']} on the
  79 wave-1 tasks.
- **Cost (journal):**
  - Wave legs on NVAPI_KEY_NEW_3: ${sum(c[m]['all_attempts'] for m in MODELS):,.2f} in total (kimi ${c['kimi']['all_attempts']:,.2f}, glm
    ${c['glm']['all_attempts']:,.2f}, deepseek ${c['deepseek']['all_attempts']:,.2f}). Of this,
    ${sum(c[m]['ungraded_attempts'] for m in MODELS):,.2f} went to attempts that were cancelled or ended on infra.
  - Pilots on NVAPI_KEY_NEW_2: ${c['pilots']['all_attempts']:,.2f}.
  - NEW_3 went from ${s['new3_at_first_leg']['remaining']:,.2f} (lowest reading at the 23:07 leg starts) to
    ${s['new3_last_reading']['remaining']:,.2f} ({s['new3_last_reading']['at']}). The difference is
    ${s['new3_at_first_leg']['remaining'] - s['new3_last_reading']['remaining']:,.2f}; the rest beyond this campaign's
    journals is r6's judge and other users.
- **Quota at 11:10:** user 11.11M free files and 11.63 TiB; project 10581 82.7M free files and 84.1 TiB. This directory
  holds 31.7k inodes for 615 attempts. The ~1,120 slots still to run need roughly 60-100k more inodes, so far more than
  4M stay free.

"""
    marker = "## Monitor window 4 (2026-10-31 08:15 - 10:45 PDT): NEW_3 floor, all legs stopped"
    assert readme.count(marker) == 1
    readme = readme.replace(marker, section + marker)
    old_status = "- Resuming needs a new OPERATOR_NOTE entry (the user's decision on the key / budget).\n"
    assert readme.count(old_status) == 1
    readme = readme.replace(old_status, old_status + (
        "- **Partial dataset collected 2026-10-31 11:09** (see \"Collection\"): 573 graded rollouts (kimi 84, glm 131,\n"
        "  deepseek 358) in `dataset/`, `per_task.csv` and `FINAL-REPORT.md` (PARTIAL banner). In ATP's view, 36 of the\n"
        "  39 ungraded ended slots are retryable and 3 are needs_review.\n"))
    old_see = 'See "Monitor window 4" (floor and stop)'
    assert readme.count(old_see) == 1
    readme = readme.replace(old_see, 'See "Collection" (partial dataset), "Monitor window 4" (floor and stop)')
    old_row = ("| `collect.py` | final dataset: `dataset/<model>.jsonl.gz`, `dataset/pilots.jsonl.gz`, `dataset/index.jsonl`, "
               "`dataset/MANIFEST.json`, `per_task.csv`, `FINAL-REPORT.md` |")
    assert readme.count(old_row) == 1
    readme = readme.replace(old_row, (
        "| `collect.py` | final dataset: `dataset/<model>.jsonl.gz`, `dataset/pilots.jsonl.gz`, `dataset/index.jsonl`, "
        "`dataset/slots.jsonl`, `dataset/MANIFEST.json`, `per_task.csv`, `FINAL-REPORT.md`, `analysis/collect-summary.json`, "
        "`analysis/collect-kimi-vs-1m.csv`; `--allow-incomplete` reads unfinished campaigns from their probe dirs |\n"
        "| `final/atp_states.py` | ATP's own `slot_state()` of every slot, read-only (atomic_json no-op, files compared "
        "before / after) -> `analysis/atp-slot-states.json` |\n"
        "| `final/collect.sbatch`, `final/submit_collect.sh` | the collection's CPU job (atp_states.py + collect.py "
        "--allow-incomplete) through the babysit pair; refuses while a tdw- leg is queued |\n"
        "| `final/verify_collection.py` | independent checks of the collection -> `analysis/collect-verify.json` |\n"
        "| `final/unit_check.py`, `final/readme_collect.py` | collect.py's slot view / row builder on a few small slots; "
        "this README's Collection section |"))
    old_prov = "| `provenance/pre-new3-20261030/` |"
    assert readme.count(old_prov) == 1
    readme = readme.replace(old_prov, "| `provenance/pre-partial-collect-20261031/` | collect.py before the partial "
                            "collection (+ SHA256SUMS) |\n" + old_prov)
    old_op = ("# 4. final collection (all 15 campaigns have results; run in a CPU job: it reads every journal once):\n"
              "/usr/bin/python3 -I -B $TD/collect.py --workers 16")
    assert readme.count(old_op) == 1
    readme = readme.replace(old_op, (
        "# 4. final collection in a CPU job (it reads every journal once): ATP's slot states + collect.py\n"
        "#    (--allow-incomplete: unfinished campaigns are read from their probe dirs), then the independent checks:\n"
        "TDW_PARTITION=cpu_interactive,cpu_dataprocessing bash $TD/final/submit_collect.sh\n"
        "/usr/bin/python3 -I -B $TD/final/verify_collection.py"))
    tmp = TD / ".README.md.tmp"
    tmp.write_text(readme)
    os.replace(tmp, TD / "README.md")
    print("README.md updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
