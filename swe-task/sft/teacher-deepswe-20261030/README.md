# teacher-deepswe-20261030

Kimi-K3, GLM-5.3 and DeepSeek-V4-Pro rollouts on every DeepSWE v1.1 task: 5 per task and model, collected as a dataset.
113 tasks x 3 models x 5 = **1,695 rollouts**, plus one-time re-runs of slots ended by a broken reply.

**User request (hengyuf, 2026-10-30, verbatim):** "我想在deepSWE 每一个task 上额外用kimi,Deepseek, GLM (Deepseek:
nvidia/deepseek-ai/deepseek-v4-pro GLM: nvidia/zai-org/glm-5.3 ), 每个模型rollout 5遍. 用一个subagent 来收集他们的rollouts"

The main session runs this as the "teacher-deepswe-rollouts" workflow. `OPERATOR_NOTE.md` holds main's instructions to
the workflow agents; it overrides this README. Agents never write to it.

**Status (2026-11-02, final collection): KIMI AND DEEPSEEK COMPLETE; GLM PARTIAL, CANCELLED BY THE USER; NO tdw-
LEG RUNS.** The final dataset, per_task.csv and FINAL-REPORT.md come from the final collection (CPU job
21180894); see "Final collection".
- **Kimi:** 565 / 565 slots graded (with the override), 371 passes, pass@1 65.7.
- **DeepSeek:** 564 graded + 1 final infra, 250 passes, pass@1 44.3
  (raw 44.2).
- **GLM (PARTIAL):** 264 of 565 graded, 182 passes, pass@1 68.9; cancelled by
  the user (OPERATOR_NOTE entry 2026-11-02 ~02:00 PST: hengyuf, verbatim, "cancel the DeepSWE teacher rollout for
  GLM"). Main wrote STOP files in babysit/glm-* at 01:46:40 PST and cancelled the 5 GLM legs (CANCELLED 01:46:45).
  These STOP files stay for good.
- **Uncovered tasks with base <= 2/10 (42) and a teacher at >= 3/5:** kimi 16,
  deepseek 9, glm 15 (PARTIAL), any teacher
  22.
- **Hub cost (journal):** $2,855.01 for the wave legs, plus
  $4.56 for the pilots. NVAPI_KEY_NEW_2 has $8,109.16 left
  (2026-11-02T02:17:01-0800). This campaign makes no more hub calls.
- Daylight saving time ended at 2026-11-01 02:00 PDT; times after that are PST.

See "Final collection" (2026-11-02), "Monitor window 16" (GLM cancelled), "Monitor window 15", "Monitor window 14", "Monitor window 13", "Monitor window 12", "Monitor window 11", "Monitor window 10", "Monitor window 9", "Monitor window 8", "Monitor window 7", "Monitor window 6", "Monitor window 5", "Hub key change 2 and resume", "Collection" (partial dataset), "Monitor window 4" (floor and stop), "Monitor window 3" (GLM 408 stall), "Monitor window 2" and "Monitor window 1".

## Hub key change 2 and resume (2026-10-31 18:15 - 18:40, OPERATOR_NOTE 18:15)

OPERATOR_NOTE.md 18:15 entry (main; sha256 6c8ce7e4...), quoting a user decision (hengyuf, verbatim as written there):
"同意上线C1+C2, 直接续跑formal, 不接受8号段留下的8个组. 另外kimi/GLM/DeepSeek 的rollout 用NEW_2 继续跑. r6 的judge key 也用NEW_2".
The r6 parts are main's. For this campaign the entry says:
- **Key:** NVAPI_KEY_NEW_2 for every new leg (main, rec and brk). It replaces NVAPI_KEY_NEW_3, which is at its floor and
  is no longer used by this campaign.
- **Floors:** NEW_2 also carries the r6 RL rubric judge. A leg starts only while NEW_2 has >= $1,700 left, and the guard
  stops legs below $1,500. At the floor: STOP files, no tdw- leg running, `stopped_budget` to main.
- **Resume:** remove the 10:30 STOP files, resubmit the 9 wave-1 legs, then wave 2 as the gating allows. The 36
  retryable slots retry as ATP decides; the 3 needs_review slots stay and are listed in the final report.

**NEW_2 budget reading (key_budget.py, 18:11:36):** spend $0.0000283 of max_budget $10,000, so **$10,000.00 left**. At
10-30 23:01 the same check read spend $5,215.06 / $4,784.94 left. The hub now reports NEW_2's spend as reset; this
agent does not know why (it reads no .env file). The leg-start checks at 18:29 also read $10,000.0.

**Code changes** (temp file + `mv`, no leg running; pre-change files and their sha256: `provenance/pre-new2-20261031/`;
patch script `monitor/patches/key_new2_20261031.py`, 40 exact-text replacements, applied 18:17):
- `driver.py`:
  - `KEY = NVAPI_KEY_NEW_2`, `START_FLOOR = 1700`, `GUARD_FLOOR = 1500`.
  - `KEY_PERIODS` gives the key by leg start time: NEW_2 before 2026-10-30 21:45; NEW_3 from 21:45 to 2026-10-31 18:15;
    NEW_2 from 18:15.
  - `key_of_time()`: `check_existing` now requires every earlier leg of a campaign to carry the key of its start time.
    `main()` asserts that KEY is the key of the current period.
  - Results carry `hub_key_policy.periods` and `.switches` (both switches with their floors and sources).
- `probe_worker.py`: `KEYS = (NVAPI_KEY_NEW_2,)`, so a worker refuses NEW_3.
- `key_budget.py`: allows NVAPI_KEY_NEW_2 only. NEW_3 is refused (argparse "invalid choice", no hub call; tested), so
  not even a budget check uses it.
- `watch.py`:
  - NEW_2 with floors $1,700 / $1,500; `--budget-floor` default 1,900.
  - The low / start-floor / floor flags are now kept per key NAME (`watch-state.json` "budget_flags"). Before, NEW_3's
    flags of 10:30 would have kept the NEW_2 floor event and its STOP files from firing.
- `progress.py`:
  - Header: NEW_2, its floors and the key history.
  - Spend baseline: the first NEW_2 leg from 18:15.
  - Cost section: one NEW_3 line (spend from its first leg to its last reading) and one pilots line.
- `verify_pilot.py`: `key_at()` with the three periods.
- `setup/offline_checks.py`: the budget check reads NEW_2 (ok needs >= $1,700).
- `final/verify_collection.py`: wave-leg cost may be attributed to NEW_3 or NEW_2 (pilots NEW_2).
- `monitor/poll.sh`: comment only.
- **Not changed:** `collect.py`'s report text (it still describes the NEW_3 stop of the partial collection); the final
  collection must update it. Its cost attribution is already per leg key (costs.py: the leg running at each reply's
  timestamp).
- `legs.jsonl`: a `key_switch` record (at 2026-10-31T18:15:00-0700, NEW_3 -> NEW_2, floors 1,700 / 1,500, source
  OPERATOR_NOTE 18:15). Every leg's `start` / `incomplete` / `complete` record names its key, and so does each
  campaign's `tdw-legs.jsonl` (written by probe_worker.py).

**Checks:**
- `monitor/keycheck.py` (new, read-only; ATP controller python): all 12 recorded `tdw-legs.jsonl` entries (3 pilots
  NEW_2, 9 wave-1 NEW_3) and all 12 earlier `start` records in legs.jsonl carry the key of their start time. 0
  mismatches; the key of "now" = driver.KEY = NEW_2.
- Every patched file compiles, and `bash -n` passes on poll.sh.
- `driver.py --dry-run` passed for glm w1-g1 (52 s) and, as submit.sh's preflight, for all 9 wave-1 campaigns.
- The full `setup/offline_checks.py` did not finish: it runs 18 dry-runs of ~52 s each, and the agent's 9-min timeout
  killed it before it wrote output.

**Resume:**
- The 15 STOP files of 10:30 were removed after checking each one's 10:30 watch.py text. Copies are in
  `monitor/removed-stop-20261031/`.
- **Partition probe** (`sbatch --test-only`, 2 CPU / 16G / 23 h, 18:28):
  - cpu_dataprocessing: at once on both accounts.
  - cpu_interactive: at once.
  - cpu_datamover: 22:49 (rl_algo) / 23:26 (agents_dev).
  - As at the wave-1 launch, `TDW_PARTITION=cpu_dataprocessing,cpu_datamover` (cpu_interactive stays free for r6's
    p_cpu_short jobs).
- **Submitted 18:28:56-18:29:45:** `bash submit.sh <model> w1 0 1 2` for kimi, glm and deepseek (leg 2 of each
  campaign). All 9 winners started 18:29:10-18:29:48 on cpu_dataprocessing, and each loser completed with "sibling
  candidate won":

| campaign | winner (account) | loser | driver start line |
|---|---|---|---|
| kimi w1-g0 | 21128606 (agents_dev) | 21128607 | 18:29:14, NEW_2, $10,000.0 left, cap 8 |
| kimi w1-g1 | 21128610 (rl_algo) | 21128609 | 18:29:19, NEW_2, $10,000.0, cap 8 |
| kimi w1-g2 | 21128612 (rl_algo) | 21128611 | 18:29:26, NEW_2, $10,000.0, cap 8 |
| glm w1-g0 | 21128614 (agents_dev) | 21128615 | 18:29:34, NEW_2, $10,000.0, cap 8 |
| glm w1-g1 | 21128619 (rl_algo) | 21128618 | 18:29:38, NEW_2, $10,000.0, cap 8 |
| glm w1-g2 | 21128621 (agents_dev) | 21128622 | 18:29:39, NEW_2, $10,000.0, cap 8 |
| deepseek w1-g0 | 21128624 (rl_algo) | 21128623 | 18:29:41, NEW_2, $10,000.0, cap 5 |
| deepseek w1-g1 | 21128627 (rl_algo) | 21128626 | 18:29:46, NEW_2, $10,000.0, cap 5 |
| deepseek w1-g2 | 21128629 (agents_dev) | 21128630 | 18:29:52, NEW_2, $10,000.0, cap 5 |

- **Per-campaign checks:** each campaign's `tdw-legs.jsonl` gained an entry 18:29:54-18:30:33 with hub_key NEW_2, the
  model id and context 262144, and each worker.log wrapper line names NEW_2.
- **ATP admission after the resume:** ATP kept the admission state from before the stop.
  - kimi: API 8 / rollouts 8 in each campaign.
  - glm and deepseek: API 2 (deepseek w1-g2: 4) and rollout limit 1, with the last pressure "rate_limit" (glm w1-g0:
    "upstream_timeout").
  - ATP doubles a rollout limit after a healthy completion in the current epoch (`deepswe_admission._grow_rollouts`),
    so GLM and DeepSeek ramp up as rollouts finish. Nothing here changes ATP's state.
- **First poll (18:35):** 30 rollouts in flight (kimi 24, glm 3, deepseek 3); 679 hub replies, all 200.

## Final collection (2026-11-02 PST): Kimi and DeepSeek complete, GLM PARTIAL (cancelled by the user)

This section follows OPERATOR_NOTE entry 2026-11-02 ~02:00 PST, item 3: GLM is reported as PARTIAL. Its graded slots
count as they stand, and its 5 slots in flight at the cancel are listed as cancelled (not run to the end, not
graded). GLM is left out of every "complete" claim. Full numbers and lists are in `FINAL-REPORT.md`.

**Outputs** (all from one run):
- `dataset/kimi.jsonl.gz` (565 rows), `glm.jsonl.gz` (264, graded slots only) and
  `deepseek.jsonl.gz` (565). Each row is one rollout: the Responses-API transcript in "output",
  patch, grade, re-run provenance, flags and hub cost.
- `dataset/pilots.jsonl.gz` (3) and `dataset/index.jsonl` (every row without its transcript).
- `dataset/slots.jsonl` (1698 lines): every planned slot (1,695) and the 3 pilots, with state and
  cost.
- `dataset/MANIFEST.json` (`complete` false, `model_status`), `per_task.csv` and `FINAL-REPORT.md`.
- `analysis/collect-summary.json`, `analysis/collect-kimi-vs-1m.csv`, `analysis/collect-verify.json` and
  `analysis/atp-slot-states.json`.

**Job.** CPU job 21180894 (cpu_interactive), 02:24:49-02:28:14 PST, exit 0. It was submitted through the babysit
pair with `final/submit_collect.sh` (`TDW_PARTITION=cpu_interactive,cpu_dataprocessing`) after `sbatch --test-only`
probes of the 3 CPU partitions. It ran `final/atp_states.py` (ATP's own slot states: kimi 564 completed + 1 needs_review; glm 264 completed, 17 retryable, 3 needs_review,
5 active, 276 pending; deepseek 564 completed + 1 needs_review; the probe files were unchanged by the check), then `collect.py --allow-incomplete --workers 16`. An earlier run, job 21180602
(02:17:47-02:21:08 PST), produced the same numbers. Only the wording of the report's cost section differed, and
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
- collect.py's 16 checks are all 0.
- `final/verify_collection.py` is ok, with no problems:
  - ledger: 1695 wave slots (5 per model per task) + 3 pilots;
  - gz line counts = index = MANIFEST, and every sha256 = MANIFEST;
  - every row's status = its terminal.json reward (1397 rows; 1 final infra row: deepseek
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
| kimi | complete | 565 / 565 | 371 | 65.7 | 65.7 | 65.7 | 66.6 | 101 / 113 | 1.6% | $1.93 | $1,116.06 |
| glm | **PARTIAL** (cancelled by the user) | 264 / 565 | 182 | 66.7 | 69.2 | 68.9 | 68.9 | 69 / 84 | 26.9% | $2.60 | $727.82 |
| deepseek | complete | 564 / 565 | 250 | 44.2 | 44.3 | 44.3 | 44.3 | 91 / 113 | 22.7% | $1.77 | $1,011.13 |

- Kimi counts w2-g1 13-0 (scriggo r0) as a graded fail (OPERATOR_NOTE 11-01 11:15 / 14:15). In ATP's view it is
  564 graded and pass@1 65.8, or 66.7 without its 8 graded agent timeouts (all fails), as
  in PROGRESS.md.
- DeepSeek has 1 final infra slot: w1-g0 langchain-request-coalescing r4, where the verifier was killed at 1,800 s
  (needs_review).
- GLM: 9 infra endings, 6 ATP-retryable and 3 needs_review (the pwntools r0 / r2 verifier kills and the w2-g0
  fastapi-deprecation r2 sandbox 410). GLM also has 1 broken-ended graded slot, w1-g0 arktype r4 (a fail); its planned
  brk re-run was cancelled with GLM.

**Wave 1 (79 uncovered tasks) vs wave 2 (34 covered tasks):**

| model x wave | graded / planned | pass@1 excl. infra | pass@5: tasks >= 1 pass / graded | base, same slots |
|---|---|---|---|---|
| kimi w1 | 395 / 395 | 55.2 | 67 / 79 | 31.4 |
| kimi w2 | 170 / 170 | 90.0 | 34 / 34 | 17.9 |
| glm w1 (PARTIAL) | 252 / 395 | 68.7 | 61 / 75 | 31.6 |
| glm w2 (PARTIAL) | 12 / 170 | 75.0 | 8 / 9 | 14.2 |
| deepseek w1 | 394 / 395 | 42.6 | 62 / 79 | 31.4 |
| deepseek w2 | 170 / 170 | 48.2 | 29 / 34 | 17.9 |

**Matched task sets:**
- On the 84 tasks where all three have a graded slot, pass@1 is kimi
  60.2, glm 68.9 and deepseek
  45.3.
- On the 21 tasks where all three have all 5 graded: kimi 62.9,
  glm 74.3, deepseek 42.9.

**Uncovered tasks with base <= 2/10 (42 of the 79) and a teacher at >= 3/5:**
- Kimi: 16 tasks (arktype-json-schema-refs-dependencies, bandit-interprocedural-taint-checks, clack-async-autocomplete-options, expr-try-catch-errors, go-critic-doc-link-checker, httpx-multipart-response-parsing, igel-persist-feature-schema, katex-multicolumn-array-spans, kombu-single-active-consumer-priority, opa-template-string-reconstruction, participle-grammar-conflict-analysis, sqlite-utils-safe-import-checkpoints, tengo-callable-instance-isolation, textual-kitty-key-phases, valibot-recursive-schema-composition, yaegi-go-embed-directives).
- DeepSeek: 9 tasks (arktype-json-schema-refs-dependencies, igel-persist-feature-schema, katex-multicolumn-array-spans, mashumaro-flattened-dataclass-fields, opa-template-string-reconstruction, sqlite-utils-safe-import-checkpoints, tengo-callable-instance-isolation, valibot-recursive-schema-composition, yaegi-go-embed-directives).
- GLM (PARTIAL): 15 yes (bandit-interprocedural-taint-checks, boa-hierarchical-evaluation-cancellation, clack-async-autocomplete-options, go-critic-doc-link-checker, ipython-session-bundle-replay, katex-multicolumn-array-spans, kombu-virtual-queue-dead-lettering, koota-pair-relation-tracking, mashumaro-flattened-dataclass-fields, opa-template-string-reconstruction, optique-conditional-option-dependencies, sqlite-utils-safe-import-checkpoints, textual-kitty-key-phases, valibot-recursive-schema-composition, yaegi-go-embed-directives), 19
  undetermined, 8 no.
- Kimi or DeepSeek (both complete): 17. Any teacher: 22.
- No teacher yet: 20. For 15 of these, only GLM's
  missing slots could still have changed the answer.

**Kimi here (256k) vs Baihe's 1M-context Kimi export (context only, different pipeline and serving period):**
- avg@5: 65.7 vs 66.7 on all 113 tasks; wave 1 55.2 vs
  55.2; wave 2 90.0 vs 93.5.
- pass@5: 101 vs 96 tasks. All 5 passing: 49 vs
  45.
- Per task: higher here 29, lower 27, tied 57.
- Uncovered base <= 2/10 tasks at >= 3/5: 16 vs 18.

**GLM at the cancel:**
- 5 attempts in flight: w1-g0 8-3, w1-g1 20-1, w1-g2 8-0, w2-g0 2-0 and w2-g1 0-3. Tasks and
  sandbox ids are in FINAL-REPORT.md; main deletes those sandboxes.
- 20 slots had ended ungraded earlier and were never retried:
  10 interrupted by main's stops, 1 cancelled by the 10-31 floor stop and 9
  infra endings.
- 276 slots never started.
- Of GLM's $727.82 hub spend, $31.71 went to attempts that
  ended ungraded or were cut off.

**Cost (journal; each hub reply is attributed to the leg running when it arrived):**
- By model, all attempts: kimi $1,116.06, glm $727.82 (PARTIAL),
  deepseek $1,011.13; pilots $4.56.
- NVAPI_KEY_NEW_3 (wave-1 legs, 10-30 23:07 PDT until the 10-31 10:30-10:40 floor stop): $1,128.40.
- NVAPI_KEY_NEW_2: $1,726.61 for the legs from 10-31 18:29, plus
  $4.56 for the pilots.
- **NVAPI_KEY_NEW_2 now: $1,890.84 spent / $8,109.16 left (2026-11-02T02:17:01-0800).**
  Since the hub's 10-31 18:11 spend reset, NEW_2's spend grew by $1,890.84.
  This campaign accounts for $1,726.61 of it. The rest is other use of the key (the r6
  judge, budget checks) and any gap between the journal and LiteLLM's figure.

**Quota (after the collection):** user 10.06M free files, 9.57 TiB free (lfs quota, 02:29 PST); project 10581 79.29M free files, 78.83 TiB free. The dataset files take
about 0.33 GB and a handful of inodes.

## Monitor window 16 (2026-11-02 00:15 - 01:56 PST; GLM cancelled by the user)

All times are PST.
- **OPERATOR_NOTE:** sha256 0a44b261... (main's 21:20 entry) at every poll from 00:16 to 01:39. At 01:46:40 main
  appended the entry headed "2026-11-02 ~02:00 PST (main) - USER DECISION, CANCEL GLM" (sha256 c0cd92d1..., 11,079
  bytes). It quotes hengyuf verbatim: "cancel the DeepSWE teacher rollout for GLM". For this campaign it says:
  1. The 5 GLM campaigns are cancelled for good. Main writes their STOP files and cancels their legs. The STOP files
     are never removed, no GLM leg is resubmitted, and no GLM re-run (brk) or recovery (rec) leg runs. This overrides
     every earlier GLM entry.
  2. Kimi and DeepSeek continue to completion under the existing rules (both were already complete).
  3. The final collection reports GLM as PARTIAL: the graded GLM slots count as they stand, the slots in flight at the
     cancel are listed as cancelled (not run, not graded), and GLM stays out of any "complete" claim.
  4. Main deletes the sandboxes of the cancelled GLM legs by their exact run labels; no other cleanup tool is built.
- **The cancel (checked 01:48-01:54):**
  - STOP files in babysit/glm-w1-g0, glm-w1-g1, glm-w1-g2, glm-w2-g0 and glm-w2-g1, written 01:46:40. Main's note: "user
    cancelled the GLM teacher rollouts ...; never resubmit this campaign".
  - sacct: legs 21154933 / 21154935 / 21154938 (wave 1) and 21167343 / 21167279 (wave 2) CANCELLED at 01:46:45.
    squeue: no tdw- job. No tdw or babysit process of this user runs.
  - No file under probes/glm changed after 01:47:30 (`find -newermt`), so no GLM worker is left and no hub call is made.
  - This agent wrote no STOP file, cancelled nothing and cleaned up no sandbox (main does that, item 4).
- **In flight at the cancel** (cancelled: not run to the end, not graded; no ending file). Sandbox ids are from each
  attempt's agent-sandbox.json; ATP's owner labels are atp-model `glm` and atp-attempt `<slot>-attempt-0`.

| campaign | slot | task | started | sandbox id |
|---|---|---|---|---|
| w1-g0 | 8-3 attempt-0 | happy-dom-deterministic-intersectionobserver | 01:34:57 | b06c19f4-17f4-458c-ab6a-5adf2c070ebc |
| w1-g1 | 20-1 attempt-0 | sqlfmt-create-table-ddl-formatting | 01:30:47 | 5db7305c-7fbc-4ae4-b608-6f5679722260 |
| w1-g2 | 8-0 attempt-0 | helm-unified-manifest-stream | 01:23:58 | 2a6ca37e-d77b-422d-9cc6-04d44e7b51c6 |
| w2-g0 | 2-0 attempt-0 | arcane-drift-detection-baselines | 01:42:46 | cc83fd2e-6c0a-4641-a791-f802e7964000 |
| w2-g1 | 0-3 attempt-0 | adaptix-name-mapping-aliases | 01:00:37 | 4dc4e786-0688-4810-9598-505cd4e06f72 |

**GLM at the cancel (PROGRESS.md 01:52):**
- 264 of 565 slots graded, 182 passes (pass@1 68.9%), 0 agent timeouts:
  - wave 1: 252 of 395 graded, 173 passes (68.7%);
  - wave 2: 12 of 170 graded, 9 passes (75.0%).
- Per campaign (graded / passes): w1-g0 51 / 32, w1-g1 92 / 63, w1-g2 109 / 78, w2-g0 7 / 6, w2-g1 5 / 3.
- Not graded: 301 = 5 in flight at the cancel + 20 ended ungraded (interrupted or infra; 18 in wave 1, 2 in wave 2;
  not retried) + 276 never started.
- 1 broken-ended slot (glm w1-g0 1-4); its planned re-run is cancelled.

**Endings before the cancel (GLM; kimi and deepseek had none, both complete):** 7 graded (3 pass, 4 fail): wave 1 5 (3
pass), wave 2 2 (0 pass). No infra ending, no interrupted attempt, no agent timeout.

| ended | campaign | slot | task | result | minutes |
|---|---|---|---|---|---|
| 00:40:56 | w1-g1 | 17-0 | participle-grammar-conflict-analysis | pass, f2p 91/91 | 99.5 |
| 01:00:36 | w2-g1 | 9-4 | obsidian-linter-auto-table-of-contents | fail, f2p 37/41 | 53.6 |
| 01:06:04 | w1-g0 | 23-4 | textual-richlog-follow-state | pass, f2p 20/20 | 99.3 |
| 01:23:55 | w1-g2 | 12-3 | koota-deferred-mutation-buffer | fail, f2p 70/71 | 80.6 |
| 01:30:47 | w1-g1 | 24-2 | updo-policy-alerting | fail, f2p 15/17 | 49.8 |
| 01:34:57 | w1-g0 | 3-1 | claude-code-by-agents-recursive-delegation | pass, f2p 7/7 | 28.8 |
| 01:42:44 | w2-g0 | 13-4 | python-statemachine-state-data-scoping | fail, f2p 70/72 | 92.4 |

- w1-g1 17-0 was window 15's watch item (runs of LiteLLM's 360 s 408s on a 424 KB request). Its try 4 got a 200 at
  00:12:44 after 331 s. It then made steady calls and passed at 00:40:56, 80 min before its deadline.

**The 21:20 stop rule (until the cancel):**
- ATP held all 5 GLM campaigns at rollout limit 1 (API limit 1-2) at every poll, so no wave-2 campaign had 2 rollouts
  in flight. No rollout ran short of API slots (gate waits 0.0-0.1 min). The rule never applied.
- GLM-5.3 429s stayed low (`monitor/glm429bins.py`, last 30 min at each poll):

| poll | rollout replies | budget checks |
|---|---|---|
| 00:16 | 21 / 389 (5.4%) | 0 / 38 (0%) |
| 00:26 | 10 / 293 (3.4%) | 0 / 37 (0%) |
| 00:36 | 0 / 225 (0%) | 0 / 35 (0%) |
| 00:47 | 0 / 301 (0%) | 0 / 33 (0%) |
| 00:57 | 5 / 330 (1.5%) | 0 / 33 (0%) |
| 01:07 | 8 / 382 (2.1%) | 0 / 32 (0%) |
| 01:18 | 15 / 335 (4.5%) | 1 / 33 (3.0%) |
| 01:29 | 20 / 375 (5.3%) | 2 / 34 (5.9%) |
| 01:39 | 21 / 429 (4.9%) | 2 / 33 (6.1%) |
| 01:54 | 18 / 348 (5.2%) | 1 / 27 (3.7%) |

- The 20% rule of OPERATOR_NOTE 11:15 was not exceeded.
- glm429bins.py's `auto_restart_rule_met` was true from 00:36. That rule (OPERATOR_NOTE 15:35) applies only after a
  wave-2 stop, and wave 2 was running, so it meant nothing here.

**Health:** health.py's 30-min infra share was 0 at every poll (1-4 endings per 30 min). There was no sandbox error, no
OpenSandbox API error spike and no verifier error.

**Agent timeouts (OPERATOR_NOTE 02:15):** glm 0 of its 7 endings. kimi and deepseek had no endings (both complete).
The Kimi auto-pause rule did not apply.

**progress.py changes** (no job runs or imports progress.py; temp file + os.replace; patch scripts in
`monitor/patches/`, pre-patch copies in `provenance/`):
1. **01:10, `progress_eta_live_20261102.py`** (sha256 a122600b -> dd583fe9):
   - The problem: at 01:07 only 9 GLM wave-1 rollouts had finished in the last 3 h (fewer than 10). projection() then
     fell back from "finished per hour over the last 3 h" to the campaigns' manifest caps (24 in flight for wave 1, 8
     for wave 2) and printed a finish at 11-02 22:27, while ATP held 1 rollout per campaign.
   - The fix: a running wave without that basis now uses the attempts in flight (at least 1 per running campaign).
     A wave that already runs (GLM wave 2) starts now, not at its wave-1 ETA.
   - `--check` printed the old and new projections on the live data first. At 01:10: glm wave 1 at 4.2/h -> 11-03
     12:36, wave 2 at 2.4/h -> 11-04 20:46.
   - The rows before 01:10 in PROGRESS.md's table keep their old ETAs.
2. **01:52, `progress_glm_cancel_20261102.py`** (sha256 dd583fe9 -> a930e58c): a model whose campaigns carry main's
   cancel STOP note is reported as cancelled:
   - a header bullet (PARTIAL; graded slots as they stand; in-flight slots cancelled; no brk / rec leg);
   - 0 rollouts and $0 to go and no ETA for its waves (the overall ETA counts only the other models);
   - "cancelled" in the model x wave ETA cell and in the per-campaign jobs cell;
   - a projection line with the not-graded split.
   The counts are unchanged. `--check` first wrote a full report into a scratch directory (since removed). TD's
   PROGRESS.md, rows, cache and progress.py stayed byte-identical during the check (sha256 checked).

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- $8,132.22 (00:16), $8,128.45 (00:26), $8,126.51 (00:36), $8,124.29 (00:47), $8,121.16 (00:57), $8,118.32 (01:07),
  $8,116.39 (01:18), $8,114.23 (01:29), $8,112.00 (01:39) and $8,109.83 (01:52-01:54).
- That is $20.51 since 00:12. The campaign's GLM estimate rose $18.74 ($664.23 -> $682.97). Far above the floors.

**Quota (watch.py, 01:54):** user 10.08M free files and 9.57 TiB; project 79.33M free files.

**Polling:**
- 10 polls: 00:16, 00:26, 00:36, 00:47, 00:57, 01:07, 01:18, 01:29, 01:39 and 01:54, plus the cancel checks at
  01:48-01:54.
- Each poll read OPERATOR_NOTE and ran `monitor/w16poll.sh` (= `monitor/w15poll.sh` with window-16 file names and
  start 00:15).
- One line per poll is in `monitor/w16-polls.log`.
- PROGRESS.md was regenerated at 00:37, 01:07, 01:10 (with the ETA patch; it replaced the 01:07 row), 01:39 and 01:52
  (with the cancel patch), and once more right before the agent returned.
- This section and the status block were written with `monitor/patches/readme_window16.py`.
- No background process of this agent runs.

## Monitor window 15 (2026-11-01 21:10 - 11-02 00:15 PST)

All times are PST.
- **OPERATOR_NOTE:** sha256 0a44b261... at every poll from 21:13 to 00:12 (written 21:11:49, 10,096 bytes). Its newest
  entry is headed 21:20 PST:
  - main accepts GLM wave 2 at its campaigns' own cap 4 and keeps all 5 GLM campaigns running through the 429 bursts;
  - wave 2 is stopped again only if a wave-2 campaign has 2 or more rollouts in flight while GLM-5.3 429s exceed 30%
    over 30 min, or if a GLM rollout looks headed for the agent timeout because of API-slot starvation;
  - the wave-1 legs are resubmitted at their 23 h limit (~10:10 PST on 11-02).
  No later entry came, and no "r6 resumed" entry.
- **Nothing to act on:** no leg ended, nothing was submitted or cancelled, no STOP file was written or removed, and no
  sandbox cleanup was needed. No code changed; the only new files are the poll tools below and this README section.

**The 21:20 stop rule:**
- ATP's adaptive admission held all 5 GLM campaigns at rollout limit 1 (API limit 1-2) at every poll. So no wave-2
  campaign had 2 rollouts in flight, even while the 30-min 429 share was above 30% (22:26-23:08).
- No rollout ran short of API slots: gate waits were at most 0.4 min per attempt (`monitor/inflight.py`).
- Two long rollouts were checked against the agent timeout:
  - **w1-g2 24-0** (valibot-recursive-schema-composition) passed at 22:03:55 after 104.6 min. At 97 min it had 246
    completed model calls, 57 429s (all retried) and 0.2 min of gate wait.
  - **w2-g1 13-3 attempt-1** (scriggo-method-declarations) passed at 23:16:24 after 106.4 min, f2p 48/48. At 89 min it
    had 356 model replies (p50 4.2 s, p90 15.9 s; 49.9 min inside 200 calls) and 174 429s with 25.7 min of ATP retry
    waits (29% of its time). It made 4-5 calls per minute in every 10-min bin.
- **Still in flight, at risk: w1-g1 17-0** (participle-grammar-conflict-analysis, started 23:01:28, deadline 02:01:28):
  - From 23:02:52 one call got six 360 s 408s in a row (LiteLLM's request timeout), then a 200 on try 7 at ~23:40.
  - After one more 408 and two 200s, the request JSON grew from 243 KB to 424 KB at 23:49:01. That call got 408s at
    23:55:02, 00:01:04 and 00:07:08, and try 4 went out at 00:07:12.
  - In total it had 11 replies and 13 non-200 replies in 71 min (`monitor/tail_events.py`). Earlier calls returned in
    2-4 s.
  - This is the window-3 pattern (runs of 360 s 408s on single calls). It is not API-slot starvation (gate wait 0, the
    only rollout of w1-g1), and it is a wave-1 rollout, so the 21:20 rule does not apply. ATP retries the call. In
    window 3, ATP rated such an attempt infra after 7 408s in a row and retried the slot. If not, the attempt may
    reach its deadline at 02:01:28.

**GLM-5.3 429s (`monitor/glm429bins.py`, last 30 min at each poll):**

| poll | rollout replies | budget checks |
|---|---|---|
| 21:13 | 213 / 630 (33.8%) | 9 / 37 (24.3%) |
| 21:24 | 129 / 553 (23.3%) | 6 / 36 (16.7%) |
| 21:36 | 59 / 429 (13.8%) | 7 / 36 (19.4%) |
| 21:46 | 30 / 422 (7.1%) | 6 / 34 (17.6%) |
| 21:56 | 37 / 483 (7.7%) | 6 / 34 (17.6%) |
| 22:07 | 140 / 659 (21.2%) | 8 / 34 (23.5%) |
| 22:17 | 207 / 741 (27.9%) | 9 / 34 (26.5%) |
| 22:26 | 331 / 886 (37.4%) | 12 / 34 (35.3%) |
| 22:36 | 327 / 849 (38.5%) | 12 / 34 (35.3%) |
| 22:47 | 336 / 846 (39.7%) | 13 / 34 (38.2%) |
| 22:58 | 303 / 781 (38.8%) | 12 / 33 (36.4%) |
| 23:08 | 274 / 799 (34.3%) | 12 / 33 (36.4%) |
| 23:18 | 220 / 747 (29.5%) | 12 / 34 (35.3%) |
| 23:28 | 117 / 551 (21.2%) | 10 / 33 (30.3%) |
| 23:38 | 77 / 409 (18.8%) | 10 / 35 (28.6%) |
| 23:48 | 55 / 327 (16.8%) | 5 / 34 (14.7%) |
| 23:57 | 40 / 380 (10.5%) | 3 / 35 (8.6%) |
| 00:06 | 32 / 401 (8.0%) | 1 / 35 (2.9%) |
| 00:12 | 25 / 417 (6.0%) | 0 / 36 (0%) |

- Budget calls got 429s at 22:06, 22:58, 23:08 ("Model capacity reached for nvidia/zai-org/glm-5.3 ... Model TPM:
  100000000 ... Remaining: 0") and 23:28; each reading still had a remaining figure.
- The 20% rule of OPERATOR_NOTE 11:15 item 2 was exceeded at 21:13-21:24 and 22:07-23:38. GLM is already at 2 for new
  campaigns (TDW_MAX_CONCURRENCY_OVERRIDE=2 on every GLM leg since 14:23), and nothing was paused.

**Endings (GLM; kimi and deepseek are complete):** 19 graded (13 pass, 6 fail): wave 1 11 (6 pass), wave 2 8 (7 pass).
No infra ending, no interrupted attempt, no agent timeout.

| ended | campaign | slot | task | result | minutes |
|---|---|---|---|---|---|
| 21:12:51 | w1-g1 | 5-0 | eicrud-keyset-pagination-cursor | fail, f2p 10/14 | 60.1 |
| 21:20:45 | w2-g0 | 13-3 attempt-1 | python-statemachine-state-data-scoping | pass, f2p 72/72 | 56.0 |
| 21:30:00 | w2-g1 | 5-3 attempt-1 | fastapi-implicit-head-options | fail, f2p 38/43 | 51.0 |
| 21:45:26 | w1-g0 | 8-2 | happy-dom-deterministic-intersectionobserver | fail, f2p 13/14 | 51.8 |
| 22:01:50 | w2-g0 | 1-4 attempt-1 | aiomonitor-task-snapshots-diff | pass, f2p 53/53 | 41.0 |
| 22:03:55 | w1-g2 | 24-0 | valibot-recursive-schema-composition | pass, f2p 10/10 | 104.6 |
| 22:05:11 | w1-g1 | 3-3 | cliffy-config-file-parsing | pass, f2p 37/37 | 52.3 |
| 22:24:20 | w1-g0 | 18-4 | psd-tools-blend-range-api | pass, f2p 45/45 | 38.9 |
| 22:34:05 | w1-g1 | 1-0 | awilix-async-container-initialization | fail, f2p 23/24 | 28.9 |
| 22:46:47 | w1-g0 | 3-0 | claude-code-by-agents-recursive-delegation | fail, f2p 2/7 | 22.4 |
| 22:49:08 | w2-g0 | 9-4 attempt-1 | narwhals-rolling-window-suite | pass, f2p 103/103 | 47.3 |
| 23:01:27 | w1-g1 | 25-2 | wazero-multi-module-snapshots | pass, f2p 78/78 | 27.3 |
| 23:03:53 | w1-g2 | 5-0 | expr-try-catch-errors | fail, f2p 78/79 | 59.9 |
| 23:14:09 | w2-g0 | 0-3 | abs-stepped-slices | pass, f2p 6/6 | 25.0 |
| 23:16:24 | w2-g1 | 13-3 attempt-1 | scriggo-method-declarations | pass, f2p 48/48 | 106.4 |
| 23:26:44 | w1-g0 | 13-2 | koota-entity-snapshot-rollback | pass, f2p 84/84 | 39.9 |
| 00:03:16 | w1-g2 | 13-1 | koota-query-predicates | pass, f2p 43/43 | 59.3 |
| 00:07:01 | w2-g1 | 1-4 attempt-1 | anko-default-function-arguments | pass, f2p 2/2 | 50.5 |
| 00:10:20 | w2-g0 | 5-4 | fastapi-deprecation-response-headers | pass, f2p 137/137 | 56.2 |

- Graded first attempts (PROGRESS.md 21:10 -> 00:06): glm 238 -> 255, passes 166 -> 177 (pass@1 69.7% -> 69.4%).
  Totals 1,366 -> 1,383 graded and 787 -> 798 passes. The two passes at 00:07 and 00:10 came after the 00:06 report.
- The broken-ended GLM slot (glm w1-g0 1-4) is still the 1 planned re-run. driver.py runs it in a brk campaign (cap 2)
  once w1-g0's main campaign is complete.

**Health:** health.py's 30-min infra share was 0 at every poll (0-5 endings per 30 min). There was no sandbox error, no
OpenSandbox API error spike and no verifier error.

**Agent timeouts (OPERATOR_NOTE 02:15):** glm 0 of its 19 endings. kimi and deepseek had no endings (both complete).
The Kimi auto-pause rule did not apply.

**Budget (key_budget.py, NVAPI_KEY_NEW_2; watch.py readings):**
- $8,177.55 (21:13), $8,172.26 (21:24), $8,170.56 (21:46), $8,165.99 (21:56), $8,162.89 (22:07), $8,160.05 (22:17),
  $8,156.13 (22:26), $8,152.43 (22:36), $8,146.63 (22:58), $8,144.43 (23:08), $8,142.25 (23:18), $8,139.90 (23:28),
  $8,138.54 (23:38), $8,137.14 (23:48), $8,135.27 (23:57), $8,132.22 (00:06) and $8,130.34 (00:12).
- The hub's figure moves by about $2 between checks; for example, $8,170.56 at 21:34-21:35 and $8,172.26 at 21:36.
- That is $47.21 in 3 h. The campaign's own GLM estimate rose by $45.89 ($614.09 -> $659.98, PROGRESS.md 21:10 ->
  00:06), so almost all of the spend is this campaign's (the r6 run is paused). It is far above the $1,700 / $1,500
  floors.
- progress.py estimates ~$712 still needed (all GLM).

**Quota (watch.py, 00:12):** user 10.08M free files and 9.57 TiB; project 79.39M free files. That is far from the ~4M
free-inode mark.

**Legs:**
- 5 tdw- legs run, 5 of at most 9 campaigns:
  - w1-g0 / w1-g1 / w1-g2: 21154933 / 21154935 / 21154938, since 11:10:30 on 11-01. They hit the 23 h limit ~10:10 PST
    on 11-02 and need resubmitting then (submit.sh as before).
  - w2-g0 21167343 and w2-g1 21167279 (23 h limit, ~18:22-18:24 PST on 11-02).
- No campaign is pending: Kimi and DeepSeek are complete.

**Polling:**
- 19 polls: 21:13, 21:24, 21:36, 21:46, 21:56, 22:07, 22:17, 22:26, 22:36, 22:47, 22:58, 23:08, 23:18, 23:28, 23:38,
  23:48, 23:57, 00:06 and 00:12.
- Each poll read OPERATOR_NOTE and ran `monitor/w15poll.sh`. That is `monitor/w14poll.sh` with window-15 file names,
  plus `monitor/inflight.py --model glm`.
- `monitor/tail_events.py` (new, read-only) printed the recent hub events of w1-g1 17-0 at 23:38-00:12. Ad-hoc
  read-only scans of the journals of w1-g2 24-0 (21:57) and w2-g1 13-3 (22:58) measured their retry waits and reply
  latencies. Those scans read only the first 1.5-3 KB of each journal line and printed no payload, header or
  credential.
- One line per poll is in `monitor/w15-polls.log`.
- PROGRESS.md was regenerated at 21:31, 21:56, 22:25, 22:54, 23:23, 23:48 and 00:06, and once more right before the
  agent returned.
- This section and the status block were written with `monitor/patches/readme_window15.py`.
- No background process of this agent runs.

## Monitor window 14 (2026-11-01 18:15 - 21:10 PST)

All times are PST.
- **OPERATOR_NOTE:** sha256 56f68663... at every poll from 18:16 to 21:09 (last written 15:21:15, 9,352 bytes). Its
  newest entry is still the one headed 15:35 PST: main's stop of GLM wave 2 and the AUTO-RESTART RULE. No later entry
  came, and no "r6 resumed" entry.

**GLM wave-2 auto-restart (OPERATOR_NOTE 15:35):**
- `monitor/glm429bins.py` (window 13's tool for the rule) reads the 30-min 429 shares of the rollout replies and of the
  budget checks in the four 30-min windows covering the last 60 min. The rule is met when all of them are below 10%.

| poll | rollout replies, last 30 min | budget checks, last 30 min | rule met |
|---|---|---|---|
| 18:16 | 130 / 404 (32.2%) | 2 / 25 (8.0%) | no |
| 18:31 | 68 / 301 (22.6%) | 0 / 23 (0%) | no |
| 18:41 | 43 / 216 (19.9%) | 0 / 23 (0%) | no |
| 18:52 | 8 / 155 (5.2%) | 1 / 22 (4.5%) | no (first window below 10%) |
| 19:02 | 12 / 232 (5.2%) | 1 / 20 (5.0%) | no |
| 19:12 | 13 / 325 (4.0%) | 1 / 21 (4.8%) | no |
| 19:21 | 16 / 402 (4.0%) | 2 / 21 (9.5%) | **yes** |

- At 19:21:33 all four windows (ending 18:51, 19:01, 19:11 and 19:21) were below 10%:
  - rollout replies 4.8%, 3.7%, 4.3% and 4.0%;
  - budget checks 0%, 4.8%, 4.8% and 9.5%.
- At first the drop came with slow GLM replies (`monitor/latency.py`): 18:30-18:50 had only 16-39 replies per 10 min,
  p90 latency 132-242 s and two 408s. From 18:50 latency was back to p50 5.6-8.6 s and p90 22-33 s.
- **Cap 2 (tried, then reverted):**
  - 19:13: partition probe (`sbatch --test-only`, 2 CPU / 16G / 23 h, both accounts). cpu_dataprocessing could start
    at once, cpu_interactive at 19:15 and cpu_datamover at 00:13 on 11-02. As in every earlier window,
    `TDW_PARTITION=cpu_dataprocessing,cpu_datamover`.
  - 19:21:44: `monitor/patches/glm_w2_cap2_manifest_20261101.py --apply` (window 13's prepared script). Its checks
    passed: both STOP files present, no tdw-glm-w2 job, values 4 / 4, live limits 1. It set max / initial concurrency
    4 -> 2 in both campaigns' manifest.json and settings.json, with backups in `provenance/pre-glm-w2-cap2-20261101/`.
  - 19:21: the two STOP files were moved to `monitor/removed-stop-20261101-w14/`.
  - 19:22:00: `TDW_FORCE_W2=1 TDW_MAX_CONCURRENCY_OVERRIDE=2 TDW_PARTITION=cpu_dataprocessing,cpu_datamover bash
    submit.sh glm w2 0 1` gave w2-g0 21167276 / 21167277 and w2-g1 21167278 / 21167279.
  - 19:23:01: the w2-g0 winner 21167276 failed right after staging. ATP refused the campaign: "ValueError: Campaign
    settings changed since its original source snapshot".
    - `deepswe_source.select_source` requires the manifest's settings to equal the frozen
      `source/configs/deepswe.json`, whose sha256 is in `source-manifest.json`. `create_campaign` also compares
      settings.json with the manifest.
    - **So ATP campaign settings are immutable, and the cap of an existing campaign cannot be changed.**
    - Nothing was written to the campaign (freeze_source raised before any rollout), and no sandbox was created.
  - 19:23:57: the four files were restored from the backups (`cp -p` to a temp file + `mv`; sha256 equal to
    `SHA256SUMS`). w2-g1's leg was still staging, so it never reached that check.
  - The script is now marked WITHDRAWN and `--apply` refuses. The as-applied copy is
    `provenance/glm_w2_cap2_manifest_20261101.as-applied-19-21.py`.
- **Resubmitted at 19:24:36:** `TDW_FORCE_W2=1 TDW_MAX_CONCURRENCY_OVERRIDE=2
  TDW_PARTITION=cpu_dataprocessing,cpu_datamover bash submit.sh glm w2 0`.

| campaign | winner (node, account), start | other candidates | driver start line (legs.jsonl) | ATP campaign running |
|---|---|---|---|---|
| glm w2-g0 | 21167343 (cpu1-00008, nemotron_agents_dev), 19:24:38 | 21167344 (sibling won); 21167276 failed (above); 21167277 (sibling won) | 19:24:40, NVAPI_KEY_NEW_2, $8,204.23 left (that budget call got a 429; the figure was present), max_concurrency 2 | 19:25:37 |
| glm w2-g1 | 21167279 (cpu1-00025, nemotron_rl_algo), 19:22:06 | 21167278 (sibling won) | 19:22:11, NVAPI_KEY_NEW_2, $8,204.23 left, max_concurrency 2 | 19:26:09 (Gym staging took 4 min) |

  - Both legs run with the campaigns' own cap 4: Gym logged "Querying with 4 concurrent requests".
    TDW_MAX_CONCURRENCY_OVERRIDE=2 applies only to rec / brk campaigns created in these legs.
  - ATP's live limit restarted at the saved 1 (concurrency.json, rate-limits/admission.json).
  - ATP's recovery marked the 8 attempts that main's 15:21 cancel left behind as interrupted (`owner_job_finished`,
    retryable): w2-g0 1-4, 13-3, 5-3 and 9-4 at 19:25:40; w2-g1 1-3, 1-4, 13-3 and 5-3 at 19:26:11-12. Their sandboxes
    had passed the 18,000 s TTL (~19:08-19:13).
  - The first new rollouts were w2-g0 5-2 at 19:26:40 and w2-g1 5-2 at 19:27:09.
  - The tdw-legs.jsonl lines (19:23:01 for the failed 21167276, 19:25:37 and 19:26:09) name NVAPI_KEY_NEW_2,
    nvidia/zai-org/glm-5.3, the stock prompt and context 262144.
  - Both legs have a 23 h limit, so they end ~18:22-18:24 PST on 11-02 unless done.
- progress.py's GLM-caps bullet records the restart (`monitor/patches/progress_glm_w2_restart_20261101.py`, temp file +
  `mv`, 19:28; pre-patch copy `provenance/progress.pre-glm-w2-restart-20261101.py`).
- **For main:** the rule's "cap 2" could not be applied to the two existing campaigns (above). They run with cap 4 under
  ATP's adaptive limit, which held them at 1 rollout each for the rest of the window. Main can accept this or stop wave
  2 again (STOP files in babysit/glm-w2-g0 and glm-w2-g1, scancel 21167343 / 21167279).

**GLM-5.3 429s after the restart (20% rule, OPERATOR_NOTE 11:15 item 2):**

| poll | rollout replies, last 30 min | budget checks, last 30 min |
|---|---|---|
| 19:34 | 18 / 501 (3.6%) | 5 / 28 (17.9%) |
| 19:44 | 22 / 561 (3.9%) | 7 / 32 (21.9%) |
| 19:54 | 23 / 488 (4.7%) | 7 / 34 (20.6%) |
| 20:04 | 17 / 510 (3.3%) | 4 / 34 (11.8%) |
| 20:13 | 20 / 539 (3.7%) | 3 / 34 (8.8%) |
| 20:23 | 28 / 572 (4.9%) | 1 / 35 (2.9%) |
| 20:31 | 53 / 560 (9.5%) | 1 / 34 (2.9%) |
| 20:40 | 142 / 584 (24.3%) | 7 / 35 (20.0%) |
| 20:50 | 203 / 615 (33.0%) | 12 / 34 (35.3%) |
| 20:58 | 267 / 682 (39.1%) | 14 / 35 (40.0%) |
| 21:09 | 224 / 631 (35.5%) | 11 / 35 (31.4%) |

- The budget-check share went over 20% at 19:44-19:54. The budget calls around the two leg starts (19:24-19:28) got
  429s: "Model capacity reached for nvidia/zai-org/glm-5.3 ... Model TPM: 100000000 ... Remaining: 0".
- From ~20:28 the GLM-5.3 429s were back on the rollout replies too (10-min bins of 37-44%).
- Item 2 says to lower GLM to 2 per campaign and not to pause.
  - Every GLM leg submitted since 14:23 passes TDW_MAX_CONCURRENCY_OVERRIDE=2, the cap of any GLM campaign created from
    now on. The existing campaigns' caps cannot change (above).
  - ATP's adaptive admission held all 5 GLM campaigns at API limit 1 / rollout limit 1 at 20:31-20:58 (last pressure
    rate_limit).
  - Nothing was paused.
- So no GLM campaign had more than 1 rollout in flight, and the 15:21 situation (4 rollouts sharing one API slot) did
  not come back.
- In flight at 20:58, each at 1.6-4.1 successful calls/min with gate waits of at most 0.2 min:
  - w1-g0 8-2 (5 min), w1-g1 5-0 (46 min), w1-g2 24-0 (40 min);
  - w2-g0 13-3 attempt-1 (34 min), w2-g1 5-3 attempt-1 (20 min).

**Endings (GLM; kimi and deepseek are complete):** 13 graded (9 pass, 4 fail), 1 infra and the 8 interrupted
attempts above. No agent timeout.

| ended | campaign | slot | task | result |
|---|---|---|---|---|
| 18:22:21 | w1-g1 | 24-1 | updo-policy-alerting | fail, f2p 16/17 |
| 18:25:16 | w1-g2 | 19-3 | scc-bounded-memory-spilling | fail, f2p 28/31 |
| 19:08:10 | w1-g0 | 2-3 | bandit-structured-nosec-directives | fail, f2p 67/69, p2p 280/282 |
| 19:08:32 | w1-g1 | 20-0 | sqlfmt-create-table-ddl-formatting | pass, f2p 32/32 |
| 19:28:22 | w1-g2 | 12-4 attempt-1 | koota-deferred-mutation-buffer | pass, f2p 71/71 |
| 19:36:47 | w1-g0 | 13-1 | koota-entity-snapshot-rollback | pass, f2p 84/84 |
| 19:40:54 | w1-g1 | 12-3 | kombu-virtual-queue-dead-lettering | pass, f2p 76/76 |
| 19:52:42 | w2-g0 | 5-2 | fastapi-deprecation-response-headers | infra: the agent sandbox was OOMKilled |
| 20:12:46 | w1-g1 | 9-0 | httpx-streaming-json-iteration | fail, f2p 107/108 |
| 20:19:16 | w1-g2 | 25-1 | yaegi-go-embed-directives | pass, f2p 38/38, p2p 58/58 |
| 20:24:41 | w2-g0 | 5-3 attempt-1 | fastapi-deprecation-response-headers | pass, f2p 137/137 (first graded GLM wave-2 slot) |
| 20:31:33 | w1-g0 | 23-3 | textual-richlog-follow-state | pass, f2p 20/20 |
| 20:38:56 | w2-g1 | 5-2 | fastapi-implicit-head-options | pass, f2p 43/43 |
| 20:53:37 | w1-g0 | 3-2 | claude-code-by-agents-recursive-delegation | pass, f2p 7/7 |

- Graded first attempts (PROGRESS.md 18:14 -> 20:50): glm 225 -> 237, passes 157 -> 165 (pass@1 69.8% -> 69.6%).
  Totals 1,353 -> 1,365 graded and 778 -> 786 passes. The 20:53 pass came after the 20:50 report. kimi 564 / 371 and
  deepseek 564 / 250 are unchanged (both complete).
- The broken-ended GLM slot (glm w1-g0) is still the 1 planned re-run. driver.py runs it in a brk campaign (cap 2) once
  w1-g0's main campaign is complete.

**Health:**
- **The one infra ending (w2-g0 5-2):** its agent sandbox (207cd143) ended at ~19:52:30 as "OOMKilled" while the agent
  ran pytest. OpenSandbox returned 410 `SANDBOX::ENDED` (1/1 pods failed, ContainerFailed). OpenCode got a PTY error
  and the verifier a 410. ATP retries the slot.
- All 27 sandbox-error lines in the GLM worker logs come from this one sandbox; w1-g0/g1/g2 and w2-g1 logged none.
- health.py's 30-min infra share went above 30% from this single ending: 33% at 20:03 (1 of 3), 50% at 20:13 (1 of
  2) and 33% at 20:22 (1 of 3); 25% at 19:53 and 0 at the other polls.
- One task sandbox running out of memory is no sign of an outage on the shared cell (as windows 8-13 ruled for single
  endings), so nothing was paused. No sandbox creation or upload error and no OpenSandbox API error spike showed up.

**Agent timeouts (OPERATOR_NOTE 02:15):** glm had 0 of its 14 graded / infra endings. kimi and deepseek had no
endings (both complete). The Kimi auto-pause rule did not apply.

**Budget (key_budget.py, NVAPI_KEY_NEW_2; watch.py readings):**
- $8,216.95 (18:16), $8,213.49 (18:31), $8,211.64 (18:52), $8,206.00 (19:12), $8,201.99 (19:34), $8,197.35 (20:04),
  $8,191.49 (20:13), $8,187.36 (20:31), $8,182.81 (20:40) and $8,180.34 (20:50, 20:58).
- That is $36.61 in 2 h 42 min, far above the $1,700 / $1,500 floors.
- Several budget calls themselves got a 429 (18:52, 19:24, 19:28, 19:44, 20:40, 20:50), but each reading still had a
  remaining figure.
- progress.py estimates ~$746 still needed (all GLM).

**Quota (watch.py, 20:58):** user 10.08M free files and 9.58 TiB; project 79.39M free files. That is far from the ~4M
free-inode mark.

**Legs:**
- 5 tdw- legs run, 5 of at most 9 campaigns:
  - w1-g0 / w1-g1 / w1-g2: 21154933 / 21154935 / 21154938, since 11:10:30. They hit the 23 h limit ~10:10 PST on
    11-02 and need resubmitting then (submit.sh as before).
  - w2-g0 21167343 and w2-g1 21167279.
- No campaign is pending: Kimi and DeepSeek are complete.

**Polling:**
- 18 polls: 18:16, 18:30, 18:41, 18:52, 19:02, 19:12, 19:21 (rule check), 19:33, 19:44, 19:53, 20:03, 20:13, 20:22,
  20:31, 20:40, 20:49, 20:58 and 21:09. The last one found no new ending since 20:53, 5 rollouts in flight (ATP API
  limit 1 / rollout limit 1 in all 5 GLM campaigns) and NEW_2 at $8,174.65.
- Each poll read OPERATOR_NOTE and ran `monitor/w14poll.sh`. That script runs the OPERATOR_NOTE sha, `monitor/poll.sh`
  with WINDOW_START, the `monitor/glm429bins.py` summary, `monitor/w14endings.py`, squeue and the STOP-file list. The
  18:16 poll ran poll.sh and glm429bins.py directly.
- The 19:44 call (sleep + poll in one tool call) hit the 10-min tool limit. Its endings / squeue part was rerun
  separately, and no monitor process was left behind.
- `monitor/latency.py` and `monitor/inflight.py --model glm` ran at 18:16-18:52, 19:44, 20:13, 20:49 and 20:58.
- One line per poll is in `monitor/w14-polls.log`.
- PROGRESS.md was regenerated at 18:32, 19:28 (after the text patch), 19:54, 20:23, 20:50 and 21:01 (glm 238
  graded, pass@1 69.7%; 1,366 graded and 787 passes in total), and once more right before the agent returned.
- Changes:
  - the cap-2 manifest edit and its revert (above);
  - the STOP-file removal;
  - the progress.py text patch and the WITHDRAWN note in the cap-2 script;
  - two new tools, `monitor/w14poll.sh` and `monitor/w14endings.py` (read-only, apart from what poll.sh's watch.py
    writes).
- This section and the status block were written with `monitor/patches/readme_window14.py`; the 21:09 poll was
  added with `monitor/patches/readme_window14b.py`.
- No other job was submitted or cancelled. No sandbox cleanup was run; none was needed, since the failed leg created
  no sandbox.
- At 18:18 a grep of the ATP tree that this agent ran went to the background on a tool timeout. It was stopped by PID
  at 18:20. No process of this agent runs now.

## Monitor window 13 (2026-11-01 15:21 - 18:20 PST)

All times are PST.
- **OPERATOR_NOTE:** sha256 56f68663... at every poll from 15:24 to 18:10. The file was last written at 15:21:15 (9,352
  bytes). Its newest entry, headed "15:35 PST":
  - main STOPPED GLM wave 2 (STOP files in babysit/glm-w2-g0 and glm-w2-g1, scancel of 21158897 / 21158899), so that
    its 8 in-flight rollouts are retried later instead of ending as 3-hour agent-timeout fails;
  - GLM wave 1 keeps running;
  - AUTO-RESTART RULE: when GLM-5.3 429s stay below 10% of the rollout replies AND below 10% of the budget checks for
    60 consecutive minutes, remove the two STOP files and resubmit GLM wave 2 with TDW_FORCE_W2=1
    TDW_MAX_CONCURRENCY_OVERRIDE=2 (cap 2), and report it; otherwise leave it stopped;
  - never pause wave 1; "r6 resumed" entries take precedence.
  No later entry came.

**GLM wave 2, stopped by main:**
- sacct: 21158897 (w2-g0) and 21158899 (w2-g1) ended CANCELLED at 15:21:15, after 1:14:35 / 1:14:32. The STOP files
  (103 bytes each) are dated 15:21. watch.py logged `campaign_ended` for both at 15:25.
- `monitor/stopcheck.py` (15:26) found that none of the 8 attempts has an end marker or a cleanup confirmation, so their
  agent sandboxes are orphan candidates:
  - w2-g0: 1-4, 13-3, 5-3, 9-4;
  - w2-g1: 1-3, 1-4, 13-3, 5-3.
  Both launches have only the start-of-leg recovery-0.json (no shutdown pass ran). The attempts' http.jsonl journals
  stopped at 15:21:12-15, so no hub call has been made since: the ATP workers are gone.
- As main ruled for its 05:15 cancel (OPERATOR_NOTE 05:40 item 1), the sandboxes are left to the 18,000 s TTL. The
  w2-g0 1-4 sandbox was recorded at 14:08:38, so expiry is ~19:08-19:13 PST. ATP's recovery handles the 8 attempts when
  wave 2 resumes. No cleanup tool was built or run.
- health.py and inflight.py still list the 8 attempts as running, because they have no end marker. They are not in
  flight.

**Auto-restart rule (OPERATOR_NOTE 15:35):**
- New read-only tool `monitor/glm429bins.py`, with the same sources and counting as `monitor/glm429.py`:
  - it computes the 30-min 429 share in the four 30-min windows ending at now-30, now-20, now-10 and now, which
    together cover [now-60, now];
  - the rule counts as met only if every window is below 10% for both the rollout replies and the budget checks;
  - it also prints 10-min bins.

| poll | rollout replies, last 30 min | budget checks, last 30 min | rule met |
|---|---|---|---|
| 15:26 | 350 / 615 (56.9%) | 13 / 35 (37.1%) | no |
| 15:36 | 287 / 523 (54.9%) | 15 / 29 (51.7%) | no |
| 15:47 | 270 / 504 (53.6%) | 17 / 24 (70.8%) | no |
| 15:58 | 268 / 494 (54.3%) | 14 / 23 (60.9%) | no |
| 16:09 | 277 / 537 (51.6%) | 15 / 23 (65.2%) | no |
| 16:19 | 221 / 450 (49.1%) | 12 / 21 (57.1%) | no |
| 16:29 | 189 / 441 (42.9%) | 12 / 22 (54.5%) | no |
| 16:40 | 228 / 576 (39.6%) | 8 / 22 (36.4%) | no |
| 16:50 | 238 / 600 (39.7%) | 6 / 21 (28.6%) | no |
| 17:00 | 214 / 524 (40.8%) | 6 / 22 (27.3%) | no |
| 17:11 | 159 / 391 (40.7%) | 9 / 22 (40.9%) | no |
| 17:21 | 133 / 352 (37.8%) | 10 / 21 (47.6%) | no |
| 17:31 | 165 / 436 (37.8%) | 10 / 22 (45.5%) | no |
| 17:42 | 162 / 508 (31.9%) | 7 / 22 (31.8%) | no |
| 17:52 | 161 / 539 (29.9%) | 5 / 21 (23.8%) | no |
| 18:01 | 125 / 482 (25.9%) | 5 / 22 (22.7%) | no |
| 18:11 | 109 / 389 (28.0%) | 3 / 23 (13.0%) | no |

- The 429s eased slowly, but never came near 10% of the rollout replies. GLM wave 2 stays stopped. The earliest
  possible restart is 60 min after both shares first fall below 10%.
- The budget checks fell from ~11-12 to ~7 per 10 min, because only the 3 wave-1 legs' guards (every 300 s) and
  watch.py make them now.
- **Cap 2 on resume (not covered by the entry; for main):**
  - Both wave-2 campaigns already exist. ATP reads an existing campaign's cap from its `manifest.json`
    (`settings.max_concurrency_per_model` 4, `initial_concurrency_per_model` 4, written at creation on 14:06-14:07).
    That value is both the adaptive-admission maximum and Gym's `--concurrency` (`run_deepswe.shard_concurrency`).
  - rollouts.probe() writes `settings.json` (also 4 / 4), and ATP creates the manifest, only when a campaign is
    created. So TDW_MAX_CONCURRENCY_OVERRIDE=2 sets the cap only for rec / brk campaigns created in the leg.
  - A resubmission with the override alone therefore keeps cap 4. The live limit restarts at the saved 1
    (`concurrency.json` = 1; `rate-limits/admission.json` limit 1 / rollout_limit 1) and can ramp back to 4 after
    healthy completions.
  - A hard cap of 2 needs both campaigns' `manifest.json` and `settings.json` set to 2 / 2 before the resubmission. All
    saved live limits are 1, inside ATP's bounds (shards 1 <= limit <= maximum).
  - `monitor/patches/glm_w2_cap2_manifest_20261101.py` does this:
    - dry run by default;
    - with `--apply`, it refuses unless both STOP files exist, no tdw-glm-w2-* job is queued or running, the values
      are 4 / 4 and every live limit is <= 2;
    - it backs up the four files to `provenance/pre-glm-w2-cap2-20261101/` (+ SHA256SUMS) and writes each new file
      with temp file + `os.replace`, in the originals' JSON layout.
  - Its dry runs at 18:13 and 18:14 passed ("checks OK"; no file changed, sha256 of the four files unchanged). It was NOT applied:
    the rule was not met, and a manifest edit is a change to ATP campaign state that no entry asked for by name.

**GLM wave 1:**
- The 3 legs ran all window: 21154933 / 21154935 / 21154938, on cpu_dataprocessing since 11:10:30.
- ATP's adaptive admission kept every campaign at rollout limit 1. The API limit was 1, except w1-g1 at 2 (17:20,
  18:00). Gate waits were near 0.
- 10 graded endings, 8 pass and 2 fail. No infra ending and no agent timeout:

| ended | campaign | slot | task | result |
|---|---|---|---|---|
| 15:42:35 | w1-g2 | 7-4 | happy-dom-abort-pending-body-reads | pass, f2p 14/14 |
| 15:51:10 | w1-g0 | 13-0 | koota-entity-snapshot-rollback | pass, f2p 84/84 |
| 16:11:35 | w1-g1 | 16-0 attempt-1 | opa-template-string-reconstruction | pass, f2p 5/5 |
| 16:28:49 | w1-g1 | 25-1 | wazero-multi-module-snapshots | pass, f2p 78/78 |
| 16:53:09 | w1-g0 | 23-2 | textual-richlog-follow-state | pass, f2p 20/20 |
| 16:55:13 | w1-g2 | 25-0 | yaegi-go-embed-directives | fail, f2p 0/38, p2p 0/58 |
| 17:32:15 | w1-g0 | 8-1 | happy-dom-deterministic-intersectionobserver | fail, f2p 13/14 (as GLM's 8-0 at 13:09) |
| 17:54:39 | w1-g1 | 16-3 | opa-template-string-reconstruction | pass, f2p 5/5 |
| 17:55:52 | w1-g2 | 24-2 attempt-1 | valibot-recursive-schema-composition | pass, f2p 10/10 |
| 18:02:51 | w1-g0 | 18-3 | psd-tools-blend-range-api | pass, f2p 45/45 |

- **yaegi-go-embed-directives (w1-g2 25-0) is the model's outcome:** the verifier applied model.patch (54,715 bytes)
  and then test.patch, and the Go test build failed (`interp_test [build failed]`). So every test was missing from
  the report. The other 4 GLM, 5 Kimi and 5 DeepSeek attempts on this task all reached p2p 58/58.
- In flight at 18:11:
  - w1-g0 2-3, bandit-structured-nosec-directives (8 min);
  - w1-g1 24-1, updo-policy-alerting (17 min);
  - w1-g2 19-3, scc-bounded-memory-spilling (15 min).
- Graded first attempts (PROGRESS.md, 15:20 -> 18:11): glm 215 -> 225, passes 149 -> 157 (pass@1 69.3 -> 69.8%).
  Totals 1,343 -> 1,353 graded and 770 -> 778 passes. kimi 564 / 371 and deepseek 564 / 250 are unchanged (both
  complete).
- The one broken-ended GLM slot (glm w1-g0) is still the 1 planned re-run. driver.py runs it in a brk campaign once
  w1-g0's main campaign is complete; GLM campaigns created after 14:23 get cap 2.

**Health:** health.py's 30-min infra share was 0.0 (or no ending) at every poll. No sandbox creation or upload error,
PTY closure, OpenSandbox API error or verifier error showed up. Nothing was paused.

**Agent timeouts (OPERATOR_NOTE 02:15):** glm had 0 of its 10 endings. kimi and deepseek had no endings (both
complete). The Kimi auto-pause rule did not apply.

**Budget (key_budget.py, NVAPI_KEY_NEW_2; watch.py readings):** $8,245.19 (15:25), $8,240.22 (15:58), $8,235.56
(16:29), $8,229.89 (17:00), $8,224.82 (17:31), $8,218.13 (18:01) and $8,216.95 (18:11). That is $28.24 in 2 h 46 min,
far above the $1,700 / $1,500 floors.
- Several budget calls themselves got a 429 (15:25, 15:46, 16:08, 16:29, 17:11, 17:21), but each reading still had a
  remaining figure.
- progress.py estimates ~$797 still needed (all GLM).

**Quota (watch.py, 18:11):** user 10.08M free files and 9.59 TiB; project 79.45M free files. That is far from the
~4M free-inode mark.

**Polling:**
- 17 polls: 15:24, 15:35, 15:46, 15:57, 16:08, 16:18, 16:29, 16:39, 16:49, 17:00, 17:10, 17:20, 17:30, 17:41, 17:51,
  18:01 and 18:10.
- Each poll read OPERATOR_NOTE and ran `monitor/poll.sh` with `WINDOW_START="2026-11-01 15:21"` under `timeout`, then
  `monitor/glm429bins.py`. New endings were read from the attempts' terminal.json.
- `monitor/inflight.py --model glm` ran at 15:26, 16:08, 16:49, 17:21, 17:51 and 18:11.
- One line per poll is in `monitor/w13-polls.log`.
- PROGRESS.md was regenerated at 15:46, 15:47 (after the text patch below), 16:18, 16:49, 17:21, 17:51 and 18:11, and
  once more right before the agent returned.
- progress.py's GLM-caps bullet now names main's wave-2 stop and the auto-restart rule
  (`monitor/patches/progress_glm_w2_stop_20261101.py`, temp file + `mv`, 15:47; pre-patch copy
  `provenance/progress.pre-glm-w2-stop-20261101.py`).
- No STOP file was written or removed. No job was submitted or cancelled, no manifest was edited and no sandbox cleanup
  was run. This agent started no background process.
- This section and the status block were written with `monitor/patches/readme_window13.py`.

## Monitor window 12 (2026-11-01 14:05 - 15:21 PST; early return to main)

All times are PST. The window was meant to last ~3 h. This agent returned at ~15:21 with needs_main, because of the
GLM wave-2 agent-timeout risk below.
- **OPERATOR_NOTE:** sha256 ad201caa... from 14:05 to the last poll at 15:17. The file was last written at 14:05:15
  (8,541 bytes) and holds the 14:15 PST entry:
  1. start GLM wave 2 now (both groups) with TDW_FORCE_W2=1 and TDW_MAX_CONCURRENCY_OVERRIDE=4, without
     reset_admission and without stopping the wave-1 legs;
  2. keep the 20% 429 rule (lower GLM to 2 for new campaigns), no pause by this agent;
  3. patch collect.py so that kimi w2-g1 13-0 counts as a GRADED FAIL, listed next to the 3 verifier-kill
     needs_review slots;
  4. "r6 resumed" entries take precedence.
  No later entry came.

**GLM wave 2 (OPERATOR_NOTE 14:15 item 1):**
- **Partition probe** (`sbatch --test-only`, 2 CPU / 16G / 23 h, 14:05:56):
  - cpu_dataprocessing: 15:01 on both accounts.
  - cpu_datamover: 19:40 on both accounts.
  - cpu_interactive: at once (agents_dev).
  - As in every earlier window, `TDW_PARTITION=cpu_dataprocessing,cpu_datamover`, leaving cpu_interactive free.
- **Submitted 14:06:35:** `TDW_FORCE_W2=1 TDW_MAX_CONCURRENCY_OVERRIDE=4 TDW_PARTITION=cpu_dataprocessing,cpu_datamover
  bash submit.sh glm w2 0 1`. submit.sh's dry-run preflight passed for both campaigns. Both winners started at once
  on cpu_datamover (nemotron_agents_dev), and each loser ended with "sibling candidate won".

| campaign | winner (node) | loser | driver start line (legs.jsonl) | ATP campaign running |
|---|---|---|---|---|
| glm w2-g0 | 21158897 (cpu1-00096), 14:06:40 | 21158898 | 14:06:42, NVAPI_KEY_NEW_2, $8,257.14 left, max_concurrency 4 | 14:07:31 |
| glm w2-g1 | 21158899 (cpu1-00097), 14:06:43 | 21158900 | 14:06:52, NVAPI_KEY_NEW_2, $8,257.14 left, max_concurrency 4 | 14:13:06 (Gym staging took 6 min) |

- **Per-campaign checks:**
  - `manifest.json`: max_concurrency_per_model 4, repeats 5, models [glm], 17 tasks.
  - `tdw-legs.jsonl` (14:07:31 / 14:13:07): hub_key NVAPI_KEY_NEW_2, nvidia/zai-org/glm-5.3, the stock prompt
    (962fbf3cb3ec) and context 262144.
  - Each worker.log wrapper line names NVAPI_KEY_NEW_2.
- **Admission:** each wave-2 campaign admitted 4 rollouts at its start: w2-g0 13-3, 1-4, 5-3, 9-4; w2-g1 1-3, 13-3,
  1-4, 5-3. The 429s then cut both to API limit 1 / rollout limit 1 (w2-g0 by 14:11, w2-g1 by 14:23). So no further
  wave-2 slot started in the window.
- Campaigns in flight: 5 of at most 9 (all GLM). Kimi and DeepSeek are complete.

**collect.py patch (OPERATOR_NOTE 14:15 item 3; 14:14, before any collect run):**
- Patch script `monitor/patches/collect_override_20261101.py` (exact-text replacements, temp file + `mv`; no job runs
  collect.py or verify_collection.py). The pre-patch copies are in `provenance/pre-collect-override-20261101/`, with
  sha256 before (`SHA256SUMS`) and after (`POST-PATCH.SHA256SUMS`).
- **collect.py:**
  - New `GRADED_FAIL_OVERRIDES` = {(kimi, w2-g1, scriggo-method-declarations, r0)}, applied by
    `graded_fail_override()` in `jobs_of()`.
  - The slot becomes status "fail" only while its results record is still the attempt-0 infra ending with
    evaluation_completed true and reward 0. Otherwise the new check `status_overrides_unmatched` names it.
  - The dataset row and the ledger line carry `status_override` {from infra, to fail, atp_status needs_review, reason,
    source}. `first_attempt_status` is fail too (same attempt, no re-run).
  - `MANIFEST.json` lists the applied overrides (`status_overrides`). Ledger lines also keep `verifier_error`.
  - FINAL-REPORT.md: the Definitions line names the override. "Re-runs, agent timeouts, broken replies" gains two
    lines:
    - the slots ATP left ungraded and does not retry (needs_review / final infra: glm w1-g1
      pwntools-tube-multiplexing r0 and r2, deepseek w1-g0 langchain-request-coalescing r4, verifier killed at
      1,800 s);
    - the overridden slot, with its reason and source.
- **final/verify_collection.py:** its ATP and progress.py comparisons leave out the MANIFEST `status_overrides` (ATP
  calls the slot needs_review, and progress.py counts it as final infra). A complete campaign's final_infra slots
  count on the ATP-ungraded side; before this change, the complete Kimi / DeepSeek campaigns would have failed that
  check.
- **Test** (`monitor/patches/collect_override_test_20261101.py`, read-only):
  - `jobs_of()` old vs new on all 10 results files: exactly 1 record changed (kimi w2-g1 scriggo r0, infra -> fail,
    first record = the same object). `OVERRIDE_PROBLEMS` was empty.
  - `build_row()` (export cost): status fail, reward 0.0, f2p 0/48, first_attempt_status fail, the override present,
    629 output items, patch_chars 0.
  - `ledger_entry()`: state "final", in_dataset true.
  - `render_report()` on the 2026-10-31 partial collection, with the overridden line and a final_infra line injected:
    the two new lines list the 3 verifier kills and the override. The Definitions line renders.
    status_overrides_unmatched: 0.
  - The patched verify_collection.py ran on the partial dataset (output to a temp file, deleted). The only problems
    were the 3 expected progress.py differences (today's PROGRESS-rows.json vs the 10-31 ledger), and no ATP
    mismatch.

**GLM 429 rule (OPERATOR_NOTE 11:15 item 2 / 14:15 item 2):**
- `monitor/glm429.py --minutes 30` (rollout replies 429 / all; budget checks 429 / all):

| poll | rollout replies | budget checks |
|---|---|---|
| 14:11 | 48 / 302 (15.9%) | 5 / 29 (17.2%) |
| 14:23 | 112 / 424 (26.4%) | 9 / 30 (30.0%) |
| 14:34 | 197 / 572 (34.4%) | 12 / 34 (35.3%) |
| 14:44 | 318 / 685 (46.4%) | 15 / 34 (44.1%) |
| 14:54 | 386 / 708 (54.5%) | 19 / 35 (54.3%) |
| 15:05 | 413 / 705 (58.6%) | 17 / 34 (50.0%) |
| 15:15 | 357 / 613 (58.2%) | 13 / 34 (38.2%) |
| 15:18 | 357 / 617 (57.9%) | 14 / 35 (40.0%) |

- The 429s were present on the wave-1 campaigns before wave 2 started: w1-g0 had 23 in 13:41-14:11, against 2-8 per
  30 min in window 11.
- `monitor/non200.py --model glm --minutes 75` (15:17):
  - 749 of the 751 429s were LiteLLM "RateLimitError ... OpenAIException - Too Many Requests ... model group
    'nvidia/zai-org/glm-5.3'", i.e. the upstream serving GLM-5.3.
  - 2 were the hub's "Model capacity reached for nvidia/zai-org/glm-5.3 ... Model TPM: 100000000".
  - 7 replies were 408s (LiteLLM's 360 s timeout).
- **Action at 14:23 (rule > 20%):** GLM campaigns created from now on get 2, i.e. TDW_MAX_CONCURRENCY_OVERRIDE=2 for
  every GLM leg submitted later (rec / brk campaigns of that leg).
  - No GLM leg was submitted after 14:23 in this window.
  - The 5 running GLM campaigns keep their manifest caps (8 / 4). ATP's adaptive admission already held each at API
    limit 1 / rollout limit 1.
  - `run_deepswe.py set-concurrency` would have raised their API limit from 1 to 2, so it was not used. No manifest
    was edited, and nothing was paused.
  - progress.py's GLM-caps bullet says so (`monitor/patches/progress_glm_w2_start_20261101.py`, temp file + `mv`,
    14:25; pre-patch copy `provenance/progress.pre-glm-w2-start-20261101.py`).
- key_budget.py's budget checks are GLM-5.3 calls as well. Some got 429 (watch.py 14:11, 14:54, 15:15), but every
  watch.py reading still had a remaining figure. The driver guard ignores a reading without one, and the floors are far
  away.

**Why main was asked (GLM wave-2 agent timeouts):**
- `monitor/inflight.py --model glm` at 15:18:

| rollout | elapsed | 200 replies | calls/min | API-gate wait | deadline in |
|---|---|---|---|---|---|
| w1-g0 13-0 | 23 min | 68 | 2.91 | 0.1 min | 157 min |
| w1-g1 16-0 (attempt-1) | 51 min | 110 | 2.16 | 0.1 min | 129 min |
| w1-g2 7-4 | 81 min | 170 | 2.10 | 0.1 min | 99 min |
| w2-g0 1-4 / 13-3 / 5-3 / 9-4 | 70 min | 47 / 40 / 48 / 47 | 0.57-0.69 | 35-47 min | 110 min (~17:08) |
| w2-g1 1-3 / 1-4 / 13-3 / 5-3 | 64 min | 26 / 32 / 36 / 20 | 0.31-0.56 | 33-48 min | 116 min (~17:14) |

- Each wave-2 campaign's 4 rollouts share API limit 1. The single wave-1 rollout of each wave-1 campaign does not wait
  at the gate.
- GLM's 217 graded attempts in this run needed a median of 129 logical calls (q25 96, q75 179; 15% <= 80, 29% <= 100;
  from their `generation-validity-v2-r7.json`).
- At their 15:15-15:18 pace, the 8 wave-2 rollouts would reach ~55-125 calls by their deadlines. Unless the 429s
  ease, most of them (rough expectation ~6 of 8) end on ATP's 10,800 s agent timeout. They would then be graded as
  timeout fails: flagged ended_by_agent_timeout and, as for Kimi (OPERATOR_NOTE 02:15), kept with no re-run.
- **Options for main** (not taken by this agent; OPERATOR_NOTE 11:15 / 14:15: "do not pause on your own"):
  - let them run;
  - stop GLM wave 2 now (STOP files in babysit/glm-w2-g*, scancel 21158897 / 21158899), so that ATP's recovery
    marks the 8 attempts interrupted (retryable) and re-runs them later. This loses their first ~70 min of hub calls;
  - a user decision on the agent timeout (OPERATOR_NOTE 02:15: changing it needs the user).

**Campaigns and endings (14:05-15:18):**

| ended | campaign | slot | task | result |
|---|---|---|---|---|
| 14:16:55 | glm w1-g0 | 18-2 | psd-tools-blend-range-api | pass, f2p 45/45, 234 calls |
| 14:27:08 | glm w1-g1 | 17-2 attempt-0 | participle-grammar-conflict-analysis | infra (ATP "infrastructure", retryable) |
| 14:54:46 | glm w1-g0 | 0-0 | abs-module-cache-flags | pass, f2p 20/20, 114 calls |

- **The infra ending (glm w1-g1 17-2):**
  - One hub call ran 13:42:47-14:26:18 and got 408s (LiteLLM's 360 s timeout) until ATP's retries ran out
    (`unresolved_http_outcomes`: retry_exhausted, http_attempt 7, status 408).
  - OpenCode then exited 1, and ATP's validity check said "infrastructure". The slot is retryable.
  - This was a hub reply problem, not a sandbox / OpenSandbox one.
- health.py's 30-min infra share read 0.5 (14:33, 14:43) and 1.0 (14:53): this single ending among 1-2 endings. It is
  no sign of an outage on the shared OpenSandbox cell, so nothing was paused. No sandbox creation or upload error, PTY
  closure, OpenSandbox API error or verifier error appeared.
- **Agent timeouts (OPERATOR_NOTE 02:15):** glm 0 of the 3 GLM endings. kimi and deepseek had no ending (both models
  are complete). The Kimi auto-pause rule did not apply.
- **Graded first attempts (PROGRESS.md 14:04 -> 15:04):** glm 213 -> 215, passes 147 -> 149 (pass@1 69.0 -> 69.3).
  kimi 564 / 371 and deepseek 564 / 250 are unchanged. Total 1,341 -> 1,343 graded, 768 -> 770 passes.

**Budget (key_budget.py, NVAPI_KEY_NEW_2; watch.py readings):** $8,257.14 (14:06 leg starts and 14:11), $8,255.29
(14:23), $8,253.45 (14:34), $8,251.99 (14:44), $8,249.91 (14:54), $8,248.37 (15:04) and $8,246.91 (15:15 and 15:18).
That is $10.23 in 72 min, far above the $1,700 / $1,500 floors. progress.py estimates ~$808 still needed (all GLM).

**Quota (watch.py, 15:18):** user 10.08M free files and 9.6 TiB; project 79.63M free files. That is far from the ~4M
free-inode mark.

**Polling:**
- 8 polls: 14:10, 14:23, 14:33, 14:43, 14:53, 15:04, 15:14 and 15:17.
- Each poll read OPERATOR_NOTE and ran `monitor/poll.sh` with `WINDOW_START="2026-11-01 14:05"` under `timeout`, then
  `monitor/glm429.py --minutes 30`. `monitor/inflight.py --model glm` ran at 14:54, 15:15 and 15:18.
- watch.py's events: `operator_note` (the 14:15 entry's sha256) at 14:11, and `rate_limited` at every poll.
- PROGRESS.md was regenerated at 14:25, 14:44 and 15:04, and once more right before the agent returned.
- No STOP file was written, nothing was cancelled, and no sandbox cleanup was needed (no leg ended). This agent
  started no background process.
- This section and the status block were written with `monitor/patches/readme_window12.py`.

## Monitor window 11 (2026-11-01 11:08 - 14:05 PST)

All times are PST. The window lasted ~3 h.
- **OPERATOR_NOTE:** sha256 93b4db2f... at every poll (11:08 to 14:00). Its 11:15 PST entry (written at ~11:06) said:
  - resume the GLM campaigns now: remove main's STOP files in babysit/glm-*, resubmit the 3 wave-1 GLM legs (and wave 2
    as the gating allows) with TDW_MAX_CONCURRENCY_OVERRIDE=4 for every GLM campaign, key NVAPI_KEY_NEW_2, floors
    unchanged;
  - lower GLM to 2 per campaign if GLM 429s on the rollouts or on the budget checks exceed 20% over 30 min;
  - follow a later "r6 resumed" entry;
  - keep kimi w2-g1 13-0's graded fail, no re-run.
  No later entry came.
- **Code:** progress.py only, by two one-off patch scripts (temp file + `mv`; progress.py is run by no job):
  - `monitor/patches/progress_glm_resume_20261101.py` adds the "GLM caps" header bullet to PROGRESS.md.
  - `monitor/patches/progress_glm_w2_cap_20261101.py` gives a GLM wave-2 campaign that does not exist yet the cap 4
    instead of submit.sh's GLM default of 8 (the "Campaigns now" cap column and the projection).
  - The pre-patch copies and their sha256 are in `provenance/`.
- **New files:**
  - `monitor/glm429.py` (read-only GLM 429 shares).
  - `monitor/removed-stop-20261101/` (copies of main's STOP files).
  - `monitor/.poll-w11-last.txt` and `monitor/.progress-w11-last.txt` (output of the last poll.sh and progress.py runs).
  - This section and the status block were written with `monitor/patches/readme_window11.py`.

**GLM resume (OPERATOR_NOTE 11:15):**
- **STOP files:** main's 5 STOP files (babysit/glm-w1-g0..g2, glm-w2-g0/g1; text "main 2026-11-01 05:15 PST: paused to
  protect the r6 judge ...") were copied to `monitor/removed-stop-20261101/` and removed at 11:09.
- **Partition probe** (`sbatch --test-only`, 2 CPU / 16G / 23 h, 11:09):
  - cpu_dataprocessing: 12:05 on both accounts.
  - cpu_datamover: 16:07 (agents_dev).
  - cpu_interactive: at once (agents_dev).
  - As in every earlier window, `TDW_PARTITION=cpu_dataprocessing,cpu_datamover`: cpu_interactive's QoS p_cpu_short
    is shared with the r6 run's cpu_short jobs.
- **Submitted 11:10:** `TDW_MAX_CONCURRENCY_OVERRIDE=4 TDW_PARTITION=cpu_dataprocessing,cpu_datamover bash submit.sh glm
  w1 0 1 2` (leg 3 of each campaign). All three winners started at 11:10:30 on cpu_dataprocessing (nemotron_rl_algo),
  well before the probe's 12:05. Each loser ended at 11:10:35 with "sibling candidate won".

| campaign | winner | loser | driver start line (legs.jsonl) |
|---|---|---|---|
| glm w1-g0 | 21154933 | 21154932 | 11:10:37, NVAPI_KEY_NEW_2, $8,275.76 left, max_concurrency 4 |
| glm w1-g1 | 21154935 | 21154934 | 11:11:18, NVAPI_KEY_NEW_2, $8,275.76 left, max_concurrency 4 |
| glm w1-g2 | 21154938 | 21154937 | 11:10:47, NVAPI_KEY_NEW_2, $8,275.74 left, max_concurrency 4 |

- **Per-campaign checks:**
  - Each campaign's `tdw-legs.jsonl` gained an entry at 11:11:21-11:12:41 with hub_key NVAPI_KEY_NEW_2,
    nvidia/zai-org/glm-5.3, the stock prompt (962fbf3cb3ec) and context 262144.
  - Each worker.log wrapper line names NVAPI_KEY_NEW_2.
- **ATP's recovery** (`recovery-0.json` of the new launch dirs, 11:11:24-11:12:44):
  - attempted / recovered 22 / 22 (w1-g0), 30 / 30 (w1-g1) and 37 / 37 (w1-g2); 0 remaining, 0 errors.
  - The 11 attempts that main's 05:06 cancel left without an end marker are now `interrupted` (owner_job_finished):
    w1-g0 2, w1-g1 4, w1-g2 5. Their slots are retryable. `monitor/stopcheck.py` found no other unended attempt
    besides the 3 new rollouts.
- **Concurrency, as it is (not as the leg records suggest):**
  - ATP reads the cap from each campaign's manifest.json (`max_concurrency_per_model`, written at creation): 8 for the
    three GLM wave-1 campaigns. The leg's `--max-concurrency 4` (legs.jsonl `max_concurrency: 4`) only sets
    `rollouts.PROBE_MAX_CONCURRENCY` for campaigns created in that leg: rec / brk recovery campaigns, and wave 2.
  - ATP's adaptive admission restarted each wave-1 campaign at its saved live state (`concurrency.json` = 1). It held
    rollout limit 1 and API limit 1-2 all window. Every pressure event (429 rate_limit, 408 upstream_timeout, 503
    upstream_error) starts a new rollout epoch (`deepswe_admission.note_health`). The rollout limit doubles only when a
    rollout admitted in the current epoch completes (`note_rollout_completed`). With one such event per campaign every
    12-19 min (9-15 per campaign in the window) and GLM rollouts of 35-85 min, that did not happen in this window.
  - ATP's `run_deepswe.py set-concurrency` lowers the live limits at once, but the adaptive ramp can raise them back to
    the manifest's 8, so it is no hard cap of 4. This agent did not use it, and did not edit any manifest.

**Campaigns:**
- **kimi w2-g1 complete** at 11:17:34 (job 21142493, leg 1, exit 0): first attempts 75 pass / 9 fail / 1 infra (13-0,
  needs_review), pass@1 89.3%. No rec or brk re-runs.
  - Slot 11-1 (pebble-durability-wait-apis), ATP's retry attempt-1: graded pass at 11:16:43.
  - `monitor/stopcheck.py`: 85 attempts ended, 0 orphan candidates.
  - **All 5 Kimi campaigns are done.**
- **GLM wave 1:** 7 slots graded (1 pass). Each fail had a completed evaluation and an applied patch.

| time | campaign | slot | task | result |
|---|---|---|---|---|
| 12:01 | w1-g2 | 9-1 | igel-persist-feature-schema | fail, f2p 6/24 |
| 12:16 | w1-g1 | 11-2 attempt-1 | kea-atomic-signal-selectors | fail, f2p 11/12 |
| 12:34 | w1-g0 | 7-3 | go-git-worktree-merge-conflicts | fail, f2p 15/17 |
| 13:00 | w1-g2 | 11-1 | kgateway-consistent-hash-policy | fail, f2p 0/2 |
| 13:09 | w1-g0 | 8-0 | happy-dom-deterministic-intersectionobserver | fail, f2p 13/14 |
| 13:41 | w1-g1 | 1-2 | awilix-async-container-initialization | fail, f2p 23/24 |
| 13:57 | w1-g2 | 1-2 | bandit-interprocedural-taint-checks | pass, f2p 66/66 |

  GLM's earlier attempts on these tasks scored alike: awilix 23/24 three times, igel 6/24 three times (and 2 passes).
  So the run of fails is the tasks, not the harness.
- **Not submitted:** GLM wave 2. No GLM wave-1 campaign finished, so submit.sh's gating keeps both waiting.
- **No STOP file was written and nothing was cancelled.** No sandbox cleanup was needed (the only leg that ended,
  kimi w2-g1, ended normally).

**Graded first attempts (PROGRESS.md 11:05 -> 14:01):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | agent timeouts (total) |
|---|---|---|---|---|---|
| kimi | 563 -> 564 | 370 -> 371 | 65.7 -> 65.8 | 66.7 -> 66.7 | 8 |
| glm | 206 -> 213 | 146 -> 147 | 70.9 -> 69.0 | 70.9 -> 69.0 | 0 |
| deepseek | 564 | 250 | 44.3 | 44.3 | 0 |

Total: 1,333 -> 1,341 graded, 766 -> 768 passes.

**Agent timeouts (OPERATOR_NOTE 02:15):**
- kimi: 0 of the 1 Kimi slot that ended in the window. The auto-pause rule did not trigger, and all Kimi campaigns are
  done.
- glm: 0 of 7. deepseek: no slot ended.

**Infra and health:**
- health.py's 30-min infra share read 0.0 or "no endings" at every poll. The only non-graded endings were the 11
  interrupted attempts above, which ATP's recovery marked at the leg start (not infra).
- No sandbox creation or upload error, PTY closure, OpenSandbox API error or verifier error.

**GLM-5.3 hub replies (monitor/glm429.py, NVAPI_KEY_NEW_2):**
- **Whole window (11:07-14:01):**
  - Rollout replies: 1,074 responses; 15 were 429 (1.4%), 14 were 408 (LiteLLM's 360 s timeout) and 6 were 503
    (Envoy "upstream connect error").
  - key_budget.py's budget checks: 2 429s of 131 (1.5%).
- **Per 30 min:**
  - Rollout 429s: 0 until 12:27, then 2-8 per 30 min, at most 3.7% (8 of 214 at 14:01).
  - Budget-check 429s: at most 10% (2 of 20 at 12:59).
  - Both stayed below the 20% of OPERATOR_NOTE 11:15 item 2, so GLM was not lowered to 2.
- **The 429 text** names the hub-wide model limit: "Model capacity reached for nvidia/zai-org/glm-5.3. Priority: prod,
  Rate limit type: tokens, Model TPM: 100000000, ... Remaining: 0". This campaign had 3 GLM rollouts in flight, and
  the r6 judge is paused.
- **408 runs:** the 408s came mostly from w1-g1 (12 of 14), in runs on single calls as in window 3. Slot 1-2 had 9
  replies in its first 43 min.
- **Kimi** (until 11:17): 0 non-200 replies.

**Budget (key_budget.py, NVAPI_KEY_NEW_2; watch.py readings):**
- Remaining: $8,276.13 (11:09), $8,275.20 (11:25), $8,274.41 (11:36), $8,273.34 (11:48), $8,272.36 (11:58), $8,271.32
  (12:05), $8,269.70 (12:15), $8,268.66 (12:27 and 12:38), $8,266.39 (12:49), $8,265.24 (12:59), $8,264.26 (13:10),
  $8,263.02 (13:20), $8,262.01 (13:31), $8,260.63 (13:42), $8,259.69 (13:52) and $8,258.21 (14:01).
- That is $17.92 in 2.9 h, far above the $1,700 / $1,500 floors.
- The campaign's own estimate went from $2,617.44 (10:48) to $2,633.77 (14:01): kimi $1,083.40, glm $546.19, deepseek
  $1,004.18. progress.py estimates $815 still needed (all GLM).

**Quota (watch.py, 14:01):** user 10.08M free files and 9.6 TiB; project 79.64M free files. The ~354 slots left need
~20-30k files, far from the ~4M free-inode mark.

**Polling:**
- 17 polls: 11:08, 11:25, 11:35, 11:47, 11:58, 12:05, 12:14, 12:27, 12:38, 12:49, 12:59, 13:09, 13:19, 13:30, 13:41,
  13:51 and 14:00.
- Each poll read OPERATOR_NOTE and ran `monitor/poll.sh` with `WINDOW_START="2026-11-01 11:08"` under `timeout`, then
  `monitor/glm429.py --minutes 30` and `monitor/inflight.py --model glm`. The sleeps were separate calls.
- watch.py logged one event: `operator_note` (the new sha256) at 11:09.
- PROGRESS.md was regenerated at 11:15, 11:36, 12:04, 12:27, 12:56, 13:20, 13:52 and 14:01, and once more right before
  the agent returned.

## Monitor window 10 (2026-11-01 08:10 - 11:05 PST)

All times are PST. The window lasted ~3 h.
- OPERATOR_NOTE had sha256 8f4ece6d... at every poll (08:10 to 11:01). No new entry came; GLM stayed paused by main.
- No code changed. This section and the status block were written with `monitor/patches/readme_window10.py` (temp
  file + os.replace).
- New files: `monitor/.poll-w10-last.txt` (output of the last `monitor/poll.sh` run) and `monitor/.progress-w10-last.txt`
  (output of the last `progress.py` run).

**Polling:**
- 17 polls: 08:10, 08:20, 08:32, 08:45, 08:56, 09:06, 09:16, 09:26, 09:37, 09:47, 09:57, 10:07, 10:17, 10:29, 10:40,
  10:51 and 11:01.
- Each poll read OPERATOR_NOTE and ran `monitor/poll.sh` with `WINDOW_START="2026-11-01 08:10"` under `timeout`:
  `monitor/health.py --window-min 30`, `monitor/endings.py --since "2026-11-01 08:10"` and `watch.py --once
  --stop-at-floor --partition cpu_dataprocessing,cpu_datamover`. Each poll also ran `monitor/inflight.py`. The sleeps
  were separate calls.
- watch.py logged no event in the window.
- PROGRESS.md was regenerated at 08:35, 08:57, 09:27, 09:57, 10:18 and 10:48, and once more right before the agent
  returned.

**Campaigns:**
- **kimi w2-g1** (job 21142493, cpu_datamover): 81 (08:07) -> 83 (10:48) of 85 slots graded.
  - **5-0** (fastapi-implicit-head-options): graded pass at 08:13.
  - **13-0** attempt-0 (scriggo-method-declarations, 07:22-08:21) ended **needs_review**. The agent killed its own
    OpenCode process:
    - Its last reply said "Let me kill, clean up, and restart both properly." Its bash tool call began with
      `pkill -f "opencode/bin" 2>/dev/null; ...`.
    - At 08:20:26 ATP logged `agent_control_recovered`. In the native result the agent invocation is "failed" and that
      last bash call is "incomplete". opencode_finished is false.
    - The candidate patch was empty (0 bytes). The verifier ran: f2p 0/48, p2p 1049/1049, reward 0.
    - generation-validity-v2-r7.json says failure "unknown". In ATP's `deepswe_campaign.slot_state` that means
      needs_review, not retryable, so ATP will not retry the slot. The historical-exception path does not apply: the
      campaign's frozen source has no `configs/deepswe_historical_recovery.json`.
    - `brokenreply.py`: not broken-ended (the last reply was a tool call), so no brk re-run applies.
    - This makes 4 needs_review slots. It is the first one the agent caused itself; the other 3 are the verifier kills
      that OPERATOR_NOTE 18:15 keeps as they are. Leaving it as ATP rated it (or counting it as a fail, or re-running
      it) is main's decision. The agent did nothing to it.
  - **13-1** attempt-0 (scriggo-method-declarations, 07:27-10:28): ended on ATP's 10,800 s agent timeout (180.9 min,
    545 logical calls), graded fail (f2p 0/48, empty patch). brokenreply.py: not broken-ended.
  - **11-1** (pebble-durability-wait-apis): ATP's retry attempt-1 started at 10:31:14, after the first pass ended. At
    11:01 it was 31 min old with 54 replies, all 200.
- **No leg was submitted:** the 5 campaigns left to start are GLM, paused by main.
- The agent wrote no STOP file, ran no scancel, cleaned no sandbox and changed no key.

**Graded first attempts (PROGRESS.md 08:09 -> 10:48):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | agent timeouts (total) |
|---|---|---|---|---|---|
| kimi | 561 -> 563 | 369 -> 370 | 65.8 -> 65.7 | 66.6 -> 66.7 | 7 -> 8 |
| glm | 206 -> 206 | 146 | 70.9 | 70.9 | 0 |
| deepseek | 564 -> 564 | 250 | 44.3 | 44.3 | 0 |

Total: 1,331 -> 1,333 graded, 765 -> 766 passes. kimi w2-g1: 74 passes of 83 graded (89.2%).

**Agent timeouts (OPERATOR_NOTE 02:15):**
- kimi: 1 of the 3 Kimi slots that ended in the window (13-1 at 10:28). The rule needs at least 3 timeouts and more
  than 20%, so the auto-pause did not trigger.
- glm and deepseek: no slot ended in the window.

**Infra and health:**
- The 30-min infra share read 0.5 (1 of 2) at 08:32 and 1.0 (1 of 1) at 08:45. Both readings come from the single
  13-0 ending above (health.py reason `terminal:unknown`). That was the agent's own `pkill`, not a sandbox, hub or
  verifier error, so it is no sign of an outage and nothing was paused. Every other poll read 0 or had no endings.
- No sandbox creation or upload error, PTY closure, OpenSandbox API error or verifier error in the window.

**Hub:**
- **Kimi:** no non-200 reply (health.py, latency.py). The only error was 13-1's TimeoutError at its 10:28 deadline.
  latency.py (10-min buckets 08:00-11:00, 1-4 rollouts in flight, n 5-103 per bucket): p50 7.6-33.7 s, p90
  18.5-125.1 s.
- **GLM-5.3 (no teacher GLM traffic):** key_budget.py's 4-token GLM-5.3 budget check on NVAPI_KEY_NEW_2 got a 429 on
  1 of 57 checks from 08:11 to 11:02 (08:56:52, "TPM limit=100000000, current usage=102919462"). Window 9 had 48 of
  106. Per 30 min: 08:00 0 of 6, 08:30 1 of 11, 09:00 0 of 10, 09:30 0 of 10, 10:00 0 of 9, 10:30 0 of 10.

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $8,290.60 (08:11), $8,286.92 (08:22), $8,284.69 (08:35), $8,282.35 (08:46), $8,282.07 (08:56), $8,281.76
  (09:07), $8,281.24 (09:17 and 09:27), $8,279.99 (09:38), $8,279.38 (09:48 and 09:57), $8,278.14 (10:08), $8,277.62
  (10:18), $8,277.25 (10:31), $8,277.14 (10:41), $8,276.47 (10:52) and $8,276.13 (11:02).
- That is $14.47 in 2.85 h, far above the $1,700 / $1,500 floors.
- The campaign's own estimate went from $2,597.27 (08:07) to $2,617.44 (10:48): kimi $1,081.62, glm $531.64,
  deepseek $1,004.18. progress.py estimates $880 still needed (almost all GLM).

**Quota (watch.py):** at 11:02, user 10.08M free files and 9.6 TiB; project 79.79M free files. The ~360 slots left
need ~15k files, far from the ~4M free-inode mark.

## Monitor window 9 (2026-11-01 05:15 - 08:10 PST)

All times are PST. The window lasted ~3 h.
- OPERATOR_NOTE had sha256 8f4ece6d... at every poll (05:16 to 08:07). The entries of 05:15 (GLM pause) and "05:40"
  (main's label; the file's mtime is 05:15:13) were there at the start. No new entry came.
- No code changed. This section and the status block were written with `monitor/patches/readme_window9.py` (temp file
  + os.replace).
- New helper `monitor/.watchsum-w9.py`: the poll summary of `monitor/.watch-last.json`. It lists only unfinished or
  running campaigns and shortens the budget error text.

**Polling:**
- Polls at 05:16, 05:33, 05:45, 05:55, 06:05, 06:19, 06:28, 06:39, 06:50, 06:59, 07:09, 07:22, 07:33, 07:43, 07:54
  and 08:06.
- Each poll read OPERATOR_NOTE and ran `monitor/health.py --window-min 30`, `monitor/endings.py --since "2026-11-01
  05:04"` and `watch.py --once --stop-at-floor --partition cpu_dataprocessing,cpu_datamover`, under `timeout`, with the
  sleeps in separate calls. health.py took up to 3.7 min.
- watch.py logged one event set: `campaign_ended` for glm w1-g0/g1/g2 at 05:22 (main's cancel of 05:06).
- PROGRESS.md was regenerated at 05:37, 06:05, 06:30, 06:59, 07:25, 07:54 and 08:07, and once more right before the
  agent returned.

**Campaigns:**
- **deepseek w2-g1 complete** at 06:24:33 (job 21132255, exit 0): 41 pass / 44 fail / 0 infra (48.2%). No rec or brk
  re-runs.
  - Its last slot, 11-1 (pebble-durability-wait-apis), was graded (fail) on attempt-2, ATP's third and last, at
    06:23:34.
  - attempt-2 had the same sandbox symptoms as attempt-1. From 05:31 OpenCode snapshots failed with "signal: killed"
    every ~4.5 min while the agent ran the pebble test suite (its `go test -tags invariants ./...` hit an 11-min
    timeout in sstable/rowblk). This time the sandbox kept answering, so the attempt finished.
  - `monitor/stopcheck.py`: ATP's recovery at the leg end (06:24:19) recovered 1 attempt and reported 0 remaining; 0
    orphan candidates.
- **DeepSeek is complete** (5 of 5 campaigns):
  - 565 slots: 564 graded first attempts, 250 passes (44.3%; w1 42.6, w2 48.2).
  - 1 final infra: deepseek w1-g0 langchain r4 (needs_review).
  - 0 broken-ended slots, 0 agent timeouts. Est. cost $1,004.18.
- **kimi w2-g0 complete** at 07:53:28 (job 21140525, exit 0): 78 / 7 / 0 (91.8%). No rec or brk re-runs.
  - Its one infra ending was 5-3 attempt-0 (fastapi-deprecation-response-headers): verifier "SandboxApiException:
    Failed to run command. Status code: 410" at 00:49 PST.
  - ATP retried the slot after the first pass. attempt-1 (07:30-07:52) was graded (pass).
  - stopcheck: 0 orphan candidates.
- **kimi w2-g1** (job 21142493, cpu_datamover): 30 (05:12) -> 81 (08:07) of 85 graded. 3 rollouts in flight (5-0, 13-0,
  13-1, 40-48 min old). Slot 11-1 (pebble-durability-wait-apis) waits for ATP's retry:
  - attempt-0 ran 05:13-06:12. At 05:55:32 the agent's PTY closed ("PTY connection closed unexpectedly (close code
    1011)", SandboxPtyError).
  - Snapshot and quiescence commands then timed out on that sandbox (240 s / 120 s).
  - The verifier got "SandboxApiException: Failed to run command. Status code: 410" at 06:12:10. ATP rated the attempt
    infra and retries the slot.
  - Pebble is also the task whose sandboxes failed in deepseek w2-g1 (windows 8-9). These are single-sandbox failures:
    no other campaign logged OpenSandbox errors, so this is not an outage.
- **No leg was submitted:** no pending campaign may start. The 5 left are GLM, paused by main, and every other
  campaign is done or running.
- The agent wrote no STOP file, ran no scancel, cleaned no sandbox and changed no key.
- **Campaigns at the end of the window (08:07):**

| campaign | graded / slots | state |
|---|---|---|
| kimi w2-g1 | 81 / 85 | running, 3 in flight + slot 11-1 to retry |
| glm w1-g0 | 33 / 135 | paused by main |
| glm w1-g1 | 77 / 130 | paused by main |
| glm w1-g2 | 96 / 130 | paused by main |
| glm w2-g0 / g1 | 0 / 85 each | paused by main (not started) |
| the other 9 | complete | `results/*.json` |

**Graded first attempts (PROGRESS.md 05:12 -> 08:07):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | slots terminal |
|---|---|---|---|---|---|
| kimi | 484 -> 561 (395 w1 + 166 w2) | 298 -> 369 | 65.8 (w1 55.2, w2 91.0) | 66.6 (7 timeouts, all before window 5) | 561 / 565 |
| glm | 206 -> 206 (w1) | 146 -> 146 | 70.9 | 70.9 | 206 / 565 (paused) |
| deepseek | 563 -> 564 (394 w1 + 170 w2) | 250 -> 250 | 44.3 (w1 42.6, w2 48.2) | same | 565 / 565 (done) |

Total: 1,253 -> 1,331 graded, 694 -> 765 passes. Broken-ended slots: still 1 (glm w1-g0 1-4; its re-run waits for
the end of its paused campaign).

**Endings in the window (monitor/endings.py, 05:04 -> 08:07):** kimi 77 graded + 1 infra; glm none; deepseek 1 graded.

**Agent timeouts (OPERATOR_NOTE 02:15):**
- 0 this window for every model: kimi 0 of 78 endings, glm 0 of 0, deepseek 0 of 1.
- The Kimi auto-pause rule did not trigger.
- The oldest Kimi rollout seen in flight was kimi w2-g0 9-1, at 83 min (07:23).

**Infra endings (1; not an outage):**
- The only one is kimi w2-g1 11-1 (above).
- The overall 30-min infra share was 0-6.7% at every poll. The per-campaign readings above 30% were single endings:
  1 / 1 for deepseek w2-g1 at 05:16 (its attempt-1 ending of 04:48, window 8). kimi w2-g1 read 1 / 10 to 1 / 11
  from 06:19 to 06:39.

**Hub:**
- **Kimi:** no non-200 reply in the window (health.py). One upstream stream failure (kimi w2-g1, ~07:58) was retried
  with a 200.
  - ATP's cut of kimi w2-g1 to 4 rollouts at 04:59 (window 8) was lifted by 05:33. ATP cut it to 4 again at 07:58:33
    (`rate-limits/admission.json`, last_pressure upstream_error); only 3 of its rollouts were in flight by then.
  - Latency (`monitor/latency.py`, 10-min buckets 05:10-08:00): p50 6.7-13.6 s, p90 21.6-43.1 s.
- **DeepSeek:** 200 replies only, until its last rollout ended (06:23).
- **GLM:**
  - No teacher GLM call after main's cancel (05:06).
  - key_budget.py's budget check, a 4-token GLM-5.3 call on NVAPI_KEY_NEW_2, got a 429 on 48 of 106 checks from 05:08
    to 08:02. Of those, 46 were "Too Many Requests". 2 were "Model rate limit exceeded. TPM limit=100000000, current
    usage=101294381 ..." (e.g. 06:29).
  - 429s per 30 min: 05:00 7 of 15, 05:30 14 of 22, 06:00 5 of 21, 06:30 8 of 16, 07:00 9 of 16, 07:30 4 of 15.
  - So the GLM-5.3 saturation went on without our traffic.
  - Every 429 reply still carried the spend, so the driver guards and watch.py kept reading the budget.

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $8,545.73 (05:22), $8,514.80 (05:37 and 05:46), $8,483.43 (05:56), $8,458.78 (06:09), $8,431.81 (06:19
  and 06:29), $8,412.57 (06:40), $8,391.20 (06:51), $8,350.99 (07:00), $8,337.97 (07:12), $8,321.43 (07:24), $8,306.92
  (07:34), $8,301.23 (07:44) and $8,293.05 (08:02). progress.py read $8,290.60 at 08:07.
- That is $255 in 2.75 h (~$93/h), r6's judge included. Far above the $1,700 / $1,500 floors.
- The campaign's own estimate went from $2,440.24 (05:05) to $2,597.27 (08:07): kimi $1,061.45, glm $531.64, deepseek
  $1,004.18. So the campaign spent ~$157 (~$57/h) of the key's ~$255. The rest went to the key's other users, which
  include r6's judge.
- progress.py estimates $883 still needed (almost all GLM), against ~$6,790 usable above the $1,500 guard floor.

**Quota:**
- At 08:02: user 10.08M free files and 9.60 TiB; project 79.83M free files. At 05:22: 10.64M, 9.66 TiB and 80.51M.
- The rest of the run needs ~363 more slots (GLM 359, Kimi 4), ~15k files at ~42 per attempt. That is far from the
  ~4M free-inode mark.

## Monitor window 8 (2026-11-01 02:12 - 05:15 PST)

All times are PST. The window lasted ~3 h.
- OPERATOR_NOTE was unchanged (sha256 6c8ce7e4...) until 05:06:05. Then main added the 05:15 PST entry (GLM pause;
  sha256 523387a1...). See "GLM paused by main" below.
- Code change: progress.py now reports the pause (`monitor/patches/progress_glm_pause_20261101.py`, temp file + mv;
  pre-change copy and sha256 in `provenance/pre-progress-glm-pause-20261101/`; progress.py is not run by any job).
- This section was written with `monitor/patches/readme_window8.py` (temp file + os.replace).

**Polling:**
- Polls at 02:12, 02:23, 02:37, 02:49, 03:00, 03:11, 03:21, 03:32, 03:41, 03:51, 04:04, 04:20, 04:32, 04:41, 04:51,
  05:02 and 05:08. The 05:08 check found the new OPERATOR_NOTE entry, main's STOP files and the cancelled GLM legs.
- Each poll read OPERATOR_NOTE and ran `monitor/health.py --window-min 30`, `monitor/endings.py --since "2026-11-01
  02:05"` and `watch.py --once --stop-at-floor --partition cpu_dataprocessing,cpu_datamover`, each as its own call under
  `timeout`, with the sleeps in separate calls. No call hit the tool limit; health.py took 16 s to ~3 min.
- watch.py logged one event: `rate_limited` at 05:04:48 (557 replies since its previous check: 474 200s and 83 429s,
  all GLM).
- PROGRESS.md was regenerated at 02:40, 03:11, 03:41, 04:11, 04:41, 05:05 and 05:12. The 05:12 run is the final one,
  with the pause patch; it replaced the 05:05 row (progress.py's minimum gap between rows).

**Campaigns:**
- **kimi w1-g2 complete** at 02:25:44 (job 21128612, leg 2, exit 0): 87 pass / 43 fail / 0 infra, pass@1 66.9%. No
  rec or brk re-runs.
- **kimi w2-g1 started** (wave-2 gating: 2 Kimi wave-1 campaigns finished, 1 Kimi wave-2 campaign started):
  - Partition probe (`sbatch --test-only`, 2 CPU / 16G / 23 h, 02:26, both accounts): cpu_interactive at once,
    cpu_dataprocessing 03:54, cpu_datamover 06:47.
  - As before, `TDW_PARTITION=cpu_dataprocessing,cpu_datamover bash submit.sh kimi w2 1` (02:27:54), so cpu_interactive
    stays free for r6's p_cpu_short jobs.
  - Winner 21142493 (nemotron_agents_dev, cpu_datamover, started 02:28:08 despite the probe's 06:47). The loser
    21142494 ended at 02:28:10 ("sibling candidate won").
  - Start line (legs.jsonl): NVAPI_KEY_NEW_2 with $8,886.00 left, cap 8, stock prompt, context 262144,
    nvidia/moonshotai/kimi-k3, wave 2, 17 tasks x 5.
- **kimi w1-g1 complete** at 03:00:32 (job 21128610, leg 2, exit 0): 61 / 69 / 0, pass@1 46.9%. No rec or brk re-runs.
  **Kimi wave 1 is done:** 218 of 395 passed (55.2%; 56.2% without its 7 agent-timeout slots, all from before window 5).
- **deepseek w2-g1, last slot 11-1 (pebble-durability-wait-apis):**
  - attempt-0 (10-31 23:13) ended without a grade: the verifier's OpenSandbox "command run (background submit)" timed
    out after 360 s.
  - attempt-1 (03:04-04:48) ended without a grade: "RuntimeError: Cannot finalize a candidate before the agent process
    has stopped" (SandboxPtyError).
    - From ~04:23 the sandbox (d3452148) stopped answering OpenSandbox "command status" polls (10 s timeouts).
    - OpenCode snapshots failed with "signal: killed".
    - The agent had been running the whole pebble test suite (`go test -tags invariants ./...`).
  - attempt-2 started at 04:50:49. The campaign manifest has `max_attempts` 3, so if it fails too the slot becomes
    final infra.
  - The errors came from this one sandbox; no other campaign logged OpenSandbox errors. Not an outage.
- The agent wrote no STOP file, ran no scancel, cleaned no sandbox and changed no key. The STOP files and the GLM
  scancel were main's (below).
- **Campaigns at the end of the window (05:12):**

| campaign | graded / slots | state |
|---|---|---|
| kimi w2-g0 | 59 / 85 | running, 8 in flight |
| kimi w2-g1 | 30 / 85 | running, 4 in flight |
| deepseek w2-g1 | 84 / 85 | running, 1 in flight (11-1 attempt-2) |
| glm w1-g0 | 33 / 135 | paused by main (leg cancelled 05:06:07) |
| glm w1-g1 | 77 / 130 | paused by main (leg cancelled 05:06:07) |
| glm w1-g2 | 96 / 130 | paused by main (leg cancelled 05:06:07) |
| glm w2-g0 / g1 | 0 / 85 each | paused by main (not started) |

**GLM paused by main (OPERATOR_NOTE 2026-11-01 05:15 PST):**
- The entry: the r6 rubric judge (GLM-5.3 on NVAPI_KEY_NEW_2) got HTTP 429 on 53 of its last ~140 calls (04:53-05:04).
  Main wrote STOP files for every glm-* campaign (05:06:06) and cancelled the running tdw-glm-* legs. GLM campaigns are
  not to be resubmitted until a later entry. Kimi and DeepSeek continue, and the cancelled GLM slots stay retryable.
- sacct: 21128614 (glm w1-g0), 21128619 (w1-g1) and 21128621 (w1-g2) were CANCELLED at 05:06:07 after 11 h 36 min
  (leg 2 each).
- **ATP's shutdown cleanup did not run** in these legs: their launch dirs have no cleanup.json, process.json or
  recovery-1.json. `monitor/stopcheck.py` lists 11 attempts with neither an end marker nor a cleanup confirmation
  (orphan-sandbox candidates), all in the agent phase (no verify-request.json):
  - glm w1-g0 0-4 and 2-0;
  - glm w1-g1 16-0, 23-4, 4-3 and 8-3;
  - glm w1-g2 12-4, 15-3, 24-4, 4-3 and 7-3.
- They started 03:11-04:12. The vendored agent config (`opencode_sandboxed_agent.yaml`) creates agent sandboxes with
  `ttl_s` 18,000 s, so these should expire by ~08:11-09:12. Their model endpoint, the leg's local policy server, ended
  with the leg.
- **No cleanup was done.** ATP has no standalone cleanup command. Its `recover_incomplete()` runs at the start of a
  campaign's next leg, and that is blocked by the pause. Cleaning earlier would need a new tool that calls ATP's
  recovery outside a leg, on the cell r6 shares; that is main's decision.
- No GLM leg was submitted after the entry. watch.py has no auto legs (`--auto-leg-cap 0`), and submit.sh refuses a
  campaign with a STOP file.
- Fact for the 429 question: from 04:30 our GLM traffic was 56-79 200 replies per 15 min (371-780 before), yet the
  hub kept answering "TPM limit=100000000, current usage=102-130M" (see "Hub" below).

**Graded first attempts (PROGRESS.md 02:10 -> 05:12):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | in flight at 05:05 (before the GLM cancel) |
|---|---|---|---|---|---|
| kimi | 401 -> 484 (395 w1 + 89 w2) | 224 -> 298 | 61.6 (w1 55.2, w2 89.9) | 62.5 (7 timeouts, all before window 5) | 12 |
| glm | 178 -> 206 (w1) | 126 -> 146 | 70.9 | 70.9 | 11 |
| deepseek | 553 -> 563 (394 w1 + 169 w2) | 247 -> 250 | 44.4 (w1 42.6, w2 48.5) | same | 1 |

Total: 1,132 -> 1,253 graded, 597 -> 694 passes. Broken-ended slots: still 1 (glm w1-g0 1-4, re-run planned at
campaign end).

**Endings in the window (monitor/endings.py, 02:05 -> 05:03):** kimi 88 graded; glm 29 graded + 2 infra; deepseek
11 graded + 1 infra.

**Agent timeouts (OPERATOR_NOTE 02:15):** 0 this window for every model (kimi 0 of 88 endings, glm 0 of 31, deepseek 0
of 12). The Kimi auto-pause rule did not trigger. When main cancelled the GLM legs (05:06), the oldest GLM rollout in
flight (glm w1-g2 7-3, 114 min) still had ~66 min to ATP's 10,800 s deadline, so no GLM rollout reached it.

**Infra endings (3; none was an outage):**
- glm w1-g0 11-1 (02:37:42) and glm w1-g2 24-2 (03:16:09): one GLM request got LiteLLM's 360 s 408 seven times in a
  row (`retry_exhausted`); ATP rated the attempt infra and retries the slot. The per-campaign 30-min readings of 1 / 3
  and 1 / 1 (glm w1-g0, 02:37-02:49) and 1 / 3 (glm w1-g2, 03:41) are these single hub-408 endings, not OpenSandbox
  errors (as in window 4).
- deepseek w2-g1 11-1 attempt-1 (04:48:31), above.
- The overall 30-min infra share was 0-6.2% at every poll.

**Hub:**
- **Kimi:** no non-200 reply in the last hour of the window.
  - Two upstream stream failures, both in kimi w2-g1 and both retried: 2-0 at 02:57:07 and 14-4 at 04:59:35.
  - ATP cut kimi w2-g1 to 4 rollouts after each one. It was back at 8 by ~03:53, and at 4 again from 04:59.
  - Latency (02:00-02:50, `monitor/latency.py`): p50 8-17 s, p90 23-37 s.
- **DeepSeek:** no non-200 reply in the last hour of the window.
- **GLM-5.3:**
  - 02:12-04:15: 0-9 LiteLLM 360 s 408s per 30 min. ATP held each GLM campaign at 1-8 rollouts; GLM in flight
    averaged 10.5-12.5 (02:00-03:00) and 7-7.8 (03:00-04:00). Latency (02:00-02:50) p50 3-10 s, p90 24-60 s.
  - **From ~04:15 the hub's GLM-5.3 capacity is saturated.** GLM 429s per 30-min health window: 1 (04:20), 16 (04:32),
    26 (04:41), 38 (04:51), 76 (05:02). In the hour to ~05:06 there were about 140 429s and 10 408s:
    - ~91 "Too Many Requests" from the upstream (LiteLLM model group nvidia/zai-org/glm-5.3, no fallback);
    - ~31 "Model rate limit exceeded. TPM limit=100000000, current usage=102-130M";
    - ~18 "Model capacity reached for nvidia/zai-org/glm-5.3. Priority: prod ... Model TPM: 100000000 ... Remaining: 0".
  - The `x-ratelimit-model_saturation_check-remaining-tokens` header (limit 100M) read 7.7-14.5M at 02:00-04:15 and
    0.5-1.7M from 04:15 (15-min minimums). GLM 200 replies fell from 371-780 per 15 min (02:00-04:15) to 56-79
    (04:30-05:00).
  - ATP cut every GLM campaign to 1 rollout / 1 API call. The 11 GLM rollouts in flight wait at ATP's API gate (47
    waits over 60 s and 12,823 s of gate wait in the 30 min to 05:02). No GLM attempt ended after 04:20:32.
  - Our GLM traffic fell to a few calls per minute while the saturation continued, so the load is not ours. r6's
    rubric judge uses the same model.
  - key_budget.py's budget check is itself a 4-token GLM call. From 04:21 some checks got a 429, but the reply still
    carried the spend, so the driver guards keep working (`remaining` 8,589.25 at 05:04).

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $8,902.61 (02:12), $8,886.00 (02:13), $8,868.20 (02:39), $8,830.35 (02:51), $8,807.90 (03:01), $8,784.58
  (03:11), $8,769.75 (03:23), $8,743.70 (03:34), $8,711.58 (03:55), $8,693.70 (04:05), $8,672.86 (04:23), $8,654.99
  (04:33), $8,635.32 (04:42), $8,614.06 (04:54) and $8,589.25 (05:04). progress.py read $8,563.80 at 05:12.
- That is $313 in 2.87 h (~$109/h), r6's judge included if it uses NEW_2. Far above the $1,700 / $1,500 floors.
- The campaign's own estimate went from $2,161.23 (02:10) to $2,440.24 (05:05): kimi $907.39, glm $531.64, deepseek
  $1,001.21.
- progress.py estimates $1,022 still needed, against ~$7,089 usable above the $1,500 guard floor.

**Quota:**
- At 05:04: user 10.73M free files and 9.67 TiB; project 80.63M free files.
- The user's free space fell from 11.29 TiB (04:42) to 9.69 TiB (04:54), about 1.6 TiB in 12 min. That was not this
  campaign (an attempt directory is up to ~0.4 GB).
- The user's free files fell from 11.08M (02:13) to 10.73M (05:04). This campaign writes ~42 files per attempt (131
  attempt endings this window).
- The rest of the run needs ~500 more slots (~21k files), so it stays far from the ~4M free-inode mark.

## Monitor window 7 (2026-11-01 00:15 PDT - 02:10 PST)

Daylight saving time ended during this window: at 02:00 PDT the clocks went back to 01:00 PST. The window lasted
2 h 55 min.
- progress.py now prints each history row's time with its zone. Without it the row times "11-01 01:xx" would repeat.
  - Patch: `monitor/patches/progress_tz_20261101.py` (2 exact-text replacements, temp file + mv).
  - Pre-change copy and sha256: `provenance/pre-progress-tz-20261101/`.
  - progress.py is not run by any job.
- driver.py's `key_of_time()` compares `%z`-aware times, so `KEY_PERIODS` and the leg-start key check are not affected.
- endings.py's `--since "2026-11-01 00:10"` is a PDT time (unambiguous).

**Polling:**
- Polls about every 10 min:
  - PDT: 00:15, 00:26, 00:40, 00:50, 01:00, 01:10, 01:17, 01:42-01:50 and 01:58.
  - PST: 01:08, 01:11, 01:22, 01:31, 01:40, 01:52 and 02:03.
- Endings were counted from `--since "2026-11-01 00:10"` (PDT).
- The 01:17 PDT `monitor/poll.sh` ran in one tool call together with its sleep and hit the agent's 10-min tool limit.
  - health.py took ~7 min on slow Lustre (01:17-01:24 PDT).
  - The poll's watch.py kept running and finished at ~01:34 PDT (output stamped 01:25:51). watch.py writes its state
    atomically, so nothing was lost.
  - After that, health.py, endings.py and watch.py (`--once --stop-at-floor --partition
    cpu_dataprocessing,cpu_datamover`) ran as separate calls.
- PROGRESS.md was regenerated at 00:34, 01:00, 01:35 and 01:59 PDT, and at 01:10, 01:40, 02:06 and 02:10 PST. The
  02:10 run, right before the agent returned, replaced the 02:06 row (progress.py's minimum gap between rows). The
  01:35 PDT run came 35 min after the one before it, because of the stuck poll.
- OPERATOR_NOTE did not change after its 18:15 entry (sha256 6c8ce7e4...).
- New read-only tools:
  - `monitor/non200.py`: a model's non-200 hub replies in the last N minutes, with the hub's error text (no payloads or
    headers).
  - `monitor/watchsum.py`: poll.sh's summary of `monitor/.watch-last.json`, as a script.

**Campaigns:**
- **kimi w1-g0 complete** at 01:17:45 PDT (job 21128606, leg 2, exit 0):
  - 70 pass / 65 fail / 0 infra, pass@1 51.9%.
  - No rec or brk re-runs.
  - 4 of its graded slots ended on the agent timeout, all before window 5.
- **deepseek w2-g0 complete** at 01:24:31 PST (job 21130706, leg 1, exit 0): 41 / 44 / 0, pass@1 48.2%. No rec or brk
  re-runs.
- **kimi w2-g0 started.** The wave-2 gating allowed it: 1 kimi wave-1 campaign had finished and no kimi wave-2 campaign
  had started.
  - Partition probe (`sbatch --test-only`, 2 CPU / 16G / 23 h, 01:34 PDT, both accounts): cpu_interactive at once,
    cpu_dataprocessing 02:02 PDT, cpu_datamover 04:55 PDT.
  - As before, `TDW_PARTITION=cpu_dataprocessing,cpu_datamover`, so cpu_interactive stays free for r6's p_cpu_short
    jobs.
  - `bash submit.sh kimi w2 0` ran at 01:35:07 PDT. Winner 21140525 (nemotron_rl_algo, cpu_dataprocessing, started
    01:35:31 PDT). The loser 21140524 ended at 01:35:36.
  - Start line (legs.jsonl): NVAPI_KEY_NEW_2 with $9,103.49 left, cap 8, stock prompt, context 262144,
    nvidia/moonshotai/kimi-k3, wave 2.
- **Running at the end of the window (7):**

| campaign | graded / slots |
|---|---|
| kimi w1-g1 | 124 / 130 |
| kimi w1-g2 | 128 / 130 |
| kimi w2-g0 | 11 / 85 |
| glm w1-g0 | 26 / 135 |
| glm w1-g1 | 71 / 130 |
| glm w1-g2 | 80 / 130 |
| deepseek w2-g1 | 74 / 85 |

- Nothing else could start. kimi w2-g1 waits for a second kimi wave-1 campaign to finish, and glm w2-g0/g1 wait for a
  GLM wave-1 campaign.
- No STOP file was written and no leg ended abnormally, so no sandbox cleanup was needed. watch.py logged no event.

**Graded first attempts (PROGRESS.md 00:13 PDT -> 02:06 PST):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | in flight at 02:06 PST |
|---|---|---|---|---|---|
| kimi | 323 -> 398 (387 w1 + 11 w2) | 177 -> 223 | 56.0 (w1 55.0, w2 90.9 on 11 slots) | 57.0 (7 timeouts, all before window 5) | 14 |
| glm | 152 -> 177 (w1) | 109 -> 125 | 70.6 | 70.6 | 10 |
| deepseek | 507 -> 553 (394 w1 + 159 w2) | 228 -> 247 | 44.7 (w1 42.6, w2 49.7) | same | 5 |

Total: 982 -> 1,128 graded, 514 -> 595 passes. Broken-ended slots: still 1 (glm w1-g0 1-4, re-run planned at campaign
end).

**Endings in the window (monitor/endings.py, 00:10 PDT -> 02:05 PST):**
- kimi: 76 graded + 1 infra.
- glm: 25 graded.
- deepseek: 46 graded.

**Agent timeouts (OPERATOR_NOTE 02:15):**
- 0 this window for every model: kimi 0 of 77 endings, glm 0 of 25, deepseek 0 of 46.
- The Kimi auto-pause rule did not trigger.
- At 02:05 PST the oldest Kimi rollout in flight was kimi w1-g1 16-3 at 84 min, so its deadline is ~03:41 PST.

**Infra ending (1):**
- **kimi w2-g0 5-3 attempt-0 (01:49:47 PDT):**
  - OpenSandbox reported its sandbox (9079a382) as ended by an OOM kill: the container exited with code 137,
    "SANDBOX::ENDED (OOMKilled)".
  - The PTY was lost (SandboxPtyError), and the verifier's commands got HTTP 410.
  - ATP retries the slot (awaiting at 02:06 PST).
- This was not an outage pause:
  - No other campaign had an infra ending in the window.
  - The overall 30-min infra share was 0-4.2% at every poll.
  - kimi w2-g0's own 30-min share was 1 of 4 (25%) at 01:58 PDT and 1 of 6 (16.7%) at 01:11 PST, both under 30%.

**Hub (journals, health.py's per-minute buckets, 00:15 PDT -> 02:04 PST):**
- **Kimi:**
  - 6,306 replies with status 200.
  - 5 429s, all "Reached overall message rate limit. Please try again later. Your limit will reset in 0 seconds"
    (Moonshot's upstream limit, passed on by LiteLLM). 3 came at 01:42:00-01:42:01 PDT, when kimi w2-g0 started its
    first 8 rollouts, and 2 at 01:18:22-01:18:24 PST. Each was retried.
  - 3 upstream stream failures, each retried: 01:14:54 PDT (w1-g1 11-2, "Response payload is not completed"),
    01:18:05 PDT (w1-g2 25-0) and 02:08:29 PST (w1-g1 16-4), the last two "Upstream SSE contained an error or
    non-object payload".
  - ATP's admission state (`rate-limits/admission.json`):
    - kimi w2-g0 ran at rollout limit 4 from the 429s of 01:42 PDT (again 01:18 PST) and was back at 8 at 02:05:07 PST.
    - kimi w1-g1 was at 4 after the 429s. The stream failure of 02:08:29 PST kept it at 4.
    - kimi w1-g2 was at 4 (the upstream_error of 00:08 PDT) until ~00:50 PDT. It was cut to 4 again at 01:18:05 PDT
      and was back at 8 at 01:06:47 PST.
- **GLM-5.3:**
  - 4,285 replies with status 200.
  - 18 408s (LiteLLM 360 s timeouts). 5 of them were on glm w1-g0 11-3 (00:33-01:07 PDT) and 5 on glm w1-g2 12-2
    (01:21-01:54 PST).
  - 2 429s ("Model capacity reached ... Priority: prod ... TPM 100M"), at 00:29 and 00:46 PDT.
  - GLM rollouts in flight rose from 4-7 (00:15-01:58 PDT) to 10-14 (01:11-02:03 PST).
  - glm w1-g1 reached rollout limit 8 by 01:11 PST and was at 2 from the 408 of 01:39:02 PST. glm w1-g0 and w1-g2
    were at 4 at the end of the window (their last upstream_timeout decreases: 02:09:11 and 01:54:32 PST).
- **DeepSeek:**
  - 7,494 replies with status 200.
  - 2 503s ("upstream connect error ... connection termination") on deepseek w2-g1 4-3, at 01:11 and 01:43 PDT. Each
    was retried.
  - ATP halved deepseek w2-g1 after each one. It was back at 5 at 01:29:25 PST.

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $9,259.40 (00:16 PDT), $9,237.32 (00:26), $9,197.62 (00:40), $9,178.70 (00:50), $9,162.22 (01:01),
  $9,141.60 (01:15), $9,121.90 (01:25), $9,069.27 (01:51 PDT), $9,027.29 (01:09 PST), $9,000.83 (01:23), $8,964.23
  (01:42), $8,940.00 (01:55) and $8,920.14 (02:06 PST). progress.py read $8,902.61 at 02:06 PST.
- That is $339 in 2.83 h (~$120/h), r6's judge included if it uses NEW_2.
- The campaign's own estimate went from $1,833.95 (00:13 PDT) to $2,151.46 (02:06 PST): kimi $738.15, glm $443.65,
  deepseek $969.66.
- progress.py estimates $1,249 still needed, against ~$7,400 usable above the $1,500 guard floor.

**Quota at 02:06 PST:** user 11.08M free files and 11.38 TiB; project 81.40M free files.

## Monitor window 6 (2026-10-31 21:12 - 11-01 00:15 PDT)

**Polling:**
- 17 polls, about every 10 min: 21:12, 21:23, 21:34, 21:44, 21:54, 22:13, 22:25, 22:38, 22:48, 22:58, 23:08, 23:20,
  23:30, 23:41, 23:50, 00:01 and 00:09. Endings were counted from `WINDOW_START="2026-10-31 21:12"`.
- Lustre was slow around 22:00. The 21:54 `monitor/poll.sh` took 9 min (endings.py ~5 min, watch.py ~4 min), and
  progress.py took 2 min 18 s at 22:03. From 22:13 the poll's three parts ran as separate calls, with the same arguments
  (health.py; endings.py; watch.py `--once --stop-at-floor --partition cpu_dataprocessing,cpu_datamover`), so that no
  call ran into the agent's 10-min tool limit.
- PROGRESS.md was regenerated at 21:36, 22:03, 22:32, 22:59, 23:28, 23:57 and 00:11, and once more right before the
  agent returned.
- OPERATOR_NOTE did not change after its 18:15 entry (sha256 6c8ce7e4...).
- This section was written with `monitor/patches/readme_window6.py` (temp file + mv). No code changed.

**Campaigns:**
- The same 8 legs ran all window: kimi w1-g0..g2 (21128606, 21128610, 21128612), glm w1-g0..g2 (21128614, 21128619,
  21128621), deepseek w2-g0 (21130706) and w2-g1 (21132255).
- No leg ended and no campaign finished, so nothing was submitted. No STOP file was written and no sandbox cleanup was
  needed. watch.py logged no event.
- Kimi and GLM wave 2 still wait for their models' wave-1 campaigns. At 00:11, kimi w1-g0 had 129 of its 135 slots
  graded and the other 6 in flight. progress.py's ETAs: kimi w1 11-01 01:54, deepseek w2 11-01 01:19.

**Graded first attempts (PROGRESS.md 21:11 -> 00:11):**

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | in flight at 00:11 |
|---|---|---|---|---|---|
| kimi | 191 -> 321 (w1) | 100 -> 176 | 54.8 | 56.1 (7 timeouts, all before window 5) | 22 |
| glm | 140 -> 152 (w1) | 99 -> 109 | 71.7 | 71.7 | 4 |
| deepseek | 427 -> 507 (394 w1 + 113 w2) | 190 -> 228 | 45.0 (w1 42.6, w2 53.1) | same | 10 |

Total: 758 -> 980 graded, 389 -> 513 passes. Broken-ended slots: still 1 (glm w1-g0 1-4, re-run planned at campaign
end).

**Endings in the window (monitor/endings.py, 21:12 -> 00:10):**
- kimi: 130 graded.
- glm: 12 graded + 1 infra.
- deepseek: 80 graded + 1 infra.

**Agent timeouts (OPERATOR_NOTE 02:15):**
- 0 this window for every model: kimi 0 of 130 endings, glm 0 of 13, deepseek 0 of 81.
- The Kimi auto-pause rule did not trigger.
- The Kimi rollouts that were oldest at 21:13 (up to 96 min) all ended before their deadlines. At 00:10 the oldest Kimi
  rollout in flight was kimi w1-g0 5-1 at 55 min, so its deadline is ~02:15.

**Infra endings (2), each a single event:**
- **glm w1-g0 11-2 attempt-0 (21:29):** one call got LiteLLM's 408 on all 7 HTTP attempts (`retry_exhausted`).
  OpenCode exited 1 and ATP rated the attempt infra. ATP retries the slot.
- **deepseek w2-g1 11-1 attempt-0 (23:13:44):**
  - Between 22:48 and 22:58 (worker.log, between two health scans): the agent's PTY connection closed (close code
    1011), and the OpenCode export timed out (240 s background submit) on sandbox ce5cd8c9.
  - The verification command on the same sandbox then timed out after 360 s.
  - ATP stopped the sandbox (`agent-sandbox-stopped.json`, 23:13:44) and retries the slot.
  - The campaign's 30-min infra share was 1 of 4 (25%, under 30%). No other OpenSandbox error appeared in any worker
    log, so there was no outage pause.
- The overall infra share per 30 min was 0-2.9% at every poll.

**Hub and ATP admission (journals, 21:12 -> ~23:52):**
- **Kimi:**
  - 9,082 replies, all 200.
  - 3 upstream SSE stream failures ("Upstream SSE contained an error or non-object payload"), each retried with success:
    21:25:53 (w1-g2 4-1), 21:26:06 (w1-g1 8-4) and 22:55:05 (w1-g1 20-0).
  - After each one ATP halved the campaign's rollout limit to 4. w1-g2 was back at 8 at 22:18:56. w1-g1 was back at 8
    by 22:13, cut to 4 again at 22:55:05 and back at 8 at 23:59:07.
  - A further upstream_error at 00:08:13 (ATP's admission state; after the journal scan) cut w1-g2 to 4 again.
- **DeepSeek:** 10,344 replies, all 200. Both wave-2 campaigns ran at 5 in flight all window.
- **GLM-5.3:**
  - 1,436 replies with status 200.
  - 21 429s: 17 "Model capacity reached for nvidia/zai-org/glm-5.3. Priority: prod ... Model TPM: 100000000" and 4
    "TPM limit=100000000".
  - 9 408s (LiteLLM timeouts).
  - Per clock half-hour (21:12-21:30, 21:30-22:00, ..., 23:30-23:52): 200s 147 / 387 / 241 / 255 / 265 / 141;
    429s 3 / 7 / 6 / 3 / 1 / 1; 408s 3 / 0 / 0 / 1 / 2 / 3.
- **GLM admission:**
  - ATP kept glm w1-g0 and w1-g2 at rollout limit 1 all window, with API limit 1-2.
  - glm w1-g1 grew to 2 rollouts between 23:32 and 23:41; its previous 429 / 408 had been at 22:43:29. It stayed at 2
    after the 429 of 00:01:32.
  - Each 429 / 408 starts a new admission epoch (`deepswe_admission.note_health` -> `_rollout_epoch`). The rollout
    limit doubles only after a rollout admitted in the current epoch completes (`note_rollout_completed`).
  - The median GLM rollout is 37.8 min (endings 18:30-22:20, n = 14; p25 31.1, p75 49.2, max 86.1). The GLM campaigns
    got 30 429 / 408 replies in 2.7 h, about 2 per campaign per 30 min. So the limit rarely grows.
  - Result: 13 GLM endings in 3 h (~4.4/h).
- **Budget check under 429:** watch.py's budget check at 00:01 got a 429 from GLM-5.3 ("Model capacity reached").
  LiteLLM still returned the spend headers, so the reading ($9,282.82) was valid.

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $9,628.47 (21:12), $9,582.23 (21:35), $9,520.63 (22:16), $9,478.28 (22:28), $9,435.00 (22:49), $9,412.44
  (22:59), $9,373.14 (23:21), $9,327.11 (23:41), $9,282.82 (00:01) and $9,259.40 (00:11).
- That is $369 in 2.97 h (~$124/h), r6's judge included if it uses NEW_2.
- The campaign's own estimate went from $1,441.16 (21:11) to $1,830.55 (00:11): kimi $587.35, glm $375.31, deepseek
  $867.89. progress.py estimates $1,486 still needed, against ~$7,759 usable above the $1,500 guard floor.

**Quota at 00:11:** user 11.09M free files and 11.45 TiB; project 81.57M free files.

## Monitor window 5 (2026-10-31 18:15 - 21:15 PDT)

**Polling:**
- Polls every ~10 min with `monitor/poll.sh` (`WINDOW_START="2026-10-31 18:15"`): 18:35, 18:45, 18:55, 19:08, 19:18,
  19:28, 19:42, 19:54, 20:05, 20:16, 20:27, 20:38, 20:49, 20:59 and 21:08.
- The ~19:38 poll was cut off by the agent's 10-min tool limit (sleep + poll in one call) and rerun at 19:42.
  watch.py's state is written atomically, so nothing was lost.
- PROGRESS.md was regenerated at 18:36, 19:04, 19:31, 20:01, 20:31, 20:59 and 21:09, and once more right before the
  agent returned.
- OPERATOR_NOTE did not change after its 18:15 entry (sha256 6c8ce7e4...).

**Campaigns:**
- **DeepSeek wave 1 complete** (driver exit 0; no rec or brk re-runs; legs NEW_3 then NEW_2):

| campaign | finished | pass / fail / infra | pass@1 |
|---|---|---|---|
| deepseek w1-g2 | 19:27:59 (job 21128629) | 53 / 77 / 0 | 40.8 |
| deepseek w1-g1 | 20:11:47 (job 21128627) | 63 / 67 / 0 | 48.5 |
| deepseek w1-g0 | 20:22:55 (job 21128624) | 52 / 82 / 1 | 38.8 |

  The 1 infra slot is langchain-request-coalescing deepseek-r4: ATP's needs_review, "Verifier did not produce
  /logs/verifier/reward.json". It is one of the 3 needs_review slots that OPERATOR_NOTE 18:15 keeps as they are.
- **DeepSeek wave 2:**
  - w2-g0: submitted 19:29:07 after the partition probe (cpu_dataprocessing at once). Winner 21130706 (agents_dev,
    cpu_dataprocessing), loser 21130707. Start line: NEW_2 $9,902.18 left, cap 5. settings.json has the deepseek
    spec + context 262144, cap 5.
  - w2-g1: submitted 20:17:55. Winner 21132255 (agents_dev, cpu_datamover, started 20:17:59), loser 21132256. Start
    line: NEW_2 $9,771.03 left, cap 5.
  - Both ran at 5 in flight from the start.
- **submit.sh fix (wave-2 gating):**
  - At 20:17 submit.sh refused deepseek w2-g1: "2 finished wave-1 campaign(s) and 2 wave-2 campaign(s) started".
  - Cause: the gating counted running w2-g0 twice, once for its probes dir and once for its queued / running job.
  - Fix: count distinct w2 campaign names over both sources (`monitor/patches/submit_w2_count_20261031.py`; temp file +
    mv; pre-change copy + sha256 in `provenance/pre-submit-w2-count-20261031/`).
  - Tested in isolation (deepseek with dir + job -> 1, no w2 -> 0, job only -> 1), `bash -n` OK. The resubmission
    was accepted.
- **Wave 1 still running:** kimi w1-g0..g2 and glm w1-g0..g2, all on leg 2 (jobs 21128606, 21128610, 21128612,
  21128614, 21128619, 21128621). No leg ended abnormally, so nothing was resubmitted and no sandbox cleanup was needed.
- No STOP file was written. Kimi and GLM wave 2 wait for their models' wave-1 campaigns.

**Graded first attempts:**
- Totals: 573 (18:36) -> 753 (21:09); passes 284 -> 386.
- Endings 18:15-21:09 (monitor/endings.py): kimi 105 graded, glm 9 graded, deepseek 65 graded, all graded (0 infra).

| model | graded | passes | pass@1 | pass@1 excl. agent timeouts | awaiting | final infra | broken-ended |
|---|---|---|---|---|---|---|---|
| kimi | 190 (w1) | 99 | 52.1 | 54.1 (7 timeouts, all before window 5) | 15 | 0 | 0 |
| glm | 140 (w1) | 99 | 70.7 | 70.7 | 9 | 0 | 1 (w1-g0 1-4, re-run planned at campaign end) |
| deepseek | 394 (w1) + 29 (w2) | 168 + 20 | 42.6 / 69.0 | same | 0 | 1 | 0 |

**Agent timeouts (OPERATOR_NOTE 02:15):**
- 0 this window for every model: kimi 0 of 105 endings, glm 0 of 9, deepseek 0 of 65.
- The Kimi auto-pause rule did not trigger.
- The resumed Kimi rollouts started at 18:30 or later. The oldest in flight at 20:39 (kimi w1-g0 17-2, 94 min, 212
  calls) has its deadline at ~22:05, so window 6 sees the first possible Kimi timeouts.

**Health:**
- The infra share of endings in the last 30 min was 0.0 at every poll (0 infra endings in the window).
- OpenSandbox: 2 sandbox-create failures, both retried by Gym: deepseek w1-g2 ~19:08 and deepseek w1-g1 ~19:42. Each
  was "Node didn't have enough resource: nemo.nvidia.com/sandbox-slot, used 86/86" (and 93/93), on a cell shared
  with r6. There was no error spike, so no pause.

**Hub and ATP admission:**
- **GLM-5.3:**
  - 200 replies per 30 min (GLM journals): 351 (18:30), 334 (19:00), 387 (19:30), 211 (20:00), 133 (20:30).
  - 429s: 33 from 18:29 to 21:12. Of these, 26 were "Model capacity reached for nvidia/zai-org/glm-5.3. Priority:
    prod, Rate limit type: tokens, Model TPM: 100000000" (a model-level capacity limit, not a key limit). The other 7
    were "Model rate limit exceeded. TPM limit=100000000, current usage=100-106M".
  - 408s (LiteLLM connection timeouts): 13, all from 20:00.
  - ATP kept every GLM campaign at API limit 1-2 and rollout limit 1 all window. Every 429 / 408 starts a new rollout
    epoch, so the limit never doubled: 3 GLM rollouts in flight and 9 GLM first attempts graded in 2.9 h (~3/h).
  - GLM still has ~250 wave-1 slots and 170 wave-2 slots to run. r6's judge uses the same GLM-5.3 model.
- **Kimi:** 8 rollouts per campaign (24 in flight) all window. Only 200 replies in the 30-min health scans. Median
  2.4 calls/min per running rollout (inflight.py, 20:39).
- **DeepSeek:**
  - ATP restarted the wave-1 campaigns at rollout limit 1-2 and raised them to 5 by ~19:28.
  - The wave-2 campaigns ran at 5 from the start.
  - Only 200 replies in the 30-min health scans.

**Budget (key_budget.py, NVAPI_KEY_NEW_2):**
- Remaining: $10,000.00 (18:11, 18:29), $9,989.16 (18:45), $9,902.18 (19:28), $9,795.53 (20:06), $9,677.43 (20:50)
  and $9,653.66 (21:09). That is $346 in 2.7 h (~$130/h), r6's judge included if it now uses NEW_2.
- The campaign's own estimate went from $1,111.11 (18:36) to $1,427.41 (21:09): kimi $367.66, glm $346.11, deepseek
  $713.64.
- progress.py estimates $1,880 still needed, against $8,154 usable above the $1,500 guard floor.

**Quota at 21:09:** user 11.10M free files and 11.56 TiB; project 82.04M free files.

## Collection (2026-10-31 11:09 PDT): partial dataset

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
  - The 24,772 attempt files were unchanged before vs after, and no validity cache would have been written.
  - completed: kimi 84, glm 131, deepseek 358.
  - retryable: 25 / 7 / 4.
  - needs_review: 0 / 2 / 1.
  - pending (not started): 286 / 255 / 32.
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
  - progress.py's export estimate / journal = 0.9493 (kimi), 0.9723 (glm), 0.9946 (deepseek).
- **Results (graded first attempts, all wave 1; wave 2 never started):**

| model | graded | passes | pass@1 raw | excl. broken-ended | excl. infra | excl. agent timeouts | tasks >= 1 pass / graded | compaction | $ per graded rollout | hub cost, all attempts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi | 84 / 565 | 40 | 47.1 | 47.6 | 47.6 | 51.9 | 25 / 43 | 0.0% | $1.94 | $188.86 |
| glm | 131 / 565 | 91 | 66.4 | 70.0 | 69.5 | 69.5 | 36 / 53 | 22.1% | $2.42 | $328.14 |
| deepseek | 358 / 565 | 153 | 42.5 | 42.7 | 42.7 | 42.7 | 60 / 79 | 21.2% | $1.69 | $611.40 |

  Each model graded a different part of ATP's fixed (task, repeat) admission order (kimi 43 tasks, glm 53, deepseek 79).
  On the 41 tasks all three have graded, pass@1 is kimi 48.8, glm 70.0, deepseek 36.8.
  Kimi's 7 graded agent timeouts are all fails with an empty patch (nothing committed before the 3 h kill). Kimi's
  largest prompt (191,601 tokens) stayed below the 242,144 compaction point, so 0% compaction.
- **Uncovered tasks with base <= 2/10 (42 of the 79 wave-1 tasks) where a teacher already has >= 3 of 5 passing:**
  - kimi 0: none.
  - glm 8: boa-hierarchical-evaluation-cancellation, clack-async-autocomplete-options, go-critic-doc-link-checker, ipython-session-bundle-replay, katex-multicolumn-array-spans, mashumaro-flattened-dataclass-fields, textual-kitty-key-phases, yaegi-go-embed-directives.
  - deepseek 8: arktype-json-schema-refs-dependencies, katex-multicolumn-array-spans, mashumaro-flattened-dataclass-fields, opa-template-string-reconstruction, sqlite-utils-safe-import-checkpoints, tengo-callable-instance-isolation, valibot-recursive-schema-composition, yaegi-go-embed-directives.
  - Any teacher: 13.
  - Still open for at least one teacher (no yes yet): 29.
- **Kimi vs baiheh's 1M-context Kimi export (context only):** on the same 84 slots / 43 tasks, 47.6 here vs 53.6
  (1M); without the 7 timeouts 51.9 vs 55.3. Per task: higher here 13, lower 15, tied 15. The 1M export: avg@5 66.7
  on all 113 tasks and 55.2 on the 79 wave-1 tasks.
- **Cost (journal):**
  - Wave legs on NVAPI_KEY_NEW_3: $1,128.40 in total (kimi $188.86, glm $328.14, deepseek $611.40). Of this, $42.75
    went to attempts that were cancelled or ended on infra.
  - Pilots on NVAPI_KEY_NEW_2: $4.56.
  - NEW_3 went from $2,510.64 (lowest reading at the 23:07 leg starts) to $986.41 (2026-10-31T11:07:12-0700). The
    difference is $1,524.23; the part beyond this campaign's $1,128.40 is r6's judge and other users of the key.
- **Quota at 11:10:** user 11.11M free files and 11.63 TiB; project 10581 82.7M free files and 84.1 TiB. This directory
  holds 31.7k inodes for 615 attempts. The ~1,120 slots still to run need roughly 60-100k more inodes, so far more than
  4M stay free.

## Monitor window 4 (2026-10-31 08:15 - 10:45 PDT): NEW_3 floor, all legs stopped

Polls every ~5-10 min (16 polls, 08:14-10:30) with `monitor/poll.sh`, using `WINDOW_START="2026-10-31 08:12"` (right
after window 3's last poll, so the Kimi ending at 08:14 counts here). After the floor came three more checks of the guard
stops (10:34, 10:38, 10:41). OPERATOR_NOTE did not change after its 02:15 entry (sha256 74d83874...).

PROGRESS.md was regenerated at 08:35, 09:03, 09:32, 09:59 and 10:42, and once more right before the window agent
returned. The 09:59-10:42 gap (43 min) is longer than the 30 min asked for: the agent was following the guard stops.

The only code added is the read-only `monitor/stopcheck.py`. This section was written with
`monitor/patches/readme_window4.py` (temp file + mv).

- **Budget floor (OPERATOR_NOTE 21:45):**
  - **Lowest NEW_3 readings:** $1,156.29 (08:12), $1,101.06 (09:03), $1,047.30 (09:40), $1,009.35 (10:21), $995.38
    (10:29-10:30), $987.46 (10:42). That is about $70/h with r6's judge included.
  - **STOP files:** at 10:30:59, watch.py (`--stop-at-floor`) logged `budget_floor` ($995.38). It wrote
    `babysit/<model>-<campaign>/STOP` for all 15 campaigns (9 wave-1, 6 wave-2; none has results).
  - **Guards:** the 9 leg guards fired between 10:30:29 (deepseek w1-g0) and 10:39:52 (kimi w1-g1, whose LiteLLM replica
    still read $1,009.35 at 10:34:49). Each guard sent SIGTERM to its ATP worker.
  - **Leg exits:** the drivers exited 4 between 10:32:38 and 10:40:23 (Slurm FAILED 4:0, jobs 21092988, 21092997,
    21093000, 21093001, 21093003, 21093006, 21093011, 21093013, 21093015).
  - **Leg records:** `legs.jsonl` has an `incomplete` record with the guard reason for each leg (hub key
    NVAPI_KEY_NEW_3). Nothing was cancelled by hand, and no tdw- job is queued or running.
  - **ATP's shutdown recovery** ran in every campaign. `recovery-0.json` in each launch dir was rewritten 10:31-10:40:
    every slot's latest attempt was checked, with 0 sandboxes remaining and 0 errors. There is no recovery-timeout,
    recovery-error or collector-error file.
  - **Cancelled attempts:** the 30 attempts in flight were cancelled: kimi 24 (8 per wave-1 campaign), glm 3 (w1-g0 2-2,
    w1-g1 15-2, w1-g2 11-3), deepseek 3 (w1-g0 5-2, w1-g1 24-4, w1-g2 5-2). Each has `failure.json` (CancelledError,
    artifact incomplete) and a `cleanup-confirmed.json` with no remaining sandbox. ATP retries these slots on a later leg.
  - **Kimi w1-g1 during its delay:** four attempts ended normally between 10:31 and 10:38 while its guard waited: 22-2,
    14-2 and 22-1 pass, 10-3 fail.
  - **Sandboxes:** `monitor/stopcheck.py` found no orphan sandbox candidate, so no cleanup was needed.
  - **What a resume needs:** a new OPERATOR_NOTE entry with the user's decision on the key / budget. Then the STOP files
    must be removed and the campaigns resubmitted with `submit.sh`. A leg starts only while its key is above the start
    floor.
- **Graded first attempts, 08:12 -> 10:42:**
  - kimi 50 -> 84 (+34, 19 passes); glm 122 -> 131 (+9, 6 passes); deepseek 310 -> 358 (+48, 20 passes).
  - Total 482 -> 573 (passes 239 -> 284).
  - Slots not yet graded: kimi 481, glm 434, deepseek 207. Of these, wave 1: kimi 311, glm 264, deepseek 37.
  - 39 ungraded slots wait for a retry (kimi 25, glm 9, deepseek 5): the 30 stop cancellations and 9 earlier infra
    endings.
- **pass@1 (first attempts, 10:42):**
  - kimi 47.6% (40 / 84); 51.9% without its 7 graded agent timeouts (40 / 77).
  - glm 69.5% (91 / 131).
  - deepseek 42.7% (153 / 358).
- **Agent timeouts (OPERATOR_NOTE 02:15):**
  - 0 this window for every model: kimi 0 of 34 graded endings, glm 0 of 11 endings, deepseek 0 of 48. The stop
    cancellations are not timeouts.
  - The Kimi auto-pause rule did not trigger.
  - Window 3's at-risk rollouts: kimi w1-g1 2-2 ended normally at 08:14 (fail). glm w1-g0 6-1 (78.8 min gate wait) ended
    graded pass before 08:44.
- **Infra endings before the stop: 2, both GLM.** In both, LiteLLM's 360 s 408 exhausted ATP's HTTP retries
  (`retry_exhausted`):
  - w1-g1 11-2 at 08:30 (9 HTTP attempts).
  - w1-g0 22-1 at 10:00 (11 HTTP attempts; 14 logical calls).

  That makes 4 such cases since launch. The overall 30-min infra share stayed at 0-7.7% before the stop, with 0 worker
  errors. The per-campaign 1 / 1 readings (glm w1-g1 at 08:34, glm w1-g0 at 10:04) are these single hub-408 endings,
  not OpenSandbox errors, so there was no outage pause.
- **GLM throttling:**
  - **Rates:** GLM got 2-43 429s per 30 min (peak 43 at 09:58-10:04) and 4-5 408s per 30 min all window.
  - **TPM limit:** some 429s carry LiteLLM's "Model rate limit exceeded. TPM limit=100000000, current usage=..." for
    model group `nvidia/zai-org/glm-5.3`. There were 7 in the rollout journals in the hour to 09:23 (usage up to 111.4M)
    and 5 on budget checks between 07:56 and 09:22 (usage 100.8M-109.5M).
  - **Reading:** GLM-5.3's 100M tokens/min limit was exceeded. The message does not say whether that limit is hub-wide
    or per key. This campaign had only 3-4 GLM rollouts in flight.
  - **Other 429:** a budget check at 09:51 got an upstream "Too Many Requests" from the same model group.
  - **Admission:** ATP kept the GLM campaigns at API limit 1-4 and rollout limit 1-2 all window. GLM graded only 9 first
    attempts in 2.5 h.
- **DeepSeek:** 0-21 429s per 30 min, with no LiteLLM limit text. ATP's rollout limits were 1-4, mostly 2; deepseek
  w1-g1 was at 1 from ~09:13.
- **Kimi:** 0-8 429s per 30 min. From ~09:50 ATP raised Kimi admission: w1-g2 and w1-g1 reached 8 rollouts by 09:58,
  w1-g0 by ~10:10. Those 24 rollouts were cancelled at the stop.
- **Budget checks under 429:** LiteLLM returned the spend headers on the 429 replies, so every reading still had a
  remaining value.
- **Campaign estimate:** $937.97 (08:12) -> $1,111.11 (10:42): kimi $180.64, glm $321.41, deepseek $609.06. progress.py
  estimates $2,151 still needed.
- **Quota at 10:30:** user 11.11M free files and 11.63 TiB; project 82.68M free files.

## Monitor window 3 (2026-10-31 05:15 - 08:15 PDT)

Polls every ~9 min with `monitor/poll.sh` (`WINDOW_START="2026-10-31 05:15"`). PROGRESS.md was regenerated at 05:34,
06:00, 06:27, 06:54, 07:22, 07:48, 08:07 and once more right before the window agent returned. OPERATOR_NOTE did not
change after its 02:15 entry (sha256 74d83874...). All 9 wave-1 legs ran through the window: no leg ended, nothing was
resubmitted, no STOP file was written and no sandbox cleanup was needed. No wave-1 campaign finished, so wave 2 was not
submitted. No code changed (only this README section, via `monitor/patches/readme_window3.py`).

- **Graded first attempts at 08:07:** kimi 50 (21 passes), glm 122 (85), deepseek 308 (131), 480 in total (237 passes).
  Endings in the window (attempts): kimi 13 graded (6 passes); glm 26 graded (19) + 1 infra; deepseek 92 graded (44).
- **Agent timeouts (OPERATOR_NOTE 02:15):**
  - kimi: 2 of its 13 endings. w1-g1 2-1 (05:35, 180.7 min) and w1-g1 18-1 (06:17, 180.4 min), both graded fail.
  - glm 0 of 27 endings, deepseek 0 of 92.
  - The Kimi auto-pause rule (at least 3 AND more than 20%) did not trigger. The Kimi share was 2 of 6 (33%) at 06:18
    and 2 of 13 (15%) at the end.
  - Of the three Kimi rollouts window 2 named as at risk, w1-g1 2-1 and 18-1 timed out; w1-g0 14-0 ended normally.
  - Next Kimi deadline: w1-g1 2-2 at ~09:18 (126 calls at 100 min). The other Kimi rollouts in flight have
    deadlines after 10:20.
- **pass@1 with / without the 7 graded Kimi timeouts:** kimi 42.0% (21 / 50) / 48.8% (21 / 43). glm 69.7% and deepseek
  42.5% have no timeouts.
- **NEW_3 (lowest reading per 15 min, budget-log.jsonl):**
  - $1,476.62 (05:00-05:15) -> $1,172.25 (08:00). That is about $101/h with r6's judge included ($134/h 06:00-07:00,
    $75/h 07:00-08:00).
  - It has been below the $1,200 start floor since 07:48 (progress.py read $1,186.60; watch.py logged
    `budget_start_floor` at 07:57). No leg had to start.
  - At $75-135/h the $1,000 guard floor comes around 09:25-10:25.
  - The campaign's own estimate for ended attempts rose from $663.52 (05:13) to $932.54 (08:07), about $90/h.
- **GLM stall (hub 408s + ATP admission):**
  - **408 counts per 30 min:** 1-2 at 05:16-05:52, 4-13 at 06:00-07:48, 5-7 at 07:57-08:06. From 07:45 there were also
    1-3 429s and one 502.
  - **What the 408s are:** every one is LiteLLM's 360 s request timeout (360.3-363.8 s; read from the journals).
  - **Runs on single calls:** the same call hits 360 s 2-7 times, then a retry (a new sample, temperature 1.0) returns in
    4-12 s. Examples:
    - glm w1-g0 26-1: 408s from 05:56, then 7 in a row on one call from 06:46 to 07:29. ATP rated the attempt infra
      at 07:30 and retries the slot.
    - glm w1-g1 11-2: 10 non-200 replies by 07:48.
    - glm w1-g2 15-1, 3-2, 15-0 and 19-0.
  - **ATP's response:** ATP halves the campaign's API limit on each 408. By 07:30 all three GLM campaigns were at API
    limit 1-2 and rollout limit 1.
  - **Gate waits:** at API limit 1, a rollout stuck in 360 s retries holds the campaign's only API slot. glm w1-g0 6-1
    waited 78.8 min at ATP's gate (deadline ~08:58, so it may hit the agent timeout) and glm w1-g1 19-2 waited 33.1 min.
  - **Effect:** GLM 200 replies per 30 min fell from 675-905 (05:16-06:54) to 125-190 (07:39-08:06). GLM graded first
    attempts were 121 at 07:22 and 122 at 08:07.
  - GLM wave-2 campaigns are already set to 4 in flight (OPERATOR_NOTE 02:15), and no new GLM campaign started.
- **Kimi:**
  - Latency improved: p50 31-39 s and p90 68-118 s per 10 min from 05:10 to 05:50 (p50 36-94 s in window 2's slow phase).
  - There were 0-4 Kimi 429s per 30 min.
  - ATP's rollout limits: kimi w1-g0 at 2 from ~05:52 to ~07:22 and again from ~07:57 (4 in between); kimi w1-g1 at 1 all
    window; kimi w1-g2 at 1-2.
- **DeepSeek:** 429s fell from 11 per 30 min (05:16) to 0 from ~05:43. ATP raised all three DeepSeek campaigns back to
  rollout limit 5 by 06:09. There were 1,500-2,970 replies per 30 min.
- **Late 429 burst (last poll, 08:11):**
  - All three models got 429s in the 30 min to 08:11: deepseek 8 (from ~08:05), glm 6, kimi 4.
  - ATP cut every DeepSeek campaign to API limit 4 / rollout limit 2.
  - The 429 replies carry no x-ratelimit header, only DeepSeek's `retry-after: 60`. So it is not known whether this is
    per-model deployment pressure or something key-wide on NEW_3.
- **Final PROGRESS.md (08:12):**
  - 482 graded first attempts (kimi 50, glm 122, deepseek 310), 239 passes, 24 in flight.
  - pass@1: kimi 42.0 (48.8 without the timeouts), glm 69.7, deepseek 42.9.
  - Campaign estimate $937.97.
  - NEW_3 lowest reading $1,156.29 (watch.py, 08:12).
- **Infra endings (1):** glm w1-g0 26-1 (above). The infra share per 30 min was 0-4.3%, with no OpenSandbox error spike.
- **In flight:** 21 (05:13), 23-32 during the window, 26 at 08:07 (deepseek 14, kimi 7, glm 5).
- **Quota at 08:06:** user 11.22M free files and 11.66 TiB; project 82.8M free files.

## Monitor window 2 (2026-10-31 02:15 - 05:15 PDT)

Polls every ~10 min with `monitor/poll.sh` (it now also runs `monitor/endings.py`). PROGRESS.md was regenerated at 02:24,
02:51, 03:20, 03:48, 04:17, 04:45 and 05:13. OPERATOR_NOTE did not change after its 02:15 entry (sha256 74d83874...).
All 9 wave-1 legs ran through the window: no leg ended, nothing was resubmitted, no STOP file was written and no
sandbox cleanup was needed. No wave-1 campaign finished, so wave 2 was not submitted.

- **Graded first attempts at 05:13:** kimi 37 (15 passes), glm 96 (66), deepseek 215 (87), 348 in total. Endings in the
  window: kimi 14 graded (5 passes); glm 49 graded (39) + 2 infra; deepseek 117 graded (54) + 2 infra.
- **Agent timeouts (OPERATOR_NOTE 02:15):** no slot of any model ended on ATP's 10,800 s agent timeout in this window
  (kimi 0 of 14 endings, glm 0 of 51, deepseek 0 of 119), so the Kimi auto-pause rule did not trigger. Over the whole
  run: 6 Kimi attempts, all at 02:09-02:11 (window 1): 5 graded fail + 1 rated infra by ATP (kimi w1-g2 5-4, retried).
- **pass@1 with / without the 5 graded Kimi timeouts:** kimi 40.5% (15 / 37) / 46.9% (15 / 32). glm 68.8% and deepseek
  40.5% have no timeouts.
- **Code (OPERATOR_NOTE 02:15 item 2; temp file + mv, no job runs these files):**
  - `progress.py`: new `agent_timeout(attempt)` = the attempt's `generation-validity-v2-r7.json` `budget_timeout`
    (ATP deepswe_campaign.py: a timeout at or after started.json's deadline = start + 10,800 s); without that file,
    terminal.json `opencode_run_error_type` TimeoutError at or after the deadline. Per-attempt fact `agent_timeout`
    (cached facts without it are recomputed).
  - `progress.py` report: columns "agent timeouts" and "pass@1 excl. timeouts" in the progress table (rows from 02:24;
    earlier rows show "-"), in the model x wave table and in the campaign table. The cost lines count timed-out attempts,
    including ones ATP retried.
  - `collect.py`: per row `ended_by_agent_timeout` (final attempt) and `first_attempt_ended_by_agent_timeout`.
    FINAL-REPORT.md overview and per-wave tables show pass@1 with and without them; per_task.csv gains
    `<model>_agent_timeout`.
  - Pre-change copies with SHA256SUMS: `provenance/pre-agent-timeout-20261031/`. Patch scripts: `monitor/patches/`.
  - Tests: progress.py on a scratch `--root` gave the live numbers plus the new columns. collect.py `--allow-incomplete`
    ran on a scratch root with a fake results file: kimi w1-g0 1-4 (timed out) gave True, 9-4 (normal end) gave False.
  - New read-only tools: `monitor/endings.py` (endings since a time with the agent-timeout flag; evaluates the auto-pause
    rule), `monitor/latency.py` (hub latency per 10 min; log `monitor/latency-log.txt`), `monitor/inflight.py` (running
    attempts: calls, API-gate wait, minutes to the agent deadline).
- **NEW_3:** $1,989.62 (02:17) -> $1,476.62 (05:13), about $175/h with r6's judge included. The campaign's own estimate
  for ended attempts rose from $354.70 (02:24) to $663.52 (05:13), about $110/h. At $175/h the $1,200 start floor is
  reached around 06:50 and the $1,000 guard floor around 07:55.
- **Infra endings (4).** The infra share per 30 min stayed at 0-6.5%, with no sign of an outage:
  - glm w1-g0 21-4 (superjson-error-stack-serialization): one GLM request got LiteLLM's 360 s 408 seven times in a row
    (02:34-03:18). ATP marked it `retry_exhausted` and rated the attempt infra; ATP retries the slot.
  - glm w1-g1 18-2 (pwntools-tube-multiplexing): the verifier was killed at 1,800 s. This is the second time for GLM on
    this task (18-0 in window 1). Other patches for the task verify in 17-64 s (kimi 18-0, glm 18-1 / 18-3, deepseek
    18-0..18-4), so it is these particular patches that make the tests hang.
  - deepseek w1-g0 14-4 (langchain-request-coalescing): the verifier was killed at 1,800 s.
  - deepseek w1-g0 7-0 (go-git-worktree-merge-conflicts): the agent sandbox was OOMKilled (OpenSandbox 410
    SANDBOX::ENDED). It was a single event.
  - OpenSandbox: no API-error spike, only a few "PTY socket lost; re-attached" warnings.
- **Hub pressure and ATP admission:**
  - Kimi latency is back in the slow regime. From 03:00, per 10-min bucket: p50 36-94 s, p90 115-260 s. At 02:10-02:50
    it was p50 20-38 s. There were 0-5 Kimi 408s per 30 min.
  - ATP cut kimi w1-g1 and w1-g2 to API limit 1-2. From ~03:30 to ~05:00, Kimi calls waited up to 6,100 s per 30 min at
    ATP's gate, and w1-g1's rollouts lost ~40 min each.
  - Next likely Kimi agent timeouts, among the 7 Kimi rollouts in flight at 05:13: w1-g1 2-1 (deadline ~05:35, 84 calls
    so far), w1-g0 14-0 (~05:56, 99 calls) and w1-g1 18-1 (~06:18, 36 calls). The auto-pause rule can trigger in window
    3.
  - GLM: 1-10 408s per 30 min (up to ~0.9% of calls). glm w1-g0 had most of them (6 of its 141 replies in the hour to
    03:10) and stayed at rollout limit 1 through the window.
  - DeepSeek: 429s rose to 6-11 per 30 min after 04:50 (about 0.5% of replies). ATP cut each DeepSeek campaign to
    rollout limit 2.
  - Total in flight fell from 36 (02:24) to 21 (05:13).
- **Quota at 05:13:** user 11.38M free files and 11.71 TiB; project 83.3M free files.

## Monitor window 1 (2026-10-30 23:15 - 10-31 02:15 PDT)

Polls every ~10 min: `monitor/poll.sh` = OPERATOR_NOTE digest + `monitor/health.py` (read-only: endings per campaign in
the last 30 min, graded / infra / interrupted; hub statuses and ATP API-gate waits per model; ATP admission state) +
`watch.py --once --stop-at-floor --partition cpu_dataprocessing,cpu_datamover` (no auto legs). OPERATOR_NOTE unchanged
(sha256 d2a19774...). No leg ended, so nothing was resubmitted, and no STOP file was written.

- **Graded at 02:12:** kimi 23 (10 passes), glm 47 (27), deepseek 96 (33); 0 final infra; 2 ended without a grade.
- **NEW_3 burn:** $2,510.64 left (23:06) -> $2,023.01 (02:12), about $148/h with r6's judge included. LiteLLM's spend
  figure lags and differs between hub replicas. At this rate the $1,000 guard floor is reached around 09:00 on 10-31.
- **ATP's adaptive admission** (`atp/scripts/deepswe_admission.py`, unchanged; KR4 and TR2 ran with it too):
  - Every 408 / 429 / 5xx reply halves the campaign's API limit (at most once per 60 s) and starts a new rollout epoch.
  - The rollout limit grows only after a rollout admitted in the current epoch completes without a pressure event.
  - Result: in flight fell from 63 to 32. Rollout limits at 02:12: kimi 2 / 4 / 2, glm 1 / 4 / 4, deepseek 5 / 5 / 5.
- **GLM 408s** are LiteLLM's 360 s request timeout on long non-streamed GLM generations (e.g. 24,308 completion tokens
  took 230 s). They hit about 0.5% of GLM calls.
  - glm w1-g0 had API limit 1 from 23:38 to about 01:10, with 8 rollouts sharing the one slot. Between them they waited
    30-49 min at ATP's gate in their first hour.
  - Its limit recovered at about 01:10, and its first 8 rollouts ended by 02:11, the last one at 182 min.
- **Kimi-K3 hub latency:** p50 per call 45-80 s, p90 125-200 s (10-min buckets, 23:10-00:50). KR4's pilot had p50 14.7 s.
  - **6 Kimi rollouts hit ATP's 3 h agent timeout (TimeoutError) at 02:09-02:11.** Five are graded fail with F2P 0 (truncated
    sessions, 94-131 calls, est. $3.5-4.9 each). ATP rated the sixth (kimi w1-g2 5-4) an infrastructure generation, so it
    will retry that slot.
  - The 18 Kimi rollouts that ended normally had a median of 70 min, 61.5 calls and est. $1.21.
- **First GLM broken-ended slot:** glm w1-g0 slot 1-4. Its last reply has 0 content characters, 0 reasoning characters, 0
  completion tokens and finish "stop", so it is a true empty reply, not a false positive. The brk re-run is planned for
  the end of the campaign.
- **Verifier timeout:** glm w1-g1 slot 18-0 (pwntools-tube-multiplexing). The verifier was killed at ATP's 1,800 s limit
  ("Verifier did not produce /logs/verifier/reward.json"). ATP did not grade it.
- **Rates in the last 30 min (no outage signal):** infra endings 0-6%; DeepSeek 429s up to 23 per 30 min (ATP backs off);
  OpenSandbox errors 0.

## Wave-1 launch (2026-10-30 23:07 PDT)

- **Partition probe** (`sbatch --test-only`, 2 CPU / 16G / 23 h, at 23:05): cpu_dataprocessing would start at once on both
  accounts; cpu_interactive at once (rl_algo) or at 23:08 (agents_dev); cpu_datamover not until 03:26.
- **Partition choice:** `TDW_PARTITION=cpu_dataprocessing,cpu_datamover`. cpu_interactive was left out on purpose. Its
  QoS p_cpu_short caps hengyuf at 10 nodes per user (`scontrol show assoc_mgr flags=qos qos=p_cpu_short`), and it shares
  that cap with cpu_short, where the r6 RL run's resume and autoexport jobs queue. Nine 23 h legs there would hold 9 of
  the 10 slots. For later legs (resumes, wave 2), pass the same `TDW_PARTITION` to submit.sh, or
  `--partition cpu_dataprocessing,cpu_datamover` to watch.py. The defaults of both scripts still include
  cpu_interactive.
- **Submitted:** `bash submit.sh <model> w1 0 1 2` for kimi, glm and deepseek through the schedule-helper babysit pairs.
  All 9 winners started at 23:07 on cpu_dataprocessing, and each loser exited with "sibling candidate won".

| campaign | slots | cap | winner job (account) | loser | start budget line (NVAPI_KEY_NEW_3 remaining) |
|---|---|---|---|---|---|
| kimi w1-g0 | 27 x 5 = 135 | 8 | 21092988 (rl_algo) | 21092984 | $2,510.64 |
| kimi w1-g1 | 26 x 5 = 130 | 8 | 21092997 (agents_dev) | 21092998 | $2,510.64 |
| kimi w1-g2 | 26 x 5 = 130 | 8 | 21093000 (rl_algo) | 21092999 | $2,523.58 |
| glm w1-g0 | 27 x 5 = 135 | 8 | 21093001 (agents_dev) | 21093002 | $2,510.64 |
| glm w1-g1 | 26 x 5 = 130 | 8 | 21093003 (agents_dev) | 21093004 | $2,510.64 |
| glm w1-g2 | 26 x 5 = 130 | 8 | 21093006 (rl_algo) | 21093005 | $2,510.64 |
| deepseek w1-g0 | 27 x 5 = 135 | 5 | 21093011 (rl_algo) | 21093010 | $2,523.58 |
| deepseek w1-g1 | 26 x 5 = 130 | 5 | 21093013 (rl_algo) | 21093012 | $2,510.64 |
| deepseek w1-g2 | 26 x 5 = 130 | 5 | 21093015 (rl_algo) | 21093014 | $2,523.58 |

(The two remaining figures come from different hub replicas; LiteLLM's spend figure lags.)

- **Checked per leg:**
  - The driver log names the campaign, the model id, context 262144, the stock prompt, hub key NVAPI_KEY_NEW_3 with
    its remaining budget, sandbox CELL3_SANDBOX_API_KEY, the max concurrency and broken-reply re-runs on.
  - `legs.jsonl` and each campaign's `tdw-legs.jsonl` record NVAPI_KEY_NEW_3. The campaign `settings.json` holds
    exactly the alias's spec (context_window 262144, reasoning_effort max), max_output_tokens 131072, temperature 1.0,
    top_p 0.95 and agent timeout 10800. `concurrency.json` = 8 / 8 / 5.
  - Within 2 minutes ATP admitted the full cap in every campaign: 63 attempts (24 kimi, 24 glm, 15 deepseek), each
    with hub calls. At 23:11 all 714 journal responses were status 200, with no 429 or 408.
- **DeepSeek cap 5:** this is the setup's default (hub-wide 16M tokens/min limit; see "Rate limits"). It stays inside
  the "at most 8" cap, and no OPERATOR_NOTE entry overrides it. It is fixed at campaign creation.

## Pilot results (2026-10-30)

One rollout of `geo-shapeindex-serialization` (Go, wave 1) per model, outside the 5 per task (collect.py puts them in
`dataset/pilots.jsonl.gz`). Summary: `analysis/pilot-verification.json` (`setup/pilot_summary.py`); per attempt:
`analysis/verify-<model>-pilot-0-0-attempt-0.json` (`verify_pilot.py`).

| model | job, wall | result | F2P / P2P | calls, agent window | call latency median / p90 | max prompt tokens | hub cost (NEW_2) |
|---|---|---|---|---|---|---|---|
| kimi | 21089622, 74 min | fail (reward 0) | 14/24 / 599/599 | 66, 69.0 min | 33.0 s / 150.9 s | 95,219 | $1.23 (LiteLLM usage.cost) |
| glm | 21089625, 29 min | **pass** (reward 1) | 24/24 / 599/599 | 119, 24.8 min | 4.0 s / 21.1 s | 213,443 | $2.35 (prices.json x usage) |
| deepseek | 21089628, 17 min | fail (reward 0) | 14/24 / 599/599 | 90, 12.8 min | 3.7 s / 15.3 s | 160,409 | $0.97 (prices.json x usage) |

All 20 checks of `verify_pilot.py` passed for each model:
- **Request journal (every request, not only the first):** model = the alias's hub id, max_tokens 131,072, temperature
  1.0, top_p 0.95, reasoning_effort max, streamed for kimi only; every system message = the stock OpenCode prompt
  (sha256 `962fbf3c...`) + the model line, no added sentence; OpenCode limit context = input = 262,144, output 131,072,
  no agent prompt override. Every request / response carries ATP credential `existing` (= NVIDIA_API_KEY = the
  wrapper's NVAPI_KEY_NEW_2); no ATP extra-key pool exists; ledgers and worker logs name NVAPI_KEY_NEW_2 only.
- **Image:** the task copy is byte-identical to the pinned DeepSWE dir; request / started / terminal / Gym's rollout
  record carry `public.ecr.aws/d3j8x8q7/swe-bench-202605:kh74gd4v9cty7573mr6k7va39183ngrq-v1.1`; Gym's sandbox_handle =
  the registered agent sandbox (Gym creates every sandbox with the task store's image and refuses a different request
  image).
- **Grading:** the graded patch is the committed diff (model.patch sha256 / size = terminal's = candidate_saved; the
  agent sandbox's collect command is the task's `git diff --binary <base> HEAD`); a separate verifier sandbox ran the
  task's tests; F2P / P2P totals 24 / 599 = the task's tests/config.json; reward.json = terminal.
- **Spend:** per-attempt cost recorded (above); NEW_2's spend grew by more than the pilots' summed journal cost (see
  `analysis/pilot-verification.json` joint_spend; LiteLLM's spend figure lags by up to ~10 min).
- **Sandboxes:** every agent and verifier sandbox confirmed stopped (`cleanup-confirmed.json`: 2 stopped, 0 remaining
  per attempt).

**Golden patch (ATP's grader):** `golden/golden_pilot.py` (job 21090231, `golden/submit_golden.sh`) ran ATP's
`check_deepswe.py` golden-only path unchanged on the pilot task (task moved to the front of ATP's staged dataset):
reward 1, F2P 24/24, P2P 599/599, verifier artifacts complete, cleanup complete; the staged task files equal the pinned
ones; no hub key in the process. (The DeepSWE eval's 2026-10-14 Gym-stock golden run also resolved it: 24/24, 599/599.)

**Observations for the full run:**
- Kimi-K3 on the hub was ~2.2x slower per call than in KR4's pilot (median 33.0 s vs 14.7 s; p90 150.9 s vs 42.1 s),
  with one SSE stream error that ATP retried. If this persists, Kimi rollouts take ~2x the prior 1.1 h and Kimi's ETA
  roughly doubles.
- DeepSeek-V4-Pro's priority_model remaining tokens fell to 13.8M of 16M/min with one pilot rollout in flight (other hub
  users), so the 5-per-campaign cap stays.
- NEW_3 (the run's key from 21:45) fell from $2,608.81 left (21:47) to $2,567.44 (22:33) with no tdw leg on it, about
  $45-65/h of other use (r6's judge; synth's hub_key_active.txt also names NEW_3, though no synth job was running).

## Hub key change (2026-10-30 21:45, OPERATOR_NOTE)

OPERATOR_NOTE.md 21:45 entry (main), quoting a user decision (hengyuf, verbatim as written there): "Now start the
DeepSWE new rollout (kimi. deepseek, GLM) run. use NEW_3 as the api_key, and update the rollout progress every 30
minutes in a md. You can do the three model's rollout at the same time."

- **Key:** `NVAPI_KEY_NEW_3` for every new leg of every campaign (main, rec and brk). It replaces `NVAPI_KEY_NEW_2`.
  The three pilot legs that started on NEW_2 at 21:34 finish on NEW_2 (not restarted).
- **Floors (NEW_3 is also the live r6 RL run's rubric-judge key, ~$25-50 per r6 step):** a leg starts only while NEW_3
  has >= $1,200 left; the guard (every 300 s, SIGTERM as KR4) stops legs below $1,000. At the floor: STOP files for the
  unfinished campaigns, no tdw- leg running, and main is told (`stopped_budget`). Never NEW_2 or any other key unless
  a later OPERATOR_NOTE entry says so.
- **NEW_3 at 21:44 PDT:** spend $7,391.19 of $10,000, **$2,608.81 left**, so **$1,608.81 usable** above the $1,000
  guard floor, shared with r6's judge. The setup's central cost estimate for the whole run is $3,492 (high $4,882), so
  NEW_3 alone covers well under half of it at the floors above (see "Cost" below).
- **Code changes** (temp file + `mv`; the pre-change files and their sha256 are in `provenance/pre-new3-20261030/`):
  - `driver.py`: `KEY = NVAPI_KEY_NEW_3`, `START_FLOOR = 1200`, `GUARD_FLOOR = 1000`; `check_existing` accepts an
    earlier NEW_2 leg only if it started before 21:45 (`KEY_SWITCH_AT`); results carry `hub_key_policy.switch`.
  - `probe_worker.py`: `KEYS = (NVAPI_KEY_NEW_3, NVAPI_KEY_NEW_2)` (NEW_2 only for recovery campaigns of the running
    pilot legs, whose driver code was loaded before the switch); the driver picks the key per leg.
  - `key_budget.py`: allows NEW_3 (default) and NEW_2 (pilot legs' guard); refuses every other name.
  - `progress.py`: NEW_3 budget and floors in the header and the cost section; the pilot lines name their key.
  - `watch.py`: NEW_3; new events `budget_start_floor` (< $1,200) and `budget_floor` (< $1,000), `--stop-at-floor`
    writes the STOP files; `--budget-floor` default 1,400.
  - `verify_pilot.py`: a leg's expected key = NEW_2 if it started before 21:45, else NEW_3.
  - `setup/offline_checks.py`: the budget check reads NEW_3 (ok needs >= $1,200); `collect.py`: report text.
- `legs.jsonl` records the key NAME of every leg (`hub_key`); `PROGRESS.md`'s header states the key and floors.

## Settings

The settings match kimi-rollouts-v4 (`../kimi-rollouts-v4-stock256k-20261028`, "KR4", read-only) except for the tasks
and the model alias.

| | value | source |
|---|---|---|
| models (alias = hub id) | `kimi` = `nvidia/moonshotai/kimi-k3`, `glm` = `nvidia/zai-org/glm-5.3`, `deepseek` = `nvidia/deepseek-ai/deepseek-v4-pro` | kimi / glm: synth/settings.json; deepseek: teacher-rollouts-v2's (TR2) in-memory spec (checked with ast) |
| sampling | reasoning_effort max, temperature 1.0, top_p 0.95, max_tokens 131,072 | settings.json defaults, as KR4 / TR2 |
| harness | synth `rollouts.probe` -> ATP campaign, OpenCode 1.17.11, each task's own DeepSWE grader, ATP agent timeout 10,800 s | as KR4 |
| system prompt | OpenCode stock prompt (`prompt/opencode-1.17.11-default.txt`, sha256 `962fbf3c...`); `probe_worker.py` refuses any override | as KR4 |
| context | OpenCode limit.context = limit.input = 262,144 (compaction at 242,144), limit.output = 131,072 | `context_window` 262144 in memory |
| streaming | kimi streamed, glm / deepseek not (ATP's alias default) | as KR4 / TR2 |
| tasks | the 113 Harbor task dirs of deep-swe@435ee89 (ATP's dataset_revision): `.../swe-rl-v2-sc-migration/eval/deepswe-v1.1/cache/tasks`, read-only | `tasks/waves.json` snapshot |
| waves | wave 1 = 79 tasks without an accepted synthetic task (`uncovered_base_k10.csv`); wave 2 = the other 34 (exactly the targets with accepted > 0 in `synth_targets.json`) | `setup/make_tasks.py` |
| campaigns | per model: `w1-g0..g2` (27 / 26 / 26 tasks = `wave1[K::3]`), `w2-g0..g1` (17 / 17 = `wave2[K::2]`), 5 repeats; `pilot` = 1 rollout of `geo-shapeindex-serialization` | `driver.py` |
| in flight | per campaign: kimi 8, glm 8, **deepseek 5** (cap 8; see "Rate limits"); pilots 1; at most 9 campaigns queued or running at once | `submit.sh` |
| wave order | a model's wave-2 campaign starts only after one more of its wave-1 campaigns has finished (`submit.sh` refuses otherwise; `TDW_FORCE_W2=1` overrides) | `submit.sh` |
| hub key | **NVAPI_KEY_NEW_2** for every leg from 2026-10-31 18:15 (OPERATOR_NOTE 18:15). A leg starts only while NEW_2 has >= $1,700 left. The guard re-checks every 300 s and stops the leg below $1,500 (NEW_2 also carries the r6 RL rubric judge). Earlier: NVAPI_KEY_NEW_3 from 2026-10-30 21:45 until its $1,000 floor stop (floors $1,200 / $1,000); NVAPI_KEY_NEW_2 for the setup and the 21:34 pilot legs (floors $400 / $150). No fallback key, ever. | `driver.py` (`KEY_PERIODS`), `key_budget.py` (refuses any other name) |
| sandbox | hengyuf's OpenSandbox credential (S/atp/.env CELL3_SANDBOX_API_KEY, ATP's own); no Baihe path exists in this copy. The cell is shared with the live r6 RL run. | `probe_worker.py` |
| broken replies | a slot ENDED by a broken reply (brokenreply.py = degenerate.py's definition) is re-run once in `<campaign>-brk<n>`, never a third time. On for all three models; see "Broken replies". | `driver.py` |
| output-cap runaways | flagged per slot (`ended_by_output_cap_runaway`), never re-run | `driver.py` |
| budget refusals | a slot whose final attempt got a hub budget refusal is re-run once in `<campaign>-rec<n>` (KR4; covers the transient-401-then-false-422 glitch) | `driver.py` |
| legs | CPU Slurm jobs via the schedule-helper babysit pair (accounts nemotron_agents_dev + nemotron_rl_algo); 23 h (pilot 8 h); CPU partitions only | `submit.sh`, `driver.sbatch` |

## Differences from KR4

- **driver.py**
  - New `--model kimi|glm|deepseek` flag. SPEC = the model's spec + `context_window` 262144. kimi / glm must equal
    settings.json; deepseek must be absent there and is added in memory.
  - Task sets `pilot` / `deepswe-w1` / `deepswe-w2` come from `tasks/waves.json`. Each task's task.toml sha256 is
    re-checked at every leg.
  - Paths are `probes/<model>/<campaign>/`, `results/<model>-<campaign>.json` and `reruns/<model>-<campaign>.json`.
    `legs.jsonl` lines carry the model.
  - The only key is NEW_2, with floors $400 (start) and $150 (guard). A refused start writes a `refused` leg line and
    exits 5. KR4 had no refused-start case: it fell back to NEW_3.
  - hengyuf's sandbox only; the Baihe branch is gone.
  - Broken-reply re-runs need a per-model "decision" in `analysis/brokenreply-validate-tr2.json` (true for all three).
  - New per-slot fields: `last_reply_stop_class`, `last_reply_finish`, `last_reply_completion_tokens`,
    `ended_by_output_cap_runaway`.
  - New results fields: `campaign`, `wave`, `hub_key_policy`, `max_output_tokens`, `temperature`, `top_p`,
    `broken_reply_reruns_enabled`, `broken_reply_rerun_decision`.
  - New `--dry-run` flag: checks and prints a campaign (tasks, spec, labels) without any write, hub or sandbox call.
  - `check_existing` also asserts the key and the model of earlier legs. Sandbox labels are checked to be <= 63
    characters.
- **probe_worker.py**
  - Env names are `TDW_HUB_KEY_NAME` (must be NVAPI_KEY_NEW_2), `TDW_PROMPT` (must be `stock`) and `TDW_MODEL`. The
    campaign's settings.json must hold exactly that model.
  - The ledger is `<campaign>/tdw-legs.jsonl` (with model / model_id). The Baihe sandbox override is removed. The
    stock-prompt check is unchanged.
- **key_budget.py**: only `NVAPI_KEY_NEW_2` is allowed. A budget check is a 4-token GLM call with that key.
- **submit.sh / driver.sbatch**
  - Job and exp names are `tdw-<model>-<campaign>`; the babysit dir is `babysit/<model>-<campaign>/` (STOP file there).
  - CPU partitions only (refuses anything else). The helper's sha256 is pinned
    (`62056f42...`, the same r3 vendored helper KR4 sourced).
  - Before submitting it checks the 9-campaign cap and wave-2 gating and runs a driver `--dry-run` preflight; it never
    submits for a campaign that already has results.
- **brokenreply.py**: `--validate-tr2` mode and `last_reply_detail()` added. `classify` / `_reverse_lines` /
  `last_reply` / `broken_end` are KR4's code (ast-identical), and `degenerate.py` is byte-identical to KR4's.
- **progress.py**: rewritten for 3 models x 2 waves (see its docstring). Cost is estimated from OpenCode export tokens,
  priced per model.
- **collect.py**: new (final phase).
- **verify_pilot.py**
  - Model-aware, with NEW_2 / CELL3 expectations and a tdw ledger.
  - New check `params_ok`: model id, max_tokens, temperature, top_p, reasoning_effort, stream for kimi only.
  - glm / deepseek costs are priced with prices.json.
  - Run it without `--previous` for DeepSWE pilots.
- **watch.py**
  - Campaign ids are `<model>-<campaign>`; NEW_2 only; new events `refused_start`, `operator_note`, `inodes_low`
    (user and project 10581) and `w2_started`.
  - `--auto-start-w2` is optional.
- **costs.py**: reads `tdw-legs.jsonl`.
- Not copied: `make_baihe_secrets.py` and `baihe_snapshot_check.py`. They only serve Baihe's sandbox credential, which
  this campaign must never use.
- Copied unchanged and not adapted: `sources.py`, `features.py`, `window.py`, `compare.py`, `phase_report.py`,
  `ratelimits.py`, `stops.py`, and `degenerate.py`'s analysis part. They still read KR4 / v3 / v5 data paths (they write
  only under this directory or to `--out`). Adapt them before using them on this campaign.

## Broken replies

Validation is `analysis/brokenreply-validate-tr2.json`: evidence, the reviewer's reading and the per-model decision.

- **Data:** TR2's GLM-5.3 and DeepSeek-V4-Pro slots, 252 each (50 tasks x 5 + pilot).
- **Flags:** the detector flagged 0 slots for either model.
- **Forward scan (`analysis/tr2-reply-scan.json`):** 0 degenerate replies among 45,010 GLM and 39,055 DeepSeek replies.
- **Tail reader:** agreed with `degenerate.scan()` on all 25 attempts checked both ways.
- **Every ending:**
  - normal final summary: 248 GLM + 251 DeepSeek;
  - harness cut-off: 4 GLM (3 agent timeouts, 1 PTY infra);
  - 1 DeepSeek output-cap runaway: 131,072 completion tokens, content `"\n\n"`, a reasoning-only stop. This one is not
    degenerate by the definition, so it is now flagged per slot and not re-run.
- **Manual reading:** the 24 sampled ordinary endings are normal summaries.
- **Task texts:** no DeepSWE task text contains `[PAD]` or a `<|...|>` token.
- **Decision:** re-runs on for kimi (KR4's validation), glm and deepseek. Caveat: no GLM / DeepSeek positive example
  exists, so the detector's recall on their own failure modes is unknown.
- **Rule source:** hengyuf approved this rule on 2026-10-28 for the Kimi v4 batch. It is applied here on the workflow's
  instruction (main, 2026-10-30); there is no separate user approval for this campaign.

## Cost, NEW_2 and capacity (setup estimates)

Note (21:45): the run now spends NVAPI_KEY_NEW_3, not NEW_2 (see "Hub key change"). NEW_3 had $2,608.81 left at 21:44,
$1,608.81 above its $1,000 guard floor and shared with the r6 judge; the NEW_2 outcomes in the table below are the
setup's projection and no longer apply.

Inputs:
- NEW_2 had $4,789.59 left at 21:23 PDT (spend $5,210.41 of $10,000).
- Usable down to the guard floor: $4,639.59.

Size factor (`analysis/size-and-pilot.json`), DeepSWE vs the synthetic sel50 tasks:
- Method: Kimi-K3 1M rollouts on DeepSWE (TARGETS, 564 rollouts) vs on the 50 sel50 tasks (TR2, 248 rollouts); a
  prompt-token model fitted on sel50 (R² 0.988).
- Result: DeepSWE rollouts are smaller, **0.69x** prompt tokens (wave 1 0.60x, wave 2 0.89x; turns 0.81x).

256k effect for GLM / DeepSeek (`analysis/cost-projection.json`): TR2's 1M rollouts were replayed with compaction at
242,144.
- GLM: 44.6% of rollouts passed that point at 1M; cost x 0.79.
- DeepSeek: 26.4% passed it; cost x 0.82.

| case ($ per final slot, wave 1 / wave 2) | kimi | glm | deepseek | total | NEW_2 outcome |
|---|---|---|---|---|---|
| stated (main's 3.8 / 4.2 / 2.0, no size scaling) | 3.80 / 3.80 | 4.20 / 4.20 | 2.00 / 2.00 | $5,650 | wave 1 fits ($3,950); wave 2: glm 97%, deepseek 0%, kimi 0% (~1,349 of 1,695 slots) |
| central (size-scaled; glm / deepseek x 256k sim) | 2.44 / 3.64 | 1.95 / 2.91 | 0.99 / 1.47 | $3,492 | everything fits, ~$1,298 left |
| high (size-scaled; glm / deepseek x 1.29 = Kimi's 1M->256k cost rise + Kimi's re-run rate) | 2.44 / 3.64 | 3.43 / 5.11 | 1.66 / 2.47 | $4,882 | kimi wave 2 ~61% (~1,629 slots) |
| worst (no size scaling, x 1.29, re-runs) | 4.08 / 4.08 | 5.72 / 5.72 | 2.77 / 2.77 | $7,101 | 93.5% of wave 1 (~1,107 slots) |

Wave-2 order follows each model's expected wave-1 finish: glm 13.2 h, deepseek 15.8 h, kimi 18.1 h (priors).

Disk and inodes (`analysis/offline-checks.json`):
- Need: ~129k inodes and ~1.35 TB (measured per slot on KR4 / TR2 campaigns: 52-89 inodes, 0.27-1.17 GB).
- Free at 21:23:
  - user quota: 13.55M files, 11.03 TiB;
  - project 10581: 86.3M files, 84 TiB.
- The run leaves far more than 4M free inodes.
- Watch: hengyuf's user quota grew by 165k files and 458 GiB in the 35 minutes from 20:53 to 21:28, from other jobs.
  watch.py alerts below 2 TiB and below 4M files.

## Rate limits (`analysis/hub-ratelimits.json`)

The hub's limits are per model deployment and hub-wide, shared with every user of the model:
- DeepSeek: priority_model 16M tokens/min. TR2 drove it below zero remaining at 16 in flight (and got 429s).
- GLM: 80M tokens/min.
- Kimi: 50M tokens/min.

| model | tokens/min per running rollout (TR2 median) | at 24 in flight | in flight here |
|---|---|---|---|
| kimi | 0.21M | 10% of 50M | 24 |
| glm | 0.66M | 20% of 80M | 24 |
| deepseek | 0.72M | **109%** of 16M | **15** (3 x 5) |

DeepSeek therefore defaults to 5 per campaign (override with `TDW_MAX_CONCURRENCY_OVERRIDE`). DeepSeek still finishes
wave 1 before Kimi, so the overall ETA does not move. GLM-5.3 is probably also the r6 RL rubric judge's model, so GLM's
share of its limit is shared with r6.

## Files

| path | role |
|---|---|
| `driver.py` | one campaign leg: key check + guard, main campaign, rec / brk recovery campaigns, results; `--dry-run` |
| `probe_worker.py` | ATP worker wrapper: NEW_2 by name, stock-prompt check, model check, `tdw-legs.jsonl` |
| `driver.sbatch` | the CPU job (babysit_enter; drops credential env vars; runs driver.py with the ATP controller python) |
| `submit.sh` | submits babysit pairs; caps, wave-2 gating, preflight |
| `key_budget.py` | NEW_2's budget (names and numbers only); appends `budget-log.jsonl` |
| `brokenreply.py` | broken-ending detector (KR4) + `--validate-tr2` |
| `progress.py` | rewrites `PROGRESS.md` (+ `PROGRESS-rows.json`, `progress-cache.json`) |
| `watch.py` | watcher (`--once` for one check) |
| `verify_pilot.py` | checks one attempt (limits, prompt, params, key, sandbox, grading, spend; pilot phase: every request's params / prompt / ATP credential, image provenance, grading counts vs the task's tests, committed diff, cost) |
| `monitor/health.py`, `monitor/poll.sh` | monitor-phase health scan (read-only; state in `monitor/health-state.json`, log `monitor/health-log.jsonl`) and the per-poll bundle (health scan + `monitor/endings.py` + `watch.py --once`) |
| `monitor/endings.py`, `monitor/latency.py`, `monitor/inflight.py` | read-only (window 2): endings since a time with the agent-timeout flag and the Kimi auto-pause rule; hub latency per 10 min (`monitor/latency-log.txt`); running attempts with calls, gate wait and time to the agent deadline |
| `monitor/stopcheck.py` | read-only (window 4): after a leg stop, ATP's shutdown recovery per campaign (`recovery-<i>.json`, recovery / collector error markers), attempts without an end marker, orphan-sandbox candidates |
| `monitor/patches/` | the one-off patch scripts of windows 2-5 (progress.py / collect.py agent-timeout flags, the README monitor-window sections; window 5: `key_new2_20261031.py` = the NEW_3 -> NEW_2 switch, `submit_w2_count_20261031.py` = submit.sh's wave-2 count, `readme_window5a.py` / `readme_window5b.py`; window 6: `readme_window6.py`; window 7: `progress_tz_20261101.py` = row times with their zone after the DST change, `readme_window7.py` / `readme_window7b.py`; window 8: `progress_glm_pause_20261101.py` = "paused by main (r6 judge 429s)" in PROGRESS.md while main's STOP files exist, `readme_window8.py`; window 9: `readme_window9.py`; the window-9 poll summary is `monitor/.watchsum-w9.py`; window 10: `readme_window10.py`; window 11: `progress_glm_resume_20261101.py` = PROGRESS.md's GLM-caps bullet, `progress_glm_w2_cap_20261101.py` = cap 4 for a GLM wave-2 campaign not yet created, `readme_window11.py`; pre-patch copies `provenance/progress.pre-glm-resume-20261101.py` / `progress.pre-glm-w2-cap-20261101.py` with `provenance/progress-pre-20261101.SHA256SUMS`; window 12: `collect_override_20261101.py` = collect.py's GRADED_FAIL_OVERRIDES (kimi w2-g1 13-0) and the matching verify_collection.py counts, tested by `collect_override_test_20261101.py`; `progress_glm_w2_start_20261101.py` = PROGRESS.md's GLM-caps bullet (wave 2 started, the 20% rule); `readme_window12.py`; pre-patch copies `provenance/pre-collect-override-20261101/` and `provenance/progress.pre-glm-w2-start-20261101.py`; window 13: `progress_glm_w2_stop_20261101.py` = PROGRESS.md's GLM-caps bullet (main's wave-2 stop and the auto-restart rule; pre-patch copy `provenance/progress.pre-glm-w2-stop-20261101.py`), `glm_w2_cap2_manifest_20261101.py` = hard cap 2 for the existing GLM wave-2 campaigns (manifest.json / settings.json 4 -> 2): applied at 19:21:44 PST in window 14, refused by ATP at the next leg start (immutable campaign settings), reverted at 19:23:57, now WITHDRAWN (`--apply` refuses; as-applied copy `provenance/glm_w2_cap2_manifest_20261101.as-applied-19-21.py`, backups `provenance/pre-glm-w2-cap2-20261101/`), `readme_window13.py`; window 14: `progress_glm_w2_restart_20261101.py` = PROGRESS.md's GLM-caps bullet (wave-2 restart, cap 4 kept; pre-patch copy `provenance/progress.pre-glm-w2-restart-20261101.py`), `readme_window14.py` / `readme_window14b.py`; window 15: `readme_window15.py`; window 16: `progress_eta_live_20261102.py` = the ETA of a running wave from the attempts in flight (not the manifest cap) and a running wave 2 starting now, `progress_glm_cancel_20261102.py` = GLM reported as cancelled by the user (header bullet, 0 to go, no ETA), both with `--check`; `readme_window16.py`) |
| `monitor/keycheck.py` | read-only (window 5): every recorded leg (`tdw-legs.jsonl`, legs.jsonl starts) carries the key of its start time under driver.py's `KEY_PERIODS` |
| `monitor/non200.py`, `monitor/watchsum.py` | read-only (window 7): a model's non-200 hub replies in the last N minutes with the hub's error text (no payloads or headers); the summary of `monitor/.watch-last.json` (poll.sh's inline code as a script) |
| `monitor/removed-stop-20261031/` | copies of the 15 STOP files of 2026-10-31 10:30, removed at the 18:15 resume |
| `monitor/removed-stop-20261101/` | copies of main's 5 GLM STOP files of 2026-11-01 05:06 PST, removed at the 11:09 PST resume (OPERATOR_NOTE 11:15) |
| `monitor/glm429bins.py`, `monitor/w13-polls.log` | read-only (window 13): the GLM wave-2 auto-restart rule of OPERATOR_NOTE 15:35 PST (30-min 429 shares of the rollout replies and budget checks in the four 30-min windows covering the last 60 min, all < 10% = met; plus 10-min bins); the window-13 poll log (one line per poll) |
| `monitor/w15poll.sh`, `monitor/tail_events.py`, `monitor/w15-polls.log` | window 15: one poll (as `monitor/w14poll.sh`, plus `monitor/inflight.py --model glm`; it writes only what poll.sh's watch.py writes); read-only: the last hub events of one attempt (time, gap, event, HTTP status, retry wait, HTTP try; first 1.5 KB of each journal line, no payload / header / credential); the window-15 poll log (one line per poll) |
| `monitor/w16poll.sh`, `monitor/w16-polls.log` | window 16: one poll (`monitor/w15poll.sh` with window-16 file names and start 00:15; it writes only what poll.sh's watch.py writes); the window-16 poll log (one line per poll, plus the GLM cancel checks) |
| `monitor/w14poll.sh`, `monitor/w14endings.py`, `monitor/w14-polls.log` | window 14: one poll (OPERATOR_NOTE sha, `monitor/poll.sh`, the `monitor/glm429bins.py` summary, endings / starts, squeue, STOP files; it writes only what poll.sh's watch.py writes); read-only per-attempt endings since a time (task, result, f2p / p2p, duration, agent-timeout flag) and the attempts started since then; the window-14 poll log (one line per poll) |
| `monitor/removed-stop-20261101-w14/` | main's 2 GLM wave-2 STOP files of 2026-11-01 15:21 PST, removed at the 19:21 PST auto-restart (OPERATOR_NOTE entry headed 15:35 PST) |
| `monitor/glm429.py` | read-only (window 11): GLM-5.3 429 shares over the last N minutes, for key_budget.py's budget checks (budget-log.jsonl) and for the GLM rollouts' hub replies (OPERATOR_NOTE 11:15 item 2: > 20% over 30 min -> GLM to 2 per campaign) |
| `collect.py` | final dataset: `dataset/<model>.jsonl.gz`, `dataset/pilots.jsonl.gz`, `dataset/index.jsonl`, `dataset/slots.jsonl`, `dataset/MANIFEST.json`, `per_task.csv`, `FINAL-REPORT.md`, `analysis/collect-summary.json`, `analysis/collect-kimi-vs-1m.csv`; `--allow-incomplete` reads campaigns without a results file from their probe dirs; `GRADED_FAIL_OVERRIDES` (window 12): kimi w2-g1 13-0 counts as a graded fail; final collection (2026-11-02): `USER_CANCELLED` (glm = cancelled_by_user, PARTIAL), slot states interrupted / cancelled_in_flight, yes / no / undetermined verdicts, per-key legs and NEW_2 readings |
| `final/atp_states.py` | ATP's own `slot_state()` of every slot, read-only (atomic_json no-op, files compared before / after) -> `analysis/atp-slot-states.json` |
| `final/collect.sbatch`, `final/submit_collect.sh` | the collection's CPU job (atp_states.py + collect.py --allow-incomplete) through the babysit pair; refuses while a tdw- leg is queued |
| `final/verify_collection.py` | independent checks of the collection -> `analysis/collect-verify.json` (window 12: ATP / progress.py counts leave out MANIFEST `status_overrides`; 2026-11-02: the interrupted / cancelled_in_flight states, complete campaigns vs their results files, the GLM STOP files, a complete campaign's final infra row) |
| `final/unit_check.py`, `final/readme_collect.py` | collect.py's slot view / row builder on a few small slots; this README's Collection section |
| `final/collect_final_20261102.py`, `final/collect_report_wording_20261102.py`, `final/verify_infra_rows_20261102.py` | the final collection's patches of collect.py / verify_collection.py (anchored replacements; pre-change copies, diffs and hashes in `provenance/pre-final-collect-20261102/`) |
| `final/independent_counts_20261102.py`, `analysis/independent-counts-20261102.json` | the final collection's independent headline counts (results files, a direct GLM probe scan, the base CSVs) |
| `final/readme_final_collect_20261102.py` | this README's "Final collection" section and status block |
| `costs.py`, `stops.py`, `degenerate.py`, ... | KR4 analysis tools (see "Differences") |
| `tasks/waves.json`, `tasks/base_k10.json` | task / wave / base-rate snapshots with their sources' sha256 |
| `setup/*.py` | setup scripts: make_tasks, size_and_pilot, cost_projection, tr2_reply_scan, hub_ratelimits, config_check, offline_checks; pilot phase: pilot_summary (-> `analysis/pilot-verification.json`) |
| `golden/` | ATP golden-patch check of one task: `golden_pilot.py` (ATP's check_deepswe golden-only, task moved first), `golden.sbatch`, `submit_golden.sh` (babysit pair); run dirs `<task>-<job>/` with `golden-summary.json`, `stage-<job>.log` |
| `analysis/` | setup evidence: offline-checks, config-check (+ ATP gym.yaml per alias), dry-run/, brokenreply-validate-tr2, tr2-reply-scan, size-and-pilot, cost-projection, kimi-price-fit, hub-ratelimits, selftest-summary |
| `provenance/kr4-source-sha256sums.txt` | sha256 of the KR4 files this copy started from |
| `provenance/pre-partial-collect-20261031/` | collect.py before the partial collection (+ SHA256SUMS) |
| `provenance/pre-new3-20261030/` | the files changed by the 21:45 hub key switch, as they were before it (+ SHA256SUMS); `provenance/verify_pilot.setup-c5bd5576.py` = verify_pilot.py before the pilot-phase checks |
| `provenance/pre-new2-20261031/` | the files changed by the 2026-10-31 18:15 switch back to NVAPI_KEY_NEW_2, as they were before it (+ SHA256SUMS) |
| `provenance/pre-submit-w2-count-20261031/` | submit.sh before the window-5 fix of its wave-2 gating count (+ SHA256SUMS) |
| `provenance/pre-collect-override-20261101/` | collect.py and final/verify_collection.py before the window-12 status-override patch (SHA256SUMS; POST-PATCH.SHA256SUMS = after) |
| `provenance/pre-final-collect-20261102/` | collect.py and final/verify_collection.py before the final collection's patches (SHA256SUMS; POST-PATCH.SHA256SUMS = after; *.diff), and the small outputs of the 10-31 partial collection (`partial-collect-20261031-outputs/`) |
| `provenance/progress.pre-eta-live-20261102.py`, `provenance/progress.pre-glm-cancel-20261102.py` | progress.py before the window-16 ETA patch (sha256 a122600b...) and before the window-16 GLM-cancel patch (dd583fe9...) |
| `prompt/`, `prices.json` | KR4's prompt files and LiteLLM price fits |

Run-time state (not code):
- campaigns: `probes/<model>/<campaign>[-rec<n>|-brk<n>]/` (ATP campaigns, each with `tdw-legs.jsonl`);
- outcomes: `results/<model>-<campaign>.json` and the re-run plans `reruns/<model>-<campaign>.json`;
- ledgers: `legs.jsonl` and `budget-log.jsonl`;
- Slurm: `babysit/<model>-<campaign>/` (helper state; `STOP` refuses new legs) and `logs/tdw-*.out`;
- watcher: `watch-state.json` and `watch-log.jsonl`.

## Operating

Run every Slurm command as a standalone Bash call, and touch only `tdw-*` jobs.

```bash
TD=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030
# probe the CPU partitions first (no job is created):
sbatch --test-only --partition=cpu_dataprocessing --account=nemotron_agents_dev --cpus-per-task=2 --mem=16G --time=23:00:00 $TD/driver.sbatch
# 1. pilots (1 rollout each, 8 h jobs), then verify each attempt:
bash $TD/submit.sh pilot kimi glm deepseek
/usr/bin/python3 -I -B $TD/verify_pilot.py $TD/probes/glm/pilot/models/glm/attempts/0-0/attempt-0
# 2. wave 1 (9 campaigns; launched 2026-10-30 23:07 with cpu_interactive left out, see "Wave-1 launch"):
for m in kimi glm deepseek; do TDW_PARTITION=cpu_dataprocessing,cpu_datamover bash $TD/submit.sh $m w1 0 1 2; done
# 3. wave 2 as wave 1 finishes (submit.sh refuses early starts), or let the watcher do it
#    (pass --partition cpu_dataprocessing,cpu_datamover: watch.py's default still includes cpu_interactive):
/usr/bin/python3 -I -B $TD/watch.py --once --auto-leg-cap 3 --auto-start-w2 --partition cpu_dataprocessing,cpu_datamover --campaigns \
  kimi-w1-g0 kimi-w1-g1 kimi-w1-g2 glm-w1-g0 glm-w1-g1 glm-w1-g2 deepseek-w1-g0 deepseek-w1-g1 deepseek-w1-g2 \
  kimi-w2-g0 kimi-w2-g1 glm-w2-g0 glm-w2-g1 deepseek-w2-g0 deepseek-w2-g1
/usr/bin/python3 -I -B $TD/progress.py
# 4. final collection in a CPU job (it reads every journal once): ATP's slot states + collect.py
#    (--allow-incomplete: unfinished campaigns are read from their probe dirs), then the independent checks:
TDW_PARTITION=cpu_interactive,cpu_dataprocessing bash $TD/final/submit_collect.sh
/usr/bin/python3 -I -B $TD/final/verify_collection.py
```

**Resuming a campaign.** A leg that ends early (23 h limit, guard stop, node loss) leaves the campaign resumable:
- Run the same `bash $TD/submit.sh <model> <w1|w2> <group>` (or `pilot <model>`) again when no `tdw-<model>-<campaign>`
  job is queued or running and `results/<model>-<campaign>.json` does not exist.
- The helper starts the next leg. ATP resumes the campaign, retries interrupted slots, then runs the rec / brk recovery
  campaigns.
- A later leg must find the same broken-reply plan (`reruns/<model>-<campaign>.json`), or it stops without re-planning.
- The campaign keeps its own concurrency cap (fixed when it was created). watch.py's auto legs pass it on.
- A `STOP` file in `babysit/<model>-<campaign>/` blocks further legs.

**Rules.**
- Never edit in place a script that a queued or running job executes (`driver.sbatch`, `driver.py`, `probe_worker.py`,
  `brokenreply.py`, `degenerate.py`, `key_budget.py`, the helper). Write a temporary file in this directory and `mv -f`
  it over the path.
- Never use another hub key than NVAPI_KEY_NEW_2 for new legs (from 2026-10-31 18:15; NEW_3 is at its floor and is
  refused by key_budget.py and probe_worker.py). Never read any .env or secrets file. Clean up sandboxes only through ATP's own recovery
  (labels `nemo-gym.nvidia.com/run` = the launch dir, `atp-model` = the alias, `atp-attempt` = `<task>-<repeat>-attempt-<n>`,
  all <= 63 characters).
- After any change, re-run the offline checks: `/usr/bin/python3 -I -B $TD/setup/offline_checks.py` (one NEW_2 budget
  check; `--skip-budget` skips it; 18 dry-runs of ~1 min each, so allow ~20 min).
