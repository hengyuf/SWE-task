<!-- Written by the main agent from workflow wf_c3ac2b74-d62's final analysis agent (returned as text; subagents cannot write report files). Script-generated companions: analysis/final-report.{md,json}, analysis/sft-usable-passes.jsonl. -->

# Kimi-K3 v4 final analysis: 194 v2 tasks × 5, stock OpenCode prompt, 256k window

**Status: finished.** The 18 campaigns ran from 2026-10-28 14:57 to 2026-10-29 21:32 PDT. That was 27 main legs plus 2 pilot legs, all on `NVAPI_KEY_NEW_2`. All 970 slots were graded and none ended as infra. There were 66 broken-reply re-runs and no third attempt. No kr4- job is queued or running. `progress.py` ran one last time; the final PROGRESS.md row is 10-29 21:41.

## Headline
- **pass@1 on all 970 slots:** 74.2% on first attempts. After the one approved re-run it is **78.8%**; excluding broken-ended first attempts it is 79.5%.
- **Broken endings fell from 28.0% (v3) to 6.8% of rollouts.** On sel50 that is 70/250 → 17/250. This explains most of v4's +19.2-point pass@1 gain over v3. Once both arms exclude broken-ended rollouts, v4 − v3 is only +5.0 points (CI −4.3 to +14.9).
- **The drop in broken replies follows the date, not the prompt.**
  - On Oct 26, v3's concurrent stock-prompt control broke as often as the sentence arm: 3.74 vs 3.62 per 1000 replies, p = 0.82.
  - v4 (stock prompt, Oct 28/29) is at 0.86 per 1000, below that Oct 26 stock control (p = 0.007).
  - v5 (stock, Oct 29) is at 0.70, the same as v4 (p = 0.75).
- **v4 vs the earlier 1M baseline on sel50:**
  - first attempts: −9.4 points (significant);
  - after re-run: −4.6 (CI includes 0);
  - excluding broken endings: −4.5 (CI includes 0).
  - The 1M arm is biased upward: these tasks were picked for having Kimi passes in exactly those rollouts.
- **Reasoning before actions barely moves.** On sel50, 56.4% of v4's action turns carry reasoning, vs 54.9% at 1M and 58.0% with v3's sentence. The differences are +1.5 and −1.6 points.
- **The 256k window causes compaction but no hard truncation.** 6.6% of first attempts compacted once (sel50: 8.0%). There were 0 context-overflow refusals, and 2 replies cut at the 131k output cap in 1,036 rollouts.
- **Cost: $3,959.12 on NEW_2, $0 on NEW_3.** That is $3.82 per graded rollout and $4.08 per final slot. progress.py's $4,364.30 also includes v5's $390.04 on the same key.
- **SFT-usable passes: 745** (sel50 175 / rest47 184 / baihe 386). 701 of them have no compaction, and they cover 192 of the 194 tasks.

## 1. The 50 sel50 tasks: three arms
The arms:
- **v4:** this campaign, Oct 28/29.
- **v3 sentence arm:** 256k plus one reasoning sentence, Oct 26.
- **Earlier Kimi baseline:** 1M window, stock prompt, Oct 17–21. These are the build-time probe rollouts, i.e. the "Kimi-K3 previous" column of v3's phase1-report.

The same code was used for all arms. It reproduces v3's own numbers exactly: features 500/500 rows, 78/21,538 broken replies, ctrl-old 5/1,338, 1M pass@1 77.4% and 54.9% reasoning.

| | v4 first attempts | v4 after re-run | v3 sentence (256k) | earlier Kimi (1M) |
|---|---|---|---|---|
| pass@1 raw | 170/250 = **68.0%** (CI 60.4–75.2) | 182/250 = **72.8%** (65.2–80.0) | 121/248 = 48.8% (40.5–57.4) | 192/248 = 77.4% (72.0–82.6) |
| pass@1 excl. broken-ended | 170/233 = **73.0%** (65.1–80.3) | 182/249 = 73.1% | 121/178 = 68.0% (58.6–77.3) | 192/248 = 77.4% |
| tasks ≥1 pass / 5 of 5 / none | 50 / 12 / 0 | 50 / 17 / 0 | 45 / 7 / 5 | 50 / 16 / 0 |
| % action turns with reasoning | **56.4** (54.7–58.1) | 56.2 (54.4–57.9) | 58.0 (56.2–59.7) | 54.9 (53.5–56.4) |
| compacted rollouts | 20/250 (**8.0%**) | 22/250 (8.8%) | 9/248 (3.6%) | 0/250; 12 (4.8%) went past 242,144 |
| largest main-session prompt | 253,134 | 253,134 | 243,395 | 505,801 |
| truncation: cut at output cap / context-overflow refusals | 1 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| broken replies, graded attempts | 26/26,664 = **0.98 / 1000** | 10/27,892 = 0.36 | 77/20,518 = 3.75 | 0/27,154 = 0 |
| broken replies, every attempt | 27/27,240 = 0.99 (no re-runs) | – | 78/21,538 = 3.62 | 0/27,354 = 0 |
| rollouts ended by a broken reply | 17/250 (**6.8%**) | 1/250 (0.4%) | 70/250 (28.0%) | 0/250 (0%) |
| rollouts with any broken reply | 24 (9.6%) | 8 (3.2%) | 75 (30.0%) | 0 |
| median turns / agent wall (min) | 93 / 57.8 | 96 / 60.7 | 73 / 55.1 | 93.5 / 35.5 |
| $ per rollout (LiteLLM) / cache hit | 3.64 / 88.6% | – | 2.99 / 84.2% | 2.82 / 96.6% |

**Paired task-bootstrap differences** (points, 95% CI, 50 tasks):

| comparison | pass@1 | % action turns with reasoning |
|---|---|---|
| v4 first − 1M | **−9.4** (−16.6 to −2.2) | **+1.5** (+0.5 to +2.6) |
| v4 after re-run − 1M | −4.6 (−11.5 to +2.3) | +1.3 (+0.2 to +2.4) |
| v4 excl. broken − 1M | −4.5 (−12.0 to +3.2) | +1.6 (+0.5 to +2.6), both arms without broken rollouts |
| v4 first − v3 | **+19.2** (+10.0 to +28.0) | **−1.6** (−2.9 to −0.2) |
| v4 after re-run − v3 | +24.0 (+14.4 to +33.2) | – |
| v4 excl. − v3 excl. | +5.0 (−4.3 to +14.9) | −1.5 (−3.1 to −0.0), both arms without broken rollouts |
| v3 − 1M | −28.6 (−36.6 to −19.8) | +3.1 (+1.7 to +4.4) |

**Broken-reply tests (Fisher):**
- v4 vs v3: the per-reply rate ratio is 0.27 (exact 95% 0.17–0.43, p = 6e-10). Rollouts ended by a broken reply: 17/250 vs 70/250, p = 3e-10.
- v4 vs 1M: rollouts ended 17/250 vs 0/250, p = 1e-5.

**What it means:**
- **v4 vs v3.** The pass@1 gap comes from broken endings. Every v3 and v4 broken ending is a fail; in v4, 65 of 66 had F2P 0 and an empty patch.
- **The broken-reply rate depends on when the hub was called, not on the prompt.**

  | date | arm | broken replies per 1000 |
  |---|---|---|
  | Oct 17–21 | 1M | 0 / 27,354 replies |
  | Oct 26 | v3 sentence | 3.62 |
  | Oct 26 | stock control | 3.74 |
  | Oct 28 | v4 | 0.82 |
  | Oct 29 | v4 | 0.95 |
  | Oct 29 | v5 (stock, 256k) | 0.70 |

  So removing the sentence is not what cut the rate.
- **The rate also varies by task.** sel50 is at 0.98 per 1000 vs rest47 at 0.46 (p = 0.026).
- **v4 vs 1M.** v4 is close to the 1M baseline once broken endings are removed or re-run (−4.5 / −4.6 points, CIs include 0). The 1M arm is biased upward: its pass@5 is 100% by construction.
- **Reasoning.** It stays near 55% of action turns in every arm. v3's sentence added only about 1.6 points over v4's stock prompt.
- **Window.**
  - Compaction fires at the right point (trigger prompt median 241,853 on sel50).
  - After it, the median compacted rollout still does 39.5 action turns; 18 of 20 commit after it, and there are no early stops.
  - Compacted sel50 rollouts pass 12/20 (60.0%) vs 68.7% for the rest (small sample).
  - v4 compacts more often than v3 (8.0% vs 3.6%) because v3's sessions were cut short by broken endings: 73 vs 93 median turns.
  - At 1M, 4.8% of rollouts went past 242,144 tokens.

## 2. v4 full set by taskset
| | sel50 | rest47 | baihe | all |
|---|---|---|---|---|
| tasks / slots | 50 / 250 | 47 / 235 | 97 / 485 | 194 / 970 |
| sandbox credential, concurrency cap | CELL3, 8 | CELL3, 8 | BAIHE, 4 | – |
| pass@1, first attempts | 170/250 = 68.0% (60.4–75.2) | 180/235 = 76.6% (69.4–83.0) | 370/485 = 76.3% (71.8–80.8) | **720/970 = 74.2%** (70.7–77.6) |
| pass@1, first, excl. broken-ended | 170/233 = 73.0% | 180/225 = 80.0% | 369/446 = 82.7% | 719/904 = 79.5% |
| pass@1 after the one re-run | 182/250 = 72.8% (65.2–80.0) | 188/235 = 80.0% (72.8–86.4) | 394/485 = 81.2% (76.7–85.6) | **764/970 = 78.8%** (75.4–82.1) |
| tasks ≥1 pass / 5 of 5 / none (after re-run) | 50 / 17 / 0 | 46 / 22 / 1 | 97 / 43 / 0 | 193 / 82 / 1 |
| broken-ended first attempts | 17 (6.8%): 16 [PAD], 1 empty | 10 (4.3%): 9 [PAD], 1 empty | 39 (8.0%): 36 [PAD], 2 empty, 1 special | 66 (6.8%): 61 [PAD], 4 empty, 1 special |
| re-runs: pass / fail / broken again / infra | 17: 12 / 4 / 1 / 0 | 10: 8 / 1 / 1 / 0 | 39: 25 / 7 / 7 / 0 | **66: 45 / 12 / 9 / 0** |
| broken replies per 1000 (first attempts) | 26/26,664 = 0.98 | 14/30,112 = 0.46 | 50/48,309 = 1.04 | 90/105,085 = 0.86 |
| rollouts with any broken reply (first) | 24 (9.6%) | 14 (6.0%) | 48 (9.9%) | 86 (8.9%) |
| % action turns with reasoning (first) | 56.4 (54.7–58.1) | 53.2 (51.8–54.6) | 54.2 (53.1–55.2) | 54.5 (53.7–55.2) |
| compacted first attempts | 20 (8.0%) | 21 (8.9%) | 23 (4.7%) | 64 (6.6%); 73 of all 1,036 graded |
| pass@1 compacted / not | 60.0 / 68.7 | 66.7 / 77.6 | 82.6 / 76.0 | 70.3 / 74.5 |
| cut at output cap / context overflow (1,036 graded) | 1 / 0 | 0 / 0 | 1 / 0 | 2 / 0 |
| median turns / agent wall (min) | 93 / 57.8 | 120 / 67.3 | 90 / 52.1 | 95 / 57.7 |
| cost / $ per final slot | $1,011.23 / 4.05 | $1,117.46 / 4.76 | $1,827.15 / 3.77 | $3,955.84 / 4.08 |
| SFT-usable passes | 175 | 184 | 386 | **745** |

The task with no pass is rest47 `syn-bitecs-cand-c05-r1b2-1b4c0811` (0/5).

## 3. Broken-reply re-runs (the rule: "这次若遇到被坏回复终止的 slot, 可以重跑1遍")
- **Scope.** 66 re-runs, one for each broken-ended first attempt. They ran in 24 `-brk<n>` campaigns and covered 59 tasks: 52 tasks with one re-run and 7 with two.
- **Outcomes.** 45 pass, 12 fail, 9 broken again, 0 infra.
  - All 66 re-runs replaced their originals, and both records are kept (`first_attempt_records`, `broken_reply_reruns`, `original_attempt`).
  - No slot had a third attempt.
  - The detector's verdict matches driver.py's recorded verdict on all 1,036 graded attempts.
- **The one pass→fail swap.** `baihe-g4 syn-inquirer-js-cand-c03-r1b0-0548d9f2 kimi-r1` passed (F2P 33/33) even though a [PAD] reply ended it. The rule re-ran it, the re-run failed, and the re-run replaced it.
- **Broken again (9, all [PAD], all F2P 0):**
  - sel50 `zsa-c02-r1b1` r0;
  - rest47 `wasmtime-c03-r1b2` r0;
  - baihe:
    - `mathjax-src-c05-r1b0` r4;
    - `remark-lint-c03-r1b4` r1;
    - `go-jsonnet-c05-r1b4` r1;
    - `remark-lint-c01-r1b0` r0;
    - `nova-c02-r1b0` r2;
    - `risor-c04-r1b1` r2;
    - `remark-lint-c03` r1.
- **Some tasks are prone to [PAD].** Re-runs broke again at 9/66 = 13.6%, vs 6.8% on first attempts (p = 0.048).
  - `zsa-c02-r1b1` and `mathjax-src-c05-r1b0` each broke in 3 of 7 rollouts.
  - risor tasks account for 10 of the 66 first-attempt broken endings.
- **Infra.** ATP retried 14 attempts inside their slots: 11 SandboxPtyError, 2 TimeoutError, 1 SandboxApiException. There were 0 hub budget refusals and 0 quota incidents, so no `-rec` campaign was needed.

## 4. Cost (LiteLLM `usage.cost` of v4's own replies; each attempt mapped to its leg's hub key)
| key | total | first attempts | re-runs | ATP-retried attempts | pilots | $ / graded rollout |
|---|---|---|---|---|---|---|
| NVAPI_KEY_NEW_2 | **$3,959.12** | $3,629.40 | $280.40 | $46.04 | $3.27 | 3.82 (1,036 rollouts) |
| NVAPI_KEY_NEW_3 | **$0** (no v4 leg started on it) | – | – | – | – | – |

**Per rollout:**
- per graded rollout: $3.82;
- per final slot (re-runs and retries included): $4.08;
- first attempt: mean $3.74, median $3.07;
- re-run: mean $4.25.

**Volume:** all 115,154 replies carry a cost. They used 12.1B prompt tokens (89.9% cached) and 83.8M completion tokens.

**Reconciliation with the key's spend:**
- NEW_2's spend went from $846.11 (10-28 14:14, the first v4 leg) to $5,210.41 (10-29 21:41), a delta of $4,364.30.
- That delta = v4 $3,959.12 + v5 $390.04 (same key, Oct 29 02:09–10:32, from v5's journals) + $15.14 unexplained (0.35%).
- So progress.py's "$4,364 / $4.21 per rollout" overstates v4.
- NEW_2 has $4,789.59 left.

## 5. SFT-usable passes: 745 of 765 graded passes
**Criteria** (v5's, with one refinement):
- a graded pass with a clean grade: evaluation completed, reward 1, every F2P and P2P test passing, no verifier or agent error, a non-empty patch and an exported session;
- no broken hub reply anywhere in the attempt, sub-sessions included;
- no [PAD] or special-token text in the main session's assistant parts.

No LLM cheating audit was run.

| taskset | tasks | final passes | usable (first / re-run) | usable without compaction | excluded | tasks with ≥1 usable | usable per task (count: tasks) |
|---|---|---|---|---|---|---|---|
| sel50 | 50 | 182 | **175** (163 / 12) | 165 | 7 | 49 | 0:1, 1:5, 2:6, 3:10, 4:12, 5:16 |
| rest47 | 47 | 188 | **184** (176 / 8) | 168 | 4 | 46 | 0:1, 2:6, 3:8, 4:12, 5:20 |
| baihe | 97 | 394 | **386** (361 / 25) | 368 | 9 | 97 | 1:6, 2:6, 3:11, 4:35, 5:39 |
| all | 194 | 764 | **745** (700 / 45) | **701** | 20 | 192 | |

- **Excluded (20).** 19 are final passes that contain a broken reply which did not end the session: either a degenerate reply that still made a tool call, or a sub-session ended by one, so the text sits in the trajectory. The 20th is the replaced broken-ended pass. No pass was excluded for its grade.
- **Refinement vs v5.** rest47 `syn-zsa-cand-c02-r1b3-38afb932 kimi-r2` is **kept**. Its only match is `<|highlight|>` inside zsa's own source file, returned by the `read` tool, and the attempt has 0 broken hub replies. v5's stricter rule would drop it, giving 744.
- **Tasks with no usable pass:**
  - sel50 `syn-gomacro-cand-c01-c696c6ca`: its single pass (1/5) has a [PAD] sub-session. This task is very long: all 5 rollouts compacted, 275 median turns, $9.74 per rollout.
  - rest47 `syn-bitecs-cand-c05-r1b2-1b4c0811`: 0/5 passes.
- **The full list,** with attempt dirs and export paths, is in `analysis/sft-usable-passes.jsonl` (745 lines).

## 6. What I ran and changed
All paths are in `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028`.

1. **`progress.py`:** the final run. It rewrote PROGRESS.md; its last row is 10-29 21:41.
2. **v4 versions of the scripts.** Each was installed by writing a temp file and running `mv -f` over it, with no kr4- job queued or running. The previous files are in `backup-pre-final-20261029/`.
   - **`sources.py`** has two new sources; the existing ones are unchanged.
     - `kimi4-all`: 1,036 graded rollouts (970 first attempts + 66 re-runs), with taskset, role and final flags.
     - `kimi-1m`: the 250 earlier 1M sel50 slots, read-only.
   - **`features.py`** now writes to `analysis/features/`.
     - New fields: broken text, prompt sizes, post-compaction stats.
     - The turn and reasoning code is unchanged; recomputed v3 and 1M rows equal v3's stored rows on every key.
     - It needs a Python with `tokenizers`; I used `july-ladder-runtime/envs/data-cpu-py311/bin/python`.
   - **`degenerate.py`:**
     - `classify()`, `scan()` and the statistics helpers are byte-identical (checked with ast), so `brokenreply.py` and driver.py's rule are unaffected.
     - `brokenreply.py --validate-v3` still gives 74/74, with precision and recall 1.0.
     - New: the v4 / v3 / 1M / ctrl-old arms, per-attempt rows with cost and tokens, and the comparisons.
   - **`phase_report.py`:** rewritten as this final report.
3. **Outputs in `analysis/`:**
   - `final-report.json` and `final-report.md` (script-generated, with a per-task table for all 194 tasks);
   - `sft-usable-passes.jsonl`;
   - `degenerate-v4-final.json` and `degenerate-v4-final-attempts.jsonl`;
   - `features/kimi4-all.jsonl`, `features/kimi3.jsonl` and `features/kimi-1m.jsonl`.

## Notes for main
- **README.md is stale.** It still says degenerate.py is byte-identical to v3's and that phase_report.py is v3-shaped. I did not edit it.
- **PROGRESS.md's cost columns include v5's spend.** Use $3,959.12 and $3.82 per rollout for v4.
- **Rules:**
  - No Slurm job was submitted or touched. The only Slurm commands were one standalone `squeue` and progress.py's own `squeue`.
  - Keys were read by name only (progress.py → key_budget.py); no key value was printed or copied.
  - v3, v5 and the synth runs were only read.
  - No permission check denied a command.
