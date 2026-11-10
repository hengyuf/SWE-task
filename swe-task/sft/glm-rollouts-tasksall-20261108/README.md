# glm-rollouts-tasksall-20261108

Self-hosted GLM-5.3 rollouts, 5 per task, on every task of `sft/tasks-all-20261106` that has no GLM-5.3 rollouts yet,
in priority order.

User request (hengyuf, 2026-11-08, verbatim): "Upstream fix should be fixed. rerun the 3 ungraded slots. Then do the
rollout for GLM 5.3 on all the tasks in  swe-task/sft/tasks-all-20261106/, you don't need to redo the rollout for the
tasks that already has the hosted GLM 5.3 rollouts. Prioritize the task that are accepted or marked as strong (first do
the accepted/strong tasks, then do partial, then weak, then none or without any labels)."

The other two parts of the request: the ATP fix (`atp-fix/`: audited, implemented and verified by workflow
atp-capture-fsync-fix; installed into `swe-task/atp` on 2026-11-08 20:51:21 as tracked hengyuf change item 6 with
`atp-fix/install.py`, which also has `verify` and `rollback`; legs staged before that ran the equivalent staged-copy
patch, reported as "capture patch patched" in their worker.log) and the re-run of the 3 ungraded slots of
`glm-rollouts-sftv3-20261107` (that dir's `rerun_infra.py`, job gr-infra-rerun1).

## Settings (same as glm-rollouts-sftv3-20261107)

| | |
|---|---|
| model | `glmlocal` = GLM-5.3 served by `glm53-serve` (TP8 x PP2 + DCP8 + MTP3, env `vllm-0.33.0-glm53` = 0.33.0 + #60477 backport + MTP/structured-output fix; strict tool calling off), reasoning effort max, temperature 1.0, top_p 0.95 |
| context / output | 262,144 context (OpenCode compacts at 242,144: compaction allowed); 262,144 output limit per reply (vLLM clamps to the context left) |
| prompt | OpenCode 1.17.11's stock system prompt |
| harness | synth `rollouts.probe` -> ATP campaign -> OpenCode in OpenSandbox -> each task's grader; no hub key anywhere |
| sandboxes | hengyuf tasks: `CELL3_SANDBOX_API_KEY` (`S/atp/.env`); Baihe tasks: `BAIHE_SANDBOX_API_KEY` (`glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env`, mode 600) |

## Targets (`make_tasks.py` -> `tasks.json`)

- Inventory: `tasks-all-20261106/all_tasks.jsonl`, 2,203 built tasks. A task is keyed `source/run/cand/task_id`
  because 17 task ids are shared by two builds; no campaign holds two tasks with the same id.
- Excluded because they already have GLM-5.3 rollouts: the 220 SFT-v3 tasks of glm-rollouts-sftv3-20261107
  (self-hosted, 5 each) and the tasks in `coverage.json` (`make_coverage.py`: hosted GLM-5.3 in
  teacher-rollouts-v2-20261025, 50 tasks x 5; 48 of them are among the 220, the other 2 are official1-scriggo gomacro
  cand-c01 and cand-c02-r1b3). `coverage-verification.json`: three independent sweeps of swe-task, glm53-serve, Baihe's
  fork and the archives + an adversarial verifier (workflow glm-coverage-sweep): no other GLM-5.3 agent rollouts on
  synthetic tasks exist (build probes use GLM only as the judge; teacher-deepswe-20261030 is DeepSWE benchmark tasks;
  glm53-serve/e2e test runs cover a subset of the 50).
- Task dir: the accepted copy for accepted tasks (5 of them carry later grading repairs: 4 official1-scriggo gomacro
  tasks and official1-wasmi wasmtime c03-r1b2; the other accepted copies differ from the build dir only in
  provenance.json), the withdrawn copy for the 5 withdrawn tasks, the repaired copy for the 2 rejected official1 tasks
  with a grading repair (official1-ytt cand-c04: repair.json repaired_task_dir; official1-fastapi cand-c05-r1b3: the
  rescore's analyze_task_dir), else the build's task dir. Digests are frozen and re-checked every leg. (tasks.json was
  re-frozen at 19:21 for those 2 repaired dirs; `tasks.json.pre-repair-dirs` is the first freeze, same campaigns.)
- Graders: 1,087 tasks (all 536 Baihe tasks, 547 official1 build dirs, 4 val1) still carry the OLD DeepSWE grader
  (tests/grader.py 13,468 bytes: no restore of model-edited tests, no removal of model-added test files); 894 use the
  current verifier grader. Decision (2026-11-08): run them as they are -- the same grader produced these tasks' build
  probe grades (sc14 / Kimi), and Baihe's tasks in glm-rollouts-sftv3 were graded the same way -- and re-grade GLM
  passes on old-grader tasks with the current verifier before they enter an SFT corpus (tier 1: 9 such tasks, tier 2:
  104, tier 3: 112, tier 4: 862).
- Snapshots: `snapshot_check.py` -> `snapshots.json`: all 2,203 OpenSandbox snapshots exist (Ready) on 2026-11-08,
  including the rejected tasks'.
- Tiers: 1 accepted or failure-match audit strong; 2 audit partial; 3 weak; 4 none or not audited (every non-accepted
  Baihe task: Baihe never ran the audit).
- Campaigns: per tier and owner, <= 25 tasks x 5 rollouts; order tier-major, hengyuf and Baihe interleaved; max
  concurrency 24 (hengyuf) / 12 (Baihe) per campaign. Names `t<tier>-<h|b><nn>`.

## How it runs

- GPU servers: 2-node `glm53-serve` jobs named `glm53-serve-ta-NN` (batch, 4 h) and `glm53-serve-ta-sNN`
  (batch_short, 2 h), submitted with `glm53-serve/scripts/submit_serve.sh`; self-healing start (in-job vLLM restarts),
  stall watchdog, node-local weights deleted on exit.
- `keeper.py` (job `gta-keeper`, `submit_keeper.sh`; `keeper-config.json` re-read every minute): keeps
  `servers-target.txt` main slots + `short_target` batch_short slots alive (replacement `lead_minutes` before the end),
  releases campaigns (`GO-<campaign>`) in tier order while the active campaigns' load (min(concurrency, unfinished
  slots)) fits `per_server` x ready servers and the owner caps (hengyuf 240, Baihe 48 sandboxes), and keeps one driver
  leg per active campaign (`bash submit.sh <campaign>`, job `gta-<campaign>`, at most `max_legs` per campaign). It stops
  when every campaign has its results file (and then cancels the servers: `cancel_running_when_done`).
- Each leg (`driver.sbatch`, CPU job, babysit pair) runs `router.py` (127.0.0.1; backends = ready `glm53-serve-ta-*`
  jobs; tool-call-id session affinity; least-loaded placement with a soft cap of 30; drain < 45 min left; failover) and
  `driver.py --campaign <name>`. Slots ended by a sandbox-side failure are re-run once in `<campaign>-rec<n>`.
- Results: `results/glmlocal-<campaign>.json` (records, first_attempt_records, sandbox_failure_reruns, stats,
  task_keys). Progress: `progress.py` -> `PROGRESS.md` / `progress.json`.

- Keeper safety: child jobs get a scrubbed environment (no BABYSIT_* of the keeper's own babysit job: they made the
  helper reuse the keeper's leg number; no resource-shaping SLURM_*); every leg uses a fresh babysit state dir
  (`babysit/<campaign>/kNN`); a campaign whose legs hit `max_legs` / `max_submit_failures` or that has
  `babysit/<campaign>/STOP` is PARKED (holds no load; `keeper-status.json` campaigns.parked); when `breaker_parked` (3)
  campaigns got parked within `breaker_hours` (2) no new campaign is released; user-held leftovers of half-failed
  submissions are cancelled after 15 min; at most `max_servers` server slots at once (pending + running; 18 since
  2026-11-09 00:10 so that all 8 main-pool replacements can queue ~2.5 h ahead -- the batch queue waits hours -- while
  the old slots still run; when they overlap up to ~18 servers run, more than the ~12 servers' worth of load the owner
  caps allow, which only lowers per-server load); the
  keeper chains its successor (`--dependency=afterany`) 30 min before its 23 h limit.
- Server turnover: main-pool slots are 4 h batch jobs and the batch queue waited 3-5 h on 2026-11-08/09, so since
  2026-11-09 03:05 `lead_minutes` is 230 (a replacement is queued ~10 min after a server starts). Before that (lead
  150) the first 8 servers expired at 03:10-03:14 with their replacements still queued (estimate 04:14): ~1 h on 2
  servers for ~230 sessions (slow, not failed).
- Release order is tier-major. A campaign blocked only by its owner's sandbox cap is skipped, so when hengyuf's 240
  in-flight cap is full, Baihe's next (tier-4) campaigns may use spare servers ahead of hengyuf's tier-2/3 ones.

## Incidents

- 2026-11-08 23:22 - 2026-11-09 00:05: GitHub release downloads (OpenCode 1.17.11 and ripgrep 15.1.0, fetched by
  ATP staging in EVERY leg) answered HTTP 504 on and off (each asset in turn); t2-h00..h05 legs died in staging (2-4
  keeper legs each; max_legs raised to 8). Fixes, all local to this campaign (the shared ATP app is unchanged):
  * `release-cache/`: the two release tarballs, read-only, sha256-verified (`release-cache.json`): ripgrep's equals
    the release's own .sha256 asset; both are byte-identical to what job 21548031 (t1-h00) downloaded from GitHub
    at 20:42, and their binaries equal that job's staged bin/opencode and bin/rg.
  * `stage_wrapper.py`: driver.py makes rollouts.probe() run ATP's job-local deepswe_stage.py through it; it serves
    exactly those two URLs from the cache (sha256 re-checked; any mismatch falls back to GitHub) and starts the inner
    CPython-3.13 staging run through itself too; nothing else changes. stage.log shows a
    "glm-rollouts-tasksall stage_wrapper: <url> from the verified cache" line per asset. Live from the t2 legs
    started at 00:01-00:05.
  * driver.py also retries a network-caused staging failure inside the leg (up to 6 tries, 5 min apart, fresh
    job-local ATP scratch) instead of burning keeper legs.

- 2026-11-09 03:10-04:50: the first 8 batch servers expired before their replacements started (2 servers for ~230
  sessions). Rollouts slowed down and 62 of the 66 agent timeouts (ATP's 3 h limit) up to 09:00 overlapped that window
  (otherwise ~1 per 2-3 h). `rerun_dip.py` (job gta-dip1, `rerun_dip.sbatch`) re-runs those slots once, in rounds as
  their campaigns finish (round 1, 09:20: 21 slots of the 6 finished campaigns; run it again for the rest);
  `results/glmlocal-dip1.json` maps each re-run to its slot and `final_report.py` applies them (originals kept).

- 2026-11-09 12:27-12:30: all 7 servers reached their 4 h limit while the 8 replacements (queued since ~08:30) were
  estimated to start only at 16:23 (batch priority ~76k vs ~100k for others): a ~4 h outage. The 13 running legs and
  gta-dip1 were cancelled at 12:24 (in-flight attempts become retryable for the next leg instead of burning the 3 h
  agent timeout). keeper.py now does this itself: no ready server for `outage_minutes` (10) -> no releases and no new
  legs, and legs that got past their backend wait ("probe: staging ATP" in their log) are cancelled once; legs still
  waiting for a backend are left alone and start when a server is back. gta-dip1 was resubmitted (it resumes its
  round: dip1r1-h-n1 done, n2/n3 pending).

## Operate

- Scale servers: edit `servers-target.txt` (batch pool) or `short_target` in `keeper-config.json`.
- Stop releasing: `"release": false` in `keeper-config.json`. Stop one campaign: `touch babysit/<campaign>/STOP` and
  scancel its `gta-<campaign>` job (it then counts as parked). Un-park: remove the STOP file, or raise `max_legs` /
  `max_submit_failures` (all parked campaigns come back at once). Stop everything: `touch keeper.STOP` (the keeper
  cancels the PENDING server slots and exits), then scancel the `gta-*` and `glm53-serve-ta-*` jobs.
- Reload keeper code: scancel the `gta-keeper` job and `bash submit_keeper.sh` (state lives in files).
- Do not archive or clean `glm-rollouts-sftv3-20261107` before this campaign ends: Baihe legs read their sandbox
  credential from its `secrets/baihe-opensandbox.env`.
- A leg that keeps exiting in ATP recovery (a sandbox that no longer answers): `sandbox_delete.py --sandbox
  hengyuf|baihe --id <uuid>` (see glm-rollouts-sftv3-20261107/README.md, incident 17:23).
