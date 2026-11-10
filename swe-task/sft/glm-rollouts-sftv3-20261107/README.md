# glm-rollouts-sftv3-20261107

Self-hosted GLM-5.3 rollouts on the SFT-v3 tasks: 5 per task, 220 tasks = 1,100 rollouts.

User request (hengyuf, 2026-11-07, verbatim): "I have a v3 SFT dataset collected from these tasks. Now use GLM to rollout
this task (5 each task). You can launch a bunch of workers (2node job) to raise the concurrency (each rollout set a 256k
output limit, and allow compaction)."

## Settings

| | |
|---|---|
| tasks | `tasks.json` (`make_tasks.py`): every task with >= 1 conversation in `sft/corpus/sft-v3-20261029` = 220 (hengyuf 124 = 91 v2 + 33 official2 on `CELL3_SANDBOX_API_KEY`; Baihe 96 v2 on `BAIHE_SANDBOX_API_KEY`). Not included: the 7 v2/v5 tasks without any SFT-v3 conversation (5 gomacro, nova-c04-r1b0, wasmtime-c03-r1b2). Digests re-checked every leg |
| model | `glmlocal` = GLM-5.3 served here (`glm53-serve`, TP8 x PP2 + DCP8 + MTP3, env `vllm-0.33.0-pr60477`, strict tool calling off), reasoning effort max, temperature 1.0, top_p 0.95 (as the hosted GLM runs) |
| context | 262,144 (OpenCode `limit.context = limit.input`); compaction allowed: OpenCode 1.17.11 compacts at `limit.input - min(20,000, output)` = 242,144 (`analysis/opencode-src/overflow-v1.17.11.ts`) |
| output limit | 262,144 per reply (OpenCode `limit.output` and the request `max_tokens`; vLLM clamps it to the context left) |
| prompt | OpenCode's stock system prompt (as kimi-rollouts-v4) |
| harness | synth `rollouts.probe` -> ATP campaign -> OpenCode in OpenSandbox -> each task's grader |

## How it runs

- `driver.sbatch` (CPU job per campaign leg, schedule-helper babysit pair via `submit.sh`) starts `router.py` on
  127.0.0.1 and `driver.py`.
- `router.py` spreads the leg's model calls over the GPU servers: RUNNING jobs named `glm53-serve-roll-*` whose
  `glm53-serve/logs/serve/<job>.status` says ready, plus `backends-extra.txt`. It pins sessions by tool-call ids
  (prefix cache), places new sessions on the least-loaded server (soft cap 30 running+waiting, held <= 8 min), drains
  servers with < 45 min left, re-sends failed calls to another server, and answers 429 only when no server exists.
- `driver.py` waits for `GO` or `GO-<campaign>`, then for a healthy server, then runs the ATP campaign. Slots whose
  OpenCode run was ended by a sandbox-side failure (`opencode_finished` false, `opencode_run_error_type` ~ `Sandbox`) are
  re-run once in `<campaign>-rec<n>`; the re-run replaces the slot (provenance in the results file).
- `keeper.py` (job `gr-keeper`, `keeper-config.json` re-read every minute) keeps the servers alive: pool `main`
  = `roll-NN` on batch (4 h, target `servers-target.txt`, replacement 90 min before the end) and pool `short` =
  `roll-sNN` on batch_short (2 h, target 2). It releases campaigns (`GO-<campaign>`) in the order hengyuf-g0, baihe-g0,
  hengyuf-g1, ... while released concurrency <= 24 x ready servers. It never cancels queued servers unless
  `scale_down` is true. It stops (and cancels pending servers) when `keeper.STOP` exists or all 8 results exist.
- Campaigns: `hengyuf-g0..3` (31 tasks x 5, 24 in flight), `baihe-g0..3` (24 x 5, 12 in flight).
- Progress: `progress.py` -> `PROGRESS.md` (graded = finished rollouts; sandbox-failed slots counted apart).
- Results: `results/glmlocal-<campaign>.json` (records, first_attempt_records, sandbox_failure_reruns, stats).

## Operate

- Resume a campaign: `bash submit.sh hengyuf|baihe GROUP` (same campaign dir; ATP resumes; pending rec re-runs too).
- Scale servers: edit `servers-target.txt` (main pool) or `short_target` in `keeper-config.json`.
- Stop everything: `touch keeper.STOP`, `touch babysit/<campaign>/STOP`, then scancel the `gr-*` and
  `glm53-serve-roll-*` jobs.

## Incidents (2026-11-07)

- GPU queue: batch 2-node start estimates moved from 12:30 to 19:27 during the morning; batch_short (max 2 servers per
  user) gave the first servers. The first campaign started 10:56 on one server.
- An early keeper version cancelled pending `roll-01` / `roll-02` (its scale-down counted manually added batch_short
  slots). Fixed (pools, opt-in scale-down); `roll-09` / `roll-10` replaced them.
- `roll-s01` hung in CUDA-graph capture ("direct DCP q-gather multimem timeout"). The glm53-serve rank script now
  restarts vLLM inside the job on startup hangs (at most 3 starts; every curl under `timeout` after a curl ignored -m 5
  for 12 min), and fails the job on token stalls (10 min).
- `roll-s03`, `roll-s04`, `roll-s05` (submitted by the keeper from inside its 1-CPU / 2-GB CPU job) inherited
  SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK through --export=ALL, so their engine steps ran with 1 CPU and 2 GB per node:
  30-min staging, 40 s per weight shard, failed or stuck starts (it looked like slow Lustre or bad nodes; it was not).
  Fixed in serve_glm53_dfw.sbatch (unset + srun --mem=0 --cpus-per-task) and keeper.py (drops those variables). From
  12:47 to the next server, hengyuf-g0's sessions waited on 429s.
- SandboxPtyError ("PTY connection closed unexpectedly", close code 1006/None) ended 330 of 748 terminal slots by 17:04,
  in bursts of 3-32 sessions per campaign process (kimi-rollouts-v4 with 8 sessions per process: 1%). Root cause: ATP's
  capture layer (`atp/scripts/deepswe_capture.py`, `run_opencode.drain`) writes + `os.fsync`s every OpenCode output chunk
  to the attempt dir on Lustre INSIDE the Gym agent event loop; a stalled Lustre fsync blocks the loop, aiohttp misses
  the PTY websocket heartbeat pongs (30 s heartbeat) and closes every PTY of that process. Fix (17:12): `capture_patch.py`,
  applied by `probe_worker.py` to each leg's STAGED job-local copy only (scratch/scripts/deepswe_capture.py; the shared
  ATP tree and the campaigns' verified source snapshots are untouched), runs the same write + fsync in a worker thread.
  All 8 campaign drivers were restarted at 17:12 to load it (~111 in-flight sessions became retryable attempts).
  ATP grades such slots' half-done workspaces as fails; `driver.py` re-runs every sandbox-failed slot once
  (`<campaign>-rec<n>`, the re-run replaces the slot). The first 7 (11:55) and 15 hengyuf-g0 slots idle through the
  12:47-13:30 server outage were of the same kind.
- 17:23: baihe-g0's leg exited without running ("probe incomplete, worker exit 0"): ATP's fail-closed recovery could
  not stop sandbox 30a80753-... (attempt 20-0; every OpenSandbox command to it timed out after 240 s since ~16:40), so
  `recover_model` gave up after 3 tries and the whole shard stopped. Fix: `sandbox_delete.py --sandbox baihe --id <id>`
  (lifecycle DELETE /v1/sandboxes/<id> with the campaign's own credential; HTTP 204) during the next leg's recovery;
  recovery-1 then recovered it and the leg ran (18:53). Use the same tool if a campaign leg keeps exiting in recovery
  (`models/glmlocal/shards/0/launch-*/recovery-*.json` lists the stuck attempt; its `agent-sandbox.json` has the id).

## Final results (2026-11-07 23:44; `analysis/FINAL-REPORT.md`, `analysis/final-summary.json`, `analysis/final-rollouts.jsonl`)

- 1,100 rollouts (220 tasks x 5): **882 pass / 215 fail / 3 infra = 80.4% of graded**. Per task: 109 tasks 5/5, 54 4/5,
  27 3/5, 16 2/5, 8 1/5, 6 0/5.
- Same 220 tasks, Kimi-K3 (v4/v5 final records): 80.1%. Hosted GLM-5.3 (teacher-rollouts-v2, 1M window) on the 48
  overlapping tasks: hosted 77.6% vs self-hosted 78.3%.
- OpenCode compaction in 397 rollouts (307 pass); 575 passes without compaction (the SFT-v3 rule). No reply reached
  131,072 tokens; 260 rollouts had a reply >= 32k. Median 140 LLM calls and 125k generated tokens per rollout.
- 469 slots finished on an ATP retry (attempt-1/2): the pre-fix PTY drops and the 17:12 driver restart. Only 1 slot
  needed the driver's end-of-campaign sandbox re-run (pass).
- 3 infra (ATP generation-validity "unknown", attempt-1): hengyuf-g0 litestar-c01-r1b1 r1, yq-c05 r0; hengyuf-g3
  goleveldb-c01-r1b1 r1. RE-RUN 2026-11-08 (user: "rerun the 3 ungraded slots"): `rerun_infra.py` /
  `rerun_infra.sbatch` (job 21541211, router on the glm53-serve-ta-* servers of glm-rollouts-tasksall-20261108, i.e.
  env vllm-0.33.0-glm53 = pr60477 + the MTP/structured-output fix; same model spec, stock prompt, CELL3 sandbox; the
  staged-copy capture patch) in campaign `probes/glmlocal/infra-rerun1-hengyuf-n1`, 20:43-21:54: **all 3 pass**
  (goleveldb f2p 11/11, litestar 32/32, yq 6/6). `results/glmlocal-infra-rerun1.json` maps each re-run to its slot;
  the campaign results files are unchanged and `final_report.py` applies the re-runs (originals kept as
  replaced_attempt). Updated totals: **885 pass / 215 fail / 0 infra = 80.5%** (Kimi K3 same tasks 80.1%); the
  pre-re-run report is kept as `analysis/*.pre-infra-rerun.*`.
- Fleet stopped at 23:45 (keeper.STOP; last 5 servers cancelled; their node-local weight copies deleted).
