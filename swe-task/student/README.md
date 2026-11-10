# Student inference workers (sc14) and their maintainer

Persistent vLLM workers for the synthetic-task pipeline's student, the local checkpoint
`swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (alias `sc14`, served as `policy`). Each worker is one 4 h
`batch` job on one 8-GPU node running one TP8 replica. A maintainer (`fleet.py daemon`, itself a CPU job) keeps 10 of
them covered on `batch` by resubmitting; a worker submitted by hand on another partition (e.g. w99 on `interactive`)
is extra capacity. Workers publish themselves in `registry/`; one router per synth job (`router.py`, started by
`client.py`) sends calls only to workers the registry says are routable.

Other users who want to call these workers from their own jobs: see [USAGE.md](USAGE.md).

```
student/
├── worker.sbatch      one worker job: resolves the job's start/end, then srun into the production container
├── serve_student.sh   the worker itself (in the container): vLLM + registry record + watchdogs + clean stop
├── fleet.py           maintainer CLI (stdlib only, /usr/bin/python3): status | reconcile | daemon | submit | stop | ...
├── fleetd.sbatch      CPU job running `fleet.py daemon` (cpu_datamover via fleet.json, 7 days, keeps a successor queued)
├── fleet.json         maintainer config (re-read every pass)
├── router.py          the per-synth-job student endpoint in front of the workers (aiohttp, ATP controller python)
├── client.py          ensure_router / wait_for_workers and the registry routing rules (stdlib only)
├── registry/          <job_id>.json, written only by worker jobs; archive/ = records of jobs that ended
├── state/             events.jsonl, status.json, jobs.json, waits.json, excluded_nodes.jsonl, breaker.json, daemon.json,
│                      locks, STOP
├── logs/              swe-task-student-wNN-<job>.out (worker), vllm-<job>.log, vllm-<job>.cmd.txt, parity-<job>.{txt,json},
│                      hang-<job>.txt (startup hangs only), swe-task-student-fleetd-<job>.out (daemon)
├── cache/             compile/<node>/ (vLLM, Triton, Inductor, FlashInfer caches per node), hf_home/
└── tests/             test_fleet.py (maintainer, worker), test_router.py (router, client)
```

## Commands

All commands run from any login or compute node. `F=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/fleet.py`.

| Command | What it does |
|---|---|
| `python3 -B $F status` (`--json`) | Read-only table: every fleet job (squeue state, node, elapsed, time left, pending reason) joined with its registry record (status, ready time, heartbeat age, base URL); counts, coverage, effective horizon, routable workers, pending reasons, daemon heartbeat, breaker, excluded nodes. |
| `python3 -B $F start-daemon` | Submits `fleetd.sbatch` unless a daemon job is already pending or running. Clears the STOP flag. `--partition/--time` override `daemon_partition/daemon_time` for this submission only. |
| `python3 -B $F stop` | Lists the fleet jobs (workers + daemon). With `--yes`: sets the STOP flag, cancels the daemon job(s) first, then all workers. Never touches anything else. |
| `python3 -B $F submit --slot NN [--partition P] [--time T] [--account A]` | One worker in slot NN (manual / validation; default account: the first worker account). `--dry-run` prints the sbatch argv; `--test-only` runs `sbatch --test-only` (validates, submits nothing). A pending batch job with the standard walltime gets its sibling on the other worker account from the daemon once it queued `pair_delay_min` (pair completion). |
| `python3 -B $F reconcile [--dry-run]` | One maintenance pass by hand (`--dry-run`: prints the plan, writes nothing). |
| `python3 -B $F daemon [--interval 60] [--until EPOCH] [--wait-lock]` | The maintenance loop (what fleetd.sbatch runs). Only one instance runs at a time (cross-host flock `state/fleet.lock`). |
| `python3 -B $F unexclude NODE...` / `reset-breaker` | Undo a node exclusion / close the circuit breaker. |
| `python3 -B $F worker-accounts [ACCOUNT... \| --clear]` | Show or set fleet.json `worker_accounts` (under the pass lock; the daemon reads it on its next pass). `--clear` = single-account on `account`. See *Worker accounts*; `status` adds `(daemon's last pass: ...)` while the daemon's last completed pass used other accounts than fleet.json says. |
| `touch state/RELOAD` | The running daemon exits with status 3 at its next loop turn (it removes the flag); its batch loop restarts it 60 s later with the fleet.py on disk. A flag older than the daemon is only removed. Needs a daemon from 2026-11-02 or later. |

**Drain on demand (before any early scancel of a serving worker).** `touch registry/<job_id>.drain`: on its next
heartbeat (<= 30 s) the worker sets `drain_at` to now and publishes `draining`, so routers stop sending it new calls
and move its conversations (the worker stays the only writer of its record). Then wait until its vLLM `/metrics` shows
0 running and 0 waiting (at most 90 s), `scancel` it, and delete the flag. Workers started before 2026-10-20 03:18 PDT
run the previous serve_student.sh (backup `serve_student.sh.pre-drain-flag`) and ignore the flag. A plain scancel of a
busy worker aborts its in-flight calls (`finish_reason: abort`), which the Gym agent turns into a lost rollout.
Routers started after the load-aware change (2026-10-20, see Router) also read the flag themselves: from their next
registry re-read (<= 10 s) the flagged worker gets calls only when every other routable worker is leaving too, even
before the worker publishes `draining` (a `draining` worker otherwise takes calls once the ready ones are saturated).
They also retry aborted buffered calls on another worker, so a cut call costs time, not the rollout.

Logs: `tail -f logs/swe-task-student-fleetd-*.out` (daemon events), `state/events.jsonl` (every action;
`"incident": true` marks what needs a human), `logs/swe-task-student-wNN-<job>.out` (worker), `logs/vllm-<job>.log`.

## How it works

**Worker** (`worker.sbatch` → `serve_student.sh`). The batch script is frozen at submit time, so it only reads the job's
start/end (`SLURM_JOB_START_TIME`, `SLURM_JOB_END_TIME`, else `squeue %e`) and execs
`srun --container-image=<RL image> --container-mounts=/lustre/fsw:/lustre/fsw,<RL source>:/opt/nemo-rl:ro`.
Everything else is in `serve_student.sh`, which is read when the job starts, so a fix also reaches jobs that are
already queued. The worker:

1. writes its registry record with status `starting` (heartbeat every 30 s during startup);
2. starts vLLM in the background (own process group, per-job log `logs/vllm-<job>.log`, appended);
3. waits for `GET /v1/models` = 200. Two watchdogs apply. The **progress watchdog**: no `Loading weights took` within
   600 s of the vLLM launch (normally ~200 s) is the known init hang (7 GPUs busy-waiting in NCCL, 1 idle).
   The **readiness timeout**: not ready 1200 s after the job start (normally 444 s median, 526 s max). Either way the
   worker first writes `failed` with `exclude_node: true` (routers and the maintainer see it at once), then saves
   evidence (`logs/hang-<job>.txt`: nvidia-smi, `ps` of the vLLM processes with %CPU and wchan, py-spy if ptrace is
   allowed, the log tail; bounded: at most 16 processes, 10 s of py-spy each), and exits 4. If vLLM exits during
   startup, the last error line becomes the reason;
4. runs `EVAL/tools/check_serving_parity.py` on its log (read-only reuse) → `logs/parity-<job>.{txt,json}` and
   `parity` in the record. A mismatch is recorded, never fatal;
5. serves. Every 30 s it calls `GET /health` (20 s timeout). A passing check refreshes `heartbeat_at` and the status:
   `ready`, or `draining` once inside the last 45 min of the job. 4 consecutive failures, or vLLM exiting, set
   `unhealthy` (exit 6). Every registry write attempt gets 15 s (`STUDENT_REG_TIMEOUT_S`) and a failed one is
   retried once (since 2026-10-20 15:53; before: one attempt of up to 60 s);
6. stops itself 30 s before the walltime (`stopped`, reason `walltime`). SIGTERM (scancel or walltime) also gives
   `stopped`. The cleanup first writes `stopped` (routers stop picking the worker before vLLM goes away), then sends
   TERM to the vLLM process group, then KILL. After that it SIGKILLs any process still carrying this job's
   `STUDENT_WORKER_TAG` environment marker (orphaned `VLLM::Worker_TP*`). The match is on the process environment, so
   it can never hit the worker's own shell or another job.

**Maintainer** (`fleet.py`, one pass per minute):

1. *Ended jobs* (gone from squeue, end confirmed by sacct): an `AdminComment` containing "occupied idle job reaper"
   is logged as an `incident` (not a worker failure). A worker failure (registry `failed`/`unhealthy`, a job that
   ended while `starting`, a failed job without any registry record, NODE_FAIL) excludes the node. The registry record
   moves to `registry/archive/`. A job missing from squeue that sacct does not show ended keeps counting with its last
   known state for up to 30 min, so a glitchy or empty squeue never causes a mass resubmission.
2. *Running jobs*: a worker still running more than 5 min after it wrote `failed`/`unhealthy`/`stopped` → scancel.
   No ready record 25 min after the job start (`stuck_startup_min`) → scancel and exclude the node. A `ready`/
   `draining` record whose heartbeat is older than 10 min → scancel (no exclusion), unless two or more and over half of
   the serving workers are stale at once (that is Lustre or a clock problem, logged as an incident instead). A requeued
   job is judged by its current run only: a registry record written before its current start is ignored (the job
   counts as starting) and its per-run flags start over. A pass that cannot list `registry/` (missing directory,
   permissions, a Lustre error) does nothing at all and the daemon logs `pass_skipped` (at most every 10 min): read
   as empty, it would make every running worker a stuck startup, i.e. cancel the whole fleet, exclude its nodes and
   open the breaker (until 2026-11-02 an unlistable directory read as empty). One unreadable record file only costs
   its own entry.
3. *Coverage* = RUNNING workers in `coverage_partitions` (`batch`) with more than the horizon left + PENDING workers
   there (not held; a "launch failed requeued held" job is held; the candidates of an account pair count once, and
   not at all while one of them runs; any other same-name duplicate counts by itself). Workers on other partitions (w99 on `interactive`)
   are extra capacity. The **horizon** is `horizon_min` (90 min), raised by `adaptive_horizon` to the expected queue
   wait + `startup_s` (600 s) + `horizon_margin_min` (20 min), at most `horizon_max_min` (200 min) and never more
   than the walltime - `stagger_min` - 10 min (`horizon_cap_s`; above that no running worker ever counts and the
   daemon tops up to `max_jobs`: 2026-10-17, 270 min with 4 h workers held 24 nodes). The expected wait
   is the `wait_quantile` (0.9) of the fleet's observed submit→start waits of the last `wait_window_h` (12 h; first
   runs only, in `state/waits.json`; needs ≥ 3), raised to the median squeue start estimate (`%S`) of the fleet
   workers pending there; without either the horizon stays 90 min. While coverage < `target` (10), it submits new
   workers named `swe-task-student-wNN` (lowest free 2-digit slot). There are at most `max_jobs` (24) fleet jobs in the
   queue (a pending pair counts once), and at most `max_submits_per_pass` (10) slots per pass get a job (new slots
   and pair completions alike: one sbatch each with two accounts). When the wait matches the estimate, a
   replacement is ready at least ~20 min before the old worker ends (the old one drains for its last 45 min).
4. Writes `state/status.json` (counts, rows, effective horizon, `last_reconcile_at`), `state/waits.json`, and appends
   every action to `state/events.jsonl`.

Exclusions are written to `fleet.json` `exclude_nodes` (the reason goes to `state/excluded_nodes.jsonl`) and passed
to every new job as `--exclude`. A new exclusion is also applied to the pending fleet workers (`scontrol update
JobId=<id> ExcNodeList=...`): they were submitted with the old list, and a node that fails fast would otherwise take
them one after the other. fleet.json is only rewritten when it parses (else an `exclusion_not_saved` incident). The
list starts from the RL launcher's (pool0-00058, -00484, -00502, -01460, -00613). The **circuit breaker** opens after
worker failures on 4 distinct nodes (`breaker_failures`) within 120 min (`breaker_window_min`); failures on one node
only exclude that node. A systematic fault (bad image, model path, broken env) would otherwise exclude every node it
touches and keep burning node-hours. While open: no submissions and no new exclusions. After `breaker_cooldown_min`
(60 min) it is half-open: one probe worker is submitted, and the breaker closes once that worker serves; a failed
probe doubles the cooldown (at most 4 h). `fleet.py reset-breaker` closes it by hand.

**Safety rules in the code.** fleet.py cancels only jobs of `user` whose name is exactly `job_prefix` + 2 digits or
equals `daemon_job_name`, always by explicit job id, and verifies the name and user against squeue first (the tests
cover this; another job of the user that merely shares the prefix, e.g. `swe-task-student-webui`, is never touched).
Every Slurm call is an argv list (no shell) with a scrubbed environment: PATH, HOME, USER, LOGNAME, LANG,
SLURM_CONF. This matters because sbatch always exports the caller's `SLURM_*` variables: a daemon running inside a
job would otherwise leak its own allocation into the workers, and `SQUEUE_*`/`SCANCEL_*`/`SLURM_TIME_FORMAT` would
change what the commands do. Times are read as epoch seconds (`SLURM_TIME_FORMAT=%s`). Submissions use
`--export=NONE` (the workers are self-contained, absolute paths only); `worker.sbatch` sets `SLURM_EXPORT_ENV=ALL`
and passes `srun --export=ALL`, so the step still gets the batch environment.

**Daemon continuity.** `fleetd.sbatch` (submitted on `cpu_datamover` = fleet.json `daemon_partition`, whose value
overrides the script's `cpu_long` default; 2 CPUs, 4 GB, 7 days, `--dependency=singleton`) first queues its successor
(`start-daemon --successor`, which ignores its own job and daemon jobs no longer pending or running, e.g.
COMPLETING). The successor starts only after this job has ended, including when this job dies early. It then runs the
daemon until 5 min before its walltime, and restarts it after 60 s if it exits abnormally. The daemon has a watchdog:
a pass wedged for 15 min (hung Lustre or Slurm call) exits the process, and the batch script restarts it. The daemon
never dies on a failed pass. On a Slurm or config error the pass is skipped and logged at most every 10 min.

## Serving configuration (identical to the DeepSWE eval)

`serve_student.sh` uses `EVAL/scripts/serve_vllm.sh`'s engine arguments verbatim (RL r2 `policy.generation`: vLLM
0.29, TP 8, max_model_len 262144, max_num_seqs 64, max_num_batched_tokens 8192, gpu_memory_utilization 0.75, prefix
caching, mamba align/float32, moe triton, processed_logprobs, PIECEWISE cudagraphs [1..64], nano_v3 reasoning parser
plugin, qwen3_coder tool parser, auto tool choice, prompt-token details, `--host 0.0.0.0 --port 8000`). It also uses
serve_vllm.sh's runtime environment (`VLLM_USE_V2_MODEL_RUNNER=0`, `NRL_SC_IID_LOGPROBS=1`, spawn, insecure
serialization, OMP 8, NCCL WARN, the same LD_LIBRARY_PATH and PATH) and the same container, mounts and srun flags as
`eval.sbatch` (minus the Gym-only provider overlay). `tests/test_fleet.py` diffs the argv against serve_vllm.sh
token by token, and the runtime env against what serve_vllm.sh exports. Every start is also diffed against the RL
engine log (`parity` in the registry).

The differences below don't change what is served:

- per-job log instead of `server.log` truncated by each start;
- `--seed` = job id mod 2^31 instead of 0 for every replica;
- compile caches in `cache/compile/<node>` and `HF_HOME` in `cache/hf_home`, because the eval's and the RL runtime's
  directories are read-only for this project;
- `PYTHONDONTWRITEBYTECODE=1`, for the same reason;
- the `STUDENT_WORKER_TAG` marker variable.

The per-call output cap (131072) and the context-overflow retry are the router's job (contract). The record carries
`per_call_cap` and `max_model_len` for it.

## Registry (what routers read)

`registry/<job_id>.json`, written only by the worker job. Every write is atomic: a temp file `.<job>.json.<pid>.tmp`
in the same directory, then `os.replace`. The schema is the contract's:

```
job_id, slot ("07"), node, ip, port (8000), base_url ("http://<ip>:8000/v1"), served_model_name ("policy"), model_path,
max_model_len (262144), per_call_cap (131072), seed, started_at (job start), ready_at|null, end_epoch, drain_at
(end - 45 min), heartbeat_at, status (starting|ready|draining|unhealthy|stopped|failed), reason|null
+ extras: parity (PASS|FAIL|ERROR|null), kv_cache_tokens (7,156,630 expected), partition, vllm_log,
  exclude_node (only with failed/unhealthy: whether the node is suspect)
```

`ip` is `socket.gethostbyname(socket.gethostname())`, as Gym advertises its servers (falls back to the outbound
interface address if that resolves to loopback). A worker is routable iff `status` ∈ {ready, draining}, `heartbeat_at`
is at most 120 s old, and the job is not past `end_epoch`. While serving, `heartbeat_at` only advances on a passing
`/health`, so a wedged worker drops out of routing within about 2 min, even before it declares itself `unhealthy`.
Routers started after 2026-10-20 15:53 also keep a worker whose `heartbeat_at` went stale while their own
`GET /health` of it keeps passing (at most 60 s since the last pass, `client.HEALTH_GRACE_S`): the registry is on
Lustre, and a stall can hang every worker's writes at once while vLLM serves (see the 2026-10-20 heartbeat section).
Routers rank `ready` before `draining` (last 45 min of the job) among the workers with free capacity, so a draining
worker gets calls only when no ready worker has room (load-aware routing since 2026-10-20, see Router). Known
behavior: this moves a draining worker's continuing conversations too, not only new ones (the next worker recomputes
their prefix). Records of ended jobs are moved to `registry/archive/` by the maintainer.

## Router (one per synth job)

`client.ensure_router()` starts `router.py` once per process tree on 127.0.0.1 and exports its URL as
`SYNTH_STUDENT_BASE_URL`, which ATP's Gym model server and `llm_engine.chat()` call. In a synth job the first probe
that needs the student starts it (logs in that probe's directory); later probes of the same build process reuse it
while it answers, and it stops with that process, so a router change reaches a build when its build process (re)starts
(a relaunch inside the same Slurm job counts). The router process gets no variable whose
name contains KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL and never forwards client headers.

- **Semantics** of the DeepSWE eval model server with the 131k cap: model `policy`, temperature/top_p 1.0,
  `enable_thinking: true` / `truncate_history_thinking: false`, `max_tokens := min(requested, 131072)`, and on a 400
  "maximum context length" an exact clamp to 262144 - prompt tokens (worker `/tokenize`). History thinking of ATP
  probes arrives inside the assistant content as `<think>R</think>C` (ATP `build_config` gives the local spec the
  eval's `preserve_reasoning_in_assistant_content`); dropping empty reasoning fields and renaming `reasoning_content`
  to `reasoning` only matters for callers that send reasoning fields.
- **Affinity**: rendezvous hashing of `X-Synth-Affinity`, else of the conversation's opening: first system + first
  user message, plus the first assistant turn once there is one. Turn 1 of every repeat of a task shares one
  worker's prefix cache; later turns stay on their conversation's worker. OpenCode's compaction replaces the history
  with fixed messages, so without the assistant turn every compacted conversation would hash to one worker.
- **Ranking** (load-aware since 2026-10-20), over the routable workers only:
  1. workers with room before *saturated* ones. Saturated: this router has >= `SYNTH_ROUTER_SOFT_INFLIGHT` (default
     48; 0 disables) calls in flight there, or the worker's vLLM `/metrics` shows `vllm:num_requests_waiting` > 0 or
     `vllm:num_requests_running` >= its `max_num_seqs` (64 unless the registry record says otherwise). Each router
     scrapes `/metrics` of every routable worker every 5 s (`--metrics-s`; 5 s timeout, ~60 kB, ~0.1 ms to parse); a
     sample older than 45 s is unknown, and then only the in-flight count applies. The vLLM counts include the calls
     of every router (one per synth job) and of other users, which one router's in-flight count cannot see;
  2. *leaving* workers last: a job that ends within 600 s (`client.END_MARGIN_S`: the worker stops 30 s before its end
     and aborts what is still running; 99.96% of calls take < 600 s) or one with a drain flag gets calls only when
     every other routable worker is leaving too;
  3. then ready before draining, then the rendezvous weight.

  So when most workers drain at once, draining workers with room take the calls that the saturated ready workers
  would queue; with every worker saturated the order is the old one (ready, then weight). Affinity is best effort: a
  conversation moves while its worker is saturated or leaving, or when a newly routable worker wins its key.
- **Failover**: connection errors, 429/502/503/504, one 500 and, for buffered (non-stream) calls, a 200 whose choices
  have `finish_reason: "abort"` go to the next worker (calls are buffered), up to 4 attempts. vLLM aborts the
  generations still running when its worker stops (drain timeout, scancel, walltime) and Gym rejects that value, so
  the rollout would be lost; the retry is a fresh sample (temperature 1.0), so its reply differs, which is fine. If
  the last attempt aborted too, the call gets 503 `upstream_aborted`, never the aborted completion. Streams pass
  aborts through unchanged. 3 failed `/health` checks (every 15 s) make a worker unroutable. TCP keepalive (60 s
  idle, 15 s interval, 4 probes) fails calls on a node that died silently over in ~2 min. No read timeout; streams
  fail over only before their first byte.
- **Stale heartbeats** (2026-10-20 15:53): a registry `heartbeat_at` older than 120 s does not drop a worker while
  this router's own `GET /health` of it passed within the last 60 s; the router keeps checking such a worker only
  while that holds, so a worker that stops answering drops out until its registry heartbeat is fresh again. A router
  never adopts a worker that was already stale when it first saw it. `stopped`/`unhealthy`, `end_epoch`, 3 failed
  checks, refused connections and TCP keepalive remove workers as before. The log says `kept routable on this
  router's own /health checks`; `/router/status` shows `on_own_health` and `health_age_s`.
- **No worker**: a call is held up to `SYNTH_ROUTER_NO_WORKER_WAIT_S` (default 3600 s, the gap between two workers),
  then gets 429 with `Retry-After: 120`. Callers wait for a routable worker first (`wait_for_workers`,
  `SYNTH_STUDENT_WAIT_S`): probes up to 10800 s, capped so that ATP admission still fits before the job ends; chat
  3600 s.
- Logs: `student-router.log` (routability changes, drain flags, a `vLLM load:` line when a worker's load state
  changes) and `student-router-requests.jsonl` (one line per call, no content or headers; `aborts` = aborted
  attempts; `moved_from`/`moved_why` = the affinity's worker that the first attempt left because it was saturated,
  `inflight`/`waiting`/`running`, or leaving, `ending`/`drain_flag`) in the probe directory, or node-local scratch
  when a chat started the router. `GET /router/status` lists the workers (`vllm_running`, `vllm_waiting`,
  `metrics_age_s`, `saturated`, `leaving`, ...) and counters (`aborts`, `moved_*`, `spilled_to_draining`,
  `metrics_errors`, ...).

## Worker accounts (2026-11-02)

hengyuf 2026-11-02: "yes you can use rl_algo account" (until then, 2026-10-22: workers single-account). fleet.json
`worker_accounts` (`["nemotron_agents_dev", "nemotron_rl_algo"]`; `[]` or absent = `[account]`, the old behaviour,
proven identical Slurm calls by `tests/test_fleet_accounts.py` EquivalenceTest) lists the accounts every slot gets a
candidate on:

- **New slot**: ONE job, on the first usable account (`nemotron_agents_dev`; the next one when that sbatch fails, or
  while it is in its failure backoff). Name `swe-task-student-wNN`, the exact exemption `--comment`, the same
  `--exclude`, `--export=NONE`. A pending pair counts once for coverage and for `max_jobs` (which bounds nodes held or
  wanted); `max_submits_per_pass` counts slots that get a job, new or completed.
- **Delayed sibling** (`pair_delay_min`, default 15, fleet.json): a slot still pending (not held) 15 min after its
  first submission gets a candidate on each account it lacks (pair completion, below); the first candidate to start
  is the slot's worker. With fast queues (tonight: near-instant starts) a slot never has two candidates, so nothing
  starts twice; a slow queue is hedged after 15 min. (The first version of 2026-11-02 submitted both at once; in the
  simulation an instant queue then started ~2x the jobs and cancelled half of them.)
- **Settlement** (every pass, before policing): once a candidate serves (ready/draining, fresh heartbeat), the other
  candidates of its slot that are pending or still starting are cancelled (`scancel --state=PENDING` for pending
  ones, so one that started meanwhile is left for the next pass). While the first one is still starting, the pending
  sibling stays queued as its backup (a failed startup then costs no queue position); a second start is cancelled at
  once (earliest start stays). A candidate that reported ready/draining is never cancelled by settlement (serving
  duplicates after a daemon outage both serve out). Events: `sibling_cancel`, then `job_ended` with `sibling_of`.
  A cancelled candidate (`sibling_cancel` in `state/jobs.json`) is never a worker failure, exclusion or breaker
  entry, whatever its registry says (our SIGTERM can make vLLM exit before the TERM trap writes `stopped`, which
  reads `failed`), neither at its end nor while it is still listed. A job that `police()` is cancelling (stuck,
  stale, failed) never keeps its slot, so its healthy sibling is not cancelled with it. A candidate that is no longer
  a loser (its keeper died or stopped serving before the cancel took effect) loses its marks
  (`sibling_cancel_dropped`) and is an ordinary worker again. Only *account pairs* settle: the jobs of a slot, all on
  the fleet partition, with pairwise different squeue accounts (also pairs left over after a rollback); a same-name
  duplicate on the same account or on another partition (made by hand) keeps the single-account rules.
- **Pair completion**: a slot whose candidates are all pending (not held, never run, batch, 4 h, known and distinct
  accounts) and that queued `pair_delay_min` since its first submission (else since the daemon first saw the job)
  gets the missing account's candidate; the queued one keeps its queue position. This is how every new slot gets its
  sibling, how single-account jobs queued before the switch join the race, and how a failed sbatch is retried.
  Pauses with target 0, STOP or a non-closed breaker; the half-open probe is one job.
- **A failing account** (sbatch error) is skipped for the rest of the pass and, while the other one works, for 10 min
  (`account_submit_failed` incident, at most every 30 min per account); a new slot then goes to the other account.
  Every account failing is a `submit_failed`.
- Why not the helper (`schdule_helper.sh`): it appends ` cc-<token>-L<leg>-<i>` to `--comment`, which voids the
  idle-reaper exemption, uses `--export=ALL`, and its loser exits by itself without a registry record, which counts
  as a `no_registry` worker failure here.
- `status` shows an ACCOUNT column, `pending slots` (counts.pending_slots; counts.pending counts jobs) and the
  worker accounts with the sibling delay, plus `(daemon's last pass: ...)` while the daemon's last completed pass
  used other accounts (an old daemon, or no pass since a change); `stop --yes` cancels every candidate (by name).
  Rollback: `fleet.py worker-accounts --clear` (behaviour only, the preferred one: existing pairs still settle,
  nothing new is paired). A code rollback to `fleet.py.pre-dual-account` needs one job per slot first (cancel the
  extra pending candidates by explicit id): that fleet.py counts both candidates and never settles them.
- `scancel` is never run without explicit job ids (`scancel --user=hengyuf` alone would cancel every job of the user).

## Idle-reaper exemption

Every worker job carries the contract's exemption as one sbatch argv element, `--comment=<compact JSON>`
(`fleet.json` `comment`, serialized with `separators=(",", ":")`):

```
{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"inference_server","description":"Persistent 1-node 8-GPU vLLM inference worker for the swe-task synthetic-task student (4h job, continuously resubmitted); GPUs can idle between on-demand rollout bursts"}}
```

After each submission fleet.py reads the comment back (`squeue -j <id> -o %k`). A mismatch is logged as an incident
(`comment_mismatch`). Check by hand with `scontrol show job <id> | grep Comment`. A reaper cancel still shows up as a
`reaper_cancel` incident in `state/events.jsonl`.

## Uptime and queue waits

- **Per job.** From job start to the first `/v1/models` 200: 444 s median, 526 s max (eval runs; vLLM launch to ready
  is ~345 s). The worker then serves until 30 s before the walltime. That is about **3 h 52 min of serving per 4 h
  job**, never less than about 3 h 39 min (a slower start fails at the readiness timeout). The last 45 min are
  `draining`, so ~3 h 07 min are `ready`. The first start on a node compiles its caches, ~1 min more.
- **Fleet size.** Each job counts toward coverage while pending and for 240 - horizon of its 240 running minutes, so
  a target worker costs 240 / (wait + 240 - horizon) nodes: 1.6 with immediate starts (horizon at its 90 min floor),
  ~1.14 when the wait matches the adaptive horizon's estimate. `target` 10 therefore means ~11 to ~15 nodes on average
  with real waits (~16 with immediate starts), never more than `max_jobs` 24 fleet jobs including pending ones. The
  overlap is the price for replacements being ready before the old worker ends; the nodes come out of
  nemotron_agents_dev's fair share.
- **Queue waits** for 1-node batch jobs have been 5 min to 3.5 h, dominated by fair share (the r2 RL run competes for
  the same account). On 2026-10-16 18:15 PDT `sbatch --test-only` estimated about 8 h for one batch node, and about 8 h
  for a cpu_long daemon, against minutes on `interactive` and ~30 min on `cpu_short` (the daemon now runs on
  `cpu_datamover` and started within seconds). The adaptive horizon follows the waits up to `horizon_max_min`
  (200 min); with longer waits the fleet runs below target until jobs start. Coverage counts pending jobs, so the
  maintainer does not pile on more jobs because of a long wait. Remedies:
  - submit up to 2 extra workers on the interactive partition, `python3 -B $F submit --slot 99 --partition
    interactive` (QoS limit: 2 nodes per user); they are outside `coverage_partitions`, so they add capacity without
    replacing batch workers (moving a pending batch worker there with `scontrol update` would make the maintainer
    submit a batch replacement);
  - raise `horizon_max_min` (earlier replacements, more nodes), at most walltime - `stagger_min` - 10 min (190 for
    4 h workers); when the queue wait exceeds the walltime, gaps need bridges (interactive / batch_short) instead.
- **Daemon gaps.** If the daemon job itself waits long, a temporary daemon on a login node covers the gap; the flock
  keeps it to one active instance. Run it with
  `nohup python3 -B $F daemon --until $(( $(date +%s) + 8*3600 )) >> logs/fleetd-login.out 2>&1 &`.
  Alternatively set `daemon_partition` to `cpu_short` with `daemon_time` `04:00:00` in fleet.json. The chain
  continues through singleton successors with a short gap every 4 h.

## Tests

```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
/usr/bin/python3 -B -m unittest tests.test_fleet -v     # 50 tests, ~1.5 min; no Slurm job, no GPU
TMPDIR=<private dir> /usr/bin/python3 -I -B tests/test_fleet_accounts.py AccountsTest ReviewFixesTest   # ~20 s
TMPDIR=<private dir> /usr/bin/python3 -I -B tests/test_fleet_accounts.py EquivalenceTest   # vs the previous fleet.py, ~4 min
../atp/runtime/controller/bin/python -B -m unittest discover -s tests -p test_router.py -v   # 41 tests (aiohttp)
```

`test_router.py` runs the router and client against fake vLLM workers on 127.0.0.1: eval semantics and no
credentials, context clamp, affinity (sticky, spread, compaction), soft in-flight cap, ready before draining, no-worker
hold and 429, health, failover, streaming, client hang-up, `ensure_router` reuse and parent exit; since 2026-10-20 also
spill-over to draining workers (soft cap, vLLM waiting, full batch, registry `max_num_seqs`), stale `/metrics`,
affinity while workers have room, leaving workers (ending, drain flag), never an unroutable worker, abort retry,
all-abort 503, streamed abort untouched, and the `/metrics` parser; and stale registry heartbeats (kept on the router's
own /health, a dead worker still drops out, status/end still apply, a new router does not adopt stale records).
The fleet suite has four parts:

- reconcile policy against a fake Slurm: coverage math, max_jobs and per-pass caps, slot naming, stuck/failed/stale
  handling, exclusion persistence, never cancelling non-fleet jobs, reaper incidents, vanished jobs, squeue failure,
  circuit breaker, dry run, daemon single instance, start-daemon successor; plus the review regressions: exact slot
  matching, a COMPLETING daemon job, requeued workers, one bad node (no breaker, pending jobs re-excluded), half-open
  probe, held jobs, a single stale worker, adaptive horizon, waits recorded once, unreadable fleet.json;
- the registry writer (extracted from serve_student.sh): schema, escaping, no temp files left, never read partial
  under concurrent rewrites; plus the reader's tolerance of partial, temporary and foreign files;
- vLLM argv parity with serve_vllm.sh;
- `serve_student.sh` end to end with a fake vLLM server (test hooks `STUDENT_ROOT`, `STUDENT_PORT`,
  `STUDENT_TEST_SERVER`):
  - starting → ready → draining → stopped at the walltime;
  - startup hang → `failed` with evidence;
  - crash → `failed` with the error line;
  - health failures → `unhealthy`;
  - SIGTERM → `stopped`;
  - missing end time → `failed` (node not suspect);
  - hung registry writes (2026-10-20): each attempt cut at `STUDENT_REG_TIMEOUT_S`, one retry, the worker serves on;
  - no tagged process left behind;
  - runtime env equal to serve_vllm.sh's.

## Tuning change (2026-10-16 22:35 PDT)

The first deployed horizon (q90 of 12 h of waits, floor 90 min) kept ~20 batch nodes busy for 10 effective
workers once the queue became fast: the first wave's ~100-min waits stayed in the window, so each wave's
replacements started ~2 h before the old wave ended. Now (fleet.json): `wait_window_h` 3, `wait_quantile` 0.75,
`horizon_min` 45, `horizon_max_min` 150, and `stagger_min` 40 — each worker's replacement threshold is the horizon
plus 0–40 min by slot class (slot mod 10, `fleet.stagger_s`), so waves drift out of phase instead of expiring
together and a sudden slow queue only affects the few workers in handover. Expected steady state with a fast queue:
~12 nodes for 10 workers; with a slow queue the pending jobs' squeue estimates raise the horizon (up to 150 min).

## Load-aware routing and abort retry (2026-10-20)

Two problems on 2026-10-20 (reported by the fleet keeper). (1) At ~06:22 and ~10:20 most workers switched to
`draining` ahead of their wave end and the 1-2 `ready` ones took nearly all calls (up to 62 vLLM waiting per worker,
router p90 125-464 s, sc14 rollouts at the 3 h agent cap) while the draining ones ran 14-32: routers ranked ready
before draining for every call, and the soft cap counted only each router's own calls (one router per synth job).
(2) A stopped worker (drain-flag timeout; the 03:04 stop without drain) answered its in-flight calls with HTTP 200 +
`finish_reason: "abort"`, which Gym's chat completion model rejects, so ATP lost those slots (4 builds, one task
decision changed).

Change: routers rank by load first and spill over to draining workers with room, using vLLM's own `/metrics` counts
(Router: Ranking), keep leaving workers last, and retry aborted buffered calls on the next worker (Router: Failover).
`router.py` and `client.py` were installed atomically on 2026-10-20 at 14:57 PDT (backups `*.pre-spillover`).
No running router, worker or fleet process was touched, so only routers started after that run the new code: a
router lives as long as the build process that started it, and later probes of that process keep using it. A new
router says `load-aware routing` in its `listening on` log line. Registry format, drain flag and `fleet.py` are
unchanged; the router only reads the drain flag.

## Registry heartbeat stalls (2026-10-20)

At 15:22-15:26 a Lustre stall hung the registry writes of all 4 bridges (20572009-12 on 4 nodes). Their logs show
`registry write failed` twice, at the same seconds on every node (15:23:47 and 15:26:01): the writers were released by
the filesystem, not by their own 60 s timeouts. `heartbeat_at` went stale past 120 s while `/health` passed the whole
time, and every router dropped all 4 at once ("no routable worker", calls held up to 163 s).

- Routers (`router.py`, `client.worker_status(..., healthy_at)`): a stale heartbeat no longer drops a worker while the
  router's own `/health` checks of it keep passing (Router: Stale heartbeats). This is per-worker evidence rather than
  fleet.py's "half or more stale at once" heuristic. It also covers a stall on one node only and still drops a dead
  worker during a fleet-wide stall, and it has no threshold that misbehaves with 1-2 workers. fleet.py is unchanged:
  it still cancels a single worker that is stale for 10 min and logs an incident when most of them are.
- Workers (`serve_student.sh` `reg_write`): 15 s per attempt and one retry, at most ~31 s per heartbeat. A writer stuck
  in uninterruptible Lustre I/O still cannot be killed before the filesystem answers, which is why the router rule
  above is the actual fix.

Installed 2026-10-20 15:53 PDT, each file via a temp file and rename (new inode). Backups: `router.py.pre-heartbeat`,
`client.py.pre-heartbeat`, `serve_student.sh.pre-heartbeat`, `tests/test_router.py.pre-heartbeat` and
`tests/test_fleet.py.pre-heartbeat`. Who gets it: routers started after 15:53 (a router lives as long as the build
process that started it) and workers whose job starts after 15:53 (bash reads `serve_student.sh` when the step
starts; already queued jobs included). Running routers and running workers keep the old code until they end.
