# swe-task official synthesis dashboard

A web page over hengyuf's synthetic-DeepSWE pipeline (`S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`):
for each DeepSWE task, the accuracy of the three reference models and, for the tasks under conversion, the target's
failure patterns and the task generation tree round by round; plus a monitor of the student inference workers and
of the hub API budget. Only OFFICIAL runs (`S/synth/runs/official*`) are shown, never smoke1/val1; and, apart and never
audited, Baihe's batch of tasks (CONTRACT "v1.6": `/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu/outputs`,
97 tasks from 19 of his runs on 18 DeepSWE tasks, 2026-10-22); and (CONTRACT "v1.7", 2026-10-22) an Explorer that filters
the synthetic tasks of both and their trajectories and summarizes what is left. Since CONTRACT "v1.8" (2026-10-23) every
official run keeps its document and view (not only each task's latest: official2-etree builds etree again next to
official1-etree), and the Monitor shows the official2 synthesis queue (`S/synth/queue2`).
[CONTRACT.md](CONTRACT.md) is the source of truth for the file layout, the JSON schemas, the decision rules, the UI
and the security rules; this README summarizes it and covers operations.

```
dashboard/
├── CONTRACT.md       data contract (schemas, decision rules, UI, security)
├── collect.py        collector: reads the pipeline READ-ONLY, writes data/*.json atomically (stdlib only)
├── server.py         HTTP server: token, static/ + data/ only, runs collect.run_cycle() every 120 s (stdlib only)
├── static/           the single-page app: index.html, app.js (vanilla JS, no external resources), style.css
├── data/             collector output, served under /data/ (index.json, task/<task_id>.json, fleet.json,
│                     api.json, server_status.json, history/fleet.jsonl, history/hub.jsonl, run/<source>/<run>.json
│                     for a batch's runs: v1.6, Baihe's under run/baiheh/; explorer.json, the Explorer's tables: v1.7;
│                     run/hengyuf/<run>.json for an official run that is not its task's latest: v1.8)
├── state/            dir 700: collector.json (incremental-scan offsets), collector.lock, server.json (600: URL
│                     with the token), server-local.json (the same for `launch.sh local`)
├── dashboard.sbatch  the server as a CPU job (2 CPUs, 8 GB, 7 days)
├── launch.sh         start | stop | status | local | header
├── logs/             swe-task-dashboard-<job>.out (server + one line per collector cycle), launch.jsonl
└── tests/            test_collect.py, test_server.py, test_contract.py (python unittest); frontend.test.js,
                     frontend_real.test.js (node:test; explorer_fixture.js: explorer.json by the contract's rules);
                     validate_contract.py, server_real_check.py (checks)
```

## What it shows

- **Header**: a searchable DeepSWE task selector grouped as *Converting* (has an official run), *Baihe batch · not
  audited* (v1.6: converted by Baihe's batch runs), *Eligible queue* (by queue rank) and *All tasks*; tabs *Task* | *Monitor* | *Explorer* (v1.7); "updated hh:mm:ss" (the collector cycle on screen) with
  its age, a warning when the data is more than 5 minutes old, and "newer data held" while a refresh waits for a text
  selection. The selection lives in the URL hash (`#task=<id>`, `#monitor`, `#explorer&<filters>`), so links can be
  shared with anyone who has the token.
- **Task tab**
  1. Accuracy cards, always shown: Kimi K3 (teacher), Nemotron 3.5 Super VL (Baihe's student) and the local Super
     checkpoint sc14 (swonlysc-step0014, 131k cap): `passed/total`, the 5 rollouts as chips (green pass, red fail,
     grey "–" unscored/infra) labelled with their F2P fraction, sc14's mean F2P ratio, and the official-queue badges
     (eligible = sc14 <= 1 valid pass and kimi 5/5; rank by sc14 mean F2P ratio).
  2. For a task with an official run: the run bar (run, repo, Slurm state, stage, accepted count, log tail); the
     target's failure patterns (name, category, description, student slots, teacher contrast; how many evolve
     branches targeted each and how many of them were accepted) with the transferable signature and the full failure
     analysis; the plan (10 candidates, the built ones marked); "Round 0 at a glance", a table of the round-0 tasks
     (F2P/P2P, sc14 x/5, kimi clean x/5, infra of both models, decision, reason; a row opens its details); the
     generation tree: one column per round (probe slots or `built x / y`, `seeds: N valid · M invalid`), one row per
     lineage starting at its round-0 task, each evolved task under its parent. A node card shows `F2P n / P2P n`
     with an evolve task's target pattern (`P2 · <name>`), `sc14 x/5` and `kimi x/5` (clean) with their infra counts,
     and the decision (EVOLVE, DISCARD, BUILD FAILED, ACCEPTED, ACCEPTED+EVOLVE, NOT SELECTED, END, PENDING,
     ABANDONED; an accepted task stays ACCEPTED when seeds.json calls it an invalid seed, which only bars it from
     seeding the next round: CONTRACT "v1.5", a small amber `seed-invalid` badge with the reason, and a "Seed-invalid"
     callout in its drawer). While probing it shows each model's partial results (`sc14 1 pass / 3 of 5 done`) over a
     slim progress bar, then "probe done, scoring when the round's probe ends". Clicking a node opens its details in a
     drawer (resizable by its left edge; on screens from 1100 px the page makes room for it): summary, target
     pattern, gates (F2P tests, problems), the rollouts table (full agent/verifier errors behind a toggle), score and
     failure similarity, decision reason, instruction / evolve plan / synthetic failure analysis / rubric, and the
     teacher audit as a table (slot, cheated, categories, summary). Abandoned nodes (superseded probes, children of
     invalid parents) are behind a toggle.
     **Failure-match audit** (does sc14 repeat the target's failure patterns in a task? auditor agents write
     `runs/<run>/audit/failure_match/<task_id>.json`) over every *eligible* task (CONTRACT "v1.4": built and probed,
     a graded sc14 rollout, not discarded: accepted or valid in seeds.json, not withdrawn by main, not under an
     abandoned-* build root; accepted tasks and tasks rejected on pass rates or similarity alike): eligible cards carry
     an `FA match: strong | partial | weak | none | not audited` badge (green / amber / orange / red / grey); the run
     bar shows `audit n/m` over the eligible tasks with the verdicts, and the same for the accepted ones; the table
     "Failure-match audit" above the tree lists every eligible task, the accepted ones first, then the others round by
     round (round, decision, sc14, clean kimi, similarity, verdict, the summary's first line; a row opens its drawer);
     the drawer's section "Failure-pattern match (audit)" shows the verdict and summary, a row per target pattern
     (reproduced, sc14 slots, evidence), a row per sc14 rollout (status, failing tests, cause, matched pattern), other
     causes and the similarity judge's score with whether the auditor agrees. A task main withdrew after accepting it
     (metrics.json `withdrawn`, its dir moved to `runs/<run>/withdrawn/`) is never shown as accepted and is out of the
     audit's scope; its drawer says so.
  3. Without a run: "not converted yet" (and the queue rank when eligible).
  4. **Baihe's batch** (CONTRACT "v1.6", hengyuf 2026-10-22: "接到网页dashboard里, 不要做audit"): the 18 DeepSWE tasks his
     runs converted are listed in the picker under *Baihe batch · not audited* (apart from *Converting*, which stays
     hengyuf's official runs). Such a task shows its accuracy cards, then one run at a time: a run selector when it has
     two or more runs (tengo-callable-instance-isolation: pilot7-tengo-callable-isolation and pilot10-tengo-golua-inline;
     `#task=<id>&run=<run>`), the run bar (badge "Baihe batch · not audited", his tree, the batch's own totals, status
     finished / stopped, launches from launch.jsonl, accepted, *in batch*, audit "not audited"), then the same failure
     patterns, plan, round-0 table and generation tree as an official run, built from that run's own tree, with a table of
     the run's batch tasks in place of the failure-match audit. Cards of the batch's tasks end with a grey `batch · not
     audited` badge, their drawer names `outputs/<task_id>/`; no batch node carries an FA match badge or counts in any
     audit figure. A run that ended before scoring some tasks (q21-boa: job cancelled 2026-10-22, no summary.json; pilot2
     cand-c02: its score failed) shows them PENDING with why, and its rounds `unfinished`.
- **Explorer tab** (CONTRACT "v1.7", hengyuf 2026-10-22: "用来filter task 和trajectories"): every built synthetic task of
  the task and run files (hengyuf's official runs and Baihe's batch: 1,210 tasks, 11,920 trajectories on 2026-10-22) and
  every rollout of their probes, from `data/explorer.json`. One row of filters above everything: source, round 0 / 1 / 2,
  kind, decision (all / accepted / not accepted), similarity min-max (once a bound is set, the tasks the judge never
  scored are left out unless "include never judged" is ticked; the box says how many: `#explorer&smin=8` shows 62 tasks,
  "+934 never judged"), Nemotron (sc14) pass rate and Kimi pass rate (clean or raw) in percent 0-100 (on the whole
  percent shown), re-scored (all / only / none), audit verdict (strong / partial / weak / none / not audited / not
  eligible; Baihe's batch is always "not audited"), DeepSWE task (searchable checklist), and for the trajectories model,
  status (pass / fail / infra) and "Kimi clean only", plus "only tasks with ≥ 1 matching trajectory" and Reset. An input
  out of range is not applied and says why; a minimum above its maximum is flagged. The URL hash keeps every filter, so
  a filtered view can be shared (links of the first page, `sun=0|1`, still read). Then: KPI tiles (filtered tasks and
  trajectories; DeepSWE tasks covered of those the chosen sources converted, "of 113 DeepSWE tasks" alongside: Baihe
  alone reads 18 / 18; accepted; Nemotron and Kimi passes), the coverage of each DeepSWE task (bars by source; synthetic
  tasks or trajectories, a toggle), the coverage of each target's failure patterns by the audited tasks (reproduced
  yes / partial; how many filtered tasks are not counted and why), distributions (similarity, both pass rates with a
  column per exact value, audit verdicts, rounds; in aqua, a color no source wears), the task table (sortable, 50 a
  page; a row opens the node's drawer in its task view; a "re-scored" column and a `*` on each rate that metrics.json
  rewrote after the probe, see below) and the trajectory table (transcript and patch paths), each with CSV (UTF-8 BOM,
  F2P / P2P as passed and total numbers, formula-safe cells) / JSON export of the filtered rows. Hover or focus a bar
  for its values; every chart has a table; each chart panel is one tab stop (arrow keys move along its marks) and "Skip
  the charts" jumps to the task table.
  - What the rows are (the Explorer's header says it too): each official target's latest run only (older official runs
    have no task file; none on 2026-10-22). Baihe's source is only the 19 runs behind his 97-task batch
    (outputs/tasks.jsonl); his tree's other runs are never read, so never shown: pilot11-happydom-jsdom-inline,
    pilot4-dateutil-rfc5545-codex and q11-participle-grammar-conflict-analysis (30 scored tasks each, 90 in all, none
    accepted), and codex-check-20477274, pilot5-happydom-intersectionobserver, q16- / q16b-arktype-json-schema-refs-
    dependencies and smoke1 (no scored task; directory listing of 2026-10-22). Abandoned build roots
    (`runs/<run>/abandoned-<repo>`, set aside by main) are never rows; "abandoned" in the Explorer is a node's decision
    (its probe superseded, or its parent an invalid seed), and those tasks are rows.
  - Re-scored tasks (`rates_differ`): task_builder's score() computes a task's Nemotron rate, Kimi clean rate and valid
    counts from its rollout rows, so when metrics.json disagrees with the rows it was rewritten later (a regrade, a
    recovery, a withdrawal). The Explorer keeps metrics.json's values (what the pipeline decided on), marks them and
    shows both in the tooltip; 2026-10-22: 8 official tasks (bandit cand-c05-r1b1, fastapi cand-c04-r1b2 and
    cand-c05-r1b3, scriggo cand-c01, testem cand-c01 and cand-c03, ytt cand-c04 and cand-c01-r1b4).
- **Every official run** (CONTRACT "v1.8"): a task built by two official runs (official2-etree next to official1-etree:
  official2-etree and official2-aiomonitor launched at 23:22 on 2026-10-22, so official1-etree and official1-aiomonitor
  are older runs now) opens on its latest; the run selector lists it first ("latest"), then the older runs
  ("older"), then any batch runs, and `#task=<id>&run=official1-etree` shows the older run's own tree, plan and
  failure-match audits (an older run keeps its own audits; official2's tasks are not audited until audited in its own
  run dir). A link `#task=<id>&node=<n>` without `run=` whose node only an older run (or a batch run) has opens that
  run, and the address bar then names it, so links shared before official2 launched keep working; node ids repeat
  between runs, though (cand-c01 .. cand-c05 in each), so a link to an id the latest run has too opens the latest's.
  The picker marks the tasks queued in official2 but not launched yet ("queued in official2").
- **Monitor tab**: (v1.8) first, one section per synthesis queue, "Synthesis queue official2": the runner (alive / stale /
  stopped / never ran, its host, Slurm job and heartbeat age, judged on the page's own clock), the hub budget as the
  runner last checked it (key name and dollars left), "launching stopped" when its last key ran low, whether the student
  serving is on, the 14 runs (status, launch leg, winning and candidate jobs; a launched run links its task view) and
  the tail of its runner.log (the last 40 lines, redacted, the runner's pid masked; an operational view: it shows hosts,
  job ids, errors and key-switch lines with key NAMES, never a key value);
  then routable/target workers, busy workers N/M (utilization: routable workers with a running request),
  running and waiting requests, KV-cache use, generation tok/s (prompt tok/s only as "incl. prefix-cache hits"),
  router requests / error rate / p50 / p90 over 10 min (labelled with their scope, the official runs' probes),
  fleet-daemon heartbeat; a per-worker table (slot, job, partition, node, status, ready since, time left, heartbeat,
  running / waiting requests, KV %, gen tok/s, router requests and p90); 24 h charts (routable workers, running
  requests, gen tok/s, requests per cycle); the hub API panel: spend / max budget / remaining (red below $500) and
  the hub sample's age, burn rate over 1 h and 6 h with the span each was measured over, time to exhaustion, spend
  over time against the budget, and a per-run table of kimi cost / calls / tokens, codex sessions / tokens and glm
  calls, headed by the usage scan's state (complete, or catching up with the bytes left).

## Data sources (summary of CONTRACT.md)

`collect.py` reads the pipeline strictly read-only; every text copied from a file goes through `redact()` (keys,
bearer tokens, `token=`/`secret=` values, and the exact values of the keys in hengyuf's `.env`, `S/secrets` and
`S/atp/.env`, which it loads in memory only for that masking) and is truncated. It never reads environment variables
and never publishes anything from those files; symlinks out of its read roots are refused. Large logs are read incrementally from offsets saved in `state/collector.json` (keyed
by inode), never rescanned (the probe directories hold tens of GB; the first cycles catch up for at most ~35 s each).

| Output | Sources |
|---|---|
| `data/index.json` | every DeepSWE task: sc14 from `S/data/targets/sc14-out131k-kimi` (hengyuf's eval, 5 rollouts), Kimi K3 and Super VL from Baihe's `three_way_per_task.csv`; the official runs with their Slurm state (read-only `squeue`/`sacct`), current stage and failure-match audit counts over the eligible and the accepted tasks (`runs[].audit`) |
| `data/task/<task_id>.json` | one per task with an official run: job + redacted log tail, `failure_analysis`, the plan, `round-*/seeds.json`, every `cand-*` directory (author/gate results, probe metrics and rollouts, score, teacher audit verdicts); each evolve branch's target failure pattern (task_builder evolve(): pattern `branch mod n` of the run's `analysis/<task>/result.json`); a probing node's per-model pass/fail/infra so far from its graded slots; decisions follow `synth/task_builder.py` build() / seed_check() / valid_seeds() (CONTRACT "Decision rules"; a task withdrawn later, from metrics.json `withdrawn` or `runs/<run>/withdrawn/`, is never accepted; an accepted task seeds.json lists invalid stays accepted, the reason being its note `seed_invalid`, v1.5); whether each node is in the failure-match audit's scope (`audit_eligible`) and its audit (`failure_match`) from the run's `audit/failure_match/<task_id>.json` (validated, redacted, clipped; a malformed file is ignored and listed in index.json `errors`) |
| `data/fleet.json` | the fleet daemon's `student/state/status.json`, `student/registry/`, each ready worker's vLLM `/metrics` (tok/s from counter deltas; prompt tok/s includes prefix-cache hits; busy = routable with a running request), the official probes' `student-router-requests.jsonl`; 24 h of history from `data/history/fleet.jsonl` (kept 7 days) |
| `data/api.json` | hub budget from `S/tools/hub_budget.py` (a subprocess that reads the key itself; at most every 10 min, one tiny billed glm call), burn rates over the hub samples of the last 1 h / 6 h from `data/history/hub.jsonl` (30 days; no rate from fewer than 2 samples) and the exhaustion ETA from the sample's own time; kimi cost/tokens from the probes' kimi `http.jsonl`, codex tokens from the agents' `.agent/events.jsonl`, glm call counts from teacher audits and similarity judgements |
| `data/server_status.json` | written by `server.py` while it collects: the collector's health (state running / idle / stopped, cycles, failures, last duration and errors, import errors, next cycle); no token, no port |
| `data/explorer.json` | v1.7: one row per built synthetic task (a NODE with build.task_id) of the task and run files this cycle writes and one per rollout their probes list (NODE.probe.rollouts), column-headed arrays; its runs (build root, deep-link prefix), the DeepSWE targets with their failure patterns, per-source totals. Made from those very nodes (no file is read for it), so its numbers are the files'; the audit verdict counts only for an eligible official task (else "not eligible"; a batch's tasks "not audited"); the kimi clean flag from the teacher audit; transcript and patch paths derived from the build root (CONTRACT "v1.7") |
| `data/run/hengyuf/<run>.json` | v1.8, one per official run that is not its task's latest (the most recently launched run of a task owns `data/task/<task_id>.json`): the same document, rewritten only when its content changed (`generated_at` = index.json `runs[].detail_generated_at`); index.json runs[] say `latest` and `detail`, tasks[] list every run in `official_runs`; Explorer rows of every official run (an older run's rows link with `run=`) |
| index.json `queues` | v1.8, from exactly four files of `S/synth/queue2` (the official2 runner's `queue.json`, `state.json`, `heartbeat.json`, and the last 40 lines of `runner.log`, redacted; read roots: those four paths only): the runs in queue order with status, leg, winner and candidates; the runner's heartbeat (alive within 15 min); the budget as a key NAME and a number, nothing else; serving and stop-launching flags; a symlink at any of the four paths is refused. An analyzer session (`analysis/<task>/.agent/events.jsonl`) older than its run's first launch is a copy (run_queue.py keeps the old mtimes) and never that run's codex usage: it is counted where it ran; one redone in the run counts. A queue run launched through the babysit helper shows its winning job, or, while its leg is unresolved, the most active of its candidates |
| `data/run/<source>/<run>.json` | v1.6, one per run of a batch (Baihe's: `B=/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu`): the runs whose tasks are in `B/outputs/tasks.jsonl` (each task's run and target from `B/outputs/<task_id>/provenance.json` `task_dir`), built by the official code from `B/synth/runs/<run>/` (`--rounds 1`, his builder's default); the same document as a task file plus the batch fields, each node `in_batch` when its task is in the batch, never audited (`failure_match` null, `audit_eligible` false, `audit` not_audited). Read roots: `B/outputs` and those run dirs only (never his secrets, .env files, logs or other runs); no Slurm polling, no log, no usage scan. index.json lists the runs in `runs[]` (with `source`), the task's `batch_runs`, and the batch's own totals in `batches` (tasks, runs, targets, in_batch, accepted, mismatches) |

## Refresh cadence

- Collector: one cycle at server start, then every 120 s (start to start; `--interval`). One log line per cycle:
  `collect cycle N: <seconds> s, <E> errors (<runs> runs, <tasks> tasks)` (official runs). Non-fatal problems of single
  sources are listed in `index.json` "errors" and shown under the header. A batch's runs (v1.6) have ended: each is
  rebuilt only when a few stats of its tree change, else at least every 30 min (at most 2 a cycle), and its
  `data/run/...` file is rewritten only when its content changed (its `generated_at` is that cycle's, named by index.json
  `runs[].detail_generated_at`); a server's warm cycle spends well under a second on them, a fresh process about a
  minute, once.
- Older official runs (v1.8): their documents are built every cycle like the latest's, but `data/run/hengyuf/<run>.json`
  is rewritten only when its content changed; the page refetches one only when index.json names a newer
  `detail_generated_at`. The queue's four files are small and read every cycle.
- Explorer (v1.7): `data/explorer.json` is computed every cycle and rewritten only when its content changed (its
  `generated_at` = index.json `explorer.generated_at`: the cycle that last changed it); it costs ~0.2 s a cycle
  (2026-10-22: 1.2 MB, ~190 KB gzipped). The page fetches it only while the Explorer is shown, and re-renders only when
  that file changed (a cycle that leaves it alone keeps the inputs, their focus and caret).
- Hub budget sample: at most every 10 minutes (it costs one tiny glm call).
- Browser: refetches every 60 s (and on the Refresh button) with `cache: 'no-cache'` (revalidation; a batch run's
  document only when index.json names a newer `detail_generated_at`, explorer.json only when it names a newer
  `explorer.generated_at`); the header warns
  when `index.json` is older than 5 min. The view re-renders only when a new cycle landed (or you act), keeps the
  window, tree, drawer and `<pre>` scroll positions and the open collapsibles, and holds an automatic refresh while
  text is selected ("newer data held"; retried every 15 s).
  The server answers the revalidation of an unchanged file with a 304 without a body (ETag / Last-Modified), and
  gzips text responses of 2 KB or more for browsers that accept it (a task document is ~1-2 MB of JSON, about 3x
  smaller compressed; index.json about 10x).

## Launch, connect, stop

`L=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard/launch.sh`

| Command | What it does |
|---|---|
| `$L start` (`$L`) | Refuses if a `swe-task-dashboard` job of yours is pending or running. Otherwise picks the partition with `S/tools/pick_partition.py` (`sbatch --test-only` probes, earliest start) among cpu_datamover / cpu_dataprocessing / cpu_interactive whose time limit allows the walltime (7 days by default, which leaves out cpu_interactive and its 1-day cap), submits `dashboard.sbatch` with a scrubbed environment, waits for the server's `state/server.json` (up to `--wait 600` s), checks `/healthz` from the login node and prints how to connect. Options: `--time`, `--partition P` / `--partitions a,b,c`, `--port 8765`, `--reuse-token` (keeps the token and, unless `--port` is given, the previous server's port: the session cookie is per port), `--dry-run` (prints the probe result and the sbatch command, submits nothing). |
| `$L status` | The job, how to connect (URL with the token), a local server if one runs, and the collector's last cycle. |
| `$L stop` | `scancel --user=$USER --name=swe-task-dashboard` (`--dry-run` shows it). |
| `$L local` | `server.py` in the foreground on 127.0.0.1 of the current host, for VS Code's automatic port forwarding; prints the URL with the token. Serve-only while the Slurm job runs (it already collects); `--collect` / `--no-collect`, `--port`, `--reuse-token` (also keeps its port unless `--port` is given). |
| `$L header` | Writes `X-Dashboard-Token: <token>` of the Slurm job's server (`--local`: of the local one) to `$HOME/.dashboard-header` (`--file F`), mode 600, for `curl -H @file`; prints no token. |

Connecting to the Slurm job (node and port as printed by `start`/`status`; the port is 8765 unless it was taken on
that node, in which case the server took the next free one):
- VS Code: Ports view, "Forward a Port", enter `<node>:<port>`, then open `http://localhost:<port>/?token=<token>`
  (use the local port VS Code shows if it differs).
- Plain ssh: `ssh -L <port>:<node>:<port> <user>@<login node>`, then open the same URL.

The first visit stores a session cookie and the server redirects to the clean URL, so the token leaves the address
bar. Use a regular browser: VS Code's built-in Simple Browser shows pages in a cross-site frame, where the
SameSite=Strict cookie is never sent and the server forbids framing anyway. After the 7-day walltime the job ends;
`$L start` again (`--reuse-token` keeps the old token and port, so open tabs keep working; if the port was taken on
the new node, they need the URL with the token once). A "Missing or expired token" page means the server restarted
with a new token or on another port: open the URL from `$L status`.

Tools (curl, scripts) send the token in the `X-Dashboard-Token` header, from a file only you can read. Never put the
token itself on a command line: `ps` shows it to everyone on the host, and the shell history keeps it.
```
$L header                                    # $HOME/.dashboard-header, mode 600: "X-Dashboard-Token: <token>"
curl -sS -H @$HOME/.dashboard-header http://localhost:8765/data/index.json
```

Direct use: `/usr/bin/python3 -I -B server.py [--host 127.0.0.1] [--port 0] [--interval 120] [--no-collect]
[--reuse-token] [--print-url]` (`--root`, `--collector`, `--info-file` exist for tests). Manual collection without a
server: `/usr/bin/python3 -I -B collect.py once [--no-hub]`. While the server job runs, a test cycle must go to other
directories: `collect.py once --no-hub --data $T/data --state $T/state` with `T=$(mktemp -d)` (a fresh state reads the
whole log backlog: the first cycle takes the 35 s scan budget, the next one finishes it, later ones take ~4 s), then
`tests/validate_contract.py --sources --data $T/data --state $T/state`.

## Security

- **Token.** `secrets.token_urlsafe(32)`, new at every server start (or the previous one with `--reuse-token`),
  stored only in `state/server.json` (file 600, dir 700) and printed only by `launch.sh` on your terminal (`local`
  mode: by the server on your terminal). It never reaches a log: request lines are logged without query strings,
  everything the server logs (and the collector errors it records in `data/server_status.json`) has the token and
  the session value masked (`<token>`, `<session>`; also a mistyped or truncated copy: any 8 characters in a row of
  either, e.g. a token typed as a path) and every `token=...` value too, and only errors and logins are logged (at
  most 30 lines a minute).
- **Session.** `?token=<t>` on any URL sets an `HttpOnly; SameSite=Strict` session cookie (Max-Age 7 days) and
  answers 303 to the same URL without the token. The cookie never holds the token, since a browser sends the cookies
  of localhost to every port there: it is `swe_dash_<port>` (two dashboards on two ports do not clash) =
  HMAC-SHA256(token, `"session:<port>"`) in hex, checked by recomputing it. Only the token is kept, so a restart with
  `--reuse-token` on the same port keeps the sessions; a cookie of the earlier scheme (the token itself, named
  `swe_dash_<10 hex>`) is expired at the next login. Every other request needs the session cookie or the
  `X-Dashboard-Token: <t>` header (from a file, see Tools above); comparisons are constant-time; anything else gets
  403. `/healthz` is open and returns only `{"ok": true}`.
- **What is served.** `static/` and `data/` only: regular files of known types (data: `.json`/`.jsonl`). No
  directory listings, no dot or empty path segments (also percent-encoded), no hidden files (the collector's temp
  files), nothing whose real path leaves those directories (symlinks included); `state/`, `logs/` and the code are
  never served. GET and HEAD only (405 otherwise).
- **Caching.** `Cache-Control: no-store` on every response except the files of `static/` and `data/`: those carry
  `Cache-Control: no-cache` (the browser may keep a copy but revalidates it every time), a weak `ETag` (size, mtime
  and inode: the collector renames a new file over the old one; one tag for the plain and the gzipped body) and
  `Last-Modified`. `If-None-Match` (else `If-Modified-Since`) matching the file gets a 304 without a body, after the
  token check like any request.
- **Limits.** At most 32 connections are handled at once, a thread each, with a 15 s socket timeout (a client that
  stalls longer loses its slot); the next ones get a 503 (`Retry-After: 2`) at once from the accepting thread. One
  request per connection.
- **Headers on every response.** `Cache-Control` (above), `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Cross-Origin-Resource-Policy` and
  `Cross-Origin-Opener-Policy: same-origin`, and a strict Content-Security-Policy: `default-src 'self'`,
  `script-src 'self'` (no inline script, no eval), `object-src 'none'`, `base-uri 'none'`, `form-action 'none'`,
  `frame-ancestors 'none'`, `img-src 'self' data:`; inline style attributes are allowed (`style-src-attr
  'unsafe-inline'`) because app.js positions the tree cards and sizes bars with `style="..."`, `<style>` elements are
  not. **For the frontend:** index.html must load app.js as an external file (`<script src="static/app.js">`) and may
  not use inline `<script>`, `on*=` handler attributes, `javascript:` URLs or eval; data is fetched same-origin with
  the cookie (`credentials: 'same-origin'`), and a 403 means the token is missing or stale.
- **Network.** The server binds 127.0.0.1 by default and refuses any other address outside a Slurm job; only
  `dashboard.sbatch` binds 0.0.0.0, on its compute node, reached through VS Code or ssh port forwarding. Anyone on
  the cluster network can reach that port, so the token is what protects the data.
- **Environment.** `launch.sh` submits with `env -i` (nothing from your shell reaches the job), `dashboard.sbatch`
  also unsets every inherited `*KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*` variable (`--export=NONE` still loads
  the login environment), and the collector reads no credentials at all.

## Tests

```
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard
/usr/bin/python3 -I -B -m unittest discover -s tests -v       # test_collect.py, test_server.py, test_contract.py
<node> --test tests/frontend.test.js tests/frontend_real.test.js   # node v24 (e.g. the VS Code server's node binary)
DASHBOARD_DATA=$T/data <node> --test tests/frontend_real.test.js  # the real-data tests on another output (default data/)
/usr/bin/python3 -I -B tests/validate_contract.py --sources   # the live data/ against CONTRACT.md and the pipeline
/usr/bin/python3 -I -B tests/server_real_check.py             # server.py on the live static/ + data/, then stopped
```
- `test_contract.py`: a collector cycle over the synthetic tree (also with broken sources, and with withdrawn tasks)
  passes `validate_contract.py`, and copies with one planted violation each are caught; SeedPrecedenceTest (v1.5):
  accepted tasks seeds.json lists invalid conform, also with `--sources` against the synthetic tree's own accepted/,
  withdrawn/ and seeds.json; the older collector's discard of them, a broken `seed_invalid` and a withdrawn task left in
  accepted/ are caught.
- v1.6, Baihe's batch: `test_collect.py` BatchTest (a synthetic batch tree, `BatchFixture`, as his fork writes it: two
  runs on one target, a stopped run, planted files that must never be read: an audit, Slurm logs, kimi usage, a run
  outside the batch, secrets) checks the runs, documents, decisions with `--rounds 1`, unfinished probes, `in_batch`,
  never audited, nothing else read or polled, mismatches reported, unchanged runs reused and files rewritten only on a
  change, and the official output byte-identical to a tree without the batch; `test_contract.py` BatchContractTest
  (conformance with and without `--sources`, and one violation per batch rule caught); `frontend.test.js` (picker group,
  run selector, run bar, batch table, cards, drawers, escaping, the wiring of `run=`, and official views, drawers, menu and
  monitor byte-identical to the v1.5 page kept as `static/app.js.pre-baihe-batch`); `frontend_real.test.js` (every batch
  run of the live output, Baihe's known runs and totals).
- v1.7, the Explorer: `test_collect.py` ExplorerTest (on the synthetic official build and batch: every built node of the
  task and run files a row, every rollout a trajectory, in order; runs, targets and patterns, the deep-link prefixes;
  values and their sources, the audit rule (verdict / not audited / not eligible), reproduced patterns matched leniently,
  the kimi clean flags, a live probe's slots, withdrawn and abandoned tasks, the batch never audited, secrets masked,
  the re-scored flag (valid counts or a rate of metrics.json apart from the rollout rows, a metrics.json rewritten after
  the probe); rewritten only on a change, also across a process restart; a broken build never stops the cycle; every other output
  byte-identical to the v1.6 collector kept as `collect.py.pre-explorer`); `test_contract.py` ExplorerContractTest
  (conformance with and without `--sources`, and one planted violation per rule: a cell, a row, a column header, a run,
  a target, a total, an audited batch task, the stamps, the index entry, a missing transcript, a wrong rates_differ);
  `frontend.test.js` (the hash round trip with old `sun=` links, every filter and their combination, the never-judged
  rule and count, percent inputs and their rejection, inverted ranges, the statistics and the KPI's scope, exact rate
  columns, the view's sections and marks (slot 3, list roles, one tab stop per panel), plurals, sorting, pages,
  exports (BOM, F2P / P2P split, the formula guard), re-scored rows, escaping, broken documents, the wiring of
  `#explorer`: results re-rendered alone, rows opening their task view, Escape after a node visit leaving the Explorer
  and monitor URLs alone, a cycle without a new explorer.json rendering nothing, arrow keys and the skip button; and
  the task / monitor views, drawers, menu and header byte-identical to `static/app.js.pre-explorer`);
  `frontend_real.test.js` (the live explorer.json against the live task and run files, every deep link, many filter
  combinations sane and fast, escaping, `smin=8` and the never-judged count, the KPI per source, the re-scored rows,
  Escape after a node visit).
- v1.8, every official run and the synthesis queue: `test_collect.py` OfficialRunsTest (a second official run of the
  synthetic target, launched later: the latest; the older run's document byte for byte what it wrote as the latest,
  rewritten only on a change, removed once it is the latest again; runs[] / official_runs; audits per run dir, also for
  a task id both runs have; a run dir with only analysis/ is no run; the Explorer's rows of both; every other output
  byte-identical to the v1.7 collector kept as `collect.py.pre-v18`), LaunchTest (launch.jsonl as the queue runner writes
  it through the babysit helper: a resolved leg's winner, an unresolved leg's most active candidate polled and shown with
  its log, each job once) and QueueTest (the queue entry from the four files, read_at and the heartbeat's age, defaults
  when the runner has written nothing yet, liveness, the log tail's real lines with the pid masked and secrets redacted,
  only the four files read and a symlink at any of them refused wherever it points, planted secrets and notes never
  published, an analyzer session older than its run's first launch never counted as usage and a redone one counted,
  with or without queue.json, broken files never stopping the cycle); `test_contract.py` OfficialQueueContractTest (conformance with and without `--sources`, and one planted
  violation per rule: latest, detail and its stamp, official_runs, the older run's file, the Explorer's runs, the queue's
  runner, budget, runs and log tail, an unlisted run dir, a queue that disagrees with its files); `frontend.test.js` (the
  views of a task with two official runs, the selector, the older run's view and note, one-run tasks byte-identical to
  the v1.7 page kept as `static/app.js.pre-v18`, the picker's "queued in" note, the Monitor's queue section, the wiring
  of `run=<older run>`, the deep-link fallback, the Explorer's run filter and its hash, the text range inputs rejecting
  "abc", the bar labels' title and two-line clamp); `frontend_real.test.js` (every official run's document and view, the
  queue on the live index).
- `test_collect.py` (EligibilityTest): the failure-match audit's scope on the synthetic tree: which tasks are
  eligible, withdrawn tasks (metrics.json `withdrawn`, a task dir in `runs/<run>/withdrawn/`) never accepted nor
  eligible while still counted as accepted for the seed selection they took part in, uniform failure, tasks without a
  graded sc14 rollout, abandoned-* roots, and the eligible counts of `runs[].audit`. AcceptedInvalidSeedTest (v1.5):
  an accepted task that seeds.json (or the recomputed seed_check) calls an invalid seed is accepted (round 0: not
  accepted_evolve), keeps the reason as `seed_invalid`, is eligible and counted; a withdrawn one stays discarded.
- `frontend.test.js`: every renderer on the fixtures (`tests/make_fixtures.js` regenerates them: narwhals, fleet and
  api in the current collector shapes, bandit in the older ones), sparse / mutated / escaped copies, the refresh
  helpers (skip / hold / render, scroll and open-state capture and restore) and `createApp()` on a fake DOM: no
  re-render without a new cycle, scroll positions kept across one, a refresh held while text is selected, 304s, the
  drawer grip; the failure-match audit on copies of the narwhals fixture as a v1.3 collector writes them (accepted
  tasks) and as v1.4 does (every eligible task, a withdrawn one, an audit of a discarded one); the v1.5 seed notes
  (the `seed-invalid` badge on cards and tables, the drawer callout, none for a discard).
- `frontend_real.test.js`: every renderer on the live `data/` (each task view, both runs' trees and round-0 tables,
  every node card and detail panel, the monitor), the tree-layout invariants, escaping with a markup payload in every
  free-text string, the known decisions of the official runs, the v1.2 fields when present (their fallbacks when
  not), the v1.5 seed notes (official1-wasmi cand-c03-r1b2 ACCEPTED with `seed-invalid`), the failure-match audit's
  scope, cards, table and run bar against the data, runs launched minutes ago (no plan
  or candidate yet), and `createApp()` fetching the live files by their served paths. Skips when `data/index.json` does
  not exist yet.
- `validate_contract.py [--sources]`: every key, type, enum, size limit and decision rule of CONTRACT.md on the live
  output (the v1.2-v1.5 fields whenever present, e.g. each node's `audit_eligible` recomputed from its own fields);
  `--sources` also compares it with the pipeline (node set = `cand-*` dirs, `metrics.json` accept, `seeds.json`: an
  invalid entry is discarded unless metrics.json accepted it and main did not withdraw it, `accepted/`, probe dirs,
  each evolve node's focus pattern against its `evolve/.agent/prompt.md`, attached audits against their files,
  withdrawals against `metrics.json` and `withdrawn/`, and every task's accuracy against `three_way_per_task.csv` and
  the sc14 targets). v1.6: the batch's runs, documents and totals (never audited, every in_batch node accepted, the
  document's own stamp), and with `--sources` each batch run against its own tree as above plus `outputs/tasks.jsonl`,
  each task's `provenance.json` and each run's `accepted/tasks.jsonl`. v1.7: `explorer.json` (shape, stamps, and every
  run, target, row and total recomputed from the task and run files), with `--sources` the candidate dirs, transcripts
  and patches it points at.
  Read-only; safe next to a running collector. `--data`/`--state` check another output, e.g. a test cycle's.
- `server_real_check.py`: server.py on 127.0.0.1 (port 0, `--no-collect`, its info file in a temp dir) through the
  cookie flow (the per-port HMAC session cookie): `/`, the assets, every data file (plain and gzip) byte-identical to
  disk, revalidation to a 304, security and cache headers, a CSP that fits index.html, 403/404/405; always stops the
  server.
`test_server.py` starts real server processes on 127.0.0.1 with port 0 in temporary directories (dummy static/data,
`--no-collect` or a fake collector) and always stops them; it covers the token, the session cookie (per port, the
HMAC, never the token; the old scheme's cookie expired at login), header auth, path traversal (plain,
percent-encoded, double-encoded, backslashes, symlinks), directory listings, methods, security and cache headers,
ETag / Last-Modified and the 304s, gzip, the 32-handler cap (503) and the 15 s socket timeout, `/healthz`,
`server.json`, `--reuse-token`, the token (whole, cut short, glued, in malformed requests) never in the log, the
collector thread's error handling, and `launch.sh` (start / status / stop / local / header, `--reuse-token` keeping
the port) / `dashboard.sbatch` against fake Slurm commands (nothing is submitted).
