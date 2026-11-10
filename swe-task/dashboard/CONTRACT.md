# swe-task dashboard: data contract (v1)

User request (hengyuf, 2026-10-17): a web page visualizing the official synthesis runs.
- Top: choose a DeepSWE task. First show the accuracy of Kimi K3, Nemotron 3.5 Super VL and the local Super
  checkpoint sc14 (always, whether or not a conversion exists).
- For a task under conversion (an official run): the target's failure patterns, then a task generation tree: the
  round-0 tasks with a brief status (F2P/P2P tests, discard or put to evolve, rollout accuracy); click a node to
  expand its details; each task put to evolve branches into its 5 evolved tasks, annotated the same way, round by
  round.
- A monitor console: inference worker availability and utilization, and hub API usage.
Only OFFICIAL runs are recorded: `S/synth/runs/official*`. Never smoke1/val1. (v1.6: plus, apart and never audited,
the runs of a batch of tasks from another tree: Baihe's, section "v1.6". v1.7: an Explorer view filters the synthetic
tasks of both and their trajectories, from data/explorer.json: section "v1.7". v1.8: every official run keeps a document
and a view, not only each task's latest, and the official2 synthesis queue shows on the Monitor: section "v1.8".)

Paths: `S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`, `D=S/dashboard`.
- `D/collect.py`: stdlib-only collector (runs under `/usr/bin/python3 -I -B`, Python 3.10). It writes the JSON
  below into `D/data/` atomically (temp file + rename). Incremental-scan state lives in `D/state/` (mode 700).
- `D/server.py`: stdlib HTTP server. It serves `D/static/` and `D/data/`, requires a token, and runs the
  collector in a background thread.
- `D/static/{index.html,app.js,style.css}`: single-page app, vanilla JS, no external/CDN resources. Rendering
  functions are pure (data -> HTML string) so node can test them without a DOM. Auto-refreshes every 60 s.
- `D/dashboard.sbatch` + `D/launch.sh`: run the server as a small CPU Slurm job. Partition is chosen with
  `S/tools/pick_partition.py` among cpu_datamover/cpu_dataprocessing/cpu_interactive, preferring the idle one.

Every timestamp is epoch seconds (float) plus, where useful, an ISO string. Text fields copied from files go
through `redact()`, which masks `sk-...`, `nvapi-...`, `hf_...`, `Bearer <x>`, `ghp_...`, `glpat-...`, `AKIA...`,
and any 32+ char base64/hex run inside `key=`/`token=`/`secret=` assignments. Long text is truncated with a
`"…[N chars omitted]…"` marker; limits are given below. NEVER read environment variables, and NEVER emit anything
from .env files, `S/secrets` or `S/atp/.env`. The one exception (2026-10-17 security review): the key VALUES there are
loaded in memory so that redact() masks exact occurrences. Files are opened only inside the collector's read roots,
and only regular files (no symlink escapes).

## data/index.json (every cycle)
```
{ "schema": 1, "generated_at": 1792280000.0, "generated_at_iso": "2026-10-17T15:00:00-07:00",
  "collect_seconds": 3.2, "errors": ["<source>: <message>", ...],          // non-fatal collector problems
  "models": {                                                            // display order: kimi, super_vl, sc14
    "kimi":     {"label": "Kimi K3", "role": "teacher", "source": "<short source description>"},
    "super_vl": {"label": "Nemotron 3.5 Super VL", "role": "Baihe's student", "source": "..."},
    "sc14":     {"label": "Local Super ckpt sc14 (swonlysc-step0014, 131k cap)", "role": "student", "source": "..."} },
  "runs": [ {"run": "official1-narwhals", "task_id": "narwhals-rolling-window-suite", "repo": "ibis-project/ibis",
             "job_id": "20408127", "slurm_state": "RUNNING", "partition": "cpu_interactive",
             "submitted_at": 1792..., "started_at": 1792... | null, "ended_at": null,
             "status": "running|pending|finished|stopped|failed", "stage": "round-1 probe (37/200 slots)",
             "rounds_started": 2, "accepted": 1, "tasks_built": 25, "tasks_scored": 5,
             "audit": {"accepted": 1, "audited": 1, "strong": 1, "partial": 0, "weak": 0, "none": 0,      // v1.3
                       "eligible": 4, "audited_eligible": 2,                                          // v1.4
                       "eligible_verdicts": {"strong": 1, "partial": 1, "weak": 0, "none": 0}}} ],
  "tasks": [ {
      "task_id": "abs-module-cache-flags", "task_index": 0, "f2p_total": 20,
      "accuracy": {
        "kimi":     {"passed": 5, "total": 5, "unscored": 0, "rollouts": [{"f2p": "20/20", "p2p": "3/3", "score": 1}, ...]},
        "super_vl": {"passed": 2, "total": 5, "unscored": 0, "rollouts": [...]},
        "sc14":     {"passed": 4, "total": 5, "infra": 0, "rollouts": [...], "mean_f2p_ratio": 0.91} },
        // a model with no data: null. "score": 1 pass, 0 fail, null unscored/infra.
      "eligible": true,      // official-queue rule: sc14 <= 1 valid pass AND kimi 5/5
      "queue_rank": 3,       // rank among eligible tasks by sc14 mean F2P ratio, descending (1 = first); else null
      "run": "official1-narwhals" | null } ] }
```

## data/task/<task_id>.json (one per task that has an official run; every cycle)
```
{ "schema": 1, "generated_at": ..., "task_id": "...", "run": "official1-narwhals", "repo": "ibis-project/ibis",
  "build_root": "<path>", "max_rounds": 3, "branches": 5,
  "job": {"job_id": "...", "slurm_state": "...", "partition": "...", "node": "...", "elapsed": "1:02:03",
          "log_path": "...", "log_tail": ["<= 40 redacted lines, each <= 300 chars"], "stage": "<same as index>"},
  "failure_analysis": {
     "patterns": [{"name": "...", "category": "...", "description": "...", "student_slots": ["sc14-r0"],
                   "teacher_contrast": "...", "evidence": "<optional, any other string fields joined>"}],
     "transferable_signature": "<string or JSON-dumped>", "excluded": [ ... as in result.json ... ],
     "pass_stats": {"sc14": {"pass": 0, "fail": 5, "rate": 0.0}, "kimi": {...}} | null,
     "markdown": "<failure_analysis.md, <= 60000 chars>" } | null,
  "plan": {"candidates": [{"id": "c01", "title": "...", "summary": "...", "files": [...], "tests": 12,
                          "built": true}], "mask_plan_excerpt": "<= 4000 chars"} | null,
  "rounds": [ {"index": 0, "status": "done|probing|building|pending",
               "probe": {"dir": "round-0/probe-b1ade2ed", "tasks": 5, "slots_total": 50, "slots_done": 50,
                         "by_model": {"sc14": [25, 25], "kimi": [25, 25]}} | null,   // [done, total]
               "abandoned_probes": ["round-1/probe-7f97a213"],
               "seeds": {"valid": [...], "invalid": {"cand-c02": "reason"}} | null } ],
  "nodes": [ NODE, ... ] }
```
NODE (one per candidate dir `cand-<id>` / `cand-<id>-r<i>b<j>[-r<k>b<l>...]`; the renamed `*.budget-422`,
`*.failed-materialize` etc. FILES inside a candidate dir are ignored):
```
{ "id": "cand-c01-r1b2", "parent": "cand-c01" | null, "lineage": "cand-c01", "round": 1, "branch": 2,
  "kind": "mask|evolve", "focus_pattern": "<target failure pattern the evolve branch aims at>" | null,  // v1.2
  "title": "...", "summary": "<= 600 chars",
  "status": "planning|authoring|building|build_failed|built|probing|scored|abandoned",
  "build": {"ok": true|false|null, "attempts": 2, "f2p": 78, "p2p": 4903,
            "f2p_tests": ["<= 60 names"], "p2p_tests_sample": ["<= 20 names"],
            "problems": "<gate/author problems text, <= 3000 chars>" | null,
            "task_id": "syn-ibis-cand-c04-86a17614" | null},
  "probe": {"progress": {"sc14": [3, 5], "kimi": [5, 5]},                      // [done, total] while probing
            "partial": {"sc14": {"done": 3, "total": 5, "pass": 1, "fail": 2, "infra": 0}, "kimi": {...}},
            "own_done": false,              // v1.2, probing nodes only: graded slots so far; all own slots final
            "student": {"pass": 1, "fail": 4, "infra": 0, "rate": 0.2},          // from metrics/rollouts when scored
            "teacher": {"pass": 4, "fail": 1, "infra": 0, "clean_rate": 0.8, "cheated": 0},
            "rollouts": [{"slot": "sc14-r0", "model": "sc14", "status": "pass|fail|infra", "reward": 1.0,
                          "f2p": "76/76", "p2p": "4044/4044", "tool_calls": 138, "patch_chars": 19508,
                          "agent_error": "", "verifier_error": "" }]} | null,
  "score": {"accept": true, "reason": "similarity 6/10", "similarity": 6,
            "similarity_detail": "<JSON-dumped, <= 4000 chars>", "valid": [5, 5],
            "grading_failures": [], "answer_retrieval": false,
            "target_student_rate": 0.2, "target_teacher_rate": 1.0} | null,
  "decision": "evolve|discard|accepted|accepted_evolve|not_selected|end|pending|abandoned",
  "decision_reason": "<one line>",
  "children": ["cand-c01-r1b2-r2b0", ...],
  "details": {"instruction_md": "<= 20000 chars" | null, "evolve_plan_md": "<= 20000" | null,
              "synthetic_failure_analysis_md": "<= 20000" | null, "rubric_md": "<= 8000" | null,
              "teacher_audit": [{"slot": "kimi-r0", "cheated": false, "categories": ["..."],
                                 "summary": "<the judge's notes, <= 400 chars>" | null,
                                 "rubric_score": 9.8 | null}] | null},        // v1.2; before: "<JSON-dumped, <= 6000>"
  "failure_match": { ... } | null,      // v1.3: the failure-match audit of the node's task (section "v1.3" below)
  "audit_eligible": true | false,        // v1.4: the task is in the audit's scope (section "v1.4" below)
  "withdrawn": {"at": "...", "by": "main", "why": "..."} | null,   // v1.4: main withdrew it after accepting it
  "seed_invalid": "<why it may not seed evolution, one line>" | null }   // v1.5: section "v1.5" below
```
Decision rules. They mirror `S/synth/task_builder.py` build(), seed_check() and valid_seeds(); read them. build()
accepts a task on its score alone (accept()); seeds.json only picks the parents of the next round.
- `build.ok == false` (stage.json `{"ok": false}`): `discard`, "build failed: <first problem line>". No rollout.
- The round's `seeds.json` lists the node in `invalid` and the task was not accepted: `discard`, with that reason.
  v1.5: an accepted task stays accepted, the reason becomes its note NODE.seed_invalid (section "v1.5").
- The node's parent was invalid in its round's seeds.json, or its probe/round was superseded (narwhals cand-c02
  children and round-1/probe-7f97a213, from the 2026-10-17 seed-rule change): status and decision `abandoned`.
- Round 0, valid, accepted: `accepted_evolve`. Round 0, valid, not accepted: `evolve`. Since 2026-10-17 every
  valid round-0 task is a seed. Round 0, invalid, accepted (v1.5): `accepted`, "accepted (<reason>); an invalid seed,
  so not evolved".
- Round >= 1: accepted gives `accepted` (v1.5: whatever seeds.json says). Has children gives `evolve`. Valid, no
  children, and the next round has started for its lineage gives `not_selected`. Round == max_rounds gives `end`.
  Otherwise `pending`.
- Not scored yet: `pending`.
- v1.4: a task main withdrew (NODE.withdrawn) is never accepted: it is decided like a scored task that was not
  accepted, with the reason "withdrawn: <why>". The seed selection of its round still counts it as accepted when it was
  then, as task_builder did (section "v1.4"). Without seeds.json, seed_check() is recomputed with its uniform-failure
  rule too. decision_reason is one line (line breaks of copied reasons and of clip()'s marker become spaces).

## data/fleet.json (every cycle; worker metrics scraped each cycle)
```
{ "schema": 1, "generated_at": ...,
  "daemon": {"job_id": "...", "host": "...", "heartbeat_age_s": 37, "status_age_s": 40, "breaker": "closed"},
  "config": {"target": 10, "horizon_effective_min": 150, "horizon_min": 45, "stagger_min": 40, "max_jobs": 24,
             "time": "04:00:00", "coverage_partitions": ["batch"]},
  "coverage": {"value": 10, "target": 10},
  "counts": {"ready": 10, "draining": 0, "starting": 0, "pending": 0, "held": 0, "stale": 0, "unhealthy": 0,
             "failed": 0, "stopped": 0, "vanished": 0, "other": 0},
  "workers": [{"job_id": "...", "slot": "05", "partition": "batch", "node": "pool0-00874",
               "state": "RUNNING|PENDING|...", "registry_status": "ready|draining|starting|...|null",
               "ready_at": 1792..., "start": ..., "end": ..., "left_s": 13000, "heartbeat_age_s": 14,
               "base_url": "http://10.65.12.145:8000/v1", "pending_reason": null, "est_start": null,
               "metrics": {"running": 6, "waiting": 0, "kv_usage": 0.096, "prompt_tps_incl_cache": 1234.5,
                           "prompt_tps": 1234.5, "gen_tps": 88.1, "scraped_at": ..., "error": null} | null}],
  "totals": {"routable": 10, "busy_workers": 9, "running": 37, "waiting": 0, "kv_usage_mean": 0.11,
             "prompt_tps_incl_cache": ..., "prompt_tps": ..., "gen_tps": ...},
  "router": {"window_s": 600, "scope": "official-run probes", "requests": 812, "errors": 0, "error_rate": 0.0,
             "p50_s": 9.1, "p90_s": 24.0, "prompt_tokens": ..., "completion_tokens": ..., "clamps": 3,
             "by_worker": {"<job_id>": {"requests": 80, "errors": 0, "p90_s": 22.1}},
             "last_hour": {"requests": ..., "errors": ...}},
  "history": [{"t": ..., "ready": 10, "routable": 10, "running": 37, "waiting": 0, "kv": 0.11, "gen_tps": 880.2,
               "prompt_tps_incl_cache": ..., "req": 812, "err": 0, "p90_s": 24.0}] }
                                       // last 24 h, one point per cycle (append-only file data/history/fleet.jsonl,
                                       // trimmed to 7 days)
```
tps rates come from the deltas of vLLM's prompt_tokens_total/generation_tokens_total counters between two scrapes
of the same job id; the prompt counter includes prefix-cache hits, hence `prompt_tps_incl_cache` (v1.2; `prompt_tps`
is the same value under its old name, kept for one release). `busy_workers` (v1.2): routable workers whose last
scrape shows at least 1 running request. Router stats come from every official run's
`.../round-*/probe-*/student-router-requests.jsonl`, read incrementally from saved offsets: `router.scope` (v1.2)
says so, other users of the fleet are not in them.

## data/api.json (every cycle; the hub sample at most every 10 min)
```
{ "schema": 1, "generated_at": ...,
  "hub": {"spend": 468.75, "max_budget": 10000.0, "remaining": 9531.25, "status": 200, "sampled_at": ...,
          "key": "NVAPI_KEY_NEW", "error": null},
  "burn": {"per_hour_1h": 75.1, "per_hour_6h": 70.0, "window_min_1h": 58.4 | null, "window_min_6h": 351.0 | null,
           "eta_exhaust": 1792..., "eta_exhaust_iso": "..." | null},                       // window_min_*: v1.2
  "history": [{"t": ..., "spend": ..., "max_budget": ...}],    // data/history/hub.jsonl (trimmed to 30 days)
  "by_run": [{"run": "official1-bandit",
              "kimi": {"calls": 9000, "cost": 110.15, "input_tokens": ..., "cached_tokens": ..., "output_tokens": ...},
              "codex": {"sessions": 150, "input_tokens": ..., "cached_tokens": ..., "output_tokens": ..., "reasoning_tokens": ...},
              "glm": {"calls": 105}}],
  "scan": {"complete": true, "pending_bytes": 0, "files": 459, "read": 13, "sealed_probes": 6},
  "notes": ["kimi cost is hub-reported per call; codex and glm cost is not logged (tokens only)"] }
```
The hub sample comes from running `/usr/bin/python3 -I -B S/tools/hub_budget.py --min-remaining 0`, a subprocess
that reads the key itself; parse its one JSON line. The collector never touches a key. kimi usage comes from the
probe `models/kimi/attempts/*/attempt-*/http.jsonl` events with usage (`upstream_stream_completed`; inspect the
real shape). codex usage comes from each agent workdir's `.agent/events.jsonl` `turn.completed` usage. glm calls
are counted from `teacher_audit.json` entries plus the similarity calls (similarity_detail present). Everything
here is read incrementally with saved offsets, never by rescanning 40 GB of probe dirs each cycle.

## UI (static/)
- Header: title "swe-task official synthesis". A searchable task selector, grouped: "Converting" (has run),
  "Eligible queue" (by queue_rank), "All tasks". Selection is kept in the URL hash (`#task=<id>`, `#monitor`).
  Tabs: Task | Monitor. A small "updated hh:mm:ss" indicator (the collector cycle on screen) with its age, and a
  staleness warning when > 5 min old. Auto-refresh every 60 s (refresh rules: "UI additions (v1.2)" below).
- Task tab:
  1. Accuracy cards for kimi / super_vl / sc14: `passed/total` big, then 5 rollout chips (green pass, red fail,
     grey unscored/infra) labelled with the F2P fraction (an unscored or infra rollout reads "–", never its "0/0");
     the sc14 card also shows the mean F2P ratio; badges for eligible / queue rank.
  2. If a run exists:
     - run status bar: run, repo, job state, stage, accepted count, a collapsible log tail;
     - failure patterns: cards (name, category, description, student slots; teacher contrast collapsible), the
       transferable signature, and the full markdown collapsible (rendered as preformatted text);
     - plan: the 10 candidates, built ones marked (collapsible);
     - "Round 0 at a glance" (above the tree): one row per round-0 node: name, title, F2P/P2P, sc14 x/5, kimi clean
       x/5, infra counts of both models, decision badge, reason; a row opens the node's detail panel;
     - generation tree: columns = rounds 0..max_rounds, rows = lineages. Each lineage block starts from its
       round-0 node; children sit in the next column; SVG connectors run parent -> child. Compact node card:
       name, short title, `F2P n / P2P n` (+ an evolve node's target pattern tag `P<n> · <name>`), `sc14 x/5`,
       `kimi x/5` (clean), each with its infra count, decision badge (EVOLVE / DISCARD / ACCEPTED / ACCEPTED+EVOLVE /
       NOT SELECTED / END / PENDING / ABANDONED / BUILD FAILED) with colors (v1.5: an accepted invalid seed's ACCEPTED
       is preceded by a small `seed-invalid` badge). While probing: each model's partial
       results (`sc14 1 pass / 3 of 5 done`) over a slim progress bar; once the node's own slots are done, "probe
       done, scoring when the round's probe ends". Column headers: probe slots (or `built x / y` while the round's
       tasks are built) and `seeds: N valid · M invalid`. Clicking a node opens a detail panel: summary, target
       pattern, build/gates (F2P list, problems), rollouts table (long agent/verifier errors in full behind a
       collapsible), score + similarity detail, decision reason, instruction / evolve plan / synthetic failure
       analysis / rubric (collapsible preformatted), teacher audit (a table: slot, cheated, categories, summary;
       preformatted for the older string form). Abandoned nodes are hidden behind a toggle.
  3. With no run: a note "not converted yet" (plus queue rank when eligible).
- Monitor tab:
  - summary cards: routable/target, utilization as busy workers N/M (`totals.busy_workers`, else counted from
    `workers[].metrics`), running and waiting requests, mean KV usage, gen tok/s (the throughput figure; prompt tok/s
    only as "(incl. prefix-cache hits)"), requests (10 min), error rate, p50/p90 latency (router stats labelled with
    `router.scope`), daemon heartbeat;
  - worker table: slot, job, partition, node, status (colored), ready since, time left, heartbeat, running,
    waiting, KV %, gen tok/s, router requests and p90 (10 min);
  - charts as inline SVG (no libraries) over the last 24 h: routable workers, running requests, gen tok/s,
    requests per cycle;
  - API panel: spend / max / remaining with a progress bar (red when remaining < $500), the hub sample's age, burn
    $/h (1 h, 6 h) with the window each was measured over, ETA to exhaustion, a spend-over-time chart with a budget
    line, the usage scan's health (`scan.complete` / `pending_bytes`), and a per-run table (kimi $ / calls / tokens,
    codex sessions / tokens, glm calls) plus the notes.
- Robust to null/missing fields everywhere (never throw on partial data); escape every string inserted into HTML.

## Security
- The server requires a random token (`secrets.token_urlsafe(32)`), stored in `D/state/server.json` (mode 600)
  together with host/port/url. First request: `/?token=<t>`, which sets an HttpOnly SameSite=Strict cookie;
  later requests need the cookie (or header `X-Dashboard-Token`). Otherwise 403. Use constant-time compares.
  The cookie is per port and never the token (browsers send the cookies of localhost to every port there): name
  `swe_dash_<port>`, value HMAC-SHA256(token, `"session:<port>"`) in hex, verified by recomputing it; the header takes
  the token itself. Only the token is persisted (a `--reuse-token` restart on the same port keeps the sessions).
- Serve only `static/` and `data/` (no path traversal, no directory listings). All responses carry
  `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, except that the files of `static/` and `data/`
  carry `Cache-Control: no-cache` with `ETag` and `Last-Modified` (304 on a matching `If-None-Match`, else
  `If-Modified-Since`; nosniff and the CSP as everywhere). Only GET/HEAD. `/healthz` (open) returns `{"ok": true}` only.
- At most 32 connections are handled at once (beyond: 503 at once); request socket timeout 15 s.
- Nothing logged holds the token: every occurrence of it (and of the session value) and every `token=...` value is
  masked. Tools send the header from a file of mode 600 (`curl -H @$HOME/.dashboard-header`, `launch.sh header`),
  never with the token on a command line.
- Bind `127.0.0.1` by default; `--host 0.0.0.0` only inside the Slurm job (reached by port forwarding).

## Integration notes (v1.1, 2026-10-17)
Settled while integrating the three parts; `tests/validate_contract.py` enforces them (and everything above).
- One cycle, one stamp: index.json, every task file, fleet.json and api.json of a cycle carry the same `generated_at`
  (the page refetches the large task files only when index.json's `generated_at` changes).
- A run launched before its build directory exists: its task file has `repo`/`build_root` null and `rounds`/`nodes`
  []. Otherwise `rounds[]` always covers 0..max_rounds (rounds not started: `pending`).
- Every run's task has a `tasks[]` entry; a task the targets do not list gets a stub (accuracy all null, eligible false,
  `task_index` after the listed ones), so the page still shows its conversion.
- Fractions: `"None/None"` becomes null; a real `"0/0"` stays.
- fleet.json `counts` are the fleet daemon's own buckets (not recounted from `workers`). `workers[].routing`
  (student/fleet.py worker_routing: ready / draining / null) decides the status the page shows for a ready/draining
  registry record, as the daemon does: its routing, or `stale` when not routable. A failed `/metrics` scrape gives
  `metrics` with the same keys, values null and `error` set.
- api.json `burn.eta_exhaust_iso` projects at the HIGHER of the 1 h and 6 h rates (conservative while probes burst;
  the page names the one used). A dip of the hub's spend counter by < max($5, 2 %) does not start a new key stretch.
  `by_run[].kimi` / `codex` are null for a run without such usage yet. The histories are absent until the first
  cycle / hub sample.
- A broken source never stops a cycle: it adds `"<source>: <message>"` to index.json `errors`; fleet.json / api.json
  may then lack the sections that source feeds (`fleet`, `usage`, `api`).
- Additive fields (all optional for readers): job `status`, `time_left`, `submitted_at`, `started_at`, `ended_at`,
  `jobs`; rounds[] `probe.dirs`, `probe.live`, `nodes`, `built`, `scored`, `accepted`; NODE `probe.dir`, `probe.live`,
  `score.student_rate`, `score.teacher_clean_rate`, `score.cheated`; plan candidates `target_patterns`, `status`,
  `decision`; fleet `daemon.stopped_at`/`stop_flag`, workers `routing`/`parity`/`registry_reason` and the metrics token
  counters, `totals.scraped`/`scrape_errors`, `router.disconnects`; api hub `exit_code`/`last_attempt_at`/`age_s`,
  `burn.eta_exhaust`, kimi `reasoning_tokens`/`http_errors`/`files`, codex `turns_completed`/`turns_failed`, `scan`;
  index sc14 `rollouts[].slot`/`status`. data/server_status.json (server.py, only while it collects): the collector's
  health, never the token, port or URL.
- UI: `#task=<id>&node=<id>` opens a node's detail panel. Server: the session cookie is per port (Security; it was
  named after a hash of the token and held the token itself in v1.1, and such a cookie is expired at the next login);
  CSP allows inline style attributes only (`style-src-attr 'unsafe-inline'`), never inline script or `<style>`.

## Collector additions (v1.2, 2026-10-17)
All optional for readers: the page tolerates them missing (a v1.1 collector's output lacks them), and
`tests/validate_contract.py` checks them whenever they are present.
- NODE `focus_pattern`: for an evolve node, the target failure pattern its branch was aimed at, picked as task_builder
  evolve() does: `patterns[branch % len(patterns)]["name"]` of `runs/<run>/analysis/<task>/result.json`, `"overall"`
  when it lists no pattern, null when that file cannot be read. Mask nodes: null. `validate_contract.py --sources`
  compares it with the `Focus pattern for this branch: "..."` line of the node's `evolve/.agent/prompt.md`.
- NODE `probe.partial` / `probe.own_done`, on probing nodes only (`probe.live`): per model `{done, total, pass, fail,
  infra}` over the node's own slots that are final so far (graded: pass when the grading reward is 1, else fail; infra:
  the last allowed attempt ended without a grade), from the same incremental live-probe scan as `progress` (equal to
  its `[done, total]`; pass/fail/infra equal `student` / `teacher`). `own_done`: every own slot is final; the node
  stays `probing` until the whole campaign is scored and metrics.json exists.
- NODE `details.teacher_audit`: a list, one entry per audited (passing) teacher rollout in slot order: `{slot, cheated,
  categories, summary, rubric_score}`; `summary` is the judge's notes (else its evidence quotes), redacted, then cut
  to 400 chars. `[]` when the teacher passed no rollout, null when not scored. A v1.1 collector wrote the JSON-dumped
  teacher_audit.json instead (a string, <= 6000 chars): readers accept both shapes.
- fleet.json: `totals.busy_workers`; `prompt_tps_incl_cache` in the workers' metrics, the totals and the history points
  (`prompt_tps` carries the same value under its old name for one release, then goes); `router.scope`.
- api.json `burn`: each rate runs from the oldest to the newest hub sample inside its window (1 h / 6 h back from the
  newest sample, within the newest stretch of one budget and rising spend); `window_min_1h` / `window_min_6h` are the
  spans actually measured, in minutes. A window holding fewer than 2 samples gives null for both (never extrapolated
  from older samples). `eta_exhaust` (and `eta_exhaust_iso`) = the hub sample's `sampled_at` + `remaining` / rate (the
  higher of the 1 h and 6 h rates, as in v1.1 above; corrected 2026-10-20 from "the 6 h rate, else the 1 h one"), so
  the projection does not drift while the sample ages between hub calls.
- api.json `scan`, the cycle's incremental log scan: `complete` (every log read to its end), `pending_bytes` (left for
  the next cycles; 0 when complete), `files`, `read`, `sealed_probes`.

## UI additions (v1.2, 2026-10-17)
The page reads the collector additions above when present and falls back otherwise: partial results from
`probe.progress` with the so-far `student` / `teacher` counts (all slots done counting as `own_done` when the flag is
absent), no target-pattern tags or per-pattern counts without `focus_pattern`, the teacher audit as text when it is a
string, busy workers counted from `workers[].metrics` (routable workers with `running` >= 1), `totals.prompt_tps`,
scope "official-run probes", the ETA from `eta_exhaust_iso`, and the burn tiles without their windows.
- Refresh: fetches use `{cache: 'no-cache', credentials: 'same-origin'}` (an unchanged file revalidates to a 304; a
  304 keeps the copy held). The main view and the drawer re-render only when what they show changed (a new
  `generated_at`, or a user action); the header's banner is rewritten only when its markup changes. An automatic
  refresh that would re-render while the page has a non-empty text selection is held ("newer data held" next to the
  "updated" time, which keeps showing the cycle on screen) and retried every 15 s and at the next 60 s tick; clicks
  and the Refresh button re-render at once. A re-render of the same task / tab keeps the window scroll, every
  element marked `data-scroll="<stable key>"` (the tree container `tree:<task>`, `<pre>` blocks and tables keyed by
  their collapsible's key, e.g. `log:<run>`, `nd:<node>:instruction_md`), the drawer's scroll (same node) and every
  `<details data-key>` open state; another task or tab starts at the top. The capture / restore and skip / hold /
  render decisions are pure helpers (`captureScroll`, `restoreScroll`, `captureOpen`, `refreshAction`,
  `hasTextSelection`), unit-tested with the rest in `tests/frontend.test.js`.
- Drawer: from 1100 px wide, the open drawer pads the main column by its width (below that it overlays). Its width
  is adjustable with the grip on its left edge (`#drawer-grip`: drag, or Left / Right arrow keys; >= 360 px, leaving
  >= 360 px of page) and kept in `localStorage` (`swe-task-dashboard.drawer-width`); the tree re-measures its cards
  afterwards. Opening a node from a card or a drawer link scrolls its tree card into view (CSS scroll margins keep
  it clear of the sticky header and of an overlaying drawer); from the round-0 table, the row keeps the focus.
- Failure patterns: with `focus_pattern`, each pattern card shows how many evolve branches targeted it, how many of
  them were accepted and how many are still in progress (abandoned nodes left out); targets that match no pattern
  (`"overall"`) are listed below the cards. Cards number patterns P1..Pn as the tags do.
- Teacher audit table: an extra `rubric` column when an entry carries a numeric `rubric_score`; open by default when a
  slot cheated.

## Failure-match audit (v1.3, 2026-10-18)
User request (hengyuf): check, for every task accepted so far, whether the student sc14 really makes mistakes like those
of the target's failure analysis there, and show it on the page. Auditor agents (outside the dashboard) write one file
per accepted task (since v1.4: per eligible task, below); the collector only reads them. Optional for readers like the
v1.2 fields: the page shows nothing about audits when the collector does not send them, and `tests/validate_contract.py`
checks them whenever they are present.

Source: `S/synth/runs/<official run>/audit/failure_match/<task_id>.json` (UTF-8 JSON; `<task_id>` = the node's
`build.task_id`, e.g. `syn-pylama-cand-c03-r1b2-6182c7b3`), read through the read roots like every pipeline file
(`runs/` is one; regular files only, at most 1 MB). Format, as the auditors are told:
```
{ "schema": 1, "task_id": "...", "run": "official1-...", "cand": "cand-...", "round": 1, "audited_at": "<ISO time>",
  "auditor": "claude-subagent", "target_patterns": [{"name": "...", "category": "..."}],
  "verdict": "strong" | "partial" | "weak" | "none",                  // does sc14 repeat the target failure patterns?
  "summary": "<= 3 sentences",
  "patterns": [{"name": "<target pattern name>", "reproduced": "yes" | "partial" | "no", "slots": ["sc14-r1", ...],
                "evidence": "<= 600 chars"}],
  "student_rollouts": [{"slot": "sc14-r0", "status": "pass" | "fail" | "infra", "failing_tests": ["<= 8 names"],
                        "cause": "<= 300 chars", "pattern": "<target pattern name>" | null}],
  "other_causes": "<= 400 chars",
  "similarity_judge": {"score": 7, "agrees": true | false, "comment": "<= 300 chars"} }
```
Verdicts: `strong` most failed sc14 rollouts fail the way a target pattern describes (the same kind of mistake, not just
the same area); `partial` some do, or the match is loose; `weak` a match in one rollout only, or only superficial; `none`
the failures have other causes.

Collector (`Cycle.failure_matches`, `_failure_match`): a file is used when it parses as a JSON object, `schema` is 1
(or absent), `task_id` equals the file's name (or is absent), and `verdict` is one of the four (enums match after
trimming and lower-casing). Such a file is attached to the node of that run whose `build.task_id` it names, whatever the
node's decision. A file that parses but breaks one of these rules (or is over 1 MB) is ignored and reported in
index.json `errors` as `run <run> audit: <file>: <why>; ignored`; one that does not parse (perhaps being written), a
non-regular file, a symlink out of the read roots, and names that are not `<task_id>.json` (hidden temp files of an
atomic write, backups) are skipped silently. Unknown fields (`run`, `cand`, `round`, `auditor`, `target_patterns`, ...)
are dropped. Every text is redacted, then clipped; lists are capped; wrong types become null / [].
- NODE `failure_match`, on every node: null when no valid file names its task (always for a node not built), else
```
  { "verdict": "strong|partial|weak|none", "summary": "<= 1000 chars" | null,
    "patterns": [{"name": "<= 300" | null, "reproduced": "yes|partial|no" | null, "slots": ["<= 20 slots, <= 100 chars"],
                  "evidence": "<= 600" | null}],                                            // <= 20 entries
    "student_rollouts": [{"slot": "<= 100" | null, "status": "pass|fail|infra" | null, "failing_tests": ["<= 8, <= 300"],
                          "cause": "<= 300" | null, "pattern": "<= 300" | null}],           // <= 10 entries
    "other_causes": "<= 400" | null,
    "similarity_judge": {"score": 7.0 | null, "agrees": true | false | null, "comment": "<= 300" | null} | null,
    "audited_at": "<ISO string as written, <= 64>" | null }
```
  Empty strings become null; `slots` / `failing_tests` given as one string are split at commas and blanks.
- index.json `runs[].audit`: `{"accepted": n, "audited": n, "strong": n, "partial": n, "weak": n, "none": n}` over the
  run's accepted tasks (decision accepted / accepted_evolve; `accepted` = `runs[].accepted`): `audited` of them carry a
  `failure_match`, the four verdict counts sum to `audited`. An audit attached to a task that was not accepted is not
  counted. All zeros for a run without a tree.
- `validate_contract.py` checks the shapes and limits above, the counts against the task file's nodes, and (`--sources`)
  that each attached verdict is its file's.

UI (the page, when the collector sends `failure_match`; nothing changes for an older collector):
- Accepted node cards end their last line (the kimi count, which yields room) with a small badge `FA match: strong |
  partial | weak | none | not audited` (green / amber / orange / red / grey; tooltip: what the verdict means and the
  summary). Other cards are unchanged, also when an audit names their task.
- The drawer, after Score: section "Failure-pattern match (audit)": the verdict (callout in its color) and summary; the
  similarity judge's score with "agrees" / "disagrees" and its comment, other causes, when it was audited; a table per
  target pattern (pattern `P<n> · <name>` numbered like the pattern cards when it is one of the target's, reproduced
  yes / partial / no, sc14 slots, evidence) and one per sc14 rollout (slot, status, failing tests, cause, matched
  pattern). An accepted task without an audit says "Not audited yet"; other nodes without one have no section.
- Task view, between "Round 0 at a glance" and the tree: "Accepted tasks: failure-match audit", one row per accepted task
  by round (task and title, round, sc14 x/y, clean kimi x/y, similarity, verdict badge, the summary's first line cut to
  one line; in full on hover and in the drawer), headed by the counts (`audit n/m` and the verdicts). A row opens the
  node's drawer and keeps the focus (`data-fkey="audit:<id>"`, like the round-0 rows). No table without accepted tasks.
- Run status bar: a cell "Failure-match audit" reading `audit n/m` (audited / accepted) with the verdict counts and how
  many are not audited; from `runs[].audit`, else counted from the task file's nodes.
These UI rules hold for a collector without `audit_eligible`; v1.4 below extends badge, table, run bar and drawer from the
accepted tasks to every eligible one.

## Failure-match audit over the eligible tasks (v1.4, 2026-10-19)
User request (hengyuf, 2026-10-19): audit every task the run generated "只要没有discarded, 且student 在上面有rollout" (as
long as it is not discarded and the student has rollouts on it), not only the accepted ones; a watcher launches auditors
whenever new tasks and rollouts appear. The audit file format is unchanged (v1.3; `similarity_judge.score` may be null,
e.g. for a task never judged for similarity). Optional for readers like v1.2 / v1.3: the page keeps the v1.3 behaviour
(the accepted tasks) when the collector does not send the scope, and `tests/validate_contract.py` checks the fields
whenever they are present.

A task is ELIGIBLE (NODE `audit_eligible`) when all of these hold:
- it was built and probed: its scored dir `round-<k>/<cand>/` has metrics.json and rollouts/ (a probing task is not
  eligible until its round's campaign is scored);
- it has at least one graded sc14 rollout (rollouts/index.md status pass or fail; NODE.probe.rollouts);
- it is not discarded: not invalid in its round's seeds.json (task-level infrastructure problems: grading failures,
  uniform failure, fewer than 3 valid rollouts per model; before that file is written, seed_check() recomputed as for
  the decision) unless it was accepted (v1.5: an accepted task is never discarded), not withdrawn by main, not under an
  abandoned-* build root.
Accepted tasks, tasks rejected on pass rates and tasks rejected on similarity are all eligible; so is a scored task the
tree shows as abandoned (it was built and probed). Never eligible: build failures, invalid seeds that were not accepted,
withdrawn tasks, tasks not scored yet, tasks whose sc14 rollouts all ended without a grade.

Withdrawal. Main may withdraw an accepted task afterwards (2026-10-18: grading artifacts, a teacher that retrieved the
answer). It rewrites the task's metrics.json (`accept` false, `reason` "withdrawn: <why>", `"withdrawn": {"at", "by",
"why", "was": {<the fields it changed>}}`) and moves the task dir from runs/<run>/accepted/ to runs/<run>/withdrawn/
(logged in withdrawn/withdrawn.jsonl). Either sign is enough; a withdrawn task counts as discarded.
- NODE `withdrawn`, on every node: null, or `{"at": "<as written, <= 64>" | null, "by": "<= 100" | null, "why": "<= 600"
  | null}` from metrics.json's record (another non-null value counts as withdrawn too: a string is the why), redacted and
  clipped; all three null when only the task dir moved.
- Decision: never accepted / accepted_evolve; decided like a scored task that was not accepted (round 0: evolve; later
  rounds: evolve / not_selected / end / pending), the reason being main's rewritten one, else "withdrawn: <why>" or
  "withdrawn (its task dir is in runs/<run>/withdrawn/)". The seed selection of its round (task_builder's best valid
  non-accepted child per lineage) still counts it as accepted when it was then: `withdrawn.was.accept`, true when the
  record does not say (only accepted tasks are withdrawn). official1-bandit cand-c05-r1b1 was accepted when
  cand-c05-r1b2 was picked to seed round 2; without this, cand-c05-r1b2 read as not selected and its five scored children
  (two of them accepted, in runs/<run>/accepted/) as abandoned.

Abandoned build roots. Main sets an abandoned build aside (official1-httpx, official1-testem: runs/<run>/abandoned-<repo>,
outside build/, never read). Should one sit in build/<task>/ as `abandoned-<repo>`, the collector takes the run's other
root; a tree read from an abandoned-* root (the only one there) is shown, and none of its tasks is eligible.

Collector:
- NODE `audit_eligible` (true / false) and NODE `withdrawn` (above), on every node.
- index.json `runs[].audit` adds, over the run's eligible tasks (every node with audit_eligible, abandoned ones
  included): `"eligible": n, "audited_eligible": n, "eligible_verdicts": {"strong": n, "partial": n, "weak": n,
  "none": n}`; audited_eligible of them carry a failure_match, the four verdict counts sum to it. The v1.3 accepted-only
  counts (accepted, audited, strong, partial, weak, none) stay as they are. All zeros for a run without a tree.
- An audit file is still attached to the node it names whatever its scope (a withdrawn task's earlier audit shows in
  its drawer); only eligible nodes count.

`validate_contract.py`: audit_eligible recomputed from each node's score, graded sc14 rollouts, its round's seeds.json
(else a discard decision), withdrawal and build root; NODE.withdrawn's shape and that it is never accepted; the
decision rules with withdrawal; the index's eligible counts against the nodes, present exactly when the nodes carry
audit_eligible. `--sources`: withdrawn exactly when metrics.json says so or the task dir is in runs/<run>/withdrawn/;
an eligible node has its round-<k>/<cand>/metrics.json and rollouts/.

UI (the page, when the collector sends audit_eligible):
- Every eligible node card ends its last line with the badge `FA match: <verdict> | not audited` (colors and tooltip as
  v1.3), not only accepted cards. A card outside the scope has none, even with an audit attached; a withdrawn task's
  card tooltip names the withdrawal.
- The table between "Round 0 at a glance" and the tree becomes "Failure-match audit": one row per eligible task, grouped:
  "Accepted" first, then "Round <k> · not accepted" for the other eligible tasks round by round, each group under a
  header row with its task count and `audit n/m`. Columns: task and title, round, decision (badge; the decision reason
  as its tooltip), sc14 x/y, clean kimi x/y, similarity, verdict badge, the summary's first line. Headed by `<n> eligible
  tasks · audit n/m` with the verdict counts and how many are not audited. A row opens the node's drawer and keeps the
  focus (`data-fkey="audit:<id>"`); a group header row opens nothing. No table when no task is eligible.
- Run status bar, cell "Failure-match audit": `audit n/m` over the eligible tasks (audited_eligible / eligible) with
  their verdict counts and how many are not audited, then a line "accepted: audit a/b" with the accepted-only counts;
  from `runs[].audit`, else counted from the task file's nodes.
- Drawer: an eligible task without an audit says "Not audited yet: no failure-match audit of this eligible task"; the
  audit of a task outside the scope is shown under "Not counted: this task is outside the audit's scope, <why>"
  (withdrawn, discarded with its reason, not scored yet, ...); a withdrawn task gets a "Withdrawn" callout (by whom, when,
  why) under its decision.

## Acceptance beats an invalid seed (v1.5, 2026-10-21)
Found by the dashboard keeper: official1-wasmi round-1 cand-c03-r1b2 (`syn-wasmtime-cand-c03-r1b2-03550fa4`) is accepted
by the pipeline (runs/<run>/accepted/ and its tasks.jsonl row, metrics.json `accept` true; the build summary counts 6
accepted) and listed invalid in round-1/seeds.json ("verifier produced no grade for 1 rollout(s): kimi-r3", an
eviction). The rule "seeds.json invalid: discard" beat the acceptance: the node read DISCARD, the run "finished (5
accepted)", and `validate_contract.py --sources` failed "accepted dir [6] != accepted nodes [5]".
Decision (main, 2026-10-21): acceptance wins for display. task_builder build() accepts a task on its score alone and
copies it to accepted/ before seed_check() / valid_seeds() pick the tasks that seed the next round; seed validity only
governs evolution (with `--rounds 1` the round-1 seeds are not even used).
- Decision: a scored task whose metrics.json accepts it and that main did not withdraw (v1.4) is `accepted` /
  `accepted_evolve` whatever its round's seeds.json (or, before that file, the recomputed seed_check()) says. An invalid
  seed is never `accepted_evolve` (it seeds no branch): in round 0 it is `accepted`, "accepted (<reason>); an invalid
  seed, so not evolved". An invalid seed that was not accepted is still `discard` with its reason; a withdrawn task
  stays not accepted (so a withdrawn invalid seed is discarded). The children of an invalid seed stay `abandoned`.
- NODE `seed_invalid`, on every node: null, or one line (redacted): why the scored task may not seed evolution, the
  `invalid` reason of its round's seeds.json (null when that file lists it as valid), else seed_check() recomputed
  (grading failures, uniform failure, fewer than 3 valid rollouts per model). Null for a task not scored and for a
  failed build. A discard other than a failed build carries it as its decision_reason; on `accepted` it is the note;
  no other decision goes with it (`abandoned` aside, whose reason is about the node's parent or probe).
- Audit scope (v1.4): such an accepted task is not discarded, so it is eligible when built and probed with a graded sc14
  rollout (not withdrawn, not under an abandoned-* root). official1-wasmi cand-c03-r1b2 (audited: partial) joins them.
- Counts follow the decisions: runs[].accepted, the stage line "finished (N accepted)", rounds[].accepted and
  runs[].audit (official1-wasmi: 6 accepted, 23 eligible).
- `validate_contract.py`: seed_invalid's shape and rules (its round's seeds.json reason; null for a valid seed, a task
  not scored or a failed build; a discard's decision_reason; with a note the decision is `accepted` when the task is
  accepted, else `discard`); a node seeds.json lists invalid is `accepted` when its score accepts it and it is not
  withdrawn, else `discard`; a round-0 accepted invalid seed is `accepted`; an accepted task is never discarded;
  audit_eligible accordingly. `--sources`: each seeds.json `invalid` entry whose metrics.json accepts it and that is not
  withdrawn (metrics.json `withdrawn`, runs/<run>/withdrawn/) is `accepted` with the reason as its seed_invalid, every
  other one `discard` with its reason; accepted/ holds exactly the accepted tasks (unchanged: a withdrawn task left in
  accepted/ is a mismatch). The decision rules are checked as they are now, so an older collector's DISCARD of an
  accepted task is a problem; seed_invalid itself is optional for readers, like the v1.2-v1.4 fields.
- UI (when the collector sends seed_invalid; nothing changes for an older one): a note that did not decide the node
  (seedNote(): set, decision not `discard`) adds a small amber badge `seed-invalid` (tooltip: the reason, and that seed
  validity only decides which tasks seed the next round, never the acceptance): on the tree card before the decision
  badge, the two at the right edge, with a line `seed-invalid: <reason>` in the card's tooltip; after the decision
  badge in "Round 0 at a glance" and in the failure-match audit table. The drawer adds a callout "Seed-invalid"
  (amber) under the decision: the reason, then where it comes from (`round-<k>/seeds.json`, or "seed_check()
  recomputed") and that rule. A discard shows none: the note is its decision reason.

## Batches from another tree: Baihe's batch, not audited (v1.6, 2026-10-22)
User request (hengyuf, 2026-10-22): "另外baiheh 在 .../baiheh/swe-task-hengyu/outputs 也有一批tasks, 他没有做audit. 先把这批
task 也接到网页dashboard里, 不要做audit." Baihe has another batch of tasks there and did not audit them: add it to the page,
and do NOT audit it. Optional for readers like v1.2-v1.5: an older page ignores the batch (its runOf() takes official*
runs only), and `tests/validate_contract.py` checks every field below whenever index.json has `batches`.

A BATCH is the outputs/tasks.jsonl of another tree running a copy of this pipeline. Baihe's: `B=/lustre/fsw/portfolios/
nemotron/users/baiheh/swe-task-hengyu` (source `baiheh`, name "Baihe", label "Baihe · swe-task-hengyu"), a fork of S with
the same layout, models (sc14 student, kimi teacher, glm judge) and launch.jsonl. `B/outputs/tasks.jsonl` holds one Gym
dataset row per task (97 on 2026-10-22); `B/outputs/<task_id>/provenance.json` is that task's metrics.json, whose
`task_dir` = `B/synth/runs/<run>/build/<target>/<repo>/<cand>/task/<task_id>` names the run and DeepSWE target that built
it (97 tasks from 19 runs on 18 targets; pilot7-tengo-callable-isolation and pilot10-tengo-golua-inline both build
tengo-callable-instance-isolation). The batch's RUNS are exactly the runs that built one of its tasks; the other runs
under B/synth/runs (smoke1, codex-check-*, pilot4, pilot5, pilot11, q11, q16, q16b, ...) are never read.

Collector (`Batch`, `BATCHES`, `Cycle.batch_docs`):
- Read roots: `B/outputs` and each batch run's directory `B/synth/runs/<run>`, nothing else of B (never secrets/, .env
  files, atp/, the Slurm logs B/synth/runs/slurm/, other runs). Redaction as everywhere (`redact()`, hengyuf's secret
  values); no new secret source.
- No audit: no `audit/failure_match/` file of a batch is read (none exists) and none is written. Every batch NODE has
  `failure_match: null`, `audit_eligible: false` (it is in no audit's scope: the v1.4 scope and counts stay hengyuf's
  official runs') and `audit: {"status": "not_audited"}`; the run's `runs[].audit` and the doc's `audit` are
  `{"status": "not_audited"}` (no counts). The teacher's cheat audit `teacher_audit.json` is Baihe's own score step and is
  shown in the details as for any run.
- Cost: a batch's runs have ended, so their Slurm jobs are not polled, their logs not read and their probes never scanned
  (no live probe); their kimi/codex/router logs are not part of the usage scan (api.json `by_run`, fleet.json `router`
  stay official). Each run is built by the official code (`read_run`, `Tree`, `failure_analysis`, `plan`) after the
  official work of the cycle. A run's document is rebuilt only when its fingerprint changes (stats of its build root and
  round dirs, summary.json, plan/stage.json, launch.jsonl, accepted/ and accepted/tasks.jsonl, the target's result.json,
  its batch tasks and settings), and in any case once it is 30 min old (at most 2 such rebuilds per cycle); its file is
  rewritten only when its content changed. 2026-10-22: a warm cycle spends < 1 s on the 19 runs, a cold one (new
  process) about 50-90 s, once.
- Settings: the batch tree's own task_builder defaults (Baihe: `--rounds 1`, "evolve once, no repeated evolution in all
  runs"; `--branches 5`), unless launch.jsonl "extra" sets them (or a max_rounds.json caps them).
- Status: `finished` with `<build root>/summary.json`, else `stopped` (q21-boa-...: job 20640765 cancelled 2026-10-22
  09:15, no summary.json). A resumed run reads `stopped` until its summary.json appears. Stage: `finished (N accepted)`;
  a stopped run: `stopped: round-<k> unfinished (<s>/<n> tasks scored[, <u> probes unfinished])` (its first round not done).
- Probes: an incomplete probe of a batch run is UNFINISHED, never live and never superseded (Baihe's fork probes every
  task alone, keyed by that task: a newer probe never replaces it); the slots of its scored tasks count as done. Rounds:
  `done` (every task scored or failed to build), `unfinished` (some never will be), `pending` (never begun); no gating by
  the previous round (each lineage goes on alone in Baihe's fork). Decisions: the official rules (CONTRACT "Decision
  rules", v1.4, v1.5) with max_rounds 1, so a valid round-1 task that was not accepted is `end`. An unscored built task of
  an ended run is `pending` with the reason "its probe round-<k>/probe-<key> never finished: the run <finished|stopped>
  before scoring it" or "built, never probed: the run <...> before probing it" (pilot2 cand-c02: its score failed;
  q21: 12 round-1 tasks).
- NODE `in_batch` (batch nodes only): its `build.task_id` is one of the batch's tasks that provenance.json attributes to
  this run and target. Every task of the batch must be an accepted node of its run: each one that is not (or whose
  provenance.json is missing or malformed, or whose run/target has no build) is listed in `batches[].unmatched` and in
  index.json `errors` as `batch <source>: outputs task <task_id>: <why>`; an accepted node that is not in the batch goes to
  `batches[].not_in_batch` and `errors` too. 2026-10-22: none; 97 in_batch nodes, 97 accepted.

index.json (official entries unchanged; a `runs[]` entry WITHOUT `source` is hengyuf's official run):
```
  "runs": [ ...official..., {"run": "pilot7-tengo-callable-isolation", "task_id": "tengo-callable-instance-isolation",
            "repo": "Shopify/go-lua", "job_id": "20518448", "slurm_state": null, "partition": "cpu_dataprocessing",
            "submitted_at": 1792..., "started_at": null, "ended_at": null, "status": "finished|stopped",
            "stage": "finished (6 accepted)", "rounds_started": 2, "accepted": 6, "tasks_built": 30, "tasks_scored": 30,
            "audit": {"status": "not_audited"},
            "source": "baiheh", "source_name": "Baihe", "source_label": "Baihe · swe-task-hengyu",
            "root": "<B>/synth/runs/pilot7-tengo-callable-isolation",
            "batch": 6,                                  // its tasks in B/outputs/tasks.jsonl
            "detail": "run/baiheh/pilot7-tengo-callable-isolation.json", "detail_generated_at": 1792...} ],
  "tasks": [ {... as before, "run": null | "<official run>",
              "batch_runs": [{"source": "baiheh", "run": "pilot7-tengo-callable-isolation"},
                             {"source": "baiheh", "run": "pilot10-tengo-golua-inline"}]} ],   // only when it has some;
                                                                                     // oldest launch first
  "batches": [ {"source": "baiheh", "name": "Baihe", "label": "Baihe · swe-task-hengyu", "root": "<B>",
                "outputs": "<B>/outputs", "tasks": 97, "runs": 19, "targets": 18, "in_batch": 97, "accepted": 97,
                "unmatched": [], "not_in_batch": [], "audit": {"status": "not_audited"}} ]   // the batch's own totals
```
A batch target the targets do not list gets a stub task entry as a run's does (none: all 18 are DeepSWE tasks). A run
that builds several targets (none so far) gets one entry and one file per target, `run/<source>/<run>--<target>.json`.

data/run/<source>/<run>.json (one per batch run): the document of data/task/<task_id>.json (same keys, NODE, rounds,
decision rules) of that run's own tree, plus `source`, `source_name`, `source_label`, `root` (the run dir), `outputs`, `batch` and
`audit: {"status": "not_audited"}`. Differences: `generated_at` is the cycle that last CHANGED the file (=
runs[].detail_generated_at, never later than index.json's; "one cycle, one stamp" holds for every other file); `job` has
job_id / partition / submitted_at / jobs from launch.jsonl and null slurm_state, node, elapsed, time_left, log_path,
started_at, ended_at, `log_tail: []`, status finished / stopped; rounds[] status done / unfinished / pending plus
`unfinished_probes` (rel dirs, also in probe.dirs); NODE adds `in_batch` and `audit`, with `failure_match: null`,
`audit_eligible: false`.

`validate_contract.py`: the shapes above; runs[] batch entries (status finished / stopped, no Slurm state, audit
not_audited, root under the batch's root, detail path); every batch run named once in its task's `batch_runs`;
`batches[]` totals against its runs[] (runs, targets, accepted; runs[].batch summing to `tasks` and equal to `in_batch`
when nothing is unmatched; `unmatched` empty); one file per batch run exactly, checked like a task file with the batch
rules: generated_at = detail_generated_at <= index generated_at, the source fields, the empty job state, round statuses
and unfinished probes, every node never audited with `in_batch`, every in_batch node accepted and built, their count
runs[].batch; the official audit counts unaffected. `--sources`: each batch run's document against its own tree as for
official runs (node set, metrics.json accept, seeds.json, accepted/ = accepted nodes, probe dirs, focus patterns vs evolve
prompts, withdrawals), and every task of outputs/tasks.jsonl is the in_batch accepted node `<cand>` of the run its
provenance.json names; runs[].batch counts them per run; each run's accepted/tasks.jsonl lists exactly them.

UI (the page, when index.json has batch runs; nothing changes for official tasks):
- Picker: after "Converting" (official runs, unchanged), one group per batch, "<name> batch · not audited" (e.g. "Baihe
  batch · not audited"), its targets with their runs' status and `<n> in batch`; searching a batch run's name or repo finds
  its task. The default task stays the first converting official task.
- Task view of a task with batch runs: accuracy cards, then (the official run's view first, when it has one) a run
  selector (`#task=<id>&run=<run>`, `data-run-select`) when the task has two or more runs (official and batch together),
  e.g. pilot7 and pilot10 for tengo-callable-instance-isolation; the batch run's view: a source strip (badge "<name> batch ·
  not audited", the label, the root, the batch's own totals: tasks, runs, targets), the run bar (run, source, repo,
  status, launches from launch.jsonl "not polled", accepted, in batch, "Failure-match audit: not audited", built /
  scored, rounds), the stage, no log ("not read"), then the failure patterns, plan, round 0, the batch table and the
  generation tree as for an official run. The node drawer of a batch run reads the selected run's document.
- Batch table (in place of the failure-match audit table): the in_batch tasks (and any accepted task not in the batch,
  marked) by round: task and title, round, decision, sc14, clean kimi, similarity, audit `not audited`, task id; headed by
  `<n> tasks in <name>'s batch · not audited`. Rows open the drawer (as the audit table's rows).
- Cards: an in_batch node ends its last line with a grey badge `batch · not audited` (where an official eligible card has
  its `FA match` badge); no batch card carries an FA match badge. Round heads read `unfinished` (amber) and `<u> unfinished
  probes`. The drawer: a callout "<name> batch" (outputs/<task_id>/, not audited) on in_batch nodes, and for every scored
  batch node a section "Failure-pattern match (audit)" saying "Not audited: <name>'s batch is not audited".
- Totals stay separate: the official groups, run bars and api.json usage never count batch runs; the batch's totals come
  from index.json `batches`.

## The Explorer: filter the synthetic tasks and their trajectories (v1.7, 2026-10-22)
User request (hengyuf, 2026-10-22): "现在我想给dash board 加一个新功能, 用来filter task 和trajectories, 他应该支持一下filter: 1.
按照 evolve stage (round0 还是round 1) 2. similarity 3. nemotron rollout 分数/ kimi rollout 分数 4. audit 评分 5.trajectory 是否成功(用于进一步
filter trajectory) , 筛选完之后给出具体的task, 以及一些summary statistics (filtered 之后 task 数和trajectories数, 每个DeepSWE task 上的
coverage(画成histogram), 对应failure pattern 的coverage, e.t.c, ) 帮我添加这个功能." A view that filters the synthetic tasks (by evolve
round, similarity, the Nemotron sc14 and Kimi pass rates, the failure-match audit's verdict) and their trajectories (by
whether they passed), then lists the tasks and summary statistics: the filtered task and trajectory counts, the coverage
of each DeepSWE task (bars), the coverage of the target failure patterns, distributions. Optional for readers like
v1.2-v1.6: an older page ignores the file, and `tests/validate_contract.py` checks it whenever index.json names it.

### data/explorer.json (computed every cycle; rewritten only when its content changes)
```
{ "schema": 1, "generated_at": 1792...,       // the cycle that last CHANGED the file (= index.json explorer.generated_at)
  "sources": [{"source": "hengyuf", "name": "hengyuf", "label": "hengyuf · official runs", "audited": true},
              {"source": "baiheh", "name": "Baihe", "label": "Baihe · swe-task-hengyu", "audited": false}],
  "runs": [{"run": "official1-narwhals", "source": "hengyuf", "deepswe_task": "narwhals-rolling-window-suite",
            "build_root": "<S>/synth/runs/official1-narwhals/build/narwhals-rolling-window-suite/ibis-project__ibis",
            "max_rounds": 2, "detail": "task/narwhals-rolling-window-suite.json",
            "link": "#task=narwhals-rolling-window-suite"},
           {"run": "pilot7-tengo-callable-isolation", "source": "baiheh", ..., "max_rounds": 1,
            "detail": "run/baiheh/pilot7-tengo-callable-isolation.json",
            "link": "#task=tengo-callable-instance-isolation&run=pilot7-tengo-callable-isolation"}],
  "targets": [{"deepswe_task": "narwhals-rolling-window-suite", "runs": [12],
               "patterns": [{"name": "Extending a narrow backend exception ...", "category": "..."}, ...]}],
  "task_columns": ["run", "node", "task_id", "round", "branch", "kind", "title", "decision", "accepted", "withdrawn",
                   "in_batch", "similarity", "student_rate", "teacher_clean_rate", "teacher_rate", "valid", "audit",
                   "reproduced_yes", "reproduced_partial", "f2p", "p2p", "scored", "rates_differ"],
  "tasks": [[12, "cand-c01-r1b2", "syn-ibis-cand-c01-r1b2-1a2b3c4d", 1, 2, "evolve", "<title>", "accepted", true, false,
             false, 7, 0.2, 1.0, 1.0, [5, 5], "strong", [0, 2], [1], 78, 4903, true, false], ...],
  "trajectory_columns": ["task", "slot", "model", "status", "reward", "f2p", "p2p", "tool_calls", "patch_chars",
                         "agent_error", "verifier_error", "clean"],
  "trajectories": [[0, "kimi-r0", "kimi", "pass", 1.0, "78/78", "4903/4903", 138, 19508, false, false, true], ...],
  "totals": {"hengyuf": {"runs": 23, "tasks": 668, "trajectories": 6630, "accepted": 84, "audited": 649,
                         "unmatched_patterns": 0, "rates_differ": 8}, "baiheh": {...}} }
```
Rows are arrays under a column header (`task_columns`, `trajectory_columns`: this order, these names); a row's cells
follow its header. The source of every field (collect.py `Cycle.explorer_doc`, from the very NODE dicts the cycle writes
into the task and run files, so the numbers are the files' by construction; no other file is read):
- `sources`: hengyuf's official runs first (audited), then each configured batch (v1.6: never audited), in that order:
  the page's series slot of each source (slot 1, slot 2) follows this order, never a count.
- `runs[]`: one per document the cycle writes, in index.json `runs[]` order: data/task/<task_id>.json of each official
  target's latest run (a second official run of the same target owns no file and is left out; v1.8: every official run,
  an older one with its data/run/hengyuf/<run>.json, see "v1.8"), then data/run/<source>/<run>.json of each batch run. `build_root` and `max_rounds` are the document's, `detail` its path
  under data/, `link` the hash the task view opens that run by: `#task=<deepswe task>`, plus `&run=<run>` for a batch run
  (app.js taskHash).
- `targets[]`: each DeepSWE task of runs[], sorted by id; `runs` = its runs (indices into runs[]); `patterns` = the
  failure patterns ({name, category}) of its first run's document (failure_analysis.patterns as is, in order: P1..Pn as
  that task view numbers them; official runs come first), then any name a later run's analysis adds (2026-10-22: only
  tengo-callable-instance-isolation has two runs, both Baihe's).
- `tasks[]`: one row per NODE whose `build.task_id` is set (every built synthetic task: abandoned and withdrawn ones too;
  a failed or unfinished build has no task) of every document of runs[], in document order:
  `run` (index into runs[]); `node` = NODE.id; `task_id` = build.task_id; `round`, `branch`, `kind`, `title`, `decision` =
  the NODE's; `accepted` = decision accepted / accepted_evolve (as runs[].accepted counts it); `withdrawn` = NODE.withdrawn
  is set (v1.4); `in_batch` = NODE.in_batch is true (a batch's task of outputs/tasks.jsonl, v1.6); `similarity` =
  score.similarity when a number; `student_rate` = score.student_rate (metrics.json: sc14, the Nemotron student);
  `teacher_clean_rate` = score.teacher_clean_rate (metrics.json: kimi passes the teacher audit found clean / valid kimi
  rollouts); `teacher_rate` = probe.teacher.rate (raw: kimi passes / graded kimi rollouts of rollouts/index.md); `valid` =
  score.valid ([sc14, kimi] valid rollouts) when two numbers; all of these null for a task not scored yet. `audit` = the
  failure-match verdict (strong / partial / weak / none) of a task in the audit's scope (NODE.audit_eligible, v1.4) that
  has an audit; "not_audited" for an eligible task without one and for every task of a source that is never audited (a
  batch: NODE.audit not_audited); "not_eligible" for a task outside the scope (discarded, withdrawn, not scored yet, no
  graded sc14 rollout, an abandoned-* root), also when an audit names it (as in runs[].audit, only eligible tasks count).
  `reproduced_yes` / `reproduced_partial` = the indices (into its target's `patterns`, sorted) of the patterns the counted
  audit's patterns[] marks reproduced "yes" / "partial" (matched by name, exactly, else ignoring case and spacing; an index
  in yes is never in partial; a name that matches no target pattern counts in totals `unmatched_patterns`); null unless
  `audit` is a verdict. `f2p`, `p2p` = build.f2p / build.p2p; `scored` = the node has a score (its probe's rollouts are
  rollouts/index.md's). `rates_differ` (review 2026-10-22): for a scored task, whether its metrics.json disagrees with
  its own rollout rows (NODE.probe.rollouts, the rows of rollouts/index.md that are its trajectories) by task_builder
  score()'s formulas: student_rate vs sc14 passes / graded sc14 rows, teacher_clean_rate vs kimi passes the teacher audit
  did not find cheated (cheated not true) / graded kimi rows, valid (when two numbers) vs [graded sc14, graded kimi];
  graded = pass or fail (infra is not graded); rates equal within 1e-9, null equal to null only. score() computes all
  three from those rows, so a difference means metrics.json was rewritten after the probe (a regrade, a recovery or a
  withdrawal). The row keeps metrics.json's values (what task_builder decided on) and the page marks them; false for a
  task not scored yet. 2026-10-22: 8 rows, all official: bandit cand-c05-r1b1 (Nemotron 0.4 vs rows 0.2), fastapi
  cand-c04-r1b2 (0.8 vs 0.2) and cand-c05-r1b3 (0.2 vs 0.4), scriggo cand-c01 (Kimi clean 0.8 vs 0.5, valid [5, 5] vs
  [5, 2]), testem cand-c01 and cand-c03 (clean 0.4 vs 0.6), ytt cand-c04 (clean 0.8 vs 0.4) and cand-c01-r1b4
  (Nemotron 0.2 vs 0.25, valid [5, 5] vs [4, 5]).
- `trajectories[]`: one row per entry of NODE.probe.rollouts of those nodes, in order (a scored task: every slot of its
  rollouts/index.md, pass / fail / infra; a live probe's task: its slots graded so far, v1.2): `task` (index into tasks[]),
  `slot`, `model`, `status`, `reward`, `f2p`, `p2p`, `tool_calls`, `patch_chars` = the rollout's; `agent_error` /
  `verifier_error` = whether the rollout's error text is non-empty (the texts stay in the task file); `clean` = for a kimi
  pass, whether the teacher's cheat audit (NODE.details.teacher_audit, teacher_audit.json) found it clean: cheated false
  -> true, true -> false, no entry or null -> null; null for every other rollout.
- `totals`: per source of `sources`: runs, tasks and trajectories (rows), accepted, audited (task rows whose audit is a
  verdict), unmatched_patterns and rates_differ (task rows flagged).

What the rows do not cover (the page's header says it too, under "What the rows are"):
- an official DeepSWE target's older runs: only each target's latest official run has a task file (index.json lists
  the others; 2026-10-22: none);
- a batch's other runs: only the runs behind the batch's tasks (outputs/tasks.jsonl) are read. Baihe's: the 19 runs
  behind his 97 tasks. His tree also holds pilot11-happydom-jsdom-inline, pilot4-dateutil-rfc5545-codex and
  q11-participle-grammar-conflict-analysis (30 scored tasks each, 90 in all, none accepted: listed 2026-10-22) and five
  runs without a scored task (codex-check-20477274, pilot5-happydom-intersectionobserver, q16- and
  q16b-arktype-json-schema-refs-dependencies, smoke1); none of them is read, so none is shown;
- abandoned build roots (runs/<run>/abandoned-<repo>, set aside by main): never read, so never rows. "abandoned" in the
  rows is a node's decision (its probe was superseded, or its parent was an invalid seed); such tasks are rows.

Derived, not stored (the page and the exports spell them out; validate_contract.py checks them):
- a task's DeepSWE task and source: runs[run].deepswe_task / .source;
- its deep link: runs[run].link + "&node=" + encodeURIComponent(node) (= app.js taskHash(task, node, run)): it opens the
  node's drawer in the view of that run (an official run's view, or with `run=` a batch run's);
- its candidate dir: <build_root>/<node>;
- a trajectory's transcript and patch, for a scored task: <build_root>/round-<round>/<node>/rollouts/<slot>.md and
  <slot>.patch (synth/rollouts.py write_rollouts() writes both for every index.md row; Baihe's fork the same); a live
  probe's slots have none yet.

Writing: the document is computed every cycle after the task and run files are written, then encoded (redacted like
every output) and compared, without its generated_at, with the file on disk (cached while the process lives, else read
back once): only a different content rewrites the file, stamped with this cycle; otherwise the file and its stamp stay.
A broken build of it never stops the cycle: "explorer: <message>" in errors and index.json `explorer` null (an older file
may remain). 2026-10-22, a real cycle in the stage: 1,221,365 bytes (~190 KB gzipped, as the server sends it), 1,210
task rows (668 hengyuf, 542 Baihe) and 11,920 trajectory rows (6,630, 5,290) from 42 runs on 41 DeepSWE targets, 0.17 s
of the cycle (0.28 s in a new process, which reads the file back once); the page parses it in ~12 ms, builds its model
in ~23 ms and filters and re-renders in < 10 ms.

index.json adds `"explorer": {"path": "explorer.json", "generated_at": <the file's>, "bytes": n, "tasks": n,
"trajectories": n}` (null when this cycle could not build it). The cycle summary adds `"explorer": {"bytes", "written",
"tasks", "trajectories", "runs" (each per source), "seconds"}`.

### UI (the page, `#explorer`)
- Header: a third tab "Explorer" (index.html), `#explorer[&<filters>]`; the tab keeps the last filters. Under the title,
  "What the rows are" (collapsed) holds the scope above: each source's runs and tasks, the latest official run per
  target (and how many older ones are left out), a batch's runs behind its tasks only, abandoned build roots vs the
  abandoned decision.
- The hash holds every filter that is not its default (history.replaceState on each edit: Back leaves the Explorer),
  so a view can be shared: `src=<source,...>`, `round=<k,...>`, `kind=mask,evolve`, `audit=strong,partial,weak,none,
  not_audited,not_eligible`, `tm=sc14,kimi`, `ts=pass,fail,infra` (chips: absent = every value, `k=` = none),
  `dec=acc|not`, `smin`, `smax` (0..10), `nj=1` (with a similarity bound, also keep the tasks never judged),
  `nmin`, `nmax`, `kmin`, `kmax` (fractions 0..1, as the data holds them; the inputs take percent), `kb=raw` (the Kimi
  rate: clean by default), `rd=only|none` (re-scored tasks), `t=<deepswe task,...>` (absent: every task), `tc=1` (Kimi
  clean only), `tw=1` (only tasks with >= 1 matching trajectory), `sort=<column>:asc|desc`, `p`, `tp` (pages),
  `cov=traj`. Mistyped values are ignored. Links of the first v1.7 page still read: `sun=1` (it kept the never-judged
  tasks) is `nj=1`; `sun=0` without a similarity bound (it left them out) is `smin=0`, every judged task.
- Filters, one row above everything they scope (wrapping on narrow screens): source (hengyuf / Baihe), round 0 / 1 / 2,
  kind, decision (all / accepted / not accepted), similarity min-max with "include never judged" and its count,
  Nemotron pass rate min-max and Kimi pass rate min-max in percent (0-100) with clean / raw, re-scored (all / only /
  none), audit verdict (strong / partial / weak / none / not audited / not eligible), DeepSWE task (a searchable
  checklist with "Select shown" / "All tasks"), the trajectory filters (model Nemotron / Kimi, status pass / fail /
  infra, Kimi clean only), "only tasks with >= 1 matching trajectory", Reset. Filters AND together.
  - A range is off while both its bounds are empty; with a bound set, a task without that value fails it. So once a
    similarity bound is set, the tasks the similarity judge never scored are left out unless "include never judged" is
    ticked; next to the box the bar says how many that is ("+934 never judged" while left out, "incl. 934 never judged"
    while kept, "934 never judged, shown" without a bound; counted over the other filters). 2026-10-22: `smin=8` shows
    62 tasks (996 with `nj=1`).
  - A pass-rate bound applies to the whole percent the page shows (1/3 reads 33 %: "max 33" keeps it).
  - An input out of its range (similarity 0-10, rates 0-100 %) or not a number is not applied: the input is marked
    invalid (aria-invalid, a red ring) and the bar says why ("150 is outside 0–100 %"); the hash leaves that bound out.
    A minimum above its maximum is flagged in the bar ("min 8 > max 6: no task can pass") and above the results.
  - The trajectory filters pick the trajectories of the filtered tasks; "Kimi clean only" drops the kimi passes the
    teacher audit did not find clean.
- Results: KPI tiles (synthetic tasks and trajectories, of all; "DeepSWE tasks covered": those with >= 1 filtered task,
  of the DeepSWE tasks the chosen sources converted (within the DeepSWE task filter), "of 113 DeepSWE tasks" (index.json)
  alongside: Baihe alone reads 18 / 18; accepted among the tasks; Nemotron passes and Kimi passes of the trajectories
  shown, Kimi's clean ones too); "Coverage per DeepSWE task": horizontal bars, one row per DeepSWE task in scope (zeros
  included), the two sources stacked (slot 1 / slot 2 with a legend and a 2px surface gap), a toggle between synthetic
  tasks and trajectories (one scale at a time, never two axes); "Failure-pattern coverage": per target with counted
  audits, a bar per target pattern (P1..Pn), how many of the filtered audited tasks reproduce it, yes and partial
  stacked on an ordinal one-hue ramp (one legend, one scale for every target), under a note of how many filtered tasks
  are not counted and why (a batch: never audited; eligible, not audited yet; outside the scope); "Distributions":
  similarity 0..10 (and how many were never judged), Nemotron pass rate, Kimi pass rate (clean or raw), audit
  verdicts, rounds. The pass-rate histograms have one column per whole percent the data holds (exact values:
  2026-10-22 Nemotron 0 20 25 33 40 50 60 75 80 100, Kimi adds 67), the same columns whatever the filters; past 16
  distinct values they fall back to 10-point bins labelled 0–9 … 90–99, 100. The task table (sortable, 50 a page; a
  row opens the node's drawer in its task view through its deep link) has a "re-scored" column (sortable; a badge
  whose tooltip gives both sets of numbers, metrics.json's and its rollout rows'): a re-scored task (`rates_differ`)
  shows metrics.json's rates, with a `*` (same tooltip) on each rate its rows disagree with; valid counts alone differing
  get the badge only, and the raw Kimi rate (the rows' own) never a `*`. The trajectories (50 a page, with their
  transcript and patch paths). Counts read as words ("1 task", "2 tasks").
- Exports: CSV / JSON of the filtered tasks or trajectories, made in the browser (a Blob behind a download link). CSV:
  a UTF-8 byte-order mark first (spreadsheets read "·" and non-ASCII titles right), CRLF rows; a text cell that a
  spreadsheet would run as a formula gets an apostrophe: one that starts with = + - @ or their full-width forms
  ＝ ＋ － ＠, also after leading blanks, or with a tab, CR or LF; numbers stay numbers. Tasks add rates_differ,
  rollouts_student_rate and rollouts_teacher_clean_rate (the rows' rates), f2p_tests / p2p_tests (build counts);
  trajectories split F2P / P2P into f2p_passed, f2p_total, p2p_passed, p2p_total (never "7/8", which a spreadsheet
  reads as a date).
- Charts: HTML/CSS bars (no library, no inline script): bars <= 24px thick (14px rows, 24px columns), a 4px rounded data
  end and a square baseline, the value at the tip (columns: on the cap; empty columns carry none) in ink, hairline grid
  on clean ticks; one series in one hue without a legend; every mark (the whole row of a bar, the full-height slot of a
  column) shows a tooltip on hover and keyboard focus (#ex-tip: values first, a short line key per series; Escape hides
  it); each chart has a collapsible table. Axis labels wrap (never cut); a label's full text is its title. Colors:
  --s1 / --s2 (light #2a78d6 / #eb6834, dark #3987e5 / #d95926) for the sources only; the single-series histograms
  wear slot 3 aqua --s3 (light #1baf7a, dark #199e70), a hue no source wears; --ord-yes / --ord-partial (light #2a78d6
  / #86b6ef, dark #86b6ef / #256abf) for yes / partial. Validated with the dataviz skill's validate_palette.js against
  the dashboard's surfaces (#fcfcfb, #1a1a19), the three slots all-pairs: light worst CVD ΔE 9.2, normal 24.0, aqua
  2.74:1 on the light surface (the relief rule: every non-empty column carries its value, every chart its table); dark
  9.4 / 20.9, all >= 3:1. Color follows the source, never its rank: a source filtered alone keeps its slot.
- Keyboard and screen readers: a chart's marks are a list (role list / listitem, each named by its aria-label). Each
  chart panel (coverage, pattern coverage, distributions) is one tab stop (a roving tabindex: its first mark; the arrow
  keys move along the panel's marks, Home / End to its ends; a mark focused otherwise becomes the stop), and "Skip the
  charts: go to the task table" (shown on focus, after the KPI tiles) moves the focus to the first task row.
- Refresh: explorer.json is fetched when the Explorer is shown and index.json `explorer.generated_at` names another copy
  (no stamp: once per new cycle, a 304 when unchanged). The view is keyed on explorer.json's stamp, the filters and the
  load state, never on index.json's own stamp: a cycle that leaves explorer.json alone re-renders nothing (inputs keep
  their value, focus and caret). A new explorer.json with the same choices (sources, rounds, kinds, DeepSWE tasks)
  re-renders the header, the results, the checklist and the bar's status in place; other choices rebuild the whole view.
  A filter edit re-renders the results and the status slots alone (a DeepSWE task choice also its checklist label; Reset
  or another hash the bar too).
- Escape: hides the tooltip, closes the task menu, and closes a node's drawer only in the task view. Leaving the task view
  closes its node (another tab is never left holding one), so Escape on the Explorer or the monitor never rewrites
  their URL (review 2026-10-22: `#explorer&round=1&audit=strong`, a row, Back, Escape read `#task=...`). The task and
  monitor views, the drawers, the picker and the header render exactly as before (tests: byte-identical to the v1.6
  page kept as static/app.js.pre-explorer).

### validate_contract.py
The shape (keys, schema, column headers, each cell's type and enum, the audit / reproduced rules, accepted vs
decision), the stamps (generated_at <= index.json's, = index.json explorer.generated_at; explorer.bytes, tasks and
trajectories = the file's), the sources (official first and the only audited one, every batch of index.json), and then
every run, target, task row (rates_differ included: the rows' rates recomputed from each node's rollouts and teacher
audit), trajectory row and total recomputed from the task and run files by the rules above (an implementation
independent of collect.py's): a missing, extra, reordered or different row is a problem, named with its differing
columns. index.json `explorer` null needs an "explorer: ..." error. `--sources` also checks that every task
row's candidate dir exists and that every trajectory of a scored task has its <slot>.md and <slot>.patch on disk.

## Every official run, and the synthesis queue (v1.8, 2026-10-23)
Why (main, 2026-10-22, relaying hengyuf): hengyuf approved a new synthesis queue, "official2": 14 builds of reselected
host repos (repo_reselect/out, repo_reselect_baihe/out), round 0 plus round 1 (`--rounds 1`), at most 5 at a time, each
reusing its earlier failure analysis; its runner is `S/synth/queue2/run_queue.py` (`queue.json`), and hengyuf asked that
the new synthesis "同时更新到dash board" (show up on the dashboard as it runs). Six official2 runs build DeepSWE tasks that
official1 runs built: etree, aiomonitor, promreload, arcane, cattrs and ytt (`official2-etree` alongside
`official1-etree`, ...). Until v1.7 only a task's latest official run (the most recently launched) had a document and
Explorer rows, so launching official2-etree would have taken official1-etree's tree and rows off the dashboard. The other
eight official2 runs build tasks only Baihe's batch converted so far (kysely, bandit taint, pebble, adaptix, mobly,
obsidian, the two tengo tasks): their official run becomes the task's default view and the batch runs stay in its run
selector (v1.6). Optional for readers like v1.2-v1.7: every field below is new; an older page ignores them.

### Collector
- Runs: one entry per (official run dir `S/synth/runs/official*`, target task), as before. A run dir with neither
  `launch.jsonl` nor a build (`build/<task>/<repo>/`) is no run: a queued official2 run, whose runner has only copied its
  failure analysis into `analysis/<task>/`, shows nowhere until it launches (official2-kysely, -bandit-taint, -pebble,
  -etree and -aiomonitor launched at 23:22 on 2026-10-22, so official1-etree and official1-aiomonitor became older
  runs; the other 9 were still queued on 2026-10-23). A task's `latest` official run is its most recently launched one
  (launch.jsonl `submitted_at`, read with an offset as `-07:00` or, as run_queue.py writes it, `-0700`; no launch:
  oldest).
- Launches of the queue runner (run_queue.py, through the dual-account babysit helper): each leg first writes a
  launch.jsonl line naming its two candidate jobs (`babysit: {leg, candidates, resolved: false}`, `job_id` the first),
  then, once one won, a line with `babysit.resolved` true and `babysit.winner` (`job_id` the winner). A run's job is its
  last leg's winner when that leg is resolved; while it is not, every candidate is polled (squeue / sacct) and the most
  active one is shown, with its log: RUNNING, else CONFIGURING / COMPLETING, else PENDING, else the first. So a run never
  reads "stopped" because its losing candidate just exited. `job.jobs` lists each job once, in launch order (a resolved
  leg repeats its winner's line; the active candidate of an unresolved leg is added).
- Documents: the latest keeps `data/task/<task_id>.json`, unchanged (one cycle, one stamp). Every OLDER official run of the
  task writes `data/run/hengyuf/<run>.json` (`<run>--<task>.json` should a run build several targets; none does): the
  same document as a task file (task_doc(): the same keys, NODE, rounds and decision rules, its job with Slurm state and
  log tail, failure analysis, plan, failure-match audits), built every cycle like the latest's but rewritten only when its
  content (everything but `generated_at`) changed or the file on disk is not the one written last; its `generated_at` is
  the cycle that last changed it (= index runs[].`detail_generated_at`, never later than index.json's). The file of a run
  that is no older run any more (the latest again, or gone) is removed. Directories `data/run/` and `data/run/hengyuf/`
  700, files 600. (A batch can never be named `hengyuf`: v1.6 batch files live beside it and are pruned apart.)
- Audits stay per run dir: `runs/<run>/audit/failure_match/<task_id>.json` attaches to that run's nodes only, so an
  official1 audit never reaches an official2 node, even of the same task id; until audits are written into an official2
  run's own dir, its eligible tasks are not audited (v1.4 rules; index runs[].audit counts per run).
- index.json, official runs[] entries add `"latest": true|false` and `"detail"`: `"task/<task_id>.json"` (the latest) or
  `"run/hengyuf/<run>.json"` (an older run, which adds `"detail_generated_at"`); tasks[] add, for a task with an official
  run, `"official_runs": [{"run", "latest", "detail"}, ...]`: its latest first, then its older runs, newest launch first
  (`run` stays the latest's):
```
  "runs": [ {"run": "official1-etree", "task_id": "etree-xml-diff-patch", ..., "latest": false,
             "detail": "run/hengyuf/official1-etree.json", "detail_generated_at": 1792...},
            {"run": "official2-etree", "task_id": "etree-xml-diff-patch", ..., "latest": true,
             "detail": "task/etree-xml-diff-patch.json"}, ...],
  "tasks": [ {"task_id": "etree-xml-diff-patch", ..., "run": "official2-etree",
              "official_runs": [{"run": "official2-etree", "latest": true, "detail": "task/etree-xml-diff-patch.json"},
                                {"run": "official1-etree", "latest": false,
                                 "detail": "run/hengyuf/official1-etree.json"}]} ]
```
- Usage (api.json by_run), the router's probes (fleet.json) and the audit counts: every official run, as before, with
  one exception: a failure analyzer's codex session (`analysis/<task>/.agent/events.jsonl`) counts for a run only when
  its mtime is not earlier than that run's FIRST launch (launch.jsonl). run_queue.py prep() copies an earlier run's
  `analysis/<task>/` into a queued run with shutil.copy2, which keeps the old mtimes, and task_builder reuses that
  analysis instead of analyzing again: the copy's session is counted where it ran (official1-etree's, or Baihe's tree,
  never hengyuf's spend), while an analysis redone inside the run is written after its launch and counts; a run dir
  with neither launch.jsonl nor a build (queued) counts none. A copy counted before (by an earlier collector) is
  dropped from state/collector.json by its inode. A run whose document failed this cycle keeps its analyzer state
  untouched until the next cycle (nothing is dropped on a missing document). A run without any launch line but with a
  build counts its analyzer sessions. Nothing here reads queue.json. (2026-10-23, real data: every official1 session
  is 8-40 min after its run's
  first launch; every official2 copy predates its run's. Without the rule a cold staged cycle showed the 14 queued
  official2 runs in by_run with their copied analyzers' tokens.)
- Explorer (v1.7): rows of EVERY official run, in index.json runs[] order (so a target's older runs first), then the batch
  runs; runs[] add `"latest"` (an official run: true / false; a batch run: null); an older run's `detail` is its
  data/run/hengyuf/<run>.json and its `link` `#task=<task>&run=<run>` (rows link `&node=<id>` after it). A target's
  patterns: its first run's list, then names a later run adds (official2 reuses official1's analysis: the same list).
- The cycle summary adds `"official_runs": {"latest": n, "older": n}` and `"queues": {<name>: {"runs": n, "alive": b}}`.

### The synthesis queue: index.json "queues"
```
  "queues": [ {"name": "official2", "root": "<S>/synth/queue2", "read_at": 1792..., "max_active": 5,
               "runner": {"heartbeat_at": 1792... | null, "age_s": 37.0 | null, "host": "...", "slurm_job_id": "20712345",
                          "note": "running|draining|stopped", "alive": true},
               "budget": {"key": "NVAPI_KEY_OFFICIAL_1", "remaining": 2345.68, "at": 1792...} | null,
               "serving": true, "stop_launching": false, "updated_at": 1792... | null, "finished_at": null,
               "runs": [{"run": "official2-etree", "task": "etree-xml-diff-patch", "repo": "josephburnett/jd",
                         "batch": "hengyuf", "status": "queued|pending|running|waiting_to_resume|done|failed|stuck",
                         "leg": 1, "winner": "20712399" | null, "candidates": ["20712399", "20712400"]}, ...],
               "log_tail": ["<= 40 lines of runner.log, redacted, each <= 300 chars, the pid masked"]} ]
```
- Sources: exactly four files of the queue's directory, `queue.json`, `state.json`, `heartbeat.json` and `runner.log`
  (run_queue.py's formats); they may not exist yet. They are allowed as exact paths (`allow_files`: the directory
  resolved, the file itself not) and opened at that very path with O_NOFOLLOW: a symlink at one of the four paths is
  never followed, wherever it points (also into another read root), is read as absent and is named in the errors
  ("queue <name>: state.json is a symlink: not read"); nothing else there is ever opened (runner.lock, runner/, logs/,
  run_queue.py, runner.sbatch). A queue whose directory does not exist has no entry; a queue.json that is missing,
  unreadable or has no run list is an error ("queue <name>: ...") and no entry.
- runs: queue.json `runs[]` in queue order (`run` a safe run name, `task` a safe task id, else skipped with an error;
  `repo` and `batch`, the batch whose failure analysis the run reuses, as text); from state.json `runs[<run>]`: `status`
  (absent: "queued", the runner's own default; outside the seven: null and an error "queue <name>: <run>: unknown status
  '<s>'"), `leg` (an int >= 0, absent 0), `winner` (a job id or null) and `candidates` (job ids, <= 8).
- `read_at`: when the collector read the four files (late in the cycle, after the official runs' trees: up to minutes
  after the cycle's `generated_at`).
- runner: heartbeat.json `at` (epoch), `age_s` = read_at - at (>= 0), `host`, `slurm_job_id`, `note` (<= 40 chars);
  `alive` = a heartbeat at most 900 s old at read_at (run_queue.py HEARTBEAT_STALE_S) whose note is not "stopped". No
  heartbeat.json: every field null, alive false ("never ran"). The page does not show these two: it computes the age and
  the liveness again on its own clock (below), as it does every other age.
- budget: null until state.json has one; else EXACTLY `{key, remaining, at}`: the active hub key's environment variable
  NAME (`[A-Z][A-Z0-9_]{0,63}`, else null), its remaining dollars (2 decimals) and the time of that check. Never anything
  else of the runner's budget (its error text) or of these files: queue.json's requested_by, other settings, notes and
  analysis_from, state.json's key_switches, launch_errors, leg_repairs and budget_checked, heartbeat.json's pid are not
  published.
- `serving`, `stop_launching`: state.json's flags (false when absent); `max_active`: queue.json settings.max_active;
  `updated_at` / `finished_at`: state.json's times.
- `log_tail` (main's decision, 2026-10-23: an operational view, as requested, its secrets redacted): the raw tail of
  runner.log, its last 40 lines, each redacted (`redact()`, hengyuf's secret values masked too) and clipped to 300
  chars as a job's log tail; the runner's process id is masked (`pid 12345` -> `pid ⋯`). It may show what the runner
  logs: hosts and Slurm job ids, error texts (e.g. a failed squeue), key-switch lines carrying key NAMES
  ("switching the active hub key NVAPI_KEY_OFFICIAL_1 -> NVAPI_KEY_OFFICIAL_2"; the runner never logs a key value) and
  file paths. A clipped line ends with its "…[N chars omitted]…" marker on a line of its own, so the page can show more
  than 40 visual lines.

### UI (the page)
- Task view of a task with two or more runs: the run selector lists its latest official run first (badges "official",
  "latest", a neutral tone: recency is no status), then its older official runs (badges "official", "older"; newest
  launch first), then its batch runs (v1.6).
  `#task=<id>&run=<older run>` (`data-run-select` = the run) shows the older run: a note ("<run>: an earlier official
  run of this task. <latest> is its latest official run and its default view; this view shows this run's own tree, plan
  and failure-match audits"), then the official view of its own document (run bar, failure patterns, plan, round 0, the
  failure-match audit table, the tree), fetched from runs[].detail and again only when its detail_generated_at changes;
  its node drawer and node links keep `run=`. A view shows only its own run's document (a stale copy of another run's
  reads as loading). A task with one official run renders exactly as before (tests: byte-identical to the v1.7 page kept
  as static/app.js.pre-v18; also the picker and the monitor without queues).
- Deep links: `#task=<id>&node=<n>` without `run=` whose node is not in the default view's document: the page reads the
  task's other runs' documents in the selector's order and shows the first that has the node, replacing the hash with
  that run's `run=` (history.replaceState); a node of the default run, or of no run, leaves view and hash as they are (each
  run read once). So the links shared before official2 to official1 nodes keep working once official2 is latest, for
  every node official2 does not have; node ids repeat between runs (cand-c01 .. cand-c05 in each), and a link to an id
  the latest run has as well opens the latest run's node, its run selector one click away. Links written from v1.8 on
  carry `run=` for an older run (Explorer rows, the drawer of an older run's view).
- Picker: a task queued in a synthesis queue and not launched yet (status queued, no runs[] entry of that run) carries
  "queued in <queue>" (its tooltip names the run); searching an older official run's or a queued run's name finds its task.
- Monitor: one section per queue, above the workers, "Synthesis queue <name>": the runner (alive / stale / stopped /
  never ran, with its note, host, job and heartbeat age; the age is `fmtAgo(heartbeat_at, now)` and alive = now -
  heartbeat_at <= 900 s and the note is not "stopped", both on the page's own clock as every other age of the page, so
  a page left open sees the runner go stale), "Hub budget · <KEY>: $x remaining (checked … ago)" or "not
  checked yet", badges "launching stopped" (when set) and "student serving on / off", a line "<n> runs, at most <m> at a
  time · <counts by status> · state … ago", the table (run, linked to its view once launched: `#task=<task>`, with `run=`
  for an older official run; DeepSWE task; repo; analysis batch; status, queued muted, pending and waiting to resume amber,
  running blue, done green, failed red, stuck serious; leg; winner; candidates) and the runner.log tail (collapsible).
- Explorer: rows of every official run. A "Run" filter: family chips as quick toggles (an official run's name prefix,
  official1 / official2, then each batch source by its name, e.g. Baihe; hash `fam=<family,...>`) and a searchable run
  checklist ("Select shown" / "All runs"; an older run marked "older"; hash `run=<run,...>`, absent: every run); the task
  table marks an older run's rows "<run> · older"; coverage stays stacked by source (hengyuf / Baihe: a run family is a
  filter, never a new series color); "What the rows are": "hengyuf: every built task of every official run, <n> runs:
  each DeepSWE target's latest run (data/task/<task>.json) and <m> older official runs of the same targets
  (data/run/hengyuf/<run>.json; their rows link with run=)." (without older runs, the v1.7 sentence).
- Explorer, round-2 review leftovers: a bar label keeps its full text in a title and wraps to two lines at most
  (`-webkit-line-clamp: 2`, the row grows to fit; a label longer than two lines ends in an ellipsis on its second line,
  its title still whole; before, one line cut with an ellipsis). The range inputs are `type="text" inputmode="decimal"`
  (a numeric keypad on touch devices): whatever is typed reaches the bar's own check, so "abc", "--3" or "50%" is rejected
  visibly (aria-invalid, a red ring, "“abc” is not a number") where a number input handed the page "" (validity.badInput)
  and the bound cleared silently.

### validate_contract.py
Official runs[] `latest` / `detail` / `detail_generated_at` as above; a document's `job.jobs` each job once; per task exactly one latest, the most recently
submitted one, = tasks[].run; tasks[].official_runs exactly its official runs (latest first, then newest launch first);
data/run/ holds exactly the batch runs' and the older official runs' files, an older one checked like a task file
(official rules, its runs[] audit counts) with generated_at = detail_generated_at <= index.json's and no batch source; the
Explorer's runs, rows and totals recomputed from every official run's document (an older run's link with run=). queues[]:
the keys in order; read_at within the cycle (generated_at .. generated_at + collect_seconds); the runner (heartbeat_at
and age_s both or neither, age_s = read_at - heartbeat_at, alive as defined); budget null or exactly {key: a NAME, remaining, at}; flags booleans; runs once each, of a DeepSWE task of
tasks[], a known status (null only with its error), leg >= 0, candidates and winner job ids, the winner one of the
candidates; <= 40 log lines of <= 300 chars, no unmasked `pid <n>`. `--sources` lists synth/runs/official*: every dir with launch.jsonl or a
build is exactly one official run (the queued dirs counted apart), the latest of each task its last launch; each older
run's document is cross-checked against its run dir as task files are (node set, metrics.json, seeds.json, accepted/,
probe dirs, focus patterns, audits, withdrawals); and each queue against its four files: its runs against queue.json, and,
for each file not modified since the cycle began, the statuses / legs / winners / candidates, budget and flags against
state.json, the heartbeat against heartbeat.json and the log tail's length against runner.log.
