export const meta = {
  name: 'official4-build-watch',
  description: 'Background watcher for the official4 synthesis queue (16 kimi-vs-base-gap builds; NVAPI_KEY_NEW_3, then NVAPI_KEY_NEW_2): runner, builds, budget and disk quota; returns to main only when main must act',
  whenToUse: 'hengyuf swe-task official4 queue: start once the runner is up; relaunch with args.prev (the last snapshot) after main handles a return. args: {prev?, max_windows?}',
  phases: [{ title: 'Watch', detail: 'one low-effort window agent at a time: up to 3 checks about 10 min apart' }],
}

// official4 (hengyuf 2026-11-02: start the 23 tasks' synthesis, at most 5 DeepSWE targets in parallel) reuses the official3
// watchers (hengyuf 2026-10-27: "launch a watcher to monitor these task generation and another watcher to maintain student
// generation worker"). This one watches the builds and never acts; official4_fleet_keeper.js keeps the workers and
// official4_audit_loop.js labels the scored tasks. Main is called back only for needs_attention, the queue finishing,
// or the window cap (feedback "watchers as subagent workflow": 10-minute checks, structured snapshots).

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const MAX_WINDOWS = (args && args.max_windows) || 48
let prev = (args && args.prev) || null

const SNAP = {
  type: 'object',
  properties: {
    at: { type: 'string' },
    runner: { type: 'object', properties: { job: { type: 'string' }, alive: { type: 'boolean' }, heartbeat_age_s: { type: 'number' }, note: { type: 'string' } }, required: ['alive', 'heartbeat_age_s'] },
    builds: {
      type: 'object',
      properties: {
        running: { type: 'array', items: { type: 'string' } },
        pending: { type: 'array', items: { type: 'string' } },
        queued: { type: 'array', items: { type: 'string' } },
        done: { type: 'array', items: { type: 'string' } },
        failed_or_stuck: { type: 'array', items: { type: 'string' } },
        legs: { type: 'array', items: { type: 'string' }, description: '"<run>=<leg>" for every run not done' },
      },
      required: ['running', 'done', 'failed_or_stuck', 'legs'],
    },
    budget: { type: 'string' },
    budget_usd: { type: 'number', description: 'spend-cap dollars left (cap_left) from the last budget line; copy prev.budget_usd when there is no new budget line' },
    hub_usd: { type: 'number', description: "the hub's remaining from the last budget line" },
    hub_spend: { type: 'number', description: 'the hub spend from the last budget line (for jump detection)' },
    quota: { type: 'string' },
    log_offset: { type: 'number' },
    no_serving_since: { type: 'string', description: 'ISO time the runner first logged "no student serving" with no later recovery, else empty' },
    launch_failed_streak: { type: 'array', items: { type: 'string' }, description: '"<run>=<consecutive checks with launch failed>"' },
    events: { type: 'array', items: { type: 'string' } },
    needs_attention: { type: 'boolean' },
    attention_reason: { type: 'string' },
    newly_done: { type: 'array', items: { type: 'string' } },
    queue_finished: { type: 'boolean' },
    status_line: { type: 'string' },
  },
  required: ['at', 'runner', 'builds', 'events', 'needs_attention', 'newly_done', 'queue_finished', 'status_line'],
}

const windowPrompt = (prev, i) => `You WATCH hengyuf's official4 synthetic-task queue (window ${i}). S=${S}.
The queue runner (synth/queue4/run_queue.py, CPU job "swe-task-queue4-runner") launches the builds ("synth-build-official4-*": task_builder.py build --rounds 1, at most 5 at a time; the 16 runs are in synth/queue4/queue.json (hengyuf dropped the 7 targets Baihe's fork covers on 2026-11-02: queue.json "dropped", not runs); every run's target failure analysis shows the passing Kimi and failing sc14 rollouts of data/targets/sc14x10-kimix10-20261102, per synth/runs/<run>/target_source.json) and relaunches an unfinished build, up to 6 legs per run. Hub keys (synth/hub_key_active.txt): NVAPI_KEY_NEW_3 first. Its hub "remaining" is unreliable since the 2026-11-01 budget reset (spend reads negative), so the runner has a spend cap on it: its budget lines read "budget: NVAPI_KEY_NEW_3 remaining <hub remaining> | spend cap: <spent> of 7000 spent since the baseline, <cap left> left (hub spend <spend>)". When the cap is exceeded (two readings) the runner switches to NVAPI_KEY_NEW_2 (hengyuf 2026-11-02: "use NEW_2"; logs "budget: switching the active hub key NVAPI_KEY_NEW_3 -> NVAPI_KEY_NEW_2 (...)" and "budget: make_secrets.py rc <n>: ...") and keeps launching; NEW_2's budget lines read "budget: NVAPI_KEY_NEW_2 remaining <remaining>" (no cap part). NEW_2 also carries the r6 RL judge: below $3,000 on NEW_2 ("no key above its threshold") the runner stops launching and resuming. If NEW_2 is below its floor when NEW_3's cap trips, it logs "spend cap reached ...": no new builds, started builds keep resuming. The 23 runs are ordered by coverage: 13 never synthesized, then 3 that Baihe's fork built with 0 accepted, then 7 already covered by his fork (they should start only after hengyuf decides about the key). A separate keeper looks after the student vLLM workers; you do not touch them.

Previous snapshot (null on the first window): ${JSON.stringify(prev)}

Loop up to 3 times. Each time: sleep 590 seconds with Bash (\`sleep 590\`), then CHECK:
1. Runner: synth/queue4/heartbeat.json (age from "at", note); \`squeue --me --name=swe-task-queue4-runner --noheader --format="%i %T %M"\`. Read the new bytes of synth/queue4/runner.log from prev.log_offset (byte offset; store the new size in log_offset; start at 0 if the file is shorter). Note in events: launched, winner, "done (summary.json)", "ended unfinished", FAILED, STUCK, "launch failed", budget lines, "switching the active hub key", "no key above its threshold", "no student serving", "fleetd stuck", "queue finished", Traceback.
2. Builds: \`cd ${S}/synth/queue4 && /usr/bin/python3 -I -B run_queue.py status\` prints the run table (read-only: status and leg per run). Also \`squeue --me --format="%i %T %P %j %M %r"\` for synth-build-official4-* jobs. A run with synth/runs/<run>/build/*/*/summary.json is done. newly_done = done now but not in prev.builds.done. failed_or_stuck = status failed or stuck. legs = "<run>=<leg>" for every run not done.
3. Budget: the last "budget: NVAPI_KEY_... remaining ..." line of runner.log. For NEW_3: budget = "NEW_3 cap left $<cap left> (spent $<spent>), hub $<remaining>", budget_usd = cap left, hub_usd = hub remaining, hub_spend = hub spend. For NEW_2: budget = "NEW_2 $<remaining>", budget_usd = remaining - 3000 (what is left above the floor), hub_usd = remaining, hub_spend = 0. Copy prev's values when no new budget line appeared. Quota: \`lfs quota -h -u hengyuf /lustre/fsw\`, one line "used/limit". \`grep -l "Disk quota exceeded"\` over synth/runs/slurm/synth-build-official4-*.out files modified in the last 15 minutes (all of them on the first window).

Read-only: never cancel, hold, release, submit or edit anything; never read secrets (.env, S/secrets, S/atp/.env).

needs_attention = true, and stop looping, when any of these holds:
- the runner heartbeat is over 900 s old and no swe-task-queue4-runner job is pending or running, unless the note is "stopped";
- a run became failed or stuck;
- a run not done reached leg 4 or higher, and its leg is higher than in prev.builds.legs (official2's ytt burned all 6 legs in a crash loop);
- "launch failed" for the same run in 3 consecutive checks (track launch_failed_streak);
- "Disk quota exceeded" in a build log or runner.log, or the user quota over 49.5T;
- budget, EDGE-triggered (never again for a level main already saw): a new "switching the active hub key", "make_secrets.py failed", "make_secrets.py rc" with a non-zero rc, "spend cap reached", "counted as reached" or "no key above its threshold" line (the runner logs each once per transition); on NEW_2, budget_usd under 1000 (under $4,000 remaining) while prev was on NEW_3, null, or had budget_usd 1000 or more; on NEW_3, the runner itself now treats a hub-spend jump of more than $500 up or $50 down between readings as a counter correction/reset ("counted as a hub counter correction, not as spend" / "counted as a hub counter reset" lines; main re-baselined the cap on 2026-11-03 after the hub zeroed NEW_3's counter): report such lines in events, and raise needs_attention only for 3 or more of them in one window;
- "no student serving" or "fleetd stuck" persisting for over 60 minutes (track no_serving_since; clear it when launches resume);
- a Python Traceback in runner.log or in the runner's job log synth/queue4/logs/swe-task-queue4-runner-<job>.out (the newest one);
- "fleetd: STUCK" or "STUCK" for a run in runner.log;
- budget: a "spend cap clear again" line (main must check it was not a hub glitch); a budget line whose values read None is NOT a reading (keep prev's budget_usd/hub_usd/hub_spend and note it in events), but 3 such lines in a row, or 3 "budget check failed" lines in a row, need attention.
The 7 dropped targets (queue.json "dropped") are not runs: ignore them; their synth/runs/official4-* dirs hold only target_source.json.
Also stop looping, without needs_attention, when the runner logged "queue finished" or the heartbeat note is "stopped" (queue_finished = true).
newly_done builds are only reported (events, newly_done); do not stop for them.
Otherwise run all 3 checks and return the last snapshot.
status_line: one short Chinese line for hengyuf, e.g. "official4: 运行 5 / 完成 4 / 失败 0 / 排队 7, 最大 leg 2, NEW_3 上限余额 $3,962 (已花 $3,038), 配额 31T/50T" (on NEW_2: "..., NEW_2 $6,900 (地板 $3,000), ...").`

phase('Watch')
let last = null
for (let i = 1; i <= MAX_WINDOWS; i++) {
  const snap = await agent(windowPrompt(prev, i), { label: `window:${i}`, phase: 'Watch', schema: SNAP, model: 'sonnet', effort: 'low' })
  if (!snap) { log(`window ${i}: no result; retrying with the previous snapshot`); continue }
  const changed = !prev || snap.events.length || snap.newly_done.length || snap.status_line !== prev.status_line
  if (changed) log(snap.status_line + (snap.newly_done.length ? ` | 新完成: ${snap.newly_done.join(', ')}` : ''))
  else if (i % 6 === 0) log(`(心跳) ${snap.status_line}`)
  last = snap
  prev = snap
  if (snap.needs_attention || snap.queue_finished) {
    return { reason: snap.needs_attention ? 'needs_attention' : 'queue_finished', window: i, snapshot: snap }
  }
}
return { reason: 'window_cap', window: MAX_WINDOWS, snapshot: last }
