export const meta = {
  name: 'official3-build-watch',
  description: 'Background watcher for the official3 synthesis queue (11 group-A builds on NVAPI_KEY_NEW_3): runner, builds, budget and disk quota; returns to main only when main must act',
  whenToUse: 'hengyuf swe-task official3 queue: start once the runner is up; relaunch with args.prev (the last snapshot) after main handles a return. args: {prev?, max_windows?}',
  phases: [{ title: 'Watch', detail: 'one low-effort window agent at a time: up to 3 checks about 10 min apart' }],
}

// hengyuf 2026-10-27: "launch a watcher to monitor these task generation and another watcher to maintain student
// generation worker". This one watches the builds and never acts; official3_fleet_keeper.js keeps the workers and
// official3_audit_loop.js labels the scored tasks. Main is called back only for needs_attention, the queue finishing,
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
    budget_usd: { type: 'number' },
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

const windowPrompt = (prev, i) => `You WATCH hengyuf's official3 synthetic-task queue (window ${i}). S=${S}.
The queue runner (synth/queue3/run_queue.py, CPU job "swe-task-queue3-runner") launches the builds ("synth-build-official3-*": task_builder.py build --rounds 1, at most 5 at a time; the 11 runs are in synth/queue3/queue.json) and relaunches an unfinished build, up to 6 legs per run. Hub key: NVAPI_KEY_NEW_3 only (synth/hub_key_active.txt); the runner launches nothing new below $1,500. A separate keeper looks after the student vLLM workers; you do not touch them.

Previous snapshot (null on the first window): ${JSON.stringify(prev)}

Loop up to 3 times. Each time: sleep 590 seconds with Bash (\`sleep 590\`), then CHECK:
1. Runner: synth/queue3/heartbeat.json (age from "at", note); \`squeue --me --name=swe-task-queue3-runner --noheader --format="%i %T %M"\`. Read the new bytes of synth/queue3/runner.log from prev.log_offset (byte offset; store the new size in log_offset; start at 0 if the file is shorter). Note in events: launched, winner, "done (summary.json)", "ended unfinished", FAILED, STUCK, "launch failed", budget lines, "switching the active hub key", "no key above its threshold", "no student serving", "fleetd stuck", "queue finished", Traceback.
2. Builds: \`cd ${S}/synth/queue3 && /usr/bin/python3 -I -B run_queue.py status\` prints the run table (read-only: status and leg per run). Also \`squeue --me --format="%i %T %P %j %M %r"\` for synth-build-official3-* jobs. A run with synth/runs/<run>/build/*/*/summary.json is done. newly_done = done now but not in prev.builds.done. failed_or_stuck = status failed or stuck. legs = "<run>=<leg>" for every run not done.
3. Budget: the last "budget:" line of runner.log; budget = "NEW_3 $<remaining>", budget_usd = the number. Quota: \`lfs quota -h -u hengyuf /lustre/fsw\`, one line "used/limit". \`grep -l "Disk quota exceeded"\` over synth/runs/slurm/synth-build-official3-*.out files modified in the last 15 minutes (all of them on the first window).

Read-only: never cancel, hold, release, submit or edit anything; never read secrets (.env, S/secrets, S/atp/.env).

needs_attention = true, and stop looping, when any of these holds:
- the runner heartbeat is over 900 s old and its job is gone, unless the note is "stopped";
- a run became failed or stuck;
- a run not done reached leg 4 or higher, and its leg is higher than in prev.builds.legs (official2's ytt burned all 6 legs in a crash loop);
- "launch failed" for the same run in 3 consecutive checks (track launch_failed_streak);
- "Disk quota exceeded" in a build log or runner.log, or the user quota over 49.5T;
- a budget reading under $2,000, "no key above its threshold", or a key switch;
- "no student serving" or "fleetd stuck" persisting for over 60 minutes (track no_serving_since; clear it when launches resume);
- a Python Traceback in runner.log.
Also stop looping, without needs_attention, when the runner logged "queue finished" or the heartbeat note is "stopped" (queue_finished = true).
newly_done builds are only reported (events, newly_done); do not stop for them.
Otherwise run all 3 checks and return the last snapshot.
status_line: one short Chinese line for hengyuf, e.g. "official3: 运行 5 / 完成 2 / 失败 0 / 排队 4, 最大 leg 2, NEW_3 $9,410, 配额 31T/50T".`

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
