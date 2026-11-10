export const meta = {
  name: 'queue3-keeper',
  description: 'Background keeper for the official3 synthesis queue: keeps the sc14 student workers serving, watches the runner and builds, returns to main only when main must act',
  whenToUse: 'hengyuf swe-task official3 (group A) queue: start once the runner is up; relaunch with args.prev (the last snapshot) after main handles a return. args: {prev?, max_windows?}',
  phases: [{ title: 'Watch', detail: 'one low-effort window agent at a time: up to 3 checks about 10 min apart' }],
}

// Design (hengyuf feedback 2026-10-16, "watchers as subagent workflow"): periodic monitoring runs here, not as
// cron prompts on main. Each window sleeps about 590 s before every check, does up to 3 checks, and returns early on
// any change. Main is only called back for: needs_attention, builds that finished (main launches the failure-match
// audits and their review), the queue finishing, or the window cap.

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const MAX_WINDOWS = (args && args.max_windows) || 48
let prev = (args && args.prev) || null

const SNAP = {
  type: 'object',
  properties: {
    at: { type: 'string' },
    runner: { type: 'object', properties: { job: { type: 'string' }, alive: { type: 'boolean' }, heartbeat_age_s: { type: 'number' }, note: { type: 'string' } }, required: ['alive', 'heartbeat_age_s'] },
    fleet: { type: 'object', properties: { daemon_job: { type: 'string' }, target: { type: 'number' }, ready: { type: 'number' }, starting: { type: 'number' }, pending: { type: 'number' }, oldest_pending_min: { type: 'number' }, minutes_with_zero_ready: { type: 'number' }, breaker: { type: 'string' }, excluded: { type: 'number' }, stop_flag: { type: 'boolean' } }, required: ['ready', 'pending'] },
    builds: { type: 'object', properties: { running: { type: 'array', items: { type: 'string' } }, done: { type: 'array', items: { type: 'string' } }, failed_or_stuck: { type: 'array', items: { type: 'string' } } }, required: ['running', 'done', 'failed_or_stuck'] },
    budget: { type: 'string' },
    quota: { type: 'string' },
    log_offset: { type: 'number' },
    events: { type: 'array', items: { type: 'string' } },
    actions: { type: 'array', items: { type: 'string' } },
    needs_attention: { type: 'boolean' },
    attention_reason: { type: 'string' },
    newly_done: { type: 'array', items: { type: 'string' } },
    queue_finished: { type: 'boolean' },
    status_line: { type: 'string' },
  },
  required: ['at', 'runner', 'fleet', 'builds', 'events', 'actions', 'needs_attention', 'newly_done', 'queue_finished', 'status_line'],
}

const windowPrompt = (prev, i) => `You are the KEEPER of hengyuf's official3 synthetic-task queue (window ${i}). S=${S}.
The queue runner (synth/queue3/run_queue.py, CPU job "swe-task-queue3-runner") launches the builds ("synth-build-official3-*") and keeps the student fleet daemon ("swe-task-student-fleetd", student/fleetd_babysit.sbatch) alive. The daemon keeps the sc14 vLLM workers ("swe-task-student-wNN", 4 h each, single account) at fleet.json "target". Your job: keep the workers serving, watch the queue, and call main back only when main must act.

Previous snapshot (null on the first window): ${JSON.stringify(prev)}

Loop up to 3 times. Each time: sleep 590 seconds with Bash (\`sleep 590\`), then CHECK:
1. Runner: synth/queue3/heartbeat.json (age, note). Run \`squeue --me --name=swe-task-queue3-runner --noheader --format="%i %T %M"\`. Read the new lines of synth/queue3/runner.log since prev.log_offset (byte offset; store the new offset in log_offset). Watch for: launched, winner, "done (summary.json)", "ended unfinished", FAILED, STUCK, "launch failed", budget lines, "switching the active hub key", "stop launching", "queue finished".
2. Fleet: \`/usr/bin/python3 -I -B ${S}/student/fleet.py status\`. From it and \`squeue --me --format="%i %T %r %j %M %V"\`, get the daemon job, target, workers ready, starting and pending, the age of the oldest pending worker, breaker state, excluded nodes and the STOP flag. Track minutes_with_zero_ready across checks: it is 0 when any worker is ready, and adds about 10 per check while the target is > 0 and none is ready.
3. Builds: running "synth-build-official3-*" jobs, and runs with summary.json (synth/runs/official3-*/build/*/*/summary.json). newly_done = runs done now but not in prev.builds.done. failed_or_stuck = runs whose status is failed or stuck in synth/queue3/state.json.
4. Budget: the last budget line of runner.log. Quota: \`lfs quota -h -u hengyuf /lustre/fsw\`, one line: used and limit. Also \`grep -c "Disk quota exceeded"\` in the newest build logs (synth/runs/slurm/synth-build-official3-*.out).

KEEPER ACTIONS. These are the only commands you may run that change anything; each must have a reason in "actions":
- \`/usr/bin/python3 -I -B ${S}/student/fleet.py reset-breaker\`: only when status shows the breaker OPEN, at most once per window, and never twice in a row across windows (see prev.actions).
- \`/usr/bin/python3 -I -B ${S}/student/fleet.py reconcile\`: only when the daemon job is missing and the STOP flag is absent, as a one-off pass so workers get replaced. The runner itself resubmits the daemon; do not submit one yourself.
Never cancel, hold, release or submit any other job. Never edit queue/state/fleet files. Never read secrets (.env, S/secrets, S/atp/.env).

needs_attention = true (and stop looping) when any of these holds:
- the runner heartbeat is over 900 s old and its job is gone, unless the note is "stopped";
- a run became failed or stuck;
- the target is > 0 with zero ready workers for over 60 min, and the oldest pending replacement has waited over 2 h;
- the daemon has been missing for over 20 min while the STOP flag is absent;
- the breaker reopened right after your reset;
- excluded nodes > 15;
- a "Disk quota exceeded" error appeared, or the user quota is over 49.5T;
- a key switch, "stop launching", or a budget line under $300 appeared.
Also stop looping, without setting needs_attention, when newly_done is non-empty or the runner logged "queue finished" (queue_finished=true).
Otherwise run all 3 checks and return the last snapshot.
status_line: one short Chinese line for hengyuf, e.g. "官方3: 构建 5 运行/2 完成, worker 10 就绪, OFFICIAL_2 $412".`

phase('Watch')
let last = null
for (let i = 1; i <= MAX_WINDOWS; i++) {
  const snap = await agent(windowPrompt(prev, i), { label: `window:${i}`, phase: 'Watch', schema: SNAP, model: 'sonnet', effort: 'low' })
  if (!snap) { log(`window ${i}: no result; retrying with the previous snapshot`); continue }
  const changed = !prev || snap.events.length || snap.actions.length || snap.newly_done.length || snap.status_line !== prev.status_line
  if (changed) log(snap.status_line + (snap.actions.length ? ` | 动作: ${snap.actions.join('; ')}` : ''))
  else if (i % 6 === 0) log(`(心跳) ${snap.status_line}`)
  last = snap
  prev = snap
  if (snap.needs_attention || snap.newly_done.length || snap.queue_finished) {
    return { reason: snap.needs_attention ? 'needs_attention' : (snap.queue_finished ? 'queue_finished' : 'builds_done'), window: i, snapshot: snap }
  }
}
return { reason: 'window_cap', window: MAX_WINDOWS, snapshot: last }
