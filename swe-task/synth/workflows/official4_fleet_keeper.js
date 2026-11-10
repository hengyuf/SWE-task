export const meta = {
  name: 'official4-fleet-keeper',
  description: 'Background keeper of the sc14 student vLLM workers for the official4 queue: resets the breaker or reconciles when needed, returns to main only when main must act',
  whenToUse: 'hengyuf swe-task official4 queue: start once the runner has submitted the fleet daemon; relaunch with args.prev (the last snapshot) after main handles a return. args: {prev?, max_windows?}',
  phases: [{ title: 'Keep', detail: 'one low-effort window agent at a time: up to 3 checks about 10 min apart' }],
}

// official4 (hengyuf 2026-11-02) reuses the official3 keeper (hengyuf 2026-10-27: "another watcher to maintain student
// generation worker"). The fleet daemon (submitted by the
// queue runner) does the real work; this keeper only resets an open breaker, runs a one-off reconcile while the daemon
// is missing, and calls main back when the workers cannot be kept serving.

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const MAX_WINDOWS = (args && args.max_windows) || 48
// args.ack = {until: ISO time, reason}: main has seen the zero-ready / long-pending condition and decided to wait
const ACK = (args && args.ack) || null
let prev = (args && args.prev) || null

const SNAP = {
  type: 'object',
  properties: {
    at: { type: 'string' },
    daemon_job: { type: 'string' },
    daemon_alive: { type: 'boolean' },
    daemon_missing_since: { type: 'string', description: 'ISO time the daemon was first seen missing (STOP absent, target > 0), else empty' },
    target: { type: 'number' },
    ready: { type: 'number' },
    starting: { type: 'number' },
    pending: { type: 'number' },
    oldest_pending_min: { type: 'number' },
    expected_start: { type: 'string', description: 'earliest Slurm expected start of a pending worker, if any' },
    minutes_with_zero_ready: { type: 'number' },
    breaker: { type: 'string' },
    excluded: { type: 'number' },
    stop_flag: { type: 'boolean' },
    startup_failures: { type: 'array', items: { type: 'string' }, description: 'new worker startup failures this window: "<job> <node> <kind>"' },
    events_offset: { type: 'number' },
    events: { type: 'array', items: { type: 'string' } },
    actions: { type: 'array', items: { type: 'string' } },
    needs_attention: { type: 'boolean' },
    attention_reason: { type: 'string' },
    fleet_stopped: { type: 'boolean' },
    status_line: { type: 'string' },
  },
  required: ['at', 'target', 'ready', 'pending', 'minutes_with_zero_ready', 'stop_flag', 'events', 'actions', 'needs_attention', 'fleet_stopped', 'status_line'],
}

const windowPrompt = (prev, i) => `You KEEP the sc14 student workers serving for hengyuf's official4 synthetic-task queue (window ${i}). S=${S}.
The fleet daemon ("swe-task-student-fleetd", student/fleetd_babysit.sbatch, submitted by the queue runner through the dual-account helper) keeps the vLLM workers ("swe-task-student-wNN": 1 node x 8 GPUs, 4 h each, partition batch; account nemotron_agents_dev, and nemotron_rl_algo too once student/fleet.json "worker_accounts" lists both (hengyuf 2026-11-02: "yes you can use rl_algo account"): a slot still pending after "pair_delay_min" gets a second candidate on the other account, the first to start serves and the daemon cancels the other, so sibling_cancel events and job_ended events with sibling_of / "CANCELLED by" the user are NORMAL, never failures) at student/fleet.json "target" (10 while builds are pending or running, 0 otherwise). The builds' probes send their sc14 rollouts to these workers; with no worker ready, a probe waits up to 3 h and then fails.

Previous snapshot (null on the first window): ${JSON.stringify(prev)}

Loop up to 3 times. Each time: sleep 590 seconds with Bash (\`sleep 590\`), then CHECK:
1. \`/usr/bin/python3 -I -B ${S}/student/fleet.py status\` (read-only) and \`squeue --me --format="%i %T %P %r %j %M %V %S"\` for the swe-task-student-* jobs: daemon job and whether it runs, target, workers ready / starting / pending, the oldest pending worker's wait in minutes (from submit time %V), the earliest expected start (%S) of a pending worker, breaker state, excluded node count, and whether student/state/STOP exists.
2. minutes_with_zero_ready: 0 when any worker is ready; otherwise prev's value plus the minutes since prev.at while target > 0.
3. New lines of student/state/events.jsonl since prev.events_offset (byte offset; store the new size): worker startup failures (worker_failure events only; settlement cancels are not failures), unhealthy exclusions, breaker changes, and the incidents daemon_error, pass_skipped, sibling_cancel_failed, account_submit_failed. From \`fleet.py status\` also note how long ago the daemon's last pass was; count pending as queued slots (status "pending slots" when shown; a slot with two queued candidates counts once). For a worker that died within minutes, look at the tail of its log in student/logs/ and put the cause in events.

KEEPER ACTIONS. These are the only commands you may run that change anything; give each a reason in "actions":
- \`/usr/bin/python3 -I -B ${S}/student/fleet.py reset-breaker\`: only when status shows the breaker OPEN, at most once per window, and never when prev.actions already has a reset-breaker (two windows in a row).
- \`/usr/bin/python3 -I -B ${S}/student/fleet.py reconcile\`: only when the daemon job is missing, the STOP flag is absent and target > 0: a one-off pass so dead workers get replaced. The runner resubmits the daemon itself; never submit one.
Never cancel, hold, release or submit jobs; never edit fleet.json, student/state files or the STOP flag; never read secrets (.env, S/secrets, S/atp/.env).

needs_attention = true, and stop looping, when any of these holds:
- target > 0, zero ready workers for over 120 minutes, and the oldest pending worker has waited over 150 minutes${ACK ? ` — EXCEPT that main acknowledged this condition (${ACK.reason}): do not raise it before ${ACK.until}, unless the earliest Slurm expected start (%S) of a pending worker moves later than ${ACK.until}; at or after ${ACK.until} raise it again if no worker is ready` : ''};
- the daemon missing for over 20 minutes while the STOP flag is absent and target > 0 (track daemon_missing_since);
- the breaker reopened right after your reset in the previous window;
- excluded nodes > 15;
- 3 or more worker startup failures in this window;
- the daemon job runs but its last pass is more than 5 minutes old, or 3 or more daemon_error / pass_skipped events in this window (a daemon whose passes fail stops replacing workers silently);
- any sibling_cancel_failed event, any submit_failed event (every account failed), or 3 or more account_submit_failed events in this window (main checks the dual-account settlement). A single account_submit_failed is normally a transient Slurm "Socket timed out" that the daemon already handled by failing over to the other account and backing that account off for 10 min; report it in events only. Such a timed-out sbatch can still create its job on the controller, so a slot may show one candidate per account without a matching submit event: that is a normal pair (counted once, settled on start), not a problem. Do raise needs_attention if a slot has more than 2 candidates or 2 candidates on the SAME account.
fleet_stopped = true, and stop looping without needs_attention, when the STOP flag exists, target is 0 and no daemon job runs (the runner ended the queue).
Otherwise run all 3 checks and return the last snapshot.
status_line: one short Chinese line for hengyuf, e.g. "fleet: 目标 10, 就绪 6 / 启动 2 / 排队 2 (最老 95 分钟), breaker 关, 排除 9".`

phase('Keep')
let last = null
for (let i = 1; i <= MAX_WINDOWS; i++) {
  const snap = await agent(windowPrompt(prev, i), { label: `window:${i}`, phase: 'Keep', schema: SNAP, model: 'sonnet', effort: 'low' })
  if (!snap) { log(`window ${i}: no result; retrying with the previous snapshot`); continue }
  const changed = !prev || snap.events.length || snap.actions.length || snap.status_line !== prev.status_line
  if (changed) log(snap.status_line + (snap.actions.length ? ` | 动作: ${snap.actions.join('; ')}` : ''))
  else if (i % 6 === 0) log(`(心跳) ${snap.status_line}`)
  last = snap
  prev = snap
  if (snap.needs_attention || snap.fleet_stopped) {
    return { reason: snap.needs_attention ? 'needs_attention' : 'fleet_stopped', window: i, snapshot: snap }
  }
}
return { reason: 'window_cap', window: MAX_WINDOWS, snapshot: last }
