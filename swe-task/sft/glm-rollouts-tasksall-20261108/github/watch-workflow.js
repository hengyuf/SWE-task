export const meta = {
  name: 'glm-tasksall-watch',
  description: 'Watch the self-hosted GLM-5.3 tasks-all rollouts (tiers 3/4 resumed 2026-11-13): one cheap agent per tick runs the deterministic watch_tick.py and returns when it raises an alarm',
  phases: [ { title: 'Watch', detail: 'one cheap agent per ~5-9 min tick (runs watch_tick.py)', model: 'haiku' } ],
}
// 2026-11-13: the first version let each agent `sleep` ~19 min and then check by itself; agents can no longer sleep
// (36 windows burned in 6 min), so the wait and every rule now live in watch_tick.py (see its docstring for the alarms).
const TICK = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/github/watch_tick.py'
const SCHEMA = {
  type: 'object',
  properties: {
    state: { type: 'string', enum: ['running', 'needs_attention', 'complete'] },
    summary: { type: 'string' },
    alarms: { type: 'array', items: { type: 'object', properties: { key: { type: 'string' }, message: { type: 'string' } }, required: ['key', 'message'] } },
    snapshot: { type: 'object' },
    error: { type: 'string', description: 'empty unless the command failed or printed no JSON line' },
  },
  required: ['state', 'summary', 'alarms'],
}
const PROMPT = 'You are a read-only relay for a watcher script. Run exactly one command with the Bash tool (set its ' +
  'timeout to 600000 ms): `/usr/bin/python3 -I -B ' + TICK + ' --wait 300`. It waits a few minutes and prints ONE JSON ' +
  'line {"state", "summary", "alarms", "snapshot"}. Return exactly that content: state, summary, alarms (every ' +
  '{key, message} as printed) and snapshot. If the command fails or prints no JSON line, return state ' +
  '"needs_attention", summary "watch_tick.py failed", alarms [{"key": "tick_failed", "message": <the last 300 ' +
  'characters of its output>}] and error with the same text. Run no other command, edit nothing, submit or cancel ' +
  'nothing.'
const maxTicks = (args && args.max_ticks) || 120
let last = null
let failures = 0
for (let w = 0; w < maxTicks; w++) {
  const r = await agent(PROMPT, { label: 'watch-' + (w + 1), phase: 'Watch', schema: SCHEMA, model: 'haiku', effort: 'low' })
  if (!r) { log('tick ' + (w + 1) + ': no result'); failures++; if (failures >= 5) return { tick: w + 1, note: '5 ticks without a result' }; continue }
  failures = 0
  last = r
  if (r.state !== 'running' || (w + 1) % 6 === 0) log('tick ' + (w + 1) + ' [' + r.state + '] ' + r.summary)
  if (r.state !== 'running') return { tick: w + 1, result: r }
}
return { tick: maxTicks, result: last, note: 'tick cap reached: relaunch the watcher' }
