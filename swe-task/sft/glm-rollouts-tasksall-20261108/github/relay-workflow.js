export const meta = {
  name: 'swe-task-github-relay',
  description: 'Relay for the SWE-task GitHub issue: post the main agent outbox + hourly progress, return when the user replies',
  phases: [ { title: 'Relay', detail: 'one cheap agent per ~10 min tick (runs relay_tick.py)' } ],
}
const TICK = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/github/relay_tick.py'
const SCHEMA = {
  type: 'object',
  properties: {
    replies: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, created_at: { type: 'string' }, url: { type: 'string' }, body: { type: 'string' } },
      required: ['id', 'body'] } },
    posted: { type: 'array', items: { type: 'string' } },
    progress_posted: { type: 'boolean' },
    error: { type: 'string' },
  },
  required: ['replies'],
}
const PROMPT = 'You are a message relay. Run exactly one command with the Bash tool (set its timeout to 600000 ms): ' +
  '`/usr/bin/python3 -I -B ' + TICK + ' --wait 540`. It prints one JSON line. Return its content: replies (the list of ' +
  '{id, created_at, url, body} exactly as printed), posted (each posted item as a short string), progress_posted, and ' +
  'error (its last_error value, or an empty string). Run no other command, edit nothing, post nothing yourself.'
const maxWindows = (args && args.max_windows) || 36
for (let w = 0; w < maxWindows; w++) {
  const r = await agent(PROMPT, { label: 'relay-' + (w + 1), phase: 'Relay', schema: SCHEMA, model: 'haiku', effort: 'low' })
  if (!r) { log('tick ' + (w + 1) + ': no result'); continue }
  log('tick ' + (w + 1) + ': replies ' + r.replies.length + ', posted ' + (r.posted || []).length + (r.progress_posted ? ' (progress)' : '') + (r.error ? ', error: ' + r.error : ''))
  if (r.replies.length) return { tick: w + 1, replies: r.replies }
}
return { tick: maxWindows, replies: [], note: 'tick cap reached: relaunch the relay' }
