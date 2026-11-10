export const meta = {
  name: 'official3-audit-loop',
  description: 'As official3 rounds finish scoring, label every scored task strong/partial/weak/none (does sc14 repeat the target failure patterns? plus a Kimi K3 contrast), then review the strong/partial/accepted labels',
  whenToUse: 'hengyuf swe-task official3 queue: start with the runner; relaunch with args.seen and args.attempts from the last return after a poll cap. args: {seen?: ["<run>/<cand>"], attempts?: {key: n}, max_polls?}',
  phases: [
    { title: 'Poll', detail: 'a low-effort agent runs synth/queue3/audit_poll.py up to 3 times, about 10 min apart' },
    { title: 'Annotate', detail: 'nested failure_match_annotate.js: sonnet, medium effort, kimi_contrast' },
    { title: 'Review', detail: 'nested failure_match_review.js on strong/partial/accepted labels' },
  ],
}

// hengyuf 2026-10-27: "launch subagent to launch every generated task and compare the student rollouts with kimi ones
// (give strong/partial/weak/none labels)". Every built candidate is already run by its build's probe (sc14 x5 and
// Kimi K3 x5, the pipeline's old Kimi config: synth/settings.json, 1M context, reasoning_effort max). This loop labels
// each scored task as its round closes (seeds.json written), with the annotate workflow used for official2 (verdict =
// does sc14 repeat the target DeepSWE failure patterns there) plus a kimi_contrast field, then has the stronger
// reviewer re-check the strong/partial/accepted labels, since sonnet labels ran one level high on official2.

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const W = S + '/synth/workflows'
const MAX_POLLS = (args && args.max_polls) || 72
const MAX_ATTEMPTS = 3
const seen = new Set((args && args.seen) || [])
const attempts = Object.assign({}, (args && args.attempts) || {})
const verdicts = {}   // task_id -> {run, accepted, verdict, source}
const batches = []
const skipped = []
const gaveUp = []

const POLL = {
  type: 'object',
  properties: {
    new: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          key: { type: 'string' }, run: { type: 'string' }, round: { type: 'integer' }, cand: { type: 'string' },
          task_id: { type: 'string' }, accept: { type: 'boolean' },
        },
        required: ['key', 'run', 'round', 'cand', 'task_id'],
      },
    },
    scored: { type: 'integer' },
    audited: { type: 'integer' },
    rounds_open: { type: 'integer' },
    queue_finished: { type: 'boolean' },
    checks: { type: 'integer' },
    note: { type: 'string' },
  },
  required: ['new', 'queue_finished'],
}

const pollPrompt = i => `You POLL for newly scored official3 synthetic tasks (poll ${i}). S=${S}.
Loop up to 3 times. Each time: sleep 590 seconds with Bash (\`sleep 590\`), then run exactly this command:
  /usr/bin/python3 -I -B ${S}/synth/queue3/audit_poll.py --exclude '${[...seen].join(',')}'
It prints one JSON line. Stop looping as soon as its "new" list is non-empty, or its queue.finished is true.
Return: new (its "new" list, copied as is), scored, audited, rounds_open, queue_finished (= queue.finished), checks (how many times you ran it), note (one short line with the queue's by_status).
Run nothing else. Read-only. Never read secrets.`

const counts = () => {
  const out = { all: { strong: 0, partial: 0, weak: 0, none: 0 }, accepted: { strong: 0, partial: 0, weak: 0, none: 0 } }
  for (const v of Object.values(verdicts)) {
    if (out.all[v.verdict] !== undefined) out.all[v.verdict]++
    if (v.accepted && out.accepted[v.verdict] !== undefined) out.accepted[v.verdict]++
  }
  return out
}
const brief = c => `strong ${c.strong} / partial ${c.partial} / weak ${c.weak} / none ${c.none}`
const finish = (reason, poll) => ({
  reason, polls: poll, labels: counts(), seen: [...seen], attempts, gave_up: gaveUp, skipped, batches,
  verdicts: Object.entries(verdicts).map(([task_id, v]) => ({ task_id, ...v })),
})

for (let i = 1; i <= MAX_POLLS; i++) {
  const poll = await agent(pollPrompt(i), { label: `poll:${i}`, phase: 'Poll', schema: POLL, model: 'sonnet', effort: 'low' })
  if (!poll) { log(`poll ${i}: no result`); continue }
  const fresh = poll.new.filter(x => x && x.key && !seen.has(x.key))
  if (!fresh.length) {
    if (poll.queue_finished) {
      log(`queue finished; labels: ${brief(counts().all)}`)
      return finish('queue_finished', i)
    }
    continue
  }
  for (const x of fresh) attempts[x.key] = (attempts[x.key] || 0) + 1
  const runs = [...new Set(fresh.map(x => x.run))]
  const tasks = {}
  for (const x of fresh) (tasks[x.run] = tasks[x.run] || []).push(x.task_id)
  log(`poll ${i}: ${fresh.length} newly scored task(s) in ${runs.join(', ')}; annotating`)

  let ann = null
  try {
    ann = await workflow({ scriptPath: W + '/failure_match_annotate.js' },
      { runs, tasks, model: 'sonnet', effort: 'medium', kimi_contrast: true })
  } catch (e) {
    log(`annotate threw: ${String(e).slice(0, 200)}`)
  }
  const audited = ann && ann.audits ? ann.audits.flatMap(a => (a && a.audited) || []) : []
  const auditedIds = new Set(audited.map(a => a.task_id))
  const skippedText = ann && ann.scan ? ann.scan.runs.flatMap(r => (r.skipped || []).map(s => `${r.run}: ${s}`)) : []
  for (const x of fresh) {
    const wasSkipped = skippedText.some(s => s.includes(x.task_id))
    if (auditedIds.has(x.task_id) || wasSkipped) {
      seen.add(x.key)
      if (wasSkipped) skipped.push(skippedText.find(s => s.includes(x.task_id)).slice(0, 300))
    } else if (attempts[x.key] >= MAX_ATTEMPTS) {
      seen.add(x.key)
      gaveUp.push(x.key)
      log(`gave up on ${x.key} after ${attempts[x.key]} attempts (no audit, not skipped)`)
    }
  }
  const byId = {}
  for (const x of fresh) byId[x.task_id] = x
  for (const a of audited) {
    const x = byId[a.task_id]
    if (x) verdicts[a.task_id] = { run: x.run, accepted: !!x.accept, verdict: a.verdict, source: 'annotate' }
  }

  const items = audited
    .filter(a => byId[a.task_id] && (a.verdict === 'strong' || a.verdict === 'partial' || byId[a.task_id].accept))
    .map(a => ({ run: byId[a.task_id].run, task_id: a.task_id, cand: byId[a.task_id].cand, round: byId[a.task_id].round,
      verdict: a.verdict, accepted: !!byId[a.task_id].accept }))
  let changed = []
  if (items.length) {
    let rev = null
    try {
      rev = await workflow({ scriptPath: W + '/failure_match_review.js' }, { items, effort: 'medium' })
    } catch (e) {
      log(`review threw: ${String(e).slice(0, 200)}`)
    }
    for (const r of (rev && rev.reviews) || []) {
      if (r && r.final && verdicts[r.task_id]) {
        if (r.final !== verdicts[r.task_id].verdict) changed.push(`${r.task_id}: ${verdicts[r.task_id].verdict} -> ${r.final}`)
        verdicts[r.task_id] = { ...verdicts[r.task_id], verdict: r.final, source: 'review' }
      }
    }
  }
  const mine = audited.map(a => verdicts[a.task_id]).filter(Boolean)
  const c = { strong: 0, partial: 0, weak: 0, none: 0 }
  for (const v of mine) if (c[v.verdict] !== undefined) c[v.verdict]++
  batches.push({ poll: i, runs, labeled: audited.length, after_review: c, reviewed: items.length, changed,
    check_issues: ann && ann.check ? (ann.check.issues || []).slice(0, 5) : ['annotate returned no check'] })
  log(`batch at poll ${i}: ${audited.length} labeled (${brief(c)}); reviewed ${items.length}, changed ${changed.length}` +
    `${changed.length ? ': ' + changed.join('; ') : ''} | total ${brief(counts().all)}`)
}
return finish('poll_cap', MAX_POLLS)
