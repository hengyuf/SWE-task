export const meta = {
  name: 'failure-match-annotate',
  description: 'Annotate every eligible synthetic task (built, probed, not discarded, with sc14 rollouts) with whether sc14 repeats the target failure patterns; files feed the dashboard',
  whenToUse: 'hengyuf swe-task: after new tasks are scored (watcher ANNOTATE messages), or to backfill runs. args: {runs: [run], tasks?: {run: [task_id]}, dashboard?: bool, reaudit?: bool}',
  phases: [
    { title: 'Scan', detail: 'list eligible tasks that lack a valid audit file' },
    { title: 'Dashboard', detail: 'one-time: show audits on every eligible task (contract v1.4)' },
    { title: 'Annotate', detail: 'auditors, at most 6 tasks of one run each' },
    { title: 'Check', detail: 'validate the files and spot-check' },
  ],
}

// Saved at S/synth/workflows/failure_match_annotate.js (hengyuf, 2026-10-19). Invoke with
//   Workflow({scriptPath: '<this file>', args: {runs: ['official1-x'], tasks: {'official1-x': ['syn-...']}}})
// Without `tasks`, every eligible task of the listed runs that has no valid audit file is annotated.
// Optional (2026-10-24, hengyuf: "lower effort and lighter model for the auditing"): args.model ('sonnet', ...) and
// args.effort ('low' | 'medium' | ...) apply to every agent; omitted, agents inherit the session model and effort.
// Optional (2026-10-27, hengyuf: "compare the student rollouts with kimi ones (give strong/partial/weak/none labels)"):
// args.kimi_contrast = true adds a Kimi K3 comparison (kimi_contrast field + one summary sentence). The verdict keeps
// its meaning, so labels stay comparable with earlier batches; the dashboard collector ignores the extra field.

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const NODE_BIN = '/home/hengyuf/.vscode-server/cli/servers/Stable-2a59476c9bfcb90b3ddc372c36762471b7dfad1c/server/node'
const RUNS = (args && args.runs) || []
const ONLY = (args && args.tasks) || null
const REAUDIT = !!(args && args.reaudit)
const BATCH = 6
const AOPTS = {}
if (args && args.model) AOPTS.model = args.model
if (args && args.effort) AOPTS.effort = args.effort
const KIMI = !!(args && args.kimi_contrast)
const KIMI_RULES = `
KIMI CONTRAST (hengyuf, 2026-10-27: "compare the student rollouts with kimi ones"). The teacher Kimi K3 ran the same task
in the same probe: rollouts/kimi-r0..r4.md and .patch in the scored dir, raw grading under
round-<k>/probe-*/models/kimi/attempts/<task_idx>-<rep>/attempt-*/ -- except when the scored dir has teacher_reprobe.json
(tools/reprobe_teacher.py, 2026-11-05: Kimi re-run after its first slots broke): then the Kimi slots in rollouts/ come
from that file's "probe" directory (raw grading <probe>/models/kimi/attempts/<task_idx>-<rep>/attempt-*/, task_idx from
<probe>/manifest.json), and round-<k>/probe-* holds only the sc14 slots that count plus the discarded broken Kimi ones.
Read Kimi's rollouts too, and compare them with
sc14's on the mistakes you matched: did Kimi pass, and did it avoid the mistake sc14 made (how), or fail the same way?
Add this field to the audit file:
  "kimi_contrast": {"passed": <int>, "graded": <int>, "avoids_student_mistake": "yes" | "partly" | "no" | "n/a",
                    "note": "<= 400 chars: what Kimi did differently on the matched pattern(s), quoting a transcript line
                             or patch hunk; for verdict none, what Kimi did where sc14 failed, or n/a"}
and end the summary with one sentence on the Kimi contrast. The verdict keeps the guide's meaning (does sc14 repeat the
target failure patterns?). When Kimi fails the same way, say so in other_causes: the task is then hard for both
models rather than a student gap.`

const SCHEMA_DOC = `
Audit file, one per task, at S/synth/runs/<run>/audit/failure_match/<task_id>.json (UTF-8 JSON, mode 644; create the
directory; write atomically: temp name not ending in .json, then rename). Fields:
{ "schema": 1, "task_id": "...", "run": "official1-...", "cand": "cand-...", "round": 0 | 1 | 2, "audited_at": "<ISO time>",
  "auditor": "claude-subagent",
  "target_patterns": [{"name": "...", "category": "..."}],          // from the target's analysis/<target>/result.json
  "verdict": "strong" | "partial" | "weak" | "none",                  // does sc14 repeat the target failure patterns here?
  "summary": "<= 3 sentences, plain English",
  "patterns": [{"name": "<target pattern name>", "reproduced": "yes" | "partial" | "no",
                "slots": ["sc14-r1", ...], "evidence": "<= 600 chars: what sc14 did, quoting the transcript/patch/test"}],
  "student_rollouts": [{"slot": "sc14-r0", "status": "pass" | "fail" | "infra", "failing_tests": ["<= 8 names"],
                        "cause": "<= 300 chars: what actually went wrong", "pattern": "<target pattern name>" | null}],
  "other_causes": "<= 400 chars: failures unrelated to the target patterns (environment, unrelated bugs, no commit, ...)",
  "similarity_judge": {"score": 7 | null, "agrees": true | false | null, "comment": "<= 300 chars"} }
Verdict guide:
- strong: most failed sc14 rollouts fail in a way that matches a target pattern (same kind of mistake by the same
  mechanism, not just the same area or a shared habit).
- partial: some do, or the match is loose.
- weak: a match only in one rollout or only superficially.
- none: the failures have other causes, or sc14 did not fail at all.`

const ELIGIBLE = `A task is ELIGIBLE (hengyuf, 2026-10-19: "只要没有discarded, 且student 在上面有rollout"). All of these must hold:
- it was built and probed: it has a scored dir round-<k>/<cand>/ with metrics.json and rollouts/;
- it has at least one graded sc14 rollout (status pass or fail);
- it is NOT discarded. Discarded means listed as invalid in round-<k>/seeds.json because of a TASK-level problem
  (uniform failure, tests that cannot run, grading failures caused by the task), or withdrawn by main (metrics.json has a
  "withdrawn" field, or the task dir is under runs/<run>/withdrawn/), or under an abandoned-* build root. An invalid
  entry caused only by sandbox-platform faults does not discard the task: SandboxPtyError, "Cannot finalize a candidate
  before the agent process has stopped", needs_review slots with generation_not_quiesced, sandbox 410/404 at verify,
  hub 401/422 errors that made kimi slots fatal, student "abort" replies (Gym ValidationError) after a worker stop.
  (hengyuf's rule: "只要没有discarded, 且student 在上面有rollout"; platform outages on 2026-10-19/20 hit many slots.)
Eligible tasks therefore include accepted tasks, tasks rejected on pass rates, and tasks rejected on similarity.`

const AUDIT_RULES = `
You annotate synthetic tasks of hengyuf's swe-task pipeline. S=${S} (the same tree as
/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task). Read S/README.md first, the parts
on the pipeline and "hengyuf's decisions of 2026-10-18".
The question, from the user: in each task, does the student model sc14 really make the SAME KIND of mistake as in the
target DeepSWE task's failure analysis? The pipeline's GLM similarity judge scored this only for tasks whose pass rates
were in range. You check every eligible task independently, from the raw evidence.

Where things are, for a run R=S/synth/runs/<run>:
- Target failure analysis: R/analysis/<target task>/result.json (patterns: name, category, description, student_slots,
  teacher_contrast) and failure_analysis.md.
- Build tree R/build/<target>/<owner__repo>/ (skip abandoned-* roots):
  - candidate dirs cand-*: stage.json gives task_id and task_dir;
  - the scored dir round-<k>/<cand>/: metrics.json (rates, reason, similarity, similarity_detail),
    rollouts/index.md, and rollouts/<slot>.md (condensed transcripts with failing-test headers);
  - failure_analysis/ exists only for tasks whose rates were in range; elsewhere use the raw evidence;
  - full grading output per attempt: round-<k>/probe-*/models/sc14/attempts/<task_idx>-<rep>/attempt-*/ (ctrf.json,
    stdout.log, model.patch).
- Task text: the task dir's instruction.md and tests/ (test.patch, config.json).

Method per task:
1. Read the target patterns once per run.
2. Read every FAILED sc14 rollout of the task, and the passes briefly. When sc14 passed every graded rollout, write
   verdict "none" with the summary "sc14 passed all graded rollouts; nothing to match" and list the passes.
3. Verify claims against transcripts, patches and failing tests. A synthetic FA is only a guide.
4. Judge per target pattern; matches must be by the same mechanism.
5. Note grading artifacts in other_causes as such: a model's edits to test files that test.patch does not reset, Go
   redeclarations, hanging suites, gate-only fails, empty submissions.
6. similarity_judge: the metrics' similarity score if there is one, with agrees true/false; otherwise score null and
   agrees null.
Write one audit file per task exactly in this format:
${SCHEMA_DOC}
Rules:
- Read-only everywhere except your audit files.
- Never read secrets (.env, S/secrets, S/atp/.env).
- No Slurm, sandbox or hub calls; this is reading and judgement only.
- Be concrete and honest: if the match is weak, say so. Keep evidence short.
Finally, return a compact JSON summary.`

const SCAN_SCHEMA = {
  type: 'object',
  properties: {
    runs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          run: { type: 'string' },
          eligible: { type: 'integer' },
          already_audited: { type: 'integer' },
          todo: {
            type: 'array',
            items: {
              type: 'object',
              properties: { task_id: { type: 'string' }, cand: { type: 'string' }, round: { type: 'integer' }, decision: { type: 'string' } },
              required: ['task_id', 'cand', 'round'],
            },
          },
          skipped: { type: 'array', items: { type: 'string' } },
        },
        required: ['run', 'eligible', 'todo'],
      },
    },
    notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['runs'],
}
const AUDIT_OUT = {
  type: 'object',
  properties: {
    audited: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          task_id: { type: 'string' }, run: { type: 'string' }, verdict: { type: 'string' },
          judge_score: { type: 'number' }, agrees_with_judge: { type: 'boolean' }, one_line: { type: 'string' },
        },
        required: ['task_id', 'run', 'verdict', 'one_line'],
      },
    },
    files_written: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['audited', 'files_written', 'problems'],
}
const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    valid_files: { type: 'integer' },
    missing: { type: 'array', items: { type: 'string' } },
    malformed: { type: 'array', items: { type: 'string' } },
    distribution: { type: 'string' },
    spot_checks: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['valid_files', 'missing', 'malformed', 'distribution', 'issues'],
}

const scanPrompt = `You list the tasks to annotate in hengyuf's swe-task pipeline. S=${S}.
${ELIGIBLE}
Runs: ${RUNS.join(', ')}.
${ONLY ? 'Restrict to these task_ids (check that each is eligible): ' + JSON.stringify(ONLY) : 'Consider every eligible task of these runs.'}
${REAUDIT ? 'Re-annotate even tasks that already have a valid audit file.' : 'Leave out tasks that already have a valid audit file: S/synth/runs/<run>/audit/failure_match/<task_id>.json that parses, with schema 1, a matching task_id, and a verdict in strong/partial/weak/none.'}
For each run, return:
- eligible: the count;
- already_audited: the count;
- todo: task_id, cand, round, decision (accepted / rejected_rates / rejected_similarity / other);
- skipped: id plus the reason, for tasks scored but not eligible.
Read-only. Never read secrets.`

const dashboardPrompt = `YOUR PART: extend hengyuf's swe-task dashboard so the failure-match audit covers every eligible task, not only accepted ones (contract v1.4). D=${S}/dashboard.
Read D/CONTRACT.md (section "Failure-match audit (v1.3)"), D/README.md, D/collect.py and D/static/app.js first.

The dashboard is LIVE: job swe-task-dashboard on cpu1-00148:8765. It serves D/static from disk, and its server hot-reloads collect.py when the file changes. Keep every change working at every moment:
- write new code completely, test it, then save it in place;
- python -I -B -m py_compile before saving collect.py;
- ${NODE_BIN} --check static/app.js before saving.

${ELIGIBLE}
The audit file format is unchanged (similarity_judge.score may be null). Changes:
1. Collector:
   - mark each node eligible / not eligible per the definition. Add NODE "audit_eligible": true|false; withdrawn tasks count as discarded.
   - Extend index.json runs[].audit with "eligible", "audited_eligible" and verdict counts over eligible tasks. Keep the v1.3 accepted-only counts as they are.
2. UI:
   - every eligible node card gets the badge "FA match: <verdict> | not audited", not only accepted cards;
   - the task view's table becomes "Failure-match audit": one row per eligible task, grouped accepted first, then the other eligible tasks by round, with a decision column; rows open the drawer;
   - the run status bar shows "audit n/m" over eligible tasks, plus the accepted-only counts;
   - escape everything and stay null-safe.
3. Update CONTRACT.md to v1.4, tests/validate_contract.py, tests/test_collect.py and tests/frontend.test.js. Run python -I -B -m unittest discover -s tests and ${NODE_BIN} --test tests/*.test.js; all must pass.
Do not touch anything outside D. Never read secrets.`

phase('Scan')
const scan = await agent(scanPrompt, { ...AOPTS, label: 'scan', phase: 'Scan', schema: SCAN_SCHEMA })
if (!scan) return { error: 'scan failed' }
const batches = []
for (const r of scan.runs) {
  for (let i = 0; i < r.todo.length; i += BATCH) batches.push({ run: r.run, tasks: r.todo.slice(i, i + BATCH) })
}
log(`scan: ${scan.runs.map(r => `${r.run} ${r.todo.length}/${r.eligible}`).join(', ')}; ${batches.length} batches`)

const dashboardWork = (args && args.dashboard)
  ? agent(dashboardPrompt, { label: 'dashboard-v1.4', phase: 'Dashboard', schema: {
      type: 'object',
      properties: { files_changed: { type: 'array', items: { type: 'string' } }, tests: { type: 'string' }, notes: { type: 'string' } },
      required: ['files_changed', 'tests', 'notes'] } })
  : Promise.resolve(null)

const audits = await pipeline(
  batches,
  b => agent(`${AUDIT_RULES}${KIMI ? '\n' + KIMI_RULES : ''}\n\nYOUR RUN: ${b.run}. Annotate exactly these tasks: ${JSON.stringify(b.tasks)}`,
    { ...AOPTS, label: `annotate:${b.run}:${b.tasks[0].task_id.slice(-8)}`, phase: 'Annotate', schema: AUDIT_OUT }),
)
const dashboard = await dashboardWork
const done = audits.filter(Boolean)
const dropped = batches.filter((b, i) => !audits[i]).map(b => `${b.run}: ${b.tasks.map(t => t.task_id).join(', ')}`)
if (dropped.length) log(`batches without a result: ${dropped.length}`)

phase('Check')
const check = await agent(`You CHECK failure-match annotations in hengyuf's swe-task pipeline. S=${S}.
Expected files: one per task in ${JSON.stringify(scan.runs.map(r => ({ run: r.run, todo: r.todo.map(t => t.task_id) })))}, at S/synth/runs/<run>/audit/failure_match/<task_id>.json.
Do this:
1. Every expected file exists and is valid. Required: schema 1, task_id equals the file name, verdict in strong/partial/weak/none, the field types shown in S/dashboard/CONTRACT.md, and pattern names equal to the target's result.json names. If S/dashboard/tests/validate_contract.py exists, also run it with --sources against the live data.
2. Spot-check ${Math.max(3, Math.min(8, scan.runs.length))} audits against the raw evidence (the target result.json patterns and the sc14 rollouts in rollouts/*.md), from different runs, including at least one task that was not accepted and every "strong" verdict on an accepted task if there are few. Is the verdict defensible, and are the slots and failing tests real?
3. Report the verdict distribution over the expected files, overall and for accepted tasks.${KIMI ? `
4. Every expected file also has "kimi_contrast" with passed/graded integers, avoids_student_mistake in yes/partly/no/n/a
   and a note of at most 400 chars; spot-check two of them against rollouts/kimi-r*.md.` : ''}
Read-only: do not fix files. Never read secrets.`, { ...AOPTS, label: 'check', phase: 'Check', schema: CHECK_SCHEMA })

return { scan, audits: done, dropped, dashboard, check }
