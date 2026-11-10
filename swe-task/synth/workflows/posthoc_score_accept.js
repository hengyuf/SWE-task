export const meta = {
  name: 'posthoc-score-accept',
  description: 'Score newly in-range tasks after a post-hoc repair or recovery (teacher audit, failure analysis, similarity judge), verify adversarially, and accept the verified ones of finished builds',
  whenToUse: 'hengyuf swe-task: after recover_slots.py or a grading/test repair reports newly_in_range tasks (args: {tasks: [{run, task_id, records, note, accept_now}]})',
  phases: [
    { title: 'Score', detail: "the pipeline's own score() on the repaired/recovered records, in a scratch output dir" },
    { title: 'Verify', detail: 'adversarial re-derivation of rates and judge inputs' },
    { title: 'Accept', detail: 'accept() + sanitizing grader install for verified tasks of finished builds' },
  ],
}

// Saved at S/synth/workflows/posthoc_score_accept.js (main, 2026-10-20).

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const TASKS = (args && args.tasks) || []

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    run: { type: 'string' }, task_id: { type: 'string' },
    student_rate: { type: 'number' }, teacher_clean_rate: { type: 'number' },
    valid: { type: 'array', items: { type: 'integer' } },
    cheated: { type: 'integer' }, answer_retrieval: { type: 'boolean' },
    similarity: { type: 'number' }, accept: { type: 'boolean' }, reason: { type: 'string' },
    out_dir: { type: 'string' }, metrics_file: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['run', 'task_id', 'accept', 'reason', 'out_dir', 'problems'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' },
    verdict: { type: 'string', enum: ['accept', 'reject', 'unclear'] },
    rates_rederived: { type: 'string' },
    issues: { type: 'array', items: { type: 'string' } },
    why: { type: 'string' },
  },
  required: ['task_id', 'verdict', 'why'],
}
const ACCEPT_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' }, accepted: { type: 'boolean' },
    files_changed: { type: 'array', items: { type: 'string' } },
    validation: { type: 'string' }, problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['task_id', 'accepted', 'problems'],
}

const COMMON = `Context: hengyuf's swe-task pipeline builds synthetic SWE tasks. S=${S} (the same tree as /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task). Read S/README.md (pipeline, acceptance, grading gap, withdrawals). Acceptance: sc14 rate <= 0.2, clean kimi rate >= 0.6 (cheated passes excluded), at least 3 valid slots per model, and similarity >= 6 from the pipeline's glm judge. Code: S/synth/task_builder.py (score(), judge_pass, failure analysis + similarity, accept()), S/synth/tools/recover_slots.py (records/export), S/synth/tools/grading_repair.py (install). An earlier post-hoc re-score that drove score()'s functions: S/synth/runs/official1-fastapi/build/fastapi-implicit-head-options/litestar-org__litestar/round-1/cand-c05-r1b3/rescore-2026-10-20/inputs/rescore_driver.py.
Rules: never read or print secrets (.env files, S/secrets, S/atp/.env). The pipeline reads the hub key itself (the active key is NVAPI_KEY_OFFICIAL_1). Run Python from S/synth with ../atp/runtime/controller/bin/python3. Never modify probe data, stage.json, seeds.json or other tasks. Never touch the tree of a build that is still running (check squeue with a scrubbed env: env -i HOME=$HOME USER=$USER LANG=C.UTF-8 SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf PATH=/cm/shared/apps/slurm/current/bin:/usr/bin:/bin).`

function scorePrompt(t) {
  return `${COMMON}

YOUR JOB: score ${t.run} ${t.task_id} on its repaired/recovered slot records and report whether it meets acceptance.
Inputs: ${t.records} (the 10 slot records score() takes; reruns renamed to the slots they replace). A sidecar .json next to it may hold reusable kimi teacher verdicts. Context: ${t.note || ''}
1. Build the pipeline's Build object for this run (target task and repo from runs/${t.run}/launch.jsonl and the build root) and call task_builder.score(b, task, records, out). out is a NEW dir: S/synth/runs/${t.run}/audit/recovery/${t.task_id}/score/. task is this task's dict as the build used it: task_id, task_dir, snapshot etc. from its cand-*/stage.json. score() writes teacher_audit.json, the failure analysis and the similarity judge only when rates are in range, plus metrics.json with accept/reason. Re-judging kimi passes is fine.
2. Do NOT call accept() and do not write outside that out dir.
3. Report the rates, valid counts, cheated count, answer retrieval, the similarity score, accept/reason, the path of the new metrics.json, and any problem (glm timeouts, missing inputs).`
}

function verifyPrompt(t, s) {
  return `${COMMON}

YOUR JOB: adversarially verify the post-hoc score of ${t.run} ${t.task_id} before main accepts it. Default to doubt.
Score result: ${JSON.stringify(s)}
Context: ${t.note || ''}
- Re-derive sc14 and clean-kimi rates and valid counts from ${t.records} and the new teacher_audit.json. Check that every replaced slot really was a platform loss (recover_slots record, probe attempt files) and that each rerun ran on the same task (task_id, test.patch hash).
- Check the similarity judge's inputs in the score dir (failure analysis of THIS task's sc14 fails vs the target's analysis under S/synth/runs/${t.run}/analysis/) and that the score is plausible.
- Check the run's watch.md and audit/failure_match/${t.task_id}.json for data-quality flags against acceptance: grading artifacts, over-specified tests, leaks or released copies, timeouts under degraded serving.
Verdict: accept only if every acceptance bar holds with clean evidence; reject if one fails; unclear if evidence is missing.`
}

function acceptPrompt(t, s) {
  return `${COMMON}

YOUR JOB: post-hoc ACCEPT ${t.run} ${t.task_id} (verified: ${JSON.stringify(s)}). First confirm that the build of ${t.run} is finished (summary.json exists and no Slurm job of the run is pending or running); if not, stop and report accepted=false.
1. Run task_builder.accept(b, metrics) with the NEW metrics.json from ${s.metrics_file || s.out_dir}. It copies the packaged task dir to runs/${t.run}/accepted/, archives the starting workspace through a sandbox and adds the tasks.jsonl row.
2. Install the sanitizing grader into the accepted copy: grading_repair.py stage/validate/install for this task (reference 1, empty 0), as the repair of this run did for its other accepted tasks. If grading_repair refuses because the run is out of its scope, extend nothing; report it.
3. Bookkeeping, each with a backup <file>.pre-posthoc and an atomic write:
   - the cand dir's metrics.json: accept true, reason, plus a "posthoc" field {at, by: "main", source: "recover_slots" or "repair", records, score_dir};
   - the build's summary.json: add the task to accepted;
   - one line in runs/${t.run}/posthoc.jsonl.
4. Validate: accepted/${t.task_id}/ has task.toml, tests/ with the sanitizing grader, solution/ and instruction; tasks.jsonl has the row; the dashboard collector would see it as accepted.
Report the files changed and any problem.`
}

const results = await pipeline(
  TASKS,
  t => agent(scorePrompt(t), { label: `score:${t.task_id}`, phase: 'Score', schema: SCORE_SCHEMA }),
  (s, t) => (s && s.accept)
    ? agent(verifyPrompt(t, s), { label: `verify:${t.task_id}`, phase: 'Verify', schema: VERIFY_SCHEMA }).then(v => ({ s, v }))
    : { s, v: null },
  (sv, t) => (sv && sv.v && sv.v.verdict === 'accept' && t.accept_now)
    ? agent(acceptPrompt(t, sv.s), { label: `accept:${t.task_id}`, phase: 'Accept', schema: ACCEPT_SCHEMA }).then(a => ({ ...sv, a }))
    : { ...(sv || {}), a: null },
)
return results.map((r, i) => ({ task: TASKS[i], ...(r || {}) }))
