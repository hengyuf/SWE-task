export const meta = {
  name: 'posthoc-accept-v2',
  description: 'hengyuf-approved post-hoc acceptance: score (similarity judge) tasks that came into range via slot recovery or test repair, verify adversarially, and accept the verified ones (repaired copy packaged, sanitizing grader, bookkeeping)',
  whenToUse: 'hengyuf swe-task: after hengyuf approved post-hoc acceptance (2026-10-21 "yes accept them"); args: {tasks: [{run, task_id, source: "recovery"|"test_repair", records?, repair_dir?, scored_dir?, note}]}',
  phases: [
    { title: 'Score', detail: "the pipeline's own score() on the recovered or repaired records (skipped when already scored)" },
    { title: 'Verify', detail: 'adversarial re-derivation of rates, the repair or recovery, and the similarity inputs' },
    { title: 'Accept', detail: 'accept() of the right task dir + grader + metrics/summary bookkeeping' },
  ],
}

// Saved at S/synth/workflows/posthoc_accept_v2.js (main, 2026-10-21).

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const TASKS = (args && args.tasks) || []

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    run: { type: 'string' }, task_id: { type: 'string' },
    student_rate: { type: 'number' }, teacher_clean_rate: { type: 'number' },
    valid: { type: 'array', items: { type: 'integer' } }, cheated: { type: 'integer' }, answer_retrieval: { type: 'boolean' },
    similarity: { type: 'number' }, accept: { type: 'boolean' }, reason: { type: 'string' },
    out_dir: { type: 'string' }, metrics_file: { type: 'string' }, task_dir_for_accept: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['run', 'task_id', 'accept', 'reason', 'out_dir', 'problems'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' }, verdict: { type: 'string', enum: ['accept', 'reject', 'unclear'] },
    issues: { type: 'array', items: { type: 'string' } }, why: { type: 'string' },
  },
  required: ['task_id', 'verdict', 'why'],
}
const ACCEPT_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' }, accepted: { type: 'boolean' },
    files_changed: { type: 'array', items: { type: 'string' } }, validation: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['task_id', 'accepted', 'problems'],
}

const APPROVAL = `USER APPROVAL (hengyuf, 2026-10-21, in reply to main's question): "yes accept them". The question was whether tasks that reach the acceptance bars only after post-hoc platform-loss reruns (synth/tools/recover_slots.py) or post-hoc repairs of unstated hidden checks (workflows/task_test_repair.js) may be added to the accepted set. Main had promised to score them together and accept those that pass: similarity >= 6 from the pipeline's glm judge plus an independent adversarial verification. This approval covers the acceptance steps below, including writing accepted/, tasks.jsonl, metrics.json and summary.json for these tasks.`

const COMMON = `Context: hengyuf's swe-task pipeline builds synthetic SWE tasks. S=${S} (the same tree as /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task). Read S/README.md (pipeline, acceptance, grading gap, withdrawals) and S/synth/repair/posthoc_todo.md.
${APPROVAL}
Acceptance bars: sc14 rate <= 0.2, clean kimi >= 0.6 (cheated passes excluded), >= 3 valid slots per model, similarity >= 6.
Code: S/synth/task_builder.py (score(), judge_pass, failure analysis + similarity, accept(), dataset_row), S/synth/tools/recover_slots.py (record/export), S/synth/tools/grading_repair.py (stage/validate/install), and the earlier post-hoc score driver S/synth/runs/official1-ytt/audit/recovery/syn-yq-cand-c01-r1b4-c75b758f/score/inputs/score_driver.py.
Rules:
- Never read or print secrets (.env files, S/secrets, S/atp/.env). The pipeline reads the hub key itself.
- Run Python from S/synth with ../atp/runtime/controller/bin/python3.
- Never modify probe data, stage.json or seeds.json.
- At most 6 sandboxes at a time.
- Every run in this list is finished (summary.json; no build job); confirm before writing.`

function sourceText(t) {
  if (t.source === 'recovery') {
    return `SOURCE recovery: lost slots were rerun by recover_slots.py. Records: ${t.records || '(find them in runs/' + t.run + '/audit/recovery/' + t.task_id + '/)'}. First run \`tools/recover_slots.py record --run ${t.run} --task ${t.task_id}\` so the record reads the latest grading_repair statuses, then export the 10 slot records score() takes. The task dir to score and to accept is the packaged task dir from cand-*/stage.json; grading_repair installs the sanitizing grader after acceptance.`
  }
  return `SOURCE test_repair: an unstated hidden check was repaired in a scratch copy and every slot was regraded. Repair dir: ${t.repair_dir}. Its README.md, repair.diff, regrade.json, rates.json and teacher audits are there. Build the 10 slot records score() takes from the recorded rollouts, with each slot's status replaced by its REPAIRED grading status in regrade.json. Score with task_dir = the REPAIRED copy (${t.repair_dir}/task/${t.task_id}), so the failure analysis reads the repaired test.patch. The same repaired copy, which already has the sanitizing grader and test_paths, is the task dir to accept.`
}

function scorePrompt(t) {
  return `${COMMON}

YOUR JOB: score ${t.run} ${t.task_id} post hoc. ${sourceText(t)}
Context: ${t.note || ''}
- Build the Build object for this run and call task_builder.score(b, task, records, out) with out = S/synth/runs/${t.run}/audit/posthoc_accept/${t.task_id}/score/ (a new dir).
- Teacher audits: reuse the existing verdicts where the transcript is byte-identical. Run judge_pass only for kimi passes that have no verdict yet.
- score() runs the failure analysis and the similarity judge when the rates are in range.
- Do NOT call accept().
Report: the rates, valid counts, cheated count, answer retrieval, similarity, accept/reason, the new metrics.json path, task_dir_for_accept, and problems.`
}

function verifyPrompt(t, s) {
  return `${COMMON}

YOUR JOB: adversarially verify the post-hoc score of ${t.run} ${t.task_id} before it is accepted. Default to doubt.
Score: ${JSON.stringify(s)}
${sourceText(t)}
- Re-derive the rates and valid counts yourself from the inputs.
- For recovery: confirm each replaced slot was a platform loss and each rerun ran on the same task.
- For test_repair: confirm the repaired check really was unstated (read instruction.md), the fix is minimal, and the repaired copy is what was scored.
- Check that the similarity judge's inputs and score are plausible.
- Check the run's watch.md and audit/failure_match/${t.task_id}.json for data-quality flags against acceptance.
Verdict: accept only if every bar holds with clean evidence (similarity >= 6 included); reject if a bar fails; unclear if evidence is missing.`
}

function acceptPrompt(t, s, v) {
  return `${COMMON}

YOUR JOB: ACCEPT ${t.run} ${t.task_id} (hengyuf approved; score: ${JSON.stringify(s)}).
Decision basis: ${(v && v.why) || '(verify agreed)'}
${t.note ? 'Context: ' + t.note : ''}
Template: the first post-hoc acceptance, official1-ytt syn-yq-cand-c01-r1b4-c75b758f. Its driver, log and in-memory entry are in S/synth/runs/official1-ytt/audit/posthoc_accept/syn-yq-cand-c01-r1b4-c75b758f/ (accept_driver.py, accept_run.log, entry.json). Follow it, and save this task's entry.json in the same place under this run, so later grader rollouts (repair/grader-v3-install-20261021/work/rollout.py reads runs/<R>/audit/posthoc_accept/<T>/entry.json) find it.
The live grader S/synth/verifier/grader.py is v3 (md5 a64eea49, installed 13:42 on 10-21). Install that one. The v3 rollout already finished this run's pass.
1. Confirm the build of ${t.run} is finished (summary.json exists, no live job of the run).
2. Run task_builder.accept(b, metrics) with the NEW metrics.json, its task_dir set to the task dir to accept (${t.source === 'test_repair' ? 'the REPAIRED copy ' + (t.repair_dir || '') + '/task/' + t.task_id : 'the packaged task dir'}). It copies the task dir to runs/${t.run}/accepted/, archives the starting workspace through a sandbox, and adds the tasks.jsonl row.
3. Grader:
   - For recovery: install the sanitizing grader with grading_repair.py's own functions for this ONE task. Build its accepted entry in memory with plan_task and pass it to stage_task, validate_task (--tamper) and install_task. Do NOT run "grading_repair.py plan --run <R>" and do not edit repair/plan.json: a re-plan marks every already-installed accepted task with problems, which makes the grader-v3 rollout skip them, and plan.json keeps no fresh backup.
   - For test_repair: the copied task already has it. Re-validate it in a sandbox: reference 1, empty 0.
   - If S/synth/verifier/grader.py is newer than the copied grader (a grader v3 install may be running), note it for main instead of changing it.
   - If validation fails (reference not 1, empty not 0, or tamper), do NOT install. Undo the acceptance: move accepted/<T>/ and its .workspace archive into audit/posthoc_accept/<T>/rolled_back/, restore tasks.jsonl from your backup, and skip step 4. Report it.
4. Bookkeeping, each with a backup <file>.pre-posthoc and an atomic write:
   - the cand dir's metrics.json: accept true, reason "<reason> (post-hoc)", and a "posthoc" field {at, by: "main", approved_by: "hengyuf 2026-10-21", source: "${t.source}", score_dir, repair_or_recovery_dir};
   - the build's summary.json: add the task to accepted;
   - one line in runs/${t.run}/posthoc.jsonl.
5. Validate:
   - accepted/${t.task_id}/ has task.toml, tests/ (sanitizing grader, config test_paths) and solution/;
   - tasks.jsonl has exactly one row for it;
   - Gym's loader accepts the accepted copy (task_builder.validate_task).
Report the files changed and any problem.`
}

const results = await pipeline(
  TASKS,
  t => t.scored_dir
    ? Promise.resolve({ run: t.run, task_id: t.task_id, accept: true, reason: 'already scored', out_dir: t.scored_dir, metrics_file: `${t.scored_dir}/metrics.json`, problems: [], prior: true })
    : agent(scorePrompt(t), { label: `score:${t.task_id}`, phase: 'Score', schema: SCORE_SCHEMA }),
  (s, t) => (s && s.accept)
    ? (t.verified ? Promise.resolve({ s, v: { task_id: t.task_id, verdict: 'accept', why: t.verified } })
        : agent(verifyPrompt(t, s), { label: `verify:${t.task_id}`, phase: 'Verify', schema: VERIFY_SCHEMA }).then(v => ({ s, v })))
    : { s, v: null },
  (sv, t) => (sv && sv.v && sv.v.verdict === 'accept')
    ? agent(acceptPrompt(t, sv.s, sv.v), { label: `accept:${t.task_id}`, phase: 'Accept', schema: ACCEPT_SCHEMA }).then(a => ({ ...sv, a }))
    : { ...(sv || {}), a: null },
)
return results.map((r, i) => ({ task: TASKS[i], ...(r || {}) }))
