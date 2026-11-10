export const meta = {
  name: 'failure-match-review',
  description: 'Adversarially review strong/partial and accepted failure-match audit labels of official2 tasks; a second judge settles disputes and applies fixes',
  whenToUse: 'hengyuf swe-task: after failure_match_annotate.js, to catch inflated or malformed labels. args: {items: [{run, task_id, cand, round, verdict, accepted}], effort?, extra_eligibility?: [{run, task_id, cand, round, reason}]}',
  phases: [
    { title: 'Review', detail: 'one skeptical reviewer per label, tries to refute it' },
    { title: 'Decide', detail: 'a second judge settles each dispute and applies the fix' },
    { title: 'Eligibility', detail: 'decide skipped candidates and audit them if eligible' },
    { title: 'Check', detail: 'validate files, contract, final distribution' },
  ],
}

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const ITEMS = (args && args.items) || []
const EXTRA = (args && args.extra_eligibility) || []
const EFFORT = (args && args.effort) || 'medium'
const AUDIT_MODEL = (args && args.audit_model) || 'sonnet'

const GUIDE = `Verdict guide (hengyuf's pipeline, CONTRACT.md "Failure-match audit"):
- strong: MOST failed sc14 rollouts fail in a way that matches a target pattern: the same kind of mistake by the same mechanism, not just the same code area or a shared habit.
- partial: SOME failed rollouts match by mechanism, or the match is loose.
- weak: a match in only ONE rollout, or only superficially.
- none: the failures have other causes, or sc14 did not fail at all.
patterns[].reproduced is yes / partial / no per target pattern. student_rollouts[].pattern must be ONE exact target pattern name (from the target result.json) or null; never two names joined.`

const WHERE = `Evidence for a task in run R=S/synth/runs/<run> (S=${S}; same tree as /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task):
- target patterns: R/analysis/<target>/result.json (names, categories, descriptions) and failure_analysis.md;
- the audit file under review: R/audit/failure_match/<task_id>.json;
- scored dir: R/build/<target>/<owner__repo>/round-<k>/<cand>/ with metrics.json, rollouts/index.md, rollouts/<slot>.md (condensed transcripts, failing-test headers), rollouts/<slot>.patch, and failure_analysis/ when rates were in range;
- raw grading per attempt: R/build/<target>/<owner__repo>/round-<k>/probe-*/models/sc14/attempts/<task_idx>-<rep>/attempt-*/ (ctrf.json, stdout.log, model.patch); task_idx follows the probe manifest.json task order;
- task text: the task dir's instruction.md and tests/test.patch (task_dir is in R/build/<target>/<owner__repo>/<cand>/stage.json).
Rules: read-only except the files named in your instructions; never read secrets (.env, S/secrets, S/atp/.env); no Slurm, sandbox or hub calls.`

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' },
    original_verdict: { type: 'string' },
    reviewed_verdict: { type: 'string', enum: ['strong', 'partial', 'weak', 'none'] },
    agree: { type: 'boolean' },
    failed_rollouts: { type: 'integer' },
    matching_rollouts: { type: 'integer' },
    pattern_disputes: { type: 'array', items: { type: 'string' } },
    schema_fixes: { type: 'array', items: { type: 'string' } },
    reason: { type: 'string' },
  },
  required: ['task_id', 'original_verdict', 'reviewed_verdict', 'agree', 'failed_rollouts', 'matching_rollouts', 'schema_fixes', 'reason'],
}
const DECIDE_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' },
    final_verdict: { type: 'string', enum: ['strong', 'partial', 'weak', 'none'] },
    sided_with: { type: 'string', enum: ['auditor', 'reviewer', 'neither'] },
    file_changed: { type: 'boolean' },
    backup: { type: 'string' },
    reason: { type: 'string' },
  },
  required: ['task_id', 'final_verdict', 'sided_with', 'file_changed', 'reason'],
}
const ELIG_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' },
    eligible: { type: 'boolean' },
    why: { type: 'string' },
    verdict: { type: 'string' },
    file_written: { type: 'string' },
  },
  required: ['task_id', 'eligible', 'why'],
}
const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    valid_files: { type: 'integer' },
    malformed: { type: 'array', items: { type: 'string' } },
    contract: { type: 'string' },
    distribution_all: { type: 'string' },
    distribution_accepted: { type: 'string' },
    changes: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['valid_files', 'malformed', 'contract', 'distribution_all', 'distribution_accepted', 'changes', 'issues'],
}

const reviewPrompt = it => `You are a SKEPTICAL REVIEWER of one failure-match audit label in hengyuf's synthetic DeepSWE task pipeline.
The labels were written by lighter-model auditors, and a spot check found some inflated by one level (e.g. "strong" where only one rollout matched, or "partial" where the failures had a different mechanism than the pattern they were mapped to). Your job is to try to REFUTE the label below from the raw evidence. Do not trust the audit file's own reasoning or the pipeline's synthetic failure_analysis; check transcripts, patches and failing tests yourself.

Task: run ${it.run}, task_id ${it.task_id}, cand ${it.cand}, round ${it.round}, accepted ${it.accepted}. Current verdict: ${it.verdict}.
${WHERE}
${GUIDE}

Method:
1. Read the target patterns.
2. Read the audit file.
3. For EVERY failed sc14 rollout, decide independently whether its failure matches a target pattern by the SAME MECHANISM. Count failed rollouts, and how many of them match.
4. Apply the guide strictly to get your verdict.
5. List schema problems: a student_rollouts[].pattern that is not a single exact target name or null, a patterns[].name that is not an exact target name, or a status that contradicts the rollout header. Treat an empty-patch timeout as fail, not infra.
6. Write your review record (JSON, mode 644, atomic: temp name not ending in .json, then rename) to R/audit/failure_match_review/${it.task_id}.json, creating the directory. Include these fields: {schema: 1, task_id, original_verdict, reviewed_verdict, agree, failed_rollouts, matching_rollouts, pattern_disputes, schema_fixes, reason, stage: "review"}.
Do NOT modify the audit file itself. Return the structured result.`

const decidePrompt = (it, rv) => `You are the DECIDING JUDGE for one disputed failure-match audit label in hengyuf's synthetic DeepSWE task pipeline. An auditor wrote the label; an independent reviewer disagrees or found schema problems. Settle it from the raw evidence, then apply the result.

Task: run ${it.run}, task_id ${it.task_id}, cand ${it.cand}, round ${it.round}, accepted ${it.accepted}.
Auditor's verdict: ${it.verdict} (see the audit file).
Reviewer's verdict: ${rv.reviewed_verdict}. Failed rollouts ${rv.failed_rollouts}, matching ${rv.matching_rollouts}. Reason: ${rv.reason}
Pattern disputes: ${JSON.stringify(rv.pattern_disputes || [])}. Schema fixes requested: ${JSON.stringify(rv.schema_fixes || [])}.
${WHERE}
${GUIDE}

Method:
1. Verify the disputed points yourself against the transcripts, patches and failing tests. Do not just average the two opinions.
2. Choose the final verdict.
3. If the final verdict differs from the audit file, or a schema fix is needed:
   - copy the current file to R/audit/failure_match/${it.task_id}.json.pre-review, unless that backup already exists;
   - rewrite R/audit/failure_match/${it.task_id}.json atomically (temp name not ending in .json, then rename), keeping schema 1 and every existing field and field type;
   - update verdict, summary (<= 3 sentences, and say it was revised on review), patterns[].reproduced/slots/evidence where they change, and student_rollouts[].pattern (one exact target name or null);
   - do not add new top-level fields.
4. Update R/audit/failure_match_review/${it.task_id}.json (atomic): add final_verdict, sided_with, file_changed, decided_reason and stage "decided".
Return the structured result.`

const eligPrompt = e => `You decide whether one skipped synthetic task is ELIGIBLE for a failure-match audit in hengyuf's swe-task pipeline, and audit it if so.
Task: run ${e.run}, task_id ${e.task_id}, cand ${e.cand}, round ${e.round}. It was skipped because: ${e.reason}
${WHERE}
Eligibility (hengyuf: "只要没有discarded, 且student 在上面有rollout"):
- the task needs at least one graded sc14 rollout (pass or fail);
- it must not be discarded for a TASK-level problem (tests that cannot run, grading failures caused by the task itself);
- an invalid entry caused only by a sandbox or platform fault does NOT discard it. Platform faults include a sandbox 410/404 at verify, SandboxPtyError, a lost verifier, and hub errors.
Find the cause of "verifier produced no grade" for that rollout in the attempt's stdout.log or ctrf.json and the probe's worker.log.
If the task is eligible, write its audit file R/audit/failure_match/${e.task_id}.json in exactly the format of the other files in that directory (open two of them as templates; schema 1; atomic write), applying this guide:
${GUIDE}
If it is not eligible, write nothing. Return the structured result.`

phase('Review')
const reviewed = await pipeline(
  ITEMS,
  it => agent(reviewPrompt(it), { label: `review:${it.task_id.slice(4, 40)}`, phase: 'Review', schema: REVIEW_SCHEMA, effort: EFFORT }),
  (rv, it) => {
    if (!rv) return { item: it, review: null, decision: null }
    const needs = !rv.agree || rv.reviewed_verdict !== it.verdict || (rv.schema_fixes && rv.schema_fixes.length > 0)
    if (!needs) return { item: it, review: rv, decision: null }
    return agent(decidePrompt(it, rv), { label: `decide:${it.task_id.slice(4, 40)}`, phase: 'Decide', schema: DECIDE_SCHEMA, effort: EFFORT })
      .then(d => ({ item: it, review: rv, decision: d }))
  },
)

phase('Eligibility')
const elig = await parallel(EXTRA.map(e => () =>
  agent(eligPrompt(e), { label: `eligibility:${e.task_id.slice(4, 40)}`, phase: 'Eligibility', schema: ELIG_SCHEMA, model: AUDIT_MODEL, effort: EFFORT })))

const rows = reviewed.filter(Boolean)
const noReview = rows.filter(r => !r.review).map(r => r.item.task_id)
if (noReview.length) log(`no review result for ${noReview.length}: ${noReview.join(', ')}`)
const changed = rows.filter(r => r.decision && r.decision.file_changed)
log(`reviewed ${rows.length - noReview.length}/${ITEMS.length}; disputes ${rows.filter(r => r.decision).length}; files changed ${changed.length}`)

phase('Check')
const runs = [...new Set(ITEMS.map(i => i.run).concat(EXTRA.map(e => e.run)))]
const check = await agent(`You CHECK failure-match audit files in hengyuf's swe-task pipeline after a review pass. S=${S}.
Runs: ${runs.join(', ')}. Files: S/synth/runs/<run>/audit/failure_match/*.json (ignore *.pre-review backups and the failure_match_review/ records).
Files changed by the review: ${JSON.stringify(changed.map(r => r.item.run + '/' + r.item.task_id))}.
1. Every *.json in those dirs must parse, have schema 1, a task_id equal to the file name, a verdict in strong/partial/weak/none, patterns[].name equal to target result.json names, and student_rollouts[].pattern either one exact target name or null. List anything malformed.
2. Run python3 -I -B ${S}/dashboard/tests/validate_contract.py --sources (if its usage differs, read it and run the equivalent), and report its output.
3. Report the verdict distribution over all files of these runs, and separately over accepted tasks (from runs/<run>/accepted/tasks.jsonl).
4. List each changed file as "<task_id>: <old> -> <new>", using the .pre-review backup for the old verdict.
Read-only: do not fix anything. Never read secrets.`, { label: 'check', phase: 'Check', schema: CHECK_SCHEMA, effort: 'low' })

return {
  reviews: rows.map(r => ({ run: r.item.run, task_id: r.item.task_id, accepted: r.item.accepted, original: r.item.verdict,
    reviewed: r.review && r.review.reviewed_verdict, final: r.decision ? r.decision.final_verdict : (r.review ? r.item.verdict : null),
    sided_with: r.decision && r.decision.sided_with, file_changed: !!(r.decision && r.decision.file_changed),
    review_reason: r.review && r.review.reason, decide_reason: r.decision && r.decision.reason })),
  eligibility: elig,
  check,
}
