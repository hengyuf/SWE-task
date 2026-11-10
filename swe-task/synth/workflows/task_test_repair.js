export const meta = {
  name: 'task-test-repair',
  description: 'Repair a hidden test that checks something the instruction does not state (scratch copy only), regrade every probe slot, re-derive rates, and verify adversarially; main decides keep / withdraw / accept',
  whenToUse: 'hengyuf swe-task: a watcher found an over-specified hidden check that decides a task (args: {tasks: [{run, task_id, defect, fix}]})',
  phases: [
    { title: 'Repair', detail: 'minimal test.patch fix in a scratch copy, validation, regrade of all slots under the sanitizing grader' },
    { title: 'Verify', detail: 'adversarial check that the fix matches the instruction and the rates follow' },
  ],
}

// Saved at S/synth/workflows/task_test_repair.js (main, 2026-10-20). Earlier single-agent repairs of this kind:
// official1-ytt c04 (2026-10-20 00:13) and official1-aiomonitor c02-r1b0 (repair-2026-10-20/ under its cand dir).

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const TASKS = (args && args.tasks) || []

const REPAIR_SCHEMA = {
  type: 'object',
  properties: {
    run: { type: 'string' }, task_id: { type: 'string' },
    repair_dir: { type: 'string' }, diff_summary: { type: 'string' },
    validation: { type: 'string' },
    slots: { type: 'array', items: { type: 'object', properties: {
      slot: { type: 'string' }, recorded: { type: 'string' }, repaired: { type: 'string' }, control: { type: 'string' }, note: { type: 'string' },
    }, required: ['slot', 'recorded', 'repaired'] } },
    sc14_before: { type: 'number' }, sc14_after: { type: 'number' },
    kimi_before: { type: 'number' }, kimi_after: { type: 'number' },
    recommendation: { type: 'string', enum: ['keep', 'withdraw', 'newly_in_range', 'still_rejected', 'unclear'] },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['run', 'task_id', 'recommendation', 'problems'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    task_id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirm', 'refute', 'unclear'] },
    fix_faithful: { type: 'boolean' },
    rates_rederived: { type: 'string' },
    issues: { type: 'array', items: { type: 'string' } },
    why: { type: 'string' },
  },
  required: ['task_id', 'verdict', 'why'],
}

const COMMON = `Context: hengyuf's swe-task pipeline builds synthetic SWE tasks. S=${S} (the same tree as /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task). Read S/README.md (pipeline, acceptance, grading gap, withdrawals). Acceptance: sc14 <= 0.2, clean kimi >= 0.6 (cheated passes excluded), >= 3 valid slots per model, similarity >= 6.
This is a post-hoc DATA-QUALITY repair that main runs under hengyuf's standing decisions of 2026-10-19/20: existing tasks are re-graded and re-scored, and over-specified hidden tests found by the watchers get a post-hoc look. Earlier repairs of the same kind: official1-ytt c04 and official1-aiomonitor c02-r1b0 (see S/synth/runs/official1-aiomonitor/build/aiomonitor-task-snapshots-diff/tobymao__saq/round-1/cand-c02-r1b0/repair-2026-10-20/).
Your outputs go ONLY to a new scratch dir. You never change accepted/, metrics.json, summary.json, probe data, stage.json or seeds.json, and you never withdraw or accept anything; main decides.
Tools: S/synth/tools/grading_repair.py (stage/validate/regrade logic, the sanitizing grader S/synth/verifier/grader.py), S/synth/tools/regrade.py, S/synth/task_builder.py. Use the pipeline venv S/atp/runtime/controller/bin/python3 from S/synth. At most 6 sandboxes at a time. Never read or print secrets (.env files, S/secrets, S/atp/.env). Scrubbed Slurm env if needed: env -i HOME=$HOME USER=$USER LANG=C.UTF-8 SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf PATH=/cm/shared/apps/slurm/current/bin:/usr/bin:/bin`

function repairPrompt(t) {
  return `${COMMON}

YOUR JOB: repair one over-specified hidden check of ${t.run} ${t.task_id} in a scratch copy and re-derive the task's rates.
Defect (from the watcher): ${t.defect}
Intended fix: ${t.fix}
1. Find the task dir as packaged (and, if accepted, runs/${t.run}/accepted/${t.task_id}/ with its installed sanitizing grader). Copy it to S/synth/runs/${t.run}/audit/test_repair/${t.task_id}/task/. Edit ONLY tests/test.patch there, minimally: keep test names, node ids and every check the instruction states, and relax only the unstated detail.
2. Validate the repaired copy with the sanitizing grader: reference solution 1, empty patch 0, and test.patch applies on the clean starting state.
3. Regrade EVERY graded probe slot of both models (the final graded attempt of each; see the round's rollouts/index.md and grading_repair's plan logic) on the repaired copy with the sanitizing grader. For each slot whose outcome changes, also run a control on an unrepaired staged copy, which must reproduce the recorded outcome. Record per slot: recorded / repaired / control, and the failing tests.
4. Re-derive sc14 rate and clean kimi. Kimi passes stay clean as recorded in teacher_audit.json. A slot that newly PASSES under the repair needs the teacher audit (judge_pass from task_builder) before it counts as clean: run it and record the verdict.
5. Recommend:
   - for an accepted task: keep if both rate bars still hold, else withdraw;
   - for a rejected task: newly_in_range if both bars now hold (main then runs the similarity judge), else still_rejected;
   - unclear if the evidence is incomplete.
6. Write repair.diff, validate.json, regrade.json, rates.json and a short README.md in the scratch dir.`
}

function verifyPrompt(t, r) {
  return `${COMMON}

YOUR JOB: adversarially verify a post-hoc test repair before main acts on it. Default to doubt.
Task: ${t.run} ${t.task_id}. Defect: ${t.defect}
Repair result: ${JSON.stringify(r)}
- Read the task's instruction.md and decide independently whether the relaxed check really was unstated, so the fix is faithful. Also check that the fix does not drop stated checks or weaken tests beyond the defect.
- Open repair.diff and confirm it is minimal.
- Re-derive the rates from regrade.json and teacher_audit. For the slot(s) that decide the recommendation, re-run one regrade yourself (repaired copy, plus control).
- Confirm the recommendation follows from the acceptance bars.
Verdict: confirm, refute (say what is wrong) or unclear.`
}

const results = await pipeline(
  TASKS,
  t => agent(repairPrompt(t), { label: `repair:${t.task_id}`, phase: 'Repair', schema: REPAIR_SCHEMA }),
  (r, t) => r ? agent(verifyPrompt(t, r), { label: `verify:${t.task_id}`, phase: 'Verify', schema: VERIFY_SCHEMA }).then(v => ({ r, v })) : null,
)
return results.map((x, i) => ({ task: TASKS[i], ...(x || {}) }))
