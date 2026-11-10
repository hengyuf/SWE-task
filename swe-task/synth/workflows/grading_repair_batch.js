export const meta = {
  name: 'grading-repair-batch',
  description: 'Re-grade the tasks of finished official1 builds with the sanitizing grader (hengyuf 2026-10-20: these builds are repaired after they finish): plan, validate and regrade decision-relevant slots per run, install in accepted/, verify every decision change',
  whenToUse: 'hengyuf swe-task: post-run grading repair of builds that ran with the DeepSWE grader (args: {runs: [finished official1-* runs]})',
  phases: [
    { title: 'Prepare', detail: 'bring the runs into grading_repair.py scope, plan, smoke test' },
    { title: 'Regrade', detail: 'per run: validate tasks, regrade decision-relevant slots with controls, install grader in accepted/' },
    { title: 'Verify', detail: 'adversarial check of every decision-changing flip' },
  ],
}

// Saved at S/synth/workflows/grading_repair_batch.js (main, 2026-10-20). Invoke with
//   Workflow({scriptPath: '<this file>', args: {runs: ['official1-x', ...]}})
// Only FINISHED builds (summary.json written, no live job) may be listed.

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const RUNS = (args && args.runs) || []

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    summary: { type: 'string' },
    tool_changes: { type: 'string' },
    tests: { type: 'string' },
    per_run: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          run: { type: 'string' }, language: { type: 'string' }, mode: { type: 'string' },
          accepted_tasks: { type: 'integer' }, near_bar_rejected: { type: 'integer' },
          slots_to_regrade: { type: 'integer' }, glob_notes: { type: 'string' },
        },
        required: ['run', 'slots_to_regrade'],
      },
    },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['ok', 'summary', 'per_run'],
}
const RUN_SCHEMA = {
  type: 'object',
  properties: {
    run: { type: 'string' },
    validated: { type: 'array', items: { type: 'object', properties: { task_id: { type: 'string' }, reference: { type: 'number' }, empty: { type: 'number' }, ok: { type: 'boolean' } }, required: ['task_id', 'ok'] } },
    installed: { type: 'array', items: { type: 'string' } },
    decisions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          task_id: { type: 'string' }, status_before: { type: 'string' },
          sc14_before: { type: 'number' }, sc14_after: { type: 'number' },
          kimi_before: { type: 'number' }, kimi_after: { type: 'number' },
          flips: { type: 'array', items: { type: 'string' } },
          recommendation: { type: 'string', enum: ['keep', 'withdraw', 'newly_in_range', 'still_rejected', 'unclear'] },
          note: { type: 'string' },
        },
        required: ['task_id', 'recommendation'],
      },
    },
    regraded_slots: { type: 'integer' },
    flaky_slots: { type: 'array', items: { type: 'string' } },
    watcher_crosscheck: { type: 'array', items: { type: 'string' } },
    files_written: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['run', 'decisions', 'problems'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    checked: { type: 'integer' },
    confirmed: { type: 'array', items: { type: 'string' } },
    refuted: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, why: { type: 'string' } }, required: ['item', 'why'] } },
    final_recommendations: {
      type: 'array',
      items: { type: 'object', properties: { run: { type: 'string' }, task_id: { type: 'string' }, action: { type: 'string', enum: ['keep', 'withdraw', 'consider_accept', 'no_change'] }, why: { type: 'string' } }, required: ['run', 'task_id', 'action', 'why'] },
    },
    notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['checked', 'final_recommendations'],
}

const COMMON = `Context: hengyuf's swe-task pipeline builds synthetic SWE tasks. S=${S} (the same tree as /lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task). Read S/README.md first, especially the "Grading gap" bullet (2026-10-19 repair, 2026-10-20 JS/TS refinement) and "Withdrawals".

THE GRADING GAP. The DeepSWE grader resets only the files test.patch touches, so a model's edits to OTHER test files survive into the graded run: false FAILs (renumbered parametrize IDs, a model test redeclaring a hidden Go helper, model-added fixtures picked up by table-driven tests) and false PASSes (rewritten P2P assertions or golden fixtures). S/synth/verifier/grader.py is that grader plus one marked step: after model.patch is applied, every model.patch path matching config.json "test_paths" goes back to base_commit (deleted when base_commit lacks it); "test_paths_mode" "non_additive" (JavaScript/TypeScript only) keeps an existing test file the model only added lines to. S/synth/tools/grading_repair.py (plan / stage / validate / regrade / install; read its docstring) repaired the 11 earlier runs on 2026-10-19 and the jest tasks on 2026-10-20; records are in runs/<run>/audit/grading_repair*/ and S/synth/repair/.

hengyuf's decisions (2026-10-20): new builds switch to the sanitizing grader, and the builds that ran with the DeepSWE grader are repaired the same way once they finish ("跑完后统一修复"): validate, regrade the slots that could change a decision, re-score, and install the grader in their accepted tasks.

Runs in scope (and ONLY these; every other official1-* run is either already repaired or a LIVE build that you must never touch): ${RUNS.join(', ')}. Each has a finished build (summary.json under build/<task>/<owner__repo>/ and no live Slurm job); check that before working on a run and skip it if not.

Rules:
- Never modify probe data, metrics.json, stage.json, seeds.json, cand-*/ dirs, build code (task_builder.py, rollouts.py, probe_worker.py, common.py) or prompts. grading_repair.py itself may be extended (atomic edit, backup <file>.pre-batch2, offline tests must pass).
- Do not withdraw or accept anything; main decides.
- Do not edit S/dashboard.
- Never read or print secrets (.env files, S/secrets, S/atp/.env, key env vars).
- Use the pipeline venv S/atp/runtime/controller/bin/python3, run from S/synth. Offline tests: ../atp/runtime/controller/bin/python3 -B -m unittest discover -s tests (from S/synth). Scrubbed Slurm env if needed: env -i HOME=$HOME USER=$USER LANG=C.UTF-8 SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf PATH=/cm/shared/apps/slurm/current/bin:/usr/bin:/bin
- Sandboxes: at most 6 at a time per agent (--concurrency). A shared pool also serves 5 live builds; sandbox.owned backs off on capacity errors. On SandboxPtyError or capacity errors, retry with backoff; a slot that cannot be graded after 3 tries is reported, not guessed.`

const PREPARE_PROMPT = `${COMMON}

YOUR JOB: prepare the repair of the in-scope runs. Make no change to any task dir.
1. grading_repair.py refuses runs outside its IN_SCOPE tuple (the 11 runs of 2026-10-19). Extend it so these finished runs are in scope too, keeping the guard against live builds: a run outside the original 11 is accepted only if its build summary.json exists and no Slurm job of the run is pending or running (or simply list them explicitly, with the summary.json check). Keep every sort and lookup working for the new runs. Atomic edit, backup *.pre-batch2, and the offline suite must pass.
2. Back up S/synth/repair/plan.json to plan.json.pre-batch2 (plan --run merges into it), then run: plan --run ${RUNS.join(' ')}. Read the result per run: language and mode (top100.jsonl language; "non_additive" only for JavaScript/TypeScript), globs, "glob_gaps", any reference solution.patch touching a glob (narrow with GLOB_OVERRIDES if so), accepted and near-bar task counts, and the scheduled slots.
   GLOB GAPS: the language defaults match JS "test/**"/"tests/**" and Rust "tests/**" only at the repo root. New builds since 2026-10-20 also restore per-task test directories (synth/verifier/test_paths.py test_patch_globs: test-named dirs that test.patch touches) and, for official1-anko, main added 'vm/tests/**', 'py/tests/**', 'stdlib/*/tests/**' (gpython runs every script there as graded P2P). Make the repair match: for each run, add a RUN_GLOBS override (language defaults + the run's test directories: those in glob_gaps, those test.patch touches, and directories whose files the graded table-driven tests read) when glob_gaps or the run's layout shows test material outside the defaults (official1-sqlformatter: packages/*/test/** including the shared packages/formatter/test/behavesLikeSqlFormatter.ts; official1-wasmi: crates/*/tests/**; official1-anko: the 3 gpython script dirs). Never cover a directory that a reference solution.patch touches. Re-plan after the override and report the globs per run.
3. Smoke test end to end on one task of the first run: stage, validate (reference 1, empty 0), and regrade one scheduled slot with both graders.
Return the per-run counts and any problem. ok=false only if the tool cannot be used for these runs.`

function runPrompt(run, plan) {
  return `${COMMON}

YOUR JOB: repair and re-score run ${run}, following S/synth/repair/plan.json (the prepare step planned it). Prepare summary: ${plan ? JSON.stringify(plan.per_run.filter(r => r.run === run)) : 'n/a'}.

For run ${run}:
1. VALIDATE each task in the plan for this run, and EVERY accepted task of the run even if no slot is scheduled (stage, then validate: reference = 1, empty = 0, and --tamper for accepted tasks). A task that fails validation: try narrowing its globs once; if it still fails, report it and do not install.
2. REGRADE each scheduled slot with the NEW grader. Where the new outcome differs from the slot's recorded outcome, regrade the same slot once more with the OLD grader as a control.
   - Flip: the control reproduces the recorded outcome and the new outcome differs. Write down which test files the model edited and which graded tests changed status.
   - Flaky: the control does not reproduce the recorded outcome. List the slot under flaky_slots; it does not count as a flip.
3. CROSS-CHECK with S/synth/runs/${run}/watch.md: the build's watcher flagged grading artifacts by hand (P2P-REWRITE?, GATE-ONLY, REDECLARED, UNDEFINED-IN-TEST, PARENT-ONLY-FAIL, model-added fixtures, golden-file rewrites, false PASS/FAIL). For every flagged slot of an accepted or near-bar task, say whether the plan covered it and what the regrade gave; regrade any decision-relevant flagged slot the plan missed. Put one line per flagged slot in watcher_crosscheck.
4. RE-SCORE each task from its recorded statuses plus the confirmed flips.
   - sc14 rate = passes / valid.
   - Clean kimi = non-cheated passes / valid (keep cheated passes excluded, as recorded in teacher_audit.json and metrics).
   - Accepted tasks: "keep" if sc14 <= 0.2 and clean kimi >= 0.6 still hold; otherwise "withdraw".
   - Rejected tasks: "newly_in_range" if both now hold (main then runs the similarity judge); otherwise "still_rejected".
   - Use "unclear" when the evidence is incomplete.
5. Write one JSON per task to S/synth/runs/${run}/audit/grading_repair/<task_id>.json, with the same fields as the 2026-10-19 records (e.g. S/synth/runs/official1-arcane/audit/grading_repair/): globs, mode, validation result, every regraded slot (recorded / new / control outcome, the edited test files, changed tests), before and after rates, and the recommendation. Write atomically.
6. INSTALL the new grader, test_paths and test_paths_mode into S/synth/runs/${run}/accepted/<task_id>/tests/ for every accepted task that passed validation, regardless of the recommendation (install keeps backups). Install nothing anywhere else.
Return the per-task decisions.`
}

function verifyPrompt(results) {
  const items = []
  for (const r of results) for (const d of r.decisions || []) if (d.recommendation === 'withdraw' || d.recommendation === 'newly_in_range' || d.recommendation === 'unclear') items.push(`${r.run} ${d.task_id}: ${d.recommendation} (sc14 ${d.sc14_before}->${d.sc14_after}, kimi ${d.kimi_before}->${d.kimi_after}; flips: ${(d.flips || []).join('; ')})`)
  const problems = []
  for (const r of results) for (const p of r.problems || []) problems.push(`${r.run}: ${p}`)
  return `${COMMON}

YOUR JOB: adversarially verify the grading-repair results that would change a decision. Default to doubt.
Items:
${items.join('\n') || '(none)'}
Problems the per-run agents reported:
${problems.join('\n') || '(none)'}
The per-task evidence is in S/synth/runs/<run>/audit/grading_repair/<task_id>.json.

For each item:
- Re-derive the rates from the recorded statuses and the claimed flips.
- Open each flipped slot's model.patch and the graded test results (ctrf or junit from the regrade outputs and from the original probe attempt). Confirm the outcome change is caused by restoring or deleting the model's test-file edits, not by flakiness or a staging mistake.
- Check that cheated passes stay excluded.
- Re-run a regrade yourself (new and old grader, with S/synth/tools/grading_repair.py) for at least the single most decision-critical slot of each item.
Then give the final recommendation per task:
- keep or withdraw for accepted tasks;
- consider_accept for rejected tasks that now meet both rate bars (they still need the similarity judge);
- no_change otherwise.
Also say whether any reported problem (failed validations, flaky slots, missing installs) undermines the repair.`
}

phase('Prepare')
const plan = await agent(PREPARE_PROMPT, { label: 'prepare', phase: 'Prepare', schema: PLAN_SCHEMA })
if (!plan || !plan.ok) {
  log(`prepare failed: ${plan ? plan.summary : 'agent ended without a report'}`)
  return { plan, runs: [], verify: null }
}
log(`plan: ${plan.per_run.map(r => `${r.run} ${r.slots_to_regrade}`).join(', ')}`)

const planned = RUNS.filter(run => plan.per_run.some(r => r.run === run))
const skipped = RUNS.filter(run => !planned.includes(run))
if (skipped.length) log(`not planned (skipped): ${skipped.join(', ')}`)
const runs = await pipeline(
  planned,
  run => agent(runPrompt(run, plan), { label: `repair:${run}`, phase: 'Regrade', schema: RUN_SCHEMA }),
)
const done = runs.filter(Boolean)
const missing = planned.filter((run, i) => !runs[i])
if (missing.length) log(`no result for: ${missing.join(', ')}`)

phase('Verify')
const verify = await agent(verifyPrompt(done), { label: 'verify', phase: 'Verify', schema: VERIFY_SCHEMA })
return { plan, runs: done, missing, skipped, verify }
