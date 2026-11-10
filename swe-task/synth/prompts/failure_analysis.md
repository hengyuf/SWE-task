# Failure analysis: task {{task_id}}

You are analysing why a student coding agent ({{student}}) fails this SWE task while a teacher agent ({{teacher}})
solves it. Both ran the same OpenCode harness on the same starting repository (base commit {{base_commit}}) with
the instruction below; only committed work was graded by hidden tests.

## Inputs (read-only, in ./inputs)
- `instruction.md`: the task statement both models received.
- `tests/test.patch`, `tests/test.sh`, `tests/config.json`: the hidden tests and the fail-to-pass (F2P) /
  pass-to-pass (P2P) test ids used for grading. The models never saw these.
- `solution.patch`: the reference solution.
- `rollouts/index.md`: one row per rollout slot (status, reward, F2P/P2P passed counts, tool calls, patch size).
  `rollouts/<slot>.md`: the condensed transcript of that slot (tool outputs are truncated) ending with the final
  committed patch (truncated there; `rollouts/<slot>.patch` holds the full patch). Slots with status `infra` or `grader` are already excluded (infrastructure failure or grader
  error); `pass`/`fail` slots were graded normally (an agent timeout is a genuine failure).
- `stats.json`: pass counts per model.{{rollout_selection}}

{{sandbox}}
The sandbox holds the task's starting repository and its full offline toolchain. Use it to confirm hypotheses:
for example upload a rollout's full patch (`rollouts/<slot>.patch`), `git apply` it, apply
`tests/test.patch`, and run the relevant tests to see exactly what fails and why. Reset with
`git checkout -- . && git clean -fdq` (keeping /tmp/work) between experiments.

## What to do
1. Read the instruction, hidden tests and reference solution until you know precisely what "correct" requires.
2. Read every student transcript, then the teacher transcripts. Reproduce in the sandbox whenever a transcript
   alone does not show the cause.
3. Besides the slots already excluded, exclude any slot whose outcome was decided by infrastructure (sandbox, API
   or tool crash, interrupted run) or by a grader error (verifier crash or timeout, broken report parsing), and
   give the evidence. Do not exclude a slot just because the model behaved badly.
4. Identify the student's failure patterns: recurring behavioural root causes (for example a misread or skipped
   requirement, a missing edge case, an incomplete feature surface across modules, breaking existing behaviour,
   never committing, looping until the time limit, editing the wrong layer or API). For each pattern give its
   evidence (slot ids, short quotes or commands, which F2P/P2P tests fail and why) and its frequency (k of n
   graded student slots). Contrast with what the teacher did differently. Note any teacher pass that relies on
   cheating:
{{cheat_categories}}
5. Write a repository-agnostic transferable signature: the skill gap stated so that a different task in a different
   repository of the same language could trigger the same failure.

## Outputs (in this directory)
- `failure_analysis.md` with sections: Task summary; Outcome table; Failure patterns (one subsection each:
  description, evidence, frequency, failing tests, root cause); Teacher contrast; Excluded rollouts; Transferable
  signature.
- `result.json`:
  `{"patterns": [{"name": str, "category": str, "description": str, "student_slots": [slot ids], "teacher_contrast": str}],
    "transferable_signature": str, "excluded": [{"slot": str, "reason": str}]}`
Cite only slot ids from `rollouts/index.md` (graded `pass`/`fail` slots in `student_slots`). Ground every claim in
a transcript or a command you ran.
