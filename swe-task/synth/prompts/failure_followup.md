# Failure follow-up: task {{task_id}}

An earlier analysis studied why a student coding agent ({{student}}) fails this SWE task while a teacher agent
({{teacher}}) solves it, and named the student's failure patterns. A newer student checkpoint, {{new}}, has now run
the same task 5 times on the same starting repository (base commit {{base_commit}}), with the same instruction, the
same OpenCode harness and tools, a 3 h agent time limit, and the same hidden-test grading. Only committed work was
graded (`git diff <base> HEAD`), so an attempt that never committed scores as an empty patch.

Your job: decide, pattern by pattern, whether {{new}} still shows each earlier failure pattern, and find the failure
patterns {{new}} shows that the earlier analysis does not describe.

## Inputs (read-only, in ./inputs)
- `instruction.md`: the task statement every model received.
- `tests/test.patch`, `tests/test.sh`, `tests/config.json`: the hidden tests and the fail-to-pass (F2P) /
  pass-to-pass (P2P) test ids used for grading. The models never saw these.
- `solution.patch`: the reference solution.
- `previous/failure_analysis.md` and `previous/result.json`: the earlier analysis. Its patterns cite
  {{student}}/{{teacher}} slot ids, and `previous/evidence/` holds the scripts and results it produced.
- `rollouts/index.md`: one row per rollout slot (status, reward, F2P/P2P passed counts, tool calls, patch size).
  - It includes the {{student}}-r* and {{teacher}}-r* slots the earlier analysis studied, and the new {{new}}-r* slots.
  - `rollouts/<slot>.md` is the condensed transcript of that slot: tool outputs and thinking are truncated. It ends
    with the final committed patch, also truncated; `rollouts/<slot>.patch` holds the full patch.
  - Slots with status `infra` or `grader` are already excluded (infrastructure failure or grader error).
  - `pass`/`fail` slots were graded normally; an agent timeout is a genuine failure.
- `stats.json`: pass counts per model.

{{sandbox}}
The sandbox holds the task's starting repository and its full offline toolchain. Use it to confirm hypotheses. For
example:
1. Upload a slot's full patch (`rollouts/<slot>.patch`) and `git apply` it.
2. Apply `tests/test.patch` and run the relevant tests to see exactly what fails and why.
3. Reset with `git checkout -- . && git clean -fdq` between experiments, keeping /tmp/work.

## What to do
1. Read the instruction, hidden tests, reference solution and the earlier analysis until you know exactly what
   "correct" requires and what each earlier pattern means.
2. Read every {{new}} transcript in full. Compare each one with the earlier {{student}} slots and the {{teacher}}
   slots. Reproduce in the sandbox whenever a transcript alone does not show why a {{new}} slot failed or which tests
   fail.
3. Besides the slots already excluded, exclude any {{new}} slot whose outcome was decided by infrastructure (sandbox,
   API or tool crash, interrupted run) or by a grader error, and give the evidence. Do not exclude a slot just because
   the model behaved badly.
4. For EVERY pattern in `previous/result.json` (use its exact `name`), decide its status in the graded {{new}} slots:
   - `resolved`: no graded {{new}} slot shows it, and the slots do reach the point where it would show, for example
     they implement the part of the feature where the earlier student went wrong.
   - `reduced`: it still shows, but in clearly fewer slots or in a clearly milder form than in the earlier student
     slots.
   - `persists`: it shows at about the same frequency and severity.
   - `not_observable`: the {{new}} slots never reach the point where it could show, for example every one ends early
     for another reason. Say what stopped them.

   For each pattern give the {{new}} slots that show it, "k/n" (k slots showing it, n graded {{new}} slots), and
   evidence. Evidence means short quotes, commands, or the F2P/P2P tests that fail and why, for both the slots that
   still show it and those that avoid it.
5. Identify NEW failure patterns: recurring or decisive root causes in the {{new}} slots that the earlier analysis does
   not describe. Include agent-loop failures as well as coding mistakes, for example:
   - hitting the 3 h time limit while still iterating;
   - never committing the work;
   - ending a session on a turn without a tool call;
   - tool-call markup written inside the reasoning;
   - repetitive output.

   For each new pattern give its description, evidence, frequency (k/n), the failing tests, and whether the earlier
   {{student}} slots or the {{teacher}} slots also show it. A pattern seen in a single slot counts if it decided that
   slot's outcome.

   Mark each new pattern as decisive or latent:
   - `decisive: true` if it caused a hidden-test (F2P/P2P) failure, or the empty or uncommitted patch, of at least one
     graded {{new}} slot.
   - `decisive: false` for a latent defect that only your own extra probes reveal and that changed no graded outcome.
6. For each {{new}} slot, give its outcome (pass, fail, or excluded) and its primary cause. The cause is an earlier
   pattern name, a new pattern name, or "solved". Note any {{new}} pass that relies on cheating:
{{cheat_categories}}
7. Write an overall verdict: what {{new}} fixed, what it still gets wrong, and what is new.

## Outputs (in this directory)
- `followup.md`, with these sections:
  - Summary.
  - Outcome table: every slot of every model.
  - Earlier patterns: one subsection each, with status, frequency, evidence and the contrast with the earlier
    student.
  - New patterns: one subsection each, with description, evidence, frequency, failing tests and root cause.
  - Per-slot causes for {{new}}.
  - Excluded rollouts.
- `result.json`:
  `{"previous_patterns": [{"name": str, "status": "resolved"|"reduced"|"persists"|"not_observable", "new_slots": [slot ids], "frequency": "k/n", "evidence": str}],
    "new_patterns": [{"name": str, "category": str, "description": str, "new_slots": [slot ids], "frequency": "k/n", "decisive": bool, "failing_tests": str, "seen_in_earlier_student": bool, "seen_in_teacher": bool, "evidence": str}],
    "slot_outcomes": [{"slot": str, "status": "pass"|"fail"|"excluded", "primary_cause": str}],
    "excluded": [{"slot": str, "reason": str}],
    "summary": str}`

Rules for result.json:
- `previous_patterns` must list every earlier pattern exactly once, by its exact name.
- `new_slots` may cite only graded {{new}} slots (status pass/fail in `rollouts/index.md`, not excluded by you).
- `slot_outcomes` must cover every {{new}}-r* slot once.
- `excluded` lists the {{new}} slots you exclude. You may add an earlier slot only to record that you consider it
  excluded too.

Ground every claim in a transcript or a command you ran.
