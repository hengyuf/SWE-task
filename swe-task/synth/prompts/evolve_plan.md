# Evolve plan for a probed synthetic task in {{repo}}

A synthetic task built for the DeepSWE target {{target}} was probed with the student ({{student}}) and the teacher
({{teacher}}). {{outcome}}

## Inputs (./inputs)
- `task/`: the probed task (instruction, rubric, hidden tests, reference solution).
- `probe/index.md` and `probe/<slot>.md`: the probe rollouts. `metrics.json`: pass rates, cheat audit, similarity.
- `synthetic_failure_analysis.md` (if present): how the student failed the probed task.
- `target_failure_analysis.md`: how the student fails the target. Focus pattern for this branch: "{{focus_pattern}}".
- `understanding.md`, `mask_plan.md`: repository map and the original candidate plans.

{{sandbox}}
The sandbox holds the ORIGINAL repository state A (offline, environment ready).

## Write `evolve_plan.md`
A High-IID evolution: a NEW feature, plus its verifier tests, to author on top of A (not on top of the probed task),
designed so that (i) the student's failure on it matches the focus pattern of the target, and (ii) the student and
teacher pass rates move toward the target's. Cover: what went wrong with the probed task (too easy, too hard,
different failure, leakage) with evidence; the new feature and its observable behaviour; the files involved; the
hidden tests (fail-to-pass and pass-to-pass); why the teacher can still solve it legitimately; why memorised
upstream code cannot answer it. No hint or stub markers in the starting state.

Then write `result.json`: `{"status": "ok", "title": "<short task title>", "summary": "<one sentence>"}`.
