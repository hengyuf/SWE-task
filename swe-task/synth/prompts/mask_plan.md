# Mask plan: {{n}} candidate tasks in {{repo}}

Goal: design {{n}} synthetic SWE tasks in {{repo}} on which the student ({{student}}) fails in the SAME way it fails
the DeepSWE task {{target}}, while the teacher ({{teacher}}) can still solve them legitimately.

## Inputs (./inputs)
- `understanding.md`: map of {{repo}}.
- `target/failure_analysis.md`, `target/failure_analysis.json`: the student's failure patterns on the target and the
  transferable signature.
- `target/instruction.md`, `target/stats.json`: the target's statement style and the size of its reference solution
  and hidden tests (aim for comparable scope).

{{sandbox}}
The sandbox holds {{repo}} with its environment ready (offline). Read code and run tests there to check ideas.

## Construction (Mid-IID: mask an existing feature)
Each candidate removes the implementation of one coherent, existing feature; the task then asks for that feature.
- The feature's behaviour is covered by existing tests; they become the hidden fail-to-pass tests. Other tests of
  the same packages must keep passing after the removal (pass-to-pass).
- Removal must keep the code compiling or importing: remove behaviour behind entry points that survive (keep
  signatures that other code or the hidden tests reference), so hidden tests fail by behaviour, not by build errors.
- Scope comparable to the target: several functions or files, not a one-liner.
- Leave no hints: no TODO/TBD/FIXME, no "not implemented" stubs; replacement bodies must look like plausible simpler
  behaviour, never placeholders.
- Avoid features whose implementation is easy to recover: reproduced verbatim in docs, examples or changelogs in
  the repository, or likely memorised from upstream.
- For each candidate, name the target failure pattern(s) it should trigger and argue why the student repeats them
  and why the teacher can still succeed.

## Outputs (in this directory)
- `mask_plan.md`: rationale plus one section per candidate.
- `candidates.json`: `{"candidates": [{"id": "c01", "title": str, "summary": str, "files": [paths relative to /app
  that the solution touches], "tests": [existing tests that become hidden], "target_patterns": [pattern names],
  "why_student_fails": str, "why_teacher_solvable": str}]}`, best first, ids c01, c02, ...
- `result.json`: `{"status": "ok", "summary": str}`.
