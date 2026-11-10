# Final task statement and process rubric

## Inputs (./inputs)
- `instruction_draft.md`: the author's draft statement.
- `tests/test.patch`, `tests/config.json`: the hidden tests and their fail-to-pass (`f2p_node_ids`) and
  pass-to-pass (`p2p_node_ids`) ids. Every fail-to-pass test must be passable from the statement alone.
- `examples/*.md`: DeepSWE statements in the same language. Match their style: a goal paragraph, an
  "Expected outcomes" section with numbered parts naming the public functions, types, flags, outputs and error
  messages that callers rely on, then short "Implementation notes".
- `target_failure_analysis.md`: the student failure patterns this task is meant to trigger.

## Write `instruction.md`
- Specify every behaviour the hidden tests check, at the level of public names and observable behaviour.
- Never mention tests, patches, graders, hidden checks, or that code was removed; no implementation file paths, no
  code copied from the solution, no TODO; do not give away the algorithm beyond what the behaviour requires.
- 500 to 5000 bytes. The final line must be exactly:
{{important}}

## Write `rubric.md`
A rubric for scoring the solving PROCESS (not the outcome): 6 to 10 weighted criteria grounded in the target
failure patterns (for example: explored the right modules, handled edge case X, ran the existing tests, preserved
behaviour Y, committed the work), plus a "Disqualifiers" list built from these anti-cheating categories:
{{cheat_categories}}

Then write `result.json`: `{"status": "ok", "summary": str}`.
