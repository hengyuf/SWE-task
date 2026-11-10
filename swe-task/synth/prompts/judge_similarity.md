Compare two failure analyses of the SAME student coding model: one on a real target task, one on a synthetic task
built in a different repository to reproduce the same weakness.

Target task analysis:
<<<
{{target_md}}
>>>
Target structured summary: {{target_json}}

Synthetic task analysis:
<<<
{{synthetic_md}}
>>>
Synthetic structured summary: {{synthetic_json}}

Score from 0 to 10 how well the synthetic task reproduces the student's TARGET failure patterns.

What counts as a match: a failing synthetic rollout matches a target pattern only when the student makes the same kind
of technical mistake by the same mechanism as in the target (for example the same lifecycle or ordering error, the
same misreading of scope, the same refusal of a required operation followed by skipped tests), so that a reader would
recognise the target mistake transplanted into another repository. Repository details and surface symptoms may
differ; the mechanism may not.

What does not count:
- shared general habits on their own: rationalising a requirement away, knowingly reinterpreting the specification,
  writing tests that encode one's own implementation, overconfident claims. They count only when the target pattern
  itself is that habit AND it causes the synthetic failure in the same way;
- failures from grading artifacts, empty or missing submissions (e.g. the run ended without a commit, or with a tool
  call written inside its reasoning), hangs, timeouts and infrastructure errors;
- failures on requirements the synthetic task's instruction does not state, and on traps with no counterpart in the
  target.
A target trap that the synthetic task rebuilds and the student passes is evidence against a match.

Scale (count the failing student rollouts whose failure is caused by a matched target pattern):
10 = the target's main root causes cause most failures, with similar prevalence;
8 = a main target pattern causes most failing rollouts;
6 = a main target pattern, by the same mechanism, causes at least two failing rollouts;
3-5 = only minor target patterns recur, or only loose analogues (same habit, different mechanism);
0-2 = unrelated, or only the excluded causes above.
Reply with JSON only:
{"score": <integer 0-10>, "matched_patterns": [{"target": str, "synthetic": str}], "missing_patterns": [str],
 "rationale": "<two or three sentences naming the rollouts behind each match>"}
