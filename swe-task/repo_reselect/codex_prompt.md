# Task: choose a better host repository for each DeepSWE target (selection only)

## Goal
We synthesize SWE tasks that make our student model (Nemotron, alias `sc14`) fail the same way it fails on a real
DeepSWE task, while an open teacher model (Kimi-K3, alias `kimi`) can still solve them.

For each of the **23 DeepSWE tasks** in the manifest, the pipeline has so far used exactly one "host" repository.
Choose, from that task's **candidate pool** (normally ranks 2–5 of its similar-repository list), the one repository
whose synthetic tasks are most likely to be **accepted**: the student fails by the **same mechanism** as on the real
task, and the teacher still passes.

This is a judgment and selection job: do not build tasks, run sandboxes, builds or test suites, or change anything
outside the output directory.

## Where everything is
`S` = `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`. This is the same tree as
`/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task`. In the manifest, `S/...` paths are
relative to `S`.

**Manifest: `S/repo_reselect/manifest.json`.** It has one entry per DeepSWE task in `tasks[]`, and every path in it
exists. Each entry has:

- `deepswe_task`, `run`: the task id and its pipeline run (`S/synth/runs/<run>/`). `upstream_repo` is the real
  project of the DeepSWE task. `grade_format` is the report format the grader reads.
- `target_pass_counts`: how many of the 5 student and 5 teacher rollouts passed on the real task.
- `artifacts`:
  - **`deepswe_task_dir`**: the real task. It has `instruction.md`, `solution/solution.patch`, the hidden
    `tests/test.patch`, `tests/config.json` (fail-to-pass and pass-to-pass test ids, grade format),
    `environment/Dockerfile` and `task.toml`.
    - `task.toml`'s `language` is wrong for two tasks: httpx-deterministic-cookie-store is Python (pytest), and
      prometheus-transactional-reload-status is Go. Take the language from the Dockerfile and `grade_format`.
  - **`target_rollouts_rendered`**: the 5 student and 5 teacher rollouts on the real task.
    - `index.md` gives status, reward, F2P/P2P counts, tool calls and agent error per slot. Take slot names from it:
      `sc14-r0..4`, plus five `kimi-r<n>` whose numbers are not contiguous.
    - Each `<slot>.md` starts with a header listing the failing test ids, followed by events headed
      `### [N] thinking|call <tool>|output`. The failure analysis cites events as `[N]`; find one with
      `grep -n '^### \[N\] '`. Long outputs are cut as `…[N chars omitted]…`.
    - Student transcripts are about 200–315 KB. **Never read one whole**: read about 60 lines around each cited
      event. `<slot>.patch` is the final patch.
  - **`target_rollouts_raw_records`**: the same rollouts as JSONL, one line of about 0.5 MB per rollout. Its
    `output` list holds the untruncated events. Extract a single slot with Python; never `cat` the file.
  - **`failure_analysis_md`, `failure_analysis_json`**: why the student failed and the teacher passed. `result.json`
    has `patterns[]` (name, category, description, `student_slots`, `teacher_contrast`), `transferable_signature`
    and `excluded`. The manifest copies the patterns into `target_failure_patterns` and adds two fields:
    - `n_student_slots`: how many failing student slots the pattern explains;
    - `judge_excluded`: true for delivery patterns (no commit, empty patch, timeout), which the acceptance judge never
      counts.
  - **`current_host_build_dir`**: everything built on the current host. It is huge: `round-*/probe-*/` holds raw
    probe logs, tens of GB with single files of several GB. **Never run recursive grep, find or du over it**; open
    only these files:
    - `env/result.json` and `env/setup.sh`: the image's toolchain version, how the host was installed, and the test
      command;
    - `understanding/understanding.md` and `plan/mask_plan.md`: how the pipeline read the repo;
    - `round-*/cand-*/metrics.json`, for each probed synthetic task:
      - `student_rate`, `teacher_clean_rate`;
      - `reason`: the gate that rejected it;
      - `title`, `lineage`, and `task_dir`: the synthetic task itself, with `instruction.md`, `solution/` and
        `tests/`;
      - `similarity` (0–10) and `similarity_detail`, which exist only if the task passed the pass-rate gate;
    - `round-*/cand-*/rollouts/`: same format as above, slots `sc14-r0..4` and `kimi-r0..4`;
    - `round-*/cand-*/teacher_audit.json`: teacher cheating verdicts;
    - `round-*/cand-*/failure_analysis/`: only for tasks that passed the pass-rate gate.
  - **`accepted_tasks_jsonl`**: the accepted synthetic tasks (null if none).
  - **`audit_dir`**: one `<task_id>.json` per audited synthetic task, with these fields:
    - `verdict` (strong, partial, weak or none) and `summary`;
    - `patterns[]`, each with `reproduced` (yes, partial or no) and `evidence`;
    - `student_rollouts[]`, each slot's status and cause;
    - `other_causes`, `similarity_judge`.

    `reproduced: no` has three meanings, so read the evidence before concluding anything about the host:
    1. the student passed every slot, so there was nothing to match;
    2. the task had no surface for the pattern (the evidence starts "Not applicable");
    3. the task rebuilt the trap and the student passed it.

    The audit is also looser than the acceptance judge: it marks delivery patterns as reproduced.
  - **`run_watch_log`**: the run's operational log.
- **`current_host`**: the host used so far, with its top100 entry and its results:
  - `candidates_probed`, `similarity_judged`, `accepted`, `best_similarity`;
  - `gate_outcomes`: why each probed task was rejected: `student_too_strong`, `teacher_too_weak`,
    `similarity_below_6`, `inconclusive` or `accepted`;
  - `audit_verdicts_all_audited_tasks`;
  - `pattern_reproduction_all_audited_tasks`: per pattern, counts of yes, partial and no over audited tasks where
    the student failed at least once, plus `student_passed_all`.
- **`abandoned_hosts`**: earlier hosts dropped because their environment leaked the answer, with the reason.
- **`candidate_pool`**: the **only allowed choices**. These are the next 4 entries of `S/top100.jsonl` in rank order
  after the current host, skipping the current host, abandoned hosts, and entries flagged `builds_on_target` or
  `same_owner_as_target`.
  - Each candidate carries its rank, repo, score, sub-scores (fit, arch, dom, test, stack), why, tree_path,
    language, license, flags, and `already_host_for`.
  - The flag `needs_ext` means the tests may need a service such as a database or a browser.
  - top100's score comes from retrieval plus LLM scoring and knows nothing about failure mechanisms, so do not trust
    it for that.
- **`output`**: where this task's results go.
- **Top level `all_current_hosts`**: the host of every DeepSWE task.

Background, if you need it:
- `S/README.md`;
- `S/synth/prompts/{build_env,understanding,mask_plan,author,evolve_plan,judge_similarity}.md`;
- `/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/swe-task/DeepSWE_synth_pipeline_explained.html`.

## How hosts are used and how tasks are judged
1. **Clone.** The host's default branch is cloned at its latest commit (full clone, with submodules) into `/app` of
   the DeepSWE task's own Docker image. Only `/app` is replaced; everything else the image installed stays. All 23
   images build on one base with Python, Go, Node (npm, pnpm) and Rust.
2. **Build.** One networked build step installs what the host needs. It may install another toolchain version, and
   local services or a headless browser that run inside the container. The chosen test command then runs twice
   **offline**, and at least 20 tests must pass both times. Every later step is offline.
3. **Starting state.** The student starts from a fresh single-commit git repo with no history or tags. Git-ignored
   build outputs (dist, build, out, lib, bin, coverage…), `/app/target`, the Go build cache and ignored executables
   are deleted.
4. **Task design.** Round 0 masks existing features. Later rounds add new features, seeded only from valid round-0
   tasks, and each branch aims at one target pattern.
5. **Leak scan.** A task fails at once, with no retry, when its solution lines or a **released copy of the code it
   touches** is readable anywhere outside the task: `node_modules`, `site-packages`, `vendor` or `_vendor`,
   archives, package caches. A host whose round-0 masks all hit such copies yields nothing. Two hosts were abandoned
   this way:
   - psf/requests: the image's own pip had `pip/_vendor/requests`;
   - avajs/ava: a released ava was in `node_modules`.
6. **Probe.** Each task gets 5 student and 5 teacher rollouts. The verifier has 2 CPUs, 8 GB RAM, 20 GB disk and
   1800 s, for a **cold** rebuild plus the tests. It reads per-test results in the task's grade format: pytest JUnit
   for the Python targets, CTRF otherwise.
7. **Acceptance.** A task is accepted only if all of these hold:
   - the student passes at most 1/5, i.e. fails at least 4 of 5;
   - the teacher passes at least 3/5 cleanly. The teacher audit disqualifies passes that recall upstream code from
     memory ("Memory narrated as lookup").
   - each model has at least 3 valid rollouts;
   - the similarity judge (`S/synth/prompts/judge_similarity.md`) gives at least 6/10. A 6 needs a **main** target
     pattern to cause at least two failing student rollouts **by the same mechanism**. Minor patterns alone score
     3–5.

   The judge excludes empty or missing submissions, timeouts, hangs and infrastructure errors, and failures on
   requirements the task does not state. A rebuilt trap that the student passes counts **against** a match.

**Hard limits for a host.** A candidate that misses one of these cannot work:
- language is Python, Go, JavaScript, TypeScript or Rust;
- at least 20 tests pass offline;
- per-test results come in the grade format with stable test ids;
- a cold rebuild plus the relevant tests fit in 1800 s on 2 CPUs and 8 GB;
- at test time nothing outside the container is needed: no network, Docker daemon, KVM or cloud service;
- nothing needs git history.

## How to run this: one subagent per DeepSWE task (required)
You are the dispatcher. **You are explicitly authorized and required to spawn subagents.** Do not analyze any task
yourself, and never pass one task's findings or choices to another task's subagent.

1. **Preflight**, before spawning anything:
   - run `git ls-remote https://github.com/antchfx/xpath HEAD`;
   - create and remove `${TMPDIR:-/tmp}/$USER-repo-reselect/.probe`;
   - create and remove `S/repo_reselect/out/.preflight`.

   If any of these fails, stop and report it.
2. **Resume.** Skip a task when `S/repo_reselect/out/<task>.json` and `.md` both exist and the JSON is valid
   against the schema below; log it as `done (existing)`. Always **append** to `S/repo_reselect/out/dispatch.log`
   (task id, UTC time, status); never truncate it. Track progress through that log and the files on disk, not
   through `list_agents`.
3. **Spawn one subagent per remaining task.**
   - If your `spawn_agent` takes `task_name` and `fork_turns`:
     - `task_name` = the task id with every `-` replaced by `_`. Only lowercase letters, digits and `_` are allowed.
     - `fork_turns: "none"`. The default, `all`, would copy your whole history, including other tasks' results.
     - `message` = the TASK BRIEF below with `<DEEPSWE_TASK>` replaced by the task id.
     - Do not pass `model` or `reasoning_effort`.
   - If it instead takes `fork_context`, pass `fork_context: false` with the same message. Call `close_agent` on each
     subagent after you have checked its files, because those subagents keep their slot until closed.
4. **Keep the slots full.** Spawn until `spawn_agent` reports that the agent thread limit is reached. Then call
   `wait_agent` with `timeout_ms: 3600000`. A subagent's result arrives as a message; when one does:
   - check that task's two files;
   - log it;
   - spawn the next task.

   Do not poll with `list_agents` or `sleep`.
5. **Retry once.** If a subagent errors or its files are missing or invalid, spawn one fresh subagent for that task
   with `task_name` `<name>_retry`, since names cannot be reused. If that also fails, log the task as failed.
6. **Never end your turn early.** This is a headless run: when you send a final answer, the run ends and every
   running subagent is killed. Do not finish until all 23 tasks are logged as done or failed and `SUMMARY.md` is
   written.
7. **Write `S/repo_reselect/out/SUMMARY.md`.** It needs one table row per task: DeepSWE task | current host |
   choice (rank) or `none` | confidence | main patterns the choice can host that the current host could not express
   | main risk. Below the table, list the tasks with `none` or `low` confidence, and any failed task.

### TASK BRIEF (send to each subagent; replace `<DEEPSWE_TASK>`)
> You choose the best host repository for the DeepSWE task `<DEEPSWE_TASK>`. First read these sections of
> `S/repo_reselect/codex_prompt.md` (`S` = `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`): "Where
> everything is", "How hosts are used and how tasks are judged", "Output JSON schema (per task)" and "Rules". The "How
> to run this" section is for the dispatcher, not for you. Then take your task's entry from
> `S/repo_reselect/manifest.json`. Work only on your task, and work alone: do not call `spawn_agent`,
> `followup_task` or `send_message`. Every extra agent takes one of the run's few shared slots and stalls the other
> tasks.
>
> 1. **Understand the failure.** Read the real task: instruction, solution patch and tests. Read the failure analysis,
>    then the transcripts: start with each pattern's `student_slots`, and read around the events the analysis
>    cites.
>
>    Write a mechanism card for each target pattern:
>    - **Trigger:** what in the repository or task triggers it. Examples: "a generated parser whose generator is not
>      installed offline"; "a CLI whose output shares a terminal line with a host tool's progress output"; "an API
>      route named in the instruction that differs from the obvious REST shape".
>    - **Student:** what the student does wrong.
>    - **Teacher:** what the teacher does instead.
>    - **Main:** whether it is a main pattern, i.e. behind most of the failing student slots (`n_student_slots`).
> 2. **Learn from the current host.** Start with `gate_outcomes`, which tells you the bottleneck:
>    - if most tasks were rejected as `student_too_strong`, difficulty is the problem, and a new host must allow
>      tasks with more headroom, not just the same triggers;
>    - if most were `teacher_too_weak`, solvability is the problem;
>    - if most were `similarity_below_6`, the mechanism does not match.
>
>    Then read `pattern_reproduction_all_audited_tasks` and a few audit files: the highest verdicts present, plus
>    `none` audits where the student failed for another reason or passed a rebuilt trap. Skip audits whose summary
>    says the student passed everything.
>
>    For each pattern, settle the current host's outcome: `reproduced`, `partial`, `trap_built_student_passed`,
>    `not_expressible` or `not_attempted`. Read `abandoned_hosts` if present.
> 3. **Inspect each of the 4 candidates** with git only:
>    - get the commit with `git ls-remote https://github.com/<owner>/<repo> HEAD`;
>    - clone with `git clone --depth 1 --filter=blob:none [--sparse] https://github.com/<owner>/<repo> <dir>` into
>      `${TMPDIR:-/tmp}/$USER-repo-reselect/<DEEPSWE_TASK>/`. For big repos, add `git -C <dir> sparse-checkout add
>      <paths>`.
>    - Fetching single files from raw.githubusercontent.com is fine. Do not use api.github.com or `gh`: those calls
>      share an unauthenticated quota of 60 per hour that is already used up.
>
>    Look at the package layout, the modules that could host each mechanism, and the test suite: framework, how it
>    runs, whether it needs services or network, and the reporter. Also check the build, the dependency manifests,
>    `.gitmodules`, generated code and vendored code.
> 4. **Score each candidate from 0 to 10** on these criteria, with one sentence each, and cite your evidence: paths in
>    the clone, URLs, or transcript events.
>    - **mechanism_fit** (the most important). Weigh the patterns the way the judge does:
>      - main patterns first;
>      - `judge_excluded` patterns get zero weight;
>      - minor patterns alone cannot reach 6;
>      - "the current host never reproduced it" is only a tie-breaker.
>
>      For each pattern, say yes, partial or no. For yes and partial, name the code area and give a one-line task
>      idea: mask an existing feature, or add a new one. Where the current host already built the trap and the student
>      passed it, credit a candidate only for what it adds that made the student fail on the real task: a broader API
>      surface, a denser specification, or interplay across modules. Note whether several patterns can occur
>      together in one broad task, as they do on the real task. Surface similarity of domain or API is not enough.
>    - **teacher_solvability.** A strong agent can solve such a task in a few hours from the repo alone, but the task
>      is not easy enough for the student to pass. Recall risk: in very popular repos, the teacher may recite masked
>      code from memory. Those passes are disqualified, and new-feature tasks reduce the risk.
>    - **buildability.** Check the hard limits above. The toolchain version does not have to match the image,
>      because the build step can install another one: compare with the current host's `env/result.json` and
>      `env/setup.sh`. Check `needs_ext` against the actual tests. A local database or headless browser that the
>      build installs and the test command starts offline is fine; anything outside the container is not. Watch for
>      compiled extensions or build outputs that the starting state deletes and a cold verifier must rebuild within
>      1800 s.
>    - **leak_safety.** Look for released copies of the candidate's code in three places:
>      - **(a) The DeepSWE image itself.** Look at the real task's `environment/Dockerfile` and its dependency
>        manifests at the base commit (`go.mod`, lockfiles, `pyproject.toml`, `requirements*`). If the DeepSWE
>        project depends on the candidate, released copies of the candidate are already in the image. Examples:
>        prometheus requires `go.opentelemetry.io/collector/*`; vitest locks webdriverio; narwhals installs pandas.
>      - **(b) The candidate's own dependency tree.** Look for a released copy of itself or of its sub-packages
>        (e.g., prettier devDepends on prettier; jest-worker sits inside metro), nested Go modules, and vendored
>        copies.
>      - **(c) What the pipeline cleans.** It purges the host's own releases from yarn, npm, bun, pip, uv and cargo
>        caches and its root Go module. It does not purge installed copies, the pnpm store, global node_modules or
>        nested Go modules.
>
>      A partial copy rules out only the features it contains: jest had some and still yielded 4 accepted tasks.
>      Separately, report any dependency on the DeepSWE task's own project (`upstream_repo`) as a contamination risk,
>      quoting the line.
>    - **diversity.** All else equal, prefer a repository that is not already a host (`already_host_for`).
>
>    Then give each candidate an overall score.
> 5. **Decide.** Choose one repository from `candidate_pool`, or none: if no candidate is acceptable, write
>    `"choice": {"repo": null, "rank": null, ...}` and say why. You may name a better repository from the rest of
>    this task's `S/top100.jsonl` entries in `suggestion_outside_pool`, as a suggestion only. Give your confidence,
>    the pattern-by-pattern plan, the main risks, and what the task build should watch for.
> 6. **Write two files, then clean up.**
>    - `S/repo_reselect/out/<DEEPSWE_TASK>.json`, matching the "Output JSON schema (per task)" exactly: same keys and
>      enum values.
>    - `S/repo_reselect/out/<DEEPSWE_TASK>.md`: the mechanism cards, a short comparison of the candidates, and the
>      decision.
>    - Delete your scratch clones.
> 7. **Final answer:** exactly one line, and nothing else: `DONE <DEEPSWE_TASK> choice=<owner/repo|none> rank=<n|-> confidence=<high|medium|low>`,
>    or `FAILED <DEEPSWE_TASK>: <reason>`.

### Output JSON schema (per task)
```json
{
  "deepswe_task": "string",
  "current_host": "owner/repo",
  "current_host_bottleneck": "student_too_strong|teacher_too_weak|similarity_below_6|mixed",
  "choice": {"repo": "owner/repo or null", "rank": "integer or null", "confidence": "high|medium|low"},
  "mechanism_cards": [{
    "pattern": "name from result.json", "main": true, "judge_excluded": false,
    "trigger": "string", "student_behavior": "string", "teacher_behavior": "string",
    "current_host_outcome": "reproduced|partial|trap_built_student_passed|not_expressible|not_attempted"
  }],
  "candidates": [{
    "repo": "owner/repo", "rank": 0, "inspected_commit": "sha",
    "scores": {"mechanism_fit": 0, "teacher_solvability": 0, "buildability": 0, "leak_safety": 0, "diversity": 0},
    "overall": 0.0,
    "hard_limit_failures": ["string"],
    "patterns": [{"pattern": "string", "can_host": "yes|partial|no", "where": "code area", "task_idea": "one line"}],
    "leak_findings": ["string"], "contamination_risk": "string or null",
    "risks": ["string"], "evidence": ["path, URL or transcript reference"]
  }],
  "rationale": "at most 200 words",
  "suggestion_outside_pool": "owner/repo or null",
  "watch_items_for_build": ["string"]
}
```

## Rules (dispatcher and subagents)
- **Read-only** on everything under `S` and on Baihe's tree. Write only under `S/repo_reselect/out/`: subagents
  write their two files, and the dispatcher writes `SUMMARY.md` and `dispatch.log`. Clones go to node-local scratch,
  never under `S`, because the shared filesystem is close to its file-count quota.
- **No secrets.** Never open `.env` files, `S/secrets/`, `S/atp/.env`, codex or other auth files, and never print
  environment variables.
- **Nothing heavy or shared.**
  - Do not run Slurm commands, containers, sandboxes, GPU jobs, dependency installs, or the repositories' test
    suites. Reading code, `git`, `grep`/`rg` and small listings are enough.
  - Never run recursive grep, find or du over `S/synth/runs`.
- **Command policy.** This cluster's codex policy rejects any command, or pipe segment, that starts with `timeout`,
  `env`, `xargs`, `nice`, `stdbuf`, `setsid`, `command`, `ssh`, `ip`, `dd` or `at`. Use `git -c ...`,
  `rg`/`grep -r` or `find -exec` instead.
- **Network.** Only git over HTTPS to github.com (`ls-remote`, `clone`) and raw.githubusercontent.com: no
  api.github.com and no `gh`. Do not look up the DeepSWE task's upstream fix.
- **Evidence.** Base every claim on evidence you opened, and cite it. If something cannot be checked, say so
  instead of guessing.
