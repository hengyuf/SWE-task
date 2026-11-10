# DeepSWE mid-train synthetic tasks: LLM engine, failure analyzer, task builder

Implements the three components of `../DeepSWE Mid-train Synthetic Data.pdf`. Each component is a deterministic
Python launcher: fixed prompt templates (`prompts/`), fixed agent command lines, and controller-side gates.
No agent orchestrates other agents. Run everything inside CPU jobs (`run.sbatch`). Heavy work happens in OpenSandbox
(Cell-3) through `../opensandbox/`. Outputs go under `runs/<run>/`.

This is hengyuf's port of Baihe's pipeline (2026-10-16; what changed and why: `../README.md`). Student = `sc14`
(local checkpoint swonlysc-step0014 behind this job's student router, `../student/`), teacher = `kimi`,
judge = `glm` (both on the NVIDIA inference hub). Model settings: `settings.json` (ATP's `deepswe.json` with these
three models).

| File | Role |
|---|---|
| `llm_engine.py` | `chat(model, messages)`: hosted models on the NVIDIA inference hub (hub key = `NVAPI_KEY_NEW` (`common.HUB_KEY`) from `/lustre/fsw/portfolios/nemotron/users/hengyuf/.env`, read per call), the local student through `student.client` (router URL, no credentials; waits for a worker up to `$SYNTH_STUDENT_WAIT_S`, default 3600 s); `agent(prompt, workdir)` runs codex 0.163.0 (`../tools/codex`, `gpt-6-astra`, xhigh) headless with the project's ground rules appended to every prompt (a Cursor command line is kept but unused). Agents never get secret-like variables or `SYNTH_STUDENT_BASE_URL` |
| `rollouts.py`, `probe_worker.py` | Load the target rollouts (`../data/targets/sc14-out131k-kimi`: every line sha256-checked against `index.json`, and `index.json` against `COMPLETE.json` once per process); run student/teacher probes on synthetic tasks through ATP's own campaign code (`../atp`); render transcripts |
| `sandbox.py` | Owned sandboxes, `bin/osb-*` wrappers for agents, snapshots, and `grade()` (Gym's DeepSWE verifier path) |
| `failure_analyzer.py` | codex + task sandbox -> `failure_analysis.md` / `result.json`; GLM judges for cheated passes and failure similarity |
| `task_builder.py` | image A -> understanding -> mask plan -> author -> starting state C -> grader gates -> instruction + rubric -> Harbor task -> probe -> score -> accept or evolve |
| `smoke.py` | Phase-0 checks S1-S4 (S2's `source_payload` check gates the hosted models as in Baihe's smoke and is informational for the local student, whose thinking stays in the assistant content so ATP rewrites nothing) |

## Commands (from this directory; each is one CPU job)
```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
sbatch -J synth-tests run.sbatch -m unittest discover -s tests -v       # offline tests
sbatch -J synth-smoke run.sbatch smoke.py --run smoke1 s1 s3 s4          # sandbox/grader, codex, chat (sc14 needs a worker)
sbatch -J synth-s2 run.sbatch smoke.py --run smoke1 s2                   # probe glue (after s1; ~1-3 h + worker wait)
sbatch -J synth-fa run.sbatch failure_analyzer.py --run val1 --tasks abs-stepped-slices,fastapi-implicit-head-options,valibot-recursive-schema-composition
sbatch -J synth-build run.sbatch task_builder.py build --run val1 --task abs-stepped-slices --repo deepnoodle-ai/risor --candidates 2 --branches 2 --rounds 1
sbatch -J synth-cleanup run.sbatch task_builder.py cleanup --run val1     # delete non-accepted snapshots
```
Every stage is idempotent: a stage with a validated `stage.json`/`result.json` is skipped. To resume, resubmit the
same command in a new job. Probes need at least 3.6 h left in the job (ATP admission: 3 h agent cap + verifier +
grace) after the student wait: a probe with `sc14` first starts this job's router and waits up to
`$SYNTH_STUDENT_WAIT_S` (default 10800 s; `sbatch --export=ALL,SYNTH_STUDENT_WAIT_S=...`) for a routable worker,
but never longer than the job can afford: the wait ends by `SLURM_JOB_END_TIME` - (agent timeout 10800 s + the
largest verifier timeout + 600 s), and a probe started later than that fails at once (rerun it in a fresh job).
Workers are separate GPU jobs (`../student/README.md`). `run.sbatch` sets `#SBATCH --export=NONE` (only the
submitting shell's `SLURM_*` variables reach the job unless `--export` is given on the command line) and still drops
secret-looking variables (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`) and `SYNTH_STUDENT_BASE_URL`
at job start; the code reads its credentials from files.

Smoke order: offline tests; `s1 s3 s4`; `s2` (reads the `s1` snapshot); then the failure analyzer on the three
validation tasks and one small build. The failure analyzer needs no student; `s2`, `s4` and the build's probes do.
`../tools/launch_smoke.sh [--dry-run]` submits this order as one dependency chain (tests -> `s1 s3 s4` -> `s2`
afterany; tests -> failure analyzer -> build afterany) with a scrubbed environment plus `SLURM_CONF`, and records
the job ids in `runs/launch-<time>.json`.

## Outputs
- `runs/<run>/analysis/<task>/`: `failure_analysis.md`, `result.json`, `inputs/` (rollout transcripts, stats).
- `runs/<run>/build/<task>/<owner__repo>/`: `env/` (image A recipe and snapshot), `understanding/`, `plan/`,
  `cand-*/` (author attempts, grader runs, `task/<task_id>/` Harbor dir), `round-*/` (probe campaign and per-task
  `metrics.json`), and `summary.json`.
- `runs/<run>/accepted/<task_id>/`: accepted tasks, with `tasks.jsonl` (Gym DeepSWE rows) and workspace archives.
- `runs/slurm/<job name>-<job id>.out`: job logs.

Synthetic tasks pass Gym's own `DeepSWETaskStore` checks (`../atp/vendor/Gym` and hengyuf's swe-rl Gym copy).
`task.toml` sets `docker_image = "snapshot:<id>"`, an OpenSandbox Cell-3 snapshot of the starting state. Evaluating
them requires the 3-line provider patch in `probe_worker.py` (`patch_provider`), which is applied only to the
job-local staged Gym copy. `environment/Dockerfile` (+ `setup.sh`, `materialize.sh`, `start.patch`) rebuilds the
same image for later publishing. `environment/` and `solution/` are verifier-side material: never give them to
students.
