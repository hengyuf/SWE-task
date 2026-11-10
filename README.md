# SWE-task

Code and docs of hengyuf's SWE-task framework on the dfw cluster, mirrored from Lustre (code only: no data, results,
logs, model weights, virtualenvs or credentials).

| Path | What |
|---|---|
| `swe-task/synth/` | Synthetic SWE task pipeline (task builder, failure analysis, probes via ATP, queue runners, prompts, tools, tests) |
| `swe-task/atp/` | hengyuf's tracked copy of Baihe's ATP DeepSWE campaign app (scripts, overlays, configs; `README.md`, `PROVENANCE.json`, `hengyuf-changes.diff`; vendor/runtime not mirrored) |
| `swe-task/sft/<campaign>/` | Rollout / training-accuracy / teacher campaigns (driver, keeper, router, sbatch, reports). Current: `sft/glm-rollouts-tasksall-20261108/` (self-hosted GLM-5.3 x 5 on all tasks of `tasks-all-20261106`) |
| `swe-task/student/` | Local student (sc14) vLLM fleet: router, workers, fleet daemon |
| `swe-task/dashboard/`, `swe-task/opensandbox/`, `swe-task/tools/`, `swe-task/docs/` | Dashboard, OpenSandbox helpers, small tools, contracts |
| `glm53-serve/` | Self-hosted GLM-5.3 serving on dfw H100 (vLLM 0.33 + backports): launcher sbatch/rank scripts, env build scripts, divergence notes and patches, quality/throughput tools, swe-task E2E driver |

`tools/external/` holds the two Slurm schedule-helper scripts the sbatch/submit scripts source (they live outside the
two trees; origin paths and sha256 in its README).

## Not in this repo

- Every credential: the code reads them at run time by NAME from mode-600 files on the cluster (`swe-task/atp/.env`,
  `swe-task/secrets/`, `sft/glm-rollouts-sftv3-20261107/secrets/`), built by `tools/make_secrets.py` /
  `make_baihe_secrets.py` from the user's `.env`; none of those files is mirrored.
- Data and state: task trees and campaign state (`synth/runs`, `sft/*/probes`, `sft/*/results`, logs, dashboard
  state), JSON data over 150 KB (e.g. `glm-rollouts-tasksall-20261108/tasks.json`, the dashboard's task-snapshot test
  fixtures), the release tarballs of `release-cache/` (their manifest is mirrored).
- Runtime pieces: the ATP vendor/runtime trees (`atp/README.md` "Rebuild"), the vLLM virtualenvs (built by
  `glm53-serve/env/build_*.sh` inside the container image named in `glm53-serve/scripts/serve_glm53_dfw.sbatch`), the
  GLM-5.3 weights, and the codex binary/home state (only its `config.toml` and model catalog are mirrored).
- The scripts that built `sft/tasks-all-20261106` (its README documents the build; they ran from a scratch dir).

## Refresh

`python3 -I -B tools/sync_from_lustre.py` (dry run: per-directory summary + credential-pattern scan) and `--write`
(replaces `swe-task/` and `glm53-serve/` with the current code; `tools/MANIFEST.txt` lists every file with its sha256).
The selection rules and the scan patterns are in that script.

## Progress and decisions

The running campaign posts progress hourly, and the decisions it needs, in the repository's issue thread.
