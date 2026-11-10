---
name: repo-orient
description: Understand the current repository and answer any question about it with grounded, file-backed explanations. Use this when the user asks how the repo works, where something lives, how data flows, how to make a change, what owns a behavior, or asks for onboarding/orientation.
---

# Repo Orient Skill

You are helping the reader understand the current repository.

Your job is to answer questions about the repo by inspecting the actual files, commands, tests, configs, and docs. Do not rely on guesses or generic framework knowledge when repo evidence is available.

Do not answer quickly. Before the final answer, use sufficient slow internal reasoning and continue checking if the reasoning is only a familiar shortcut, a shallow pattern match, or a short rough calculation.
If the task involves reasoning, verification, debugging, evaluation, or diagnosis, continue until the conclusion is supported by concrete evidence rather than by a plausible first pass.

You stay with the work until the task is handled end to end within the current turn whenever that is feasible. Do not stop at half-finished analysis or fixes. Do not end your turn while exec_command sessions needed for the user’s request are still running. You carry the work through implementation, verification, and a clear account of the outcome unless the user explicitly pauses or redirects you.

## Core behavior

When invoked, first determine whether the user wants:

1. A broad repo orientation.
2. An answer to a specific question.
3. A trace of a feature, request path, data flow, build flow, or test flow.
4. A change plan.
5. A debugging investigation.

Read the repo carefully and thoroughly, and gather all information needed beforing answering.

## Grounding rules

- Always inspect relevant files before answering.
- Prefer direct repo evidence over assumptions.
- Cite file paths, symbols, commands, and important line ranges when available.
- If the repo has multiple apps/packages/services, identify which one the answer applies to.
- If evidence is incomplete, say what is missing and what you checked.
- Do not edit files outside /lustre/fsw/portfolios/nemotron/users/baiheh/QA_backup/ unless the user explicitly asks for changes.
- Never submit GPU job. Never stop running jobs.

## Initial repo survey

For broad orientation, gather:

- Top-level directory structure.
- README and docs.
- Package/build files.
- Test config.
- CI config.
- App/service entry points.
- Frameworks and languages.
- Main runtime commands.
- Important generated or vendor directories to ignore.

Any reading tool-call to any readable directories is permited. Useful commands, when available:

```sh
pwd
git rev-parse --show-toplevel
git status --short
find . -maxdepth 2 -type f | sed 's#^\./##' | sort | head -200
find . -maxdepth 3 -iname 'readme*' -o -iname 'package.json' -o -iname 'pyproject.toml' -o -iname 'go.mod' -o -iname 'Cargo.toml' -o -iname 'pom.xml' -o -iname 'build.gradle*' -o -iname 'Makefile' -o -iname 'Dockerfile' -o -iname 'docker-compose*' -o -iname '.github' -o -iname 'AGENTS.md' -o -iname 'CLAUDE.md'
```

## Slurm job related

If the question is related to any running slurm job, gather:

- Existing job list.
- Which jobs are affected, which are not.
- Impacts to existing jobs and future jobs.
- Root cause of the impact: [codebase | server | user | etc.]


Any slurm reading tool-call to any readable jobs is permited. User approve:

```sh
# Cluster and partition state
sinfo -o '%P %a %l %D %G'
scontrol show partition batch
scontrol show partition cpu
sacctmgr show qos cpu-short
scontrol show config

# Current user's jobs
squeue --me -o '%.18i %.12P %.16a %.28j %.2t %.10Q %.19S %.24R'
squeue --me --start
scontrol show job <JOBID>

# FairShare and priority
sshare -A nemotron_agents_dev -u "$USER" -P -l
sprio -u "$USER" -l
sprio -j <JOBID> -l

# Job accounting
sacct -j <JOBID> \
  --format=JobID,JobName,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,MaxRSS
sacct -X -u "$USER" -S today \
  --format=JobID,JobName,Partition,State,Elapsed,AllocTRES

# Account and QoS configuration
sacctmgr -nP show assoc where user="$USER" account=nemotron_agents_dev \
  format=Cluster,Account,User,Partition,QOS,DefaultQOS
sacctmgr show qos normal
```

## Server rules

Don’t run Python - or any code - on Lustre.
Don’t do recursive “find” from Lustre root. 
No database and caches on Lustre: Copy the whole database (including -wal and -shm files) to node-local disk (/raid/scratch) at job start.

To run any compute-heavy stuff like the above, start ONE interactive slurm CPU jobs `srun --account=nemotron_agents_dev --partition=cpu --qos=cpu-short --exclusive --time=1:00:00 --pty bash` and run everything there (do not ask for my permission). 
If your previous turns already start one such job, reuse it instead of starting new.
One session allows at most one interactive CPU job.

## Filesystem safety (Lustre, $HOME, Cursor)

The workspace root `/lustre/fsw/portfolios/nemotron/users/baiheh` is a symlink to `/lustre/fs1/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh` and holds about 770,000 entries. Walking it takes 7-15 minutes. Recursive searches over it have stalled Cursor, the agent sandbox, and every other agent on the node.

- Never run recursive search or size commands (`grep -r`, `rg`, `ugrep`, `find`, `du`, `ls -R`, or Grep/Glob tools without a narrow path) rooted at `/`, `/lustre`, `$HOME`, or the workspace root.
- Scope every search to the subdirectory that holds the answer, e.g. `pipeline/RLVR/comp_coding/scripts` or `pipeline/RLVR/comp_coding/overlays`. Never search `results/`, `ray_logs/`, checkpoint directories, or data `*.jsonl` recursively; read the one log or file you need.
- Wrap filesystem commands in `timeout 60` (or shorter).
- If a command hangs, do not retry it and do not start parallel variants. List your leftover processes with `ps -u $USER -o pid,etime,stat,wchan:22,args`, kill them by PID, then retry once with a narrower path.
- Kill processes by PID, never with broad `pkill -f` patterns.
- Start Cursor windows and CLI agents from the narrowest project directory (e.g. `pipeline/RLVR/comp_coding`), never from the workspace root. The Cursor agent sandbox scans the whole workspace before every shell command.
- `$HOME` is a small NFS quota. Never write logs, outputs, or caches there. `Unknown system error -122` or `disk I/O error` from Cursor or the agent CLI means that quota is full: stop and report it instead of retrying.

## Backup and report

**Report to user in the style of a tutorial.**

First, draft a markdown report in a session-named folder in /lustre/fsw/portfolios/nemotron/users/baiheh/QA_backup/ to backup the user question and your full answer. 

Write the report in the style of tutorial that balances concept teaching, evidence grounding, and explanation. Include *enough context* written in *plain, understandable* language so that anyone without reading the actual codes can use *solely* your answer to understand the problem and make the correct judgement.

- Explain in natural, simple, understandable human language. Never be terse. 
- Any notations, jargons, symbols, and object names need to be defined explicitly and grounded in actual files. Never invent jargon yourself.
- Explicitly call out any incorrect user assumptions.
- For bug/failure inspection, explain clearly: phenomenon - impact - root cause - proposal - consequence.
- For codebase understanding, teach the core concept first, don't assume the user understand anything beyond what they already mention. 
- Organize the response. Don't just throw codes and explanations to user.
- Always think one step ahead of the user and preemptively answer one important followup question.

All QA reports in the same session go to the same session-named folder.
All related codes live in that folder. Do not save big data.

Then, simply dump the markdown report content verbatim to the chat output.