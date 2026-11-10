#!/usr/bin/env python3
"""TEMPLATE, NOT RUN (written 2026-11-06 by the swe-task-harness subagent; review before use).

GLM-5.3 served by OUR vLLM endpoint as the coding agent on the synthetic tasks of teacher-rollouts-v2 (the 50
train-acc-v2 tasks), through the synth pipeline's own probe path (rollouts.probe -> ATP campaign -> OpenCode 1.17.11 ->
each task's own grader), with the hub GLM-5.3 runs' settings except the endpoint. Run in a CPU Slurm job (ATP's stage()
refuses non-cpu partitions; ATP admits a slot only with >= 10,800 + 1,800 + 600 s left in the job):

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth && \
    /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B \
        <THIS DIR>/glm_local_driver.py --root <NEW campaign root> --base-url http://<server-ip>:8000/v1 \
        --taskset pilot|safe14|group [--group K] [--repeats N] [--max-concurrency C] [--min-model-len N] [--dry-run]

Differences from sft/teacher-rollouts-v2-20261025/driver.py (TR2), which ran hub GLM-5.3 on the same tasks:
 * model alias ALIAS = "glmlocal" with the in-memory spec SPEC below (synth/settings.json is never edited):
   id = the server's --served-model-name; base_url read from $GLM53_BASE_URL by Gym (${oc.env:...}, as the sc14 spec
   reads $SYNTH_STUDENT_BASE_URL), so a resumed campaign keeps its spec when the server moves; api_key "EMPTY";
   context_window 262,144 (OpenCode compacts at 262,144 - 20,000 = 242,144, as teacher-deepswe-20261030; TR2 itself ran
   at the global 1,048,576 and never compacted); max_output_tokens 131,072; reasoning_effort "max";
   temperature 1.0 / top_p 0.95 from settings.json.
   NO "local": true (rollouts.probe would start the sc14 student router and wait up to 3 h for sc14 workers, and
   build_config would switch Gym to <think>-in-content history); NO chat_template_kwargs (GLM template defaults:
   reasoning effort Max, clear_thinking false = what the hub ran). The alias is not "kimi", so ATP sends
   non-streaming calls (ATP_DEEPSWE_UPSTREAM_STREAM=0), like every hub GLM call.
 * no hub key anywhere: no key_budget.py / hub_budget.py (they call the hub); this directory's probe_worker.py wrapper
   gives ATP a placeholder instead of the hub key; the local URL is not ATP_DEEPSWE_POLICY_BASE_URL, so
   deepswe_http.request_with_retry never binds a credential to these calls (it sends Gym's "Bearer EMPTY").
 * pre-flight: GET <base>/models must list SPEC["id"] with max_model_len >= --min-model-len (default 262,144; use
   393,216 for vLLM <= 0.31, which answers 400 when prompt + max_tokens > max_model_len, and ATP makes any unresolved
   400 a FATAL slot).
Writes <root>/probes/glmlocal/<name>/ (the ATP campaign) and <root>/results/glmlocal-<name>.json (TR2's compact record
shape + the hub GLM records of the same tasks for comparison). Resumable: rerun the same command in a fresh CPU job.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SYNTH = S / "synth"
TA = S / "sft/train-acc-v2-20261024"           # tasks.json: the 50 tasks (task_dir per task)
TR2 = S / "sft/teacher-rollouts-v2-20261025"    # hub GLM-5.3 results on them: results/glm-group-*.json, glm-pilot.json
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402,F401  (ATP scripts and the sandbox helper on sys.path)
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402

ALIAS = "glmlocal"
SPEC = {"id": "glm-5.3", "base_url": "${oc.env:GLM53_BASE_URL}", "api_key": "EMPTY",
        "context_window": 262144, "max_output_tokens": 131072, "sampling_overrides": {"reasoning_effort": "max"}}
PILOT_TASKS = ("syn-aiohttp-cand-c05-r1b0-7d9ec1f5", "syn-argo-cd-cand-c05-cc3e1272")  # TR2's pilot pair
# The 14 tasks whose 5 hub GLM rollouts all stayed <= 200k prompt tokens (traffic-extra.json): the 262,144 window
# (compaction at 242,144) cannot have changed them; hub GLM: 64 / 70 = 91.4%.
SAFE14 = ("syn-bitecs-cand-c05-r1b0-85a91c3b", "syn-bitecs-cand-c05-r1b4-bc99c654", "syn-dacite-cand-c01-r1b3-84fe778b",
          "syn-dacite-cand-c02-r1b4-325c5f9d", "syn-dacite-cand-c04-r1b0-eeb19c3a", "syn-dacite-cand-c05-r1b4-2e37bfb7",
          "syn-jest-cand-c04-r1b4-d8b1da89", "syn-pylama-cand-c01-r1b4-r2b4-68d85058", "syn-pylama-cand-c03-r1b2-6182c7b3",
          "syn-pylama-cand-c03-r1b4-r2b4-58e91fc7", "syn-pylama-cand-c05-r1b4-6eedf9a6",
          "syn-vscode-sqltools-cand-c01-r1b0-24c4a31b", "syn-yj-cand-c01-7e8e179e", "syn-yq-cand-c01-r1b4-c75b758f")
HUB_MARKERS = ("nvidia.com", "inference-api")


def preflight(base_url: str, min_model_len: int) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        listing = json.load(response)
    models = {entry.get("id"): entry for entry in listing.get("data") or []}
    if SPEC["id"] not in models:
        raise SystemExit(f"{base_url} does not serve {SPEC['id']!r} (served: {sorted(models)})")
    max_model_len = models[SPEC["id"]].get("max_model_len")
    if max_model_len is not None and max_model_len < min_model_len:
        raise SystemExit(f"server max_model_len {max_model_len} < {min_model_len}: calls with prompt + 131,072 > "
                         "max_model_len get a 400 on vLLM <= 0.31 (ATP: fatal slot); raise it or use vLLM >= 0.32")
    return {"served": sorted(models), "max_model_len": max_model_len}


def hub_records(task_ids) -> dict:
    """TR2's hub GLM-5.3 compact records of these tasks (5 per task)."""
    out = {}
    for path in sorted((TR2 / "results").glob("glm-group-*.json")):
        for task_id, records in json.loads(path.read_text())["records"].items():
            if task_id in task_ids:
                out[task_id] = [{key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p")}
                                for record in records]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="NEW directory for this local-GLM campaign set")
    parser.add_argument("--base-url", required=True, help="OUR endpoint, e.g. http://10.1.2.3:8000/v1")
    parser.add_argument("--taskset", choices=("pilot", "safe14", "group"), required=True)
    parser.add_argument("--group", type=int, default=0, help="group K of 8 = tasks[K::8] (TR2's grouping)")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="rollouts in flight for this campaign, fixed at creation (pipeline cap 32)")
    parser.add_argument("--min-model-len", type=int, default=262144)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(marker in args.base_url for marker in HUB_MARKERS):
        raise SystemExit("--base-url must be our own server, never the NVIDIA hub")
    assert 1 <= args.max_concurrency <= 32, "ATP stalls around 125 rollouts per process; the pipeline caps 32"
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    by_id = {task["task_id"]: task for task in tasks}
    if args.taskset == "pilot":
        chosen, name, repeats = [by_id[t] for t in PILOT_TASKS], "pilot", args.repeats or 1
    elif args.taskset == "safe14":
        repeats = args.repeats or 1
        chosen, name = [by_id[t] for t in SAFE14], f"safe14-x{repeats}"
    else:
        assert 0 <= args.group < 8
        chosen, name, repeats = tasks[args.group::8], f"group-{args.group}", args.repeats or 5
    probe_dir = args.root.resolve() / "probes" / ALIAS / name
    server = preflight(args.base_url, args.min_model_len)
    plan = {"alias": ALIAS, "spec": SPEC, "base_url": args.base_url, "server": server, "campaign": str(probe_dir),
            "tasks": [t["task_id"] for t in chosen], "repeats": repeats, "slots": len(chosen) * repeats,
            "max_concurrency": args.max_concurrency}
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return
    if (probe_dir / "settings.json").exists():  # a resumed campaign must have been created with this spec
        created = json.loads((probe_dir / "settings.json").read_text())["models"]
        assert created == {ALIAS: SPEC}, f"{probe_dir} was created with {created}"
    os.environ["GLM53_BASE_URL"] = args.base_url  # inherited by ATP staging, probe_worker and Gym (${oc.env:...})
    SETTINGS["models"][ALIAS] = dict(SPEC)        # in memory only: rollouts.probe() writes it into the campaign
    rollouts.SYNTH = HERE                          # probe() runs HERE/probe_worker.py (placeholder-key wrapper)
    rollouts.PROBE_MAX_CONCURRENCY = args.max_concurrency
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"glm-local: {name}: {len(chosen)} tasks x {repeats} via {args.base_url} (server {server}), "
        f"max concurrency {args.max_concurrency}")
    started = time.time()
    records = asyncio.run(rollouts.probe(probe_dir, [Path(t["task_dir"]) for t in chosen], models=(ALIAS,),
                                         repeats=repeats))
    compact = {task_id: [{key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error",
                                                            "verifier_error", "attempt_dir")}
                         | {"patch_chars": len(record.get("patch") or "")} for record in slot_records]
               for task_id, slot_records in records.items()}
    out = args.root.resolve() / "results" / f"{ALIAS}-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps(dict(plan, started_at=started, finished_at=time.time(), records=compact,
                                   hub_glm_tr2=hub_records(set(compact))), indent=1) + "\n")
    os.replace(tmp, out)
    log(f"glm-local: {name} done: {json.dumps(rollouts.pass_stats([r for rs in records.values() for r in rs]))}")


if __name__ == "__main__":
    main()
