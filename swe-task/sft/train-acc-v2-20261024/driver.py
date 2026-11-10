#!/usr/bin/env python3
"""train-acc-v2 (2026-10-24): one probe campaign = one model x one task group x 5 rollouts per task.

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B driver.py --model sftv2|sc14 --group K [--groups 8])

The synth pipeline's own rollout and grading path (rollouts.probe: ATP campaign, OpenCode harness, each task's own
grader), exactly as the pipeline probes sc14 and kimi when it builds a task. Differences:
* the model is served by this test's fleet (fleets/<fleet>/registry, serve_model.sh), never the pipeline's sc14 fleet:
  this job's router reads only that registry, and probe()'s worker wait is pointed at it;
* "sftv2" = the sc14 model spec (local, served as "policy", same sampling and context) with the v2 SFT checkpoint;
  "sc14" (the base model) keeps its spec;
* before the campaign the job waits for --min-workers routable workers (up to --worker-wait-s, then any >= 1), so
  the rollouts of all campaigns do not crowd the first worker up.
Runs in a CPU Slurm job (ATP admission needs >= ~3.6 h left; probe() is resumable: rerun the same group to continue).
Writes probes/<model>/group-<K>/ (the campaign) and results/<model>-group-<K>.json (compact per-slot records).
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

TA = Path(__file__).resolve().parent
SYNTH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402  (ATP scripts, the OpenSandbox helper and the student client on sys.path)
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402
import student.client as student_client  # noqa: E402

MODELS = {
    "sftv2": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
              "super-kimi-raw-sft-v2-wsd-5ep-20261023/exports/iter_0000038/hf", TA / "fleets/sftv2"),
    "sc14": ("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf",
             TA / "fleets/base"),
    # 2026-10-25 (user): the same v2 corpus trained 10 and 20 epochs; served like sftv2 (the sc14 spec, "policy").
    "sftv2e10": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
                 "super-kimi-raw-sft-v2-wsd-10ep-20261024/exports/iter_0000076/hf", TA / "fleets/sftv2e10"),
    "sftv2e20": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
                 "super-kimi-raw-sft-v2-wsd-20ep-20261024/exports/iter_0000152/hf", TA / "fleets/sftv2e20"),
    # 2026-10-27 (user, "fix the thinking pattern"): the v2 corpus with an empty think prefix, 5 and 10 epochs.
    "etp5": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
             "super-kimi-v2-empty-think-prefix-wsd-5ep-16n-ep32-20261027/exports/iter_0000038/hf", TA / "fleets/etp5"),
    "etp10": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
              "super-kimi-v2-empty-think-prefix-wsd-10ep-16n-ep32-20261027/exports/iter_0000076/hf", TA / "fleets/etp10"),
    "etp20": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
              "super-kimi-v2-empty-think-prefix-wsd-20ep-16n-ep32-20261027/exports/iter_0000152/hf", TA / "fleets/etp20"),
    # 2026-10-30 (user): the SFT-v3 corpus (sft-v3-20261029) with the empty think prefix, 5 and 10 epochs.
    "v3etp5": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
               "super-kimi-sft-v3-empty-think-prefix-wsd-5ep-16n-ep32-20261029/exports/iter_0000071/hf", TA / "fleets/v3etp5"),
    "v3etp10": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
                "super-kimi-sft-v3-empty-think-prefix-wsd-10ep-16n-ep32-20261029/exports/iter_0000142/hf",
                TA / "fleets/v3etp10"),
    # 2026-11-04 (user): the same SFT-v3 empty-think-prefix recipe, 20 epochs.
    "v3etp20": ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
                "super-kimi-sft-v3-empty-think-prefix-wsd-20ep-16n-ep32-20261029/exports/iter_0000284/hf", TA / "fleets/v3etp20"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-workers", type=int, default=8)
    parser.add_argument("--worker-wait-s", type=float, default=7200)
    args = parser.parse_args()
    checkpoint, fleet = MODELS[args.model]
    registry = fleet / "registry"
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    group = tasks[args.group::args.groups]
    probe_dir = TA / "probes" / args.model / f"group-{args.group}"
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.model not in SETTINGS["models"]:
        SETTINGS["models"][args.model] = dict(SETTINGS["models"]["sc14"], checkpoint=checkpoint)
    assert SETTINGS["models"][args.model]["local"] and SETTINGS["models"][args.model]["id"] == "policy"

    # This job's router: only this fleet's workers. probe() reuses it (ensure_router honours a live
    # SYNTH_STUDENT_BASE_URL); its own worker wait is pointed at the same registry.
    os.environ.pop("SYNTH_STUDENT_BASE_URL", None)
    base_url = student_client.ensure_router(registry=registry, log_dir=TA / "routers" / f"{args.model}-group-{args.group}")
    os.environ["SYNTH_STUDENT_BASE_URL"] = base_url
    original_wait = student_client.wait_for_workers
    student_client.wait_for_workers = lambda n=1, timeout_s=10800, registry=None, **kw: original_wait(
        n, timeout_s=timeout_s, registry=fleet / "registry", **kw)
    deadline = time.time() + args.worker_wait_s
    while True:
        try:
            workers = original_wait(args.min_workers, timeout_s=max(60.0, min(1800.0, deadline - time.time())),
                                    registry=registry, log=log)
            break
        except TimeoutError:
            if time.time() >= deadline:
                workers = original_wait(1, timeout_s=3600, registry=registry, log=log)
                break
    log(f"train-acc: {args.model} group {args.group}: {len(group)} tasks x {args.repeats}, {len(workers)} workers, "
        f"router {base_url}")
    records = asyncio.run(rollouts.probe(probe_dir, [Path(task["task_dir"]) for task in group],
                                         models=(args.model,), repeats=args.repeats))
    compact = {task_id: [{key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error",
                                                            "verifier_error", "attempt_dir")} | {"patch_chars": len(record.get("patch") or "")}
                         for record in slot_records] for task_id, slot_records in records.items()}
    out = TA / "results" / f"{args.model}-group-{args.group}.json"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps({"model": args.model, "checkpoint": checkpoint, "group": args.group,
                               "tasks": [task["task_id"] for task in group], "repeats": args.repeats,
                               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": compact}, indent=1) + "\n")
    os.replace(tmp, out)
    stats = rollouts.pass_stats([record for slot_records in records.values() for record in slot_records])
    log(f"train-acc: {args.model} group {args.group} done: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
