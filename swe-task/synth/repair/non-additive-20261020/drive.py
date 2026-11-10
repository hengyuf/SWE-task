"""Driver for the 2026-10-20 non_additive repair: runs tools/grading_repair.py's own stage/validate/regrade functions
for many tasks of official1-vitest and official1-testem under ONE shared gate of at most 6 sandboxes, retrying
sandbox/PTY errors with backoff. Run from S/synth with the controller venv:
  ../atp/runtime/controller/bin/python3 -B repair/non-additive-20261020/drive.py <jobs.json> [stage|validate|regrade|both|old]
jobs.json: [{"run": ..., "task": ..., "slots": [...], "validate": bool}]"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/tools")
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
import asyncio, contextlib, json, time
import grading_repair as gr

MAX = 6


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def entry_of(plan, run, task):
    return gr.plan_entries(plan, run, [task], False)[0]


def needs_retry(result):
    return result is None or "error" in result or result.get("status") == "infra"


async def regrade_retrying(plan, job, graders, gate, attempts=4):
    entry = entry_of(plan, job["run"], job["task"])
    pending = list(job["slots"])
    state = None
    for attempt in range(1, attempts + 1):
        slots = gr.task_slots(job["run"], entry, pending, False)
        try:
            state = await gr.regrade_task(job["run"], entry, slots, graders, gate)
        except Exception as error:  # e.g. a scratch-copy check; not retried blindly
            log(f"ERROR {job['task']}: {type(error).__name__}: {str(error)[-300:]}")
            raise
        bad = [s for s in pending if any(needs_retry(state["slots"][s].get(g)) for g in graders)]
        done = [s for s in pending if s not in bad]
        log(f"{job['run'][10:]} {job['task']}: attempt {attempt}: " + ", ".join(
            f"{s}={'/'.join(str((state['slots'][s].get(g) or {}).get('status')) for g in graders)}" for s in done)
            + (f"; RETRY {bad}: " + "; ".join(str((state['slots'][s].get(graders[0]) or {}).get('error', 'infra'))[:160]
                                             for s in bad) if bad else ""))
        if not bad:
            return state
        pending = bad
        await asyncio.sleep(90 * attempt)
    return state


async def validate_retrying(plan, job, gate, attempts=4):
    entry = entry_of(plan, job["run"], job["task"])
    for attempt in range(1, attempts + 1):
        record = await gr.validate_task(job["run"], entry, gate, tamper=True)
        errors = {n: r.get("error") or r.get("status") for n, r in record["results"].items()
                  if "error" in r or r.get("status") == "infra"}
        log(f"{job['run'][10:]} {job['task']}: validate attempt {attempt}: checks {record['checks']} "
            + " ".join(f"{n}={r.get('reward')}({r.get('f2p')},{r.get('p2p')})" for n, r in record["results"].items())
            + (f" ERRORS {errors}" if errors else ""))
        if not errors:
            return record
        await asyncio.sleep(90 * attempt)
    return record


async def main(jobs_path, what):
    jobs = json.load(open(jobs_path))
    plan = gr.load_plan()
    runs = sorted({j["run"] for j in jobs})
    with contextlib.ExitStack() as stack:
        for run in runs:
            stack.enter_context(gr.locked(run))
        if what == "stage":
            for job in jobs:
                record = gr.stage_task(job["run"], entry_of(plan, job["run"], job["task"]))
                log(f"{job['task']}: {'unchanged' if record.get('unchanged') else 'staged'} mode "
                    f"{record['test_paths_mode']} source_grader {record['source_grader']}")
            return
        gate = asyncio.Semaphore(MAX)
        coros = []
        if what in ("validate", "both"):
            coros += [validate_retrying(plan, j, gate) for j in jobs if j.get("validate")]
        if what in ("regrade", "both", "old"):
            graders = ["old"] if what == "old" else ["new"]
            coros += [regrade_retrying(plan, j, graders, gate) for j in jobs if j.get("slots")]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                log(f"FAILED: {type(result).__name__}: {str(result)[-400:]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
