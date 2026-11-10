#!/usr/bin/env python3
"""Re-probe Kimi K3 (the teacher) on scored tasks whose teacher slots were infra, then re-score them.

hengyuf 2026-11-05 ("方案2 和方案3都做"; option 2): 37 official4 tasks were scored "inconclusive: fewer than 3 valid
rollouts per model" because most of their Kimi K3 slots ended as infra (2026-11-02: the hub answered ClientResponseError,
ATP's generation-validity assessment said "fatal", opencode exited after a few calls; 155 of their 185 Kimi slots).
Acceptance needs a student (sc14) pass rate <= 0.2, so only the 19 with sc14 <= 0.2 can still be accepted. For those this
tool runs ONE teacher-only probe (rollouts.probe(models=(TEACHER,)): 5 repeats, the pipeline's settings.json Kimi
config, as every build probe) and re-scores each task with task_builder.score() on
  - the sc14 records of the round's original probe (rollouts.probe_records; checked byte for byte against the task's
    rollouts/<slot>.md, i.e. exactly what the first scoring used), plus
  - the new probe's 5 Kimi records (every old Kimi slot is replaced; none is mixed in).
Then accept() when it accepts, <round>/seeds.json and summary.json's accepted list as a resumed build writes them
(tools/regrade.py helpers; backups *.pre-teacher-reprobe), and the task's old failure-match audit is moved aside (its key
dropped from queue4/audit_seen.json) so official4_audit_loop.js labels the new scoring.

  list    --runs R [R ...] [--max-student 0.2]
          read-only: the selectable tasks (reason "inconclusive", teacher valid < 3, sc14 valid >= 3 and sc14 pass rate
          <= --max-student)
  prepare --runs R [R ...] --workdir W [--max-student 0.2]
          freeze that task set in W/selection.json (the probe directory W/probe-<sha8> is named by its task ids, so a
          rerun must never re-select)
  run     --workdir W [--concurrency 3] [--dry-run]
          the probe (resumable), then per task: move the old score outputs aside (rollouts/, teacher_audit.json,
          failure_analysis/, metrics.json -> *.pre-teacher-reprobe, metrics.json last), score(), accept(), bookkeeping,
          audit reset. Idempotent through cand/teacher_reprobe.json (moving -> aside -> scored -> done): rerun to resume.
The hub budget is checked before the probe and before each task (rescore_unscored.SpendGuard: NVAPI_KEY_NEW_2 floor
$3,000; an unreadable budget = the hub's glm is down -> wait). The probe needs a CPU Slurm job with >= ~3.6 h left when it
admits slots (agent timeout 3 h + verifier + grace). Every action is one JSON line in <build root>/reprobe.jsonl.
Run from S/synth with the pipeline venv, e.g.
  ../atp/runtime/controller/bin/python3 -B tools/reprobe_teacher.py list --runs official4-textual-richlog
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import uuid  # noqa: E402

TOOLS = Path(__file__).resolve().parent
SYNTH = TOOLS.parent
for _path in (SYNTH, TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import RUNS, STUDENT, TEACHER, hub_key_name, read_json, write_json  # noqa: E402
import regrade  # noqa: E402  (tools/regrade.py: make_build, refresh_seeds, refresh_summary, FixError)
import rescore_unscored as R  # noqa: E402  (build_roots, task_dict, holding_probe, probe_records, SpendGuard, ...)
import rollouts  # noqa: E402
import task_builder  # noqa: E402

TOOL = "tools/reprobe_teacher.py"
LOG = "reprobe.jsonl"                    # <build root>/reprobe.jsonl: one JSON line per action
SUFFIX = ".pre-teacher-reprobe"          # the first scoring's outputs, moved aside; seeds/summary/audit backups
SIDECAR = "teacher_reprobe.json"         # cand/<round>/<cand>/teacher_reprobe.json: this tool's per-task state
SCORE_OUTPUTS = ("rollouts", "teacher_audit.json", "failure_analysis", "metrics.json")  # metrics.json last
AUDIT_SEEN = SYNTH / "queue4/audit_seen.json"
REPEATS = 5
regrade.LOG = LOG  # regrade.refresh_seeds / refresh_summary log through regrade.log_change into <root>/<regrade.LOG>


def log_action(root: Path, entry: dict) -> None:
    with (root / LOG).open("a") as stream:
        stream.write(json.dumps({"at": R.now_iso(), "tool": TOOL, **entry}, sort_keys=True) + "\n")


TAKEN = ("moving", "aside", "scored", "done")  # sidecar states of a task an earlier re-probe has consumed


def selectable(run: str, max_student: float) -> list:
    """The selectable tasks; a task an earlier re-probe consumed (sidecar in TAKEN) never is ("probe_infra", i.e. Kimi
    broke again and nothing was touched, may be selected again for a new workdir)."""
    out = []
    for root in R.build_roots(run):
        for round_dir in sorted(root.glob("round-*"), key=R.round_index):
            for scored in sorted(p for p in round_dir.glob("cand-*") if p.is_dir()):
                if sidecar(scored).get("status") in TAKEN:
                    continue
                metrics = read_json(scored / "metrics.json")
                if not isinstance(metrics, dict) or not str(metrics.get("reason") or "").startswith("inconclusive"):
                    continue
                valid, rate = metrics.get("valid") or [0, 0], metrics.get("student_rate")
                if valid[0] >= 3 and valid[1] < 3 and rate is not None and rate <= max_student:
                    out.append({"run": run, "root": str(root), "round": round_dir.name, "cand": scored.name,
                                "task_id": metrics["task_id"], "task_dir": metrics["task_dir"],
                                "student_rate": rate, "valid": valid})
    return out


def probe_key(tasks: list) -> str:
    return hashlib.sha1(",".join(sorted(t["task_id"] for t in tasks)).encode()).hexdigest()[:8]


def sidecar(scored: Path) -> dict:
    return read_json(scored / SIDECAR) or {}


def save_sidecar(scored: Path, value: dict) -> None:
    write_json(scored / SIDECAR, value)


def move_aside(scored: Path) -> list:
    """The first scoring's outputs -> *<SUFFIX> (metrics.json last). Resumable: an output already aside is skipped; an
    output present while its aside copy also exists is a conflict (a person looks)."""
    moved = []
    for name in SCORE_OUTPUTS:
        path, target = scored / name, scored / f"{name}{SUFFIX}"
        if not (path.exists() or path.is_symlink()):
            continue
        if target.exists() or target.is_symlink():
            raise regrade.FixError(f"{path} and {target} both exist")
        path.rename(target)
        moved.append(name)
    return moved


def reset_audit(item: dict) -> list:
    """Move the task's failure-match audit aside -- with the review's record and its .pre-review backup of the old
    label (failure_match_review.js keeps the first .pre-review, so a stale one would swallow the new label's backup) --
    and drop its key from queue4/audit_seen.json, so the audit loop's poll lists the re-scored task again (it skips
    tasks with an audit file and keys in audit_seen.json). The dashboard reads only <task_id>.json names."""
    done = []
    audit_root = RUNS / item["run"] / "audit"
    for audit in (audit_root / "failure_match" / f"{item['task_id']}.json",
                  audit_root / "failure_match" / f"{item['task_id']}.json.pre-review",
                  audit_root / "failure_match_review" / f"{item['task_id']}.json"):
        if not audit.exists():
            continue
        target = audit.with_name(audit.name + SUFFIX)
        if target.exists():
            target = audit.with_name(f"{audit.name}{SUFFIX}.{uuid.uuid4().hex[:6]}")
        audit.rename(target)
        done.append(f"{audit.parent.name}/{audit.name} -> {target.name}")
    key = f"{item['run']}/{item['cand']}"
    seen = read_json(AUDIT_SEEN)
    if isinstance(seen, list) and key in seen:
        temporary = AUDIT_SEEN.with_name(f".{AUDIT_SEEN.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps([k for k in seen if k != key], indent=1) + "\n")
        os.replace(temporary, AUDIT_SEEN)
        done.append(f"audit_seen -= {key}")
    return done


def bookkeeping(root: Path, round_dir: Path, item: dict, metrics: dict) -> None:
    b = regrade.make_build(root)
    extra = {"tool": TOOL, "cand": item["cand"], "task_id": item["task_id"]}
    regrade.refresh_seeds(b, round_dir, root, SUFFIX, extra)
    regrade.refresh_summary(root, metrics, SUFFIX, extra)


def merged_records(item: dict, new_records: dict):
    """(old probe, sc14 records of the original probe + the new probe's Kimi records) for one task."""
    root = Path(item["root"])
    round_dir = root / item["round"]
    old_probe = R.holding_probe(round_dir, item["task_id"])
    sc14 = [r for r in R.probe_records(old_probe).get(item["task_id"], []) if r["model"] == STUDENT]
    kimi = [r for r in new_records.get(item["task_id"], []) if r["model"] == TEACHER]
    if len(sc14) != REPEATS or len(kimi) != REPEATS:
        raise regrade.FixError(f"{item['task_id']}: {len(sc14)} sc14 and {len(kimi)} kimi records (want {REPEATS} each)")
    return old_probe, sc14 + kimi


def ours(scored: Path, records: list) -> list:
    """Slots whose rollouts/<slot>.md is NOT what score() renders for these records: [] means the score outputs in
    the dir are this re-probe's (sc14 of the original probe + the new Kimi), e.g. written just before a crash."""
    return [r["slot"] for r in R.score_view(records) if not R.same_render(scored / "rollouts" / f"{r['slot']}.md", r)]


async def process(item: dict, new_records: dict, probe_dir: Path) -> str:
    root = Path(item["root"])
    round_dir = root / item["round"]
    scored = round_dir / item["cand"]
    state = sidecar(scored)
    status = state.get("status")
    if status in TAKEN and state.get("probe") != str(probe_dir):
        raise regrade.FixError(f"{scored}/{SIDECAR} belongs to probe {state.get('probe')}, not {probe_dir}")
    if status == "done":
        return "accepted" if (read_json(scored / "metrics.json") or {}).get("accept") else "scored"
    if status == "probe_infra" and state.get("probe") == str(probe_dir):
        return f"kimi infra again: {state.get('kimi_graded')}/{REPEATS} graded (nothing changed)"
    if status == "probe_infra":  # an earlier probe broke again; this is a new one: start over
        status = None
    b = regrade.make_build(root)
    stage = read_json(root / item["cand"] / "stage.json")
    task = R.task_dict(root, round_dir, item["cand"], stage)
    old_probe, records = merged_records(item, new_records)
    if status in (None, "moving"):
        if status is None:  # the first scoring must be the inconclusive one and its sc14 rollouts this probe's records
            metrics = read_json(scored / "metrics.json") or {}
            if not str(metrics.get("reason") or "").startswith("inconclusive"):
                raise regrade.FixError(f"{scored}/metrics.json is not the inconclusive scoring: {metrics.get('reason')}")
            sc14 = [r for r in R.score_view(records) if r["model"] == STUDENT]
            bad = [r["slot"] for r in sc14 if not R.same_render(scored / "rollouts" / f"{r['slot']}.md", r)]
            if bad:
                raise regrade.FixError(f"{scored}: sc14 slots {bad} differ from {old_probe.name}'s records")
            kimi = [r for r in R.score_view(records) if r["model"] == TEACHER]
            graded = [r for r in kimi if r["status"] in ("pass", "fail")]
            if len(graded) < 3:  # Kimi broke again: the first scoring stays as it is, the task can be retried later
                save_sidecar(scored, {"status": "probe_infra", "probe": str(probe_dir), "at": R.now_iso(),
                                      "kimi_graded": len(graded), "kimi": {r["slot"]: r["status"] for r in kimi}})
                log_action(root, {"action": "probe_infra", "cand": item["cand"], "task_id": item["task_id"],
                                  "kimi_graded": len(graded)})
                return f"kimi infra again: {len(graded)}/{REPEATS} graded (nothing changed)"
            state = {"status": "moving", "probe": str(probe_dir), "old_probe": str(old_probe), "at": R.now_iso(),
                     "old_metrics": {k: metrics.get(k) for k in ("student_rate", "teacher_clean_rate", "valid", "reason")}}
            save_sidecar(scored, state)
        moved = move_aside(scored)
        state.update(status="aside", moved=sorted(set(state.get("moved", [])) | set(moved)))
        save_sidecar(scored, state)
        log_action(root, {"action": "move_aside", "cand": item["cand"], "task_id": item["task_id"], "moved": moved,
                          "suffix": SUFFIX})
        status = "aside"
    if status == "aside":
        # score() returns an existing metrics.json as is: accept it only if the dir holds THIS re-probe's outputs (a
        # crash after score() wrote them), never one another tool wrote from the original probe meanwhile
        if (scored / "metrics.json").exists() and (bad := ours(scored, records)):
            raise regrade.FixError(f"{scored}: a metrics.json appeared while the task was aside, and rollouts {bad} are "
                                   "not this re-probe's; a person looks")
        R.say(f"{item['run']} {item['cand']}: score() with sc14 of {old_probe.name} + kimi of {probe_dir.name}")
        metrics = await task_builder.score(b, task, records, scored)
        state.update(status="scored", scored_at=R.now_iso(),
                     new_metrics={k: metrics.get(k) for k in ("student_rate", "teacher_clean_rate", "valid", "accept",
                                                              "reason", "similarity", "cheated")})
        save_sidecar(scored, state)
        log_action(root, {"action": "score", "cand": item["cand"], "task_id": item["task_id"],
                          "metrics": state["new_metrics"]})
        status = "scored"
    metrics = read_json(scored / "metrics.json")
    if not isinstance(metrics, dict) or (bad := ours(scored, records)):
        raise regrade.FixError(f"{scored}: state scored, but metrics.json/rollouts are not this re-probe's")
    if metrics.get("accept"):
        await task_builder.accept(b, metrics)
        log_action(root, {"action": "accept", "cand": item["cand"], "task_id": item["task_id"]})
    bookkeeping(root, round_dir, item, metrics)
    audit = reset_audit(item)
    log_action(root, {"action": "audit_reset", "cand": item["cand"], "task_id": item["task_id"], "done": audit})
    state.update(status="done", done_at=R.now_iso())
    save_sidecar(scored, state)
    return "accepted" if metrics.get("accept") else "scored"


async def run(args) -> int:
    workdir = Path(args.workdir).resolve()
    selection = read_json(workdir / "selection.json")
    if not isinstance(selection, dict) or not selection.get("tasks"):
        raise SystemExit(f"no {workdir / 'selection.json'}: run prepare first")
    tasks = selection["tasks"]
    probe_dir = workdir / f"probe-{selection['key']}"
    roots = sorted({Path(t["root"]) for t in tasks})

    def states() -> dict:
        return {t["task_id"]: sidecar(Path(t["root"]) / t["round"] / t["cand"]).get("status") for t in tasks}

    now = states()
    R.say(f"{len(tasks)} task(s); states {json.dumps({s or 'new': list(now.values()).count(s) for s in set(now.values())})}"
          f"; probe {probe_dir}")
    if all(s == "done" for s in now.values()):
        R.say("every task is done")
        return 0
    if args.dry_run:  # offline: no hub call (the budget reading is a glm call)
        R.say(f"dry run: active hub key {hub_key_name()} (budget not read)")
        for t in tasks:
            R.say(f"  {t['run']} {t['round']} {t['cand']} {t['task_id']} sc14 {t['student_rate']} valid {t['valid']} "
                  f"state {now[t['task_id']] or 'new'}")
        return 0
    guard = R.SpendGuard(roots, False)
    guard_lock = asyncio.Lock()

    async def budget_state() -> str:  # hub_budget.py makes a hub call: off the event loop, one reading at a time
        async with guard_lock:
            return await asyncio.to_thread(guard.allow)

    with R.locked_roots(roots):  # one writer per build root: rescore_unscored, regrade and a second run refuse
        complete = (probe_dir / "manifest.json").exists() and rollouts.probe_complete(probe_dir)
        if not complete:
            used = sorted(task_id for task_id, s in states().items() if s in TAKEN)
            if used:  # a rebuilt campaign would pay for rollouts those tasks never use
                raise SystemExit(f"{probe_dir} is not complete, but {len(used)} task(s) were already processed with "
                                 f"it ({used[:3]}...): a person looks")
            while (state := await budget_state()) == "wait":  # the hub's glm is down: wait, fail closed
                await asyncio.sleep(60)
            if state == "stop":
                R.say(f"budget guard says stop before the probe: {json.dumps(guard.last)}")
                return 1
        R.say(f"probe: {len(tasks)} task(s) x {REPEATS} {TEACHER} in {probe_dir} ({'complete' if complete else 'running'})")
        new_records = await rollouts.probe(probe_dir, [Path(t["task_dir"]) for t in tasks], models=(TEACHER,),
                                           repeats=REPEATS)
        if not complete:
            for root in roots:
                log_action(root, {"action": "probe", "probe": str(probe_dir), "tasks": sorted(new_records)})
        semaphore = asyncio.Semaphore(args.concurrency)
        results = {}

        async def one(item):
            async with semaphore:
                status = sidecar(Path(item["root"]) / item["round"] / item["cand"]).get("status")
                if status not in ("scored", "done"):  # finishing a scored task needs no hub spend
                    while (state := await budget_state()) == "wait":
                        await asyncio.sleep(60)
                    if state == "stop":
                        return item["task_id"], "not started (budget)"
                try:
                    return item["task_id"], await process(item, new_records, probe_dir)
                except Exception as error:  # noqa: BLE001 - rerun resumes from the sidecar
                    message = f"{type(error).__name__}: {str(error)[-400:]}"
                    R.say(f"{item['cand']} FAILED (rerun resumes): {message}")
                    log_action(Path(item["root"]), {"action": "failed", "cand": item["cand"],
                                                    "task_id": item["task_id"], "error": message})
                    return item["task_id"], f"failed: {message[:160]}"

        for task_id, outcome in await asyncio.gather(*(one(t) for t in tasks)):
            results[task_id] = outcome
    reset = [{"key": f"{t['run']}/{t['cand']}", "run": t["run"], "task_id": t["task_id"]} for t in tasks
             if states()[t["task_id"]] == "done"]
    write_json(workdir / "audit_reset.json", {"at": R.now_iso(), "tasks": reset})
    counts = {}
    for outcome in results.values():
        counts[outcome.split(":", 1)[0]] = counts.get(outcome.split(":", 1)[0], 0) + 1
    R.say(f"done: {json.dumps(counts, sort_keys=True)}; budget {json.dumps(guard.last)}")
    for task_id, outcome in sorted(results.items()):
        if not outcome.startswith(("scored", "accepted")):
            R.say(f"  {task_id}: {outcome}")
    R.say(f"{len(reset)} task(s) need a new failure-match audit ({workdir / 'audit_reset.json'}): the official4 audit loop "
          "picks them up while it runs; otherwise relaunch it WITHOUT these keys in args.seen")
    return 0 if all(o.startswith(("scored", "accepted")) for o in results.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("list")
    sub.add_argument("--runs", nargs="+", required=True)
    sub.add_argument("--max-student", type=float, default=0.2)
    sub = commands.add_parser("prepare")
    sub.add_argument("--runs", nargs="+", required=True)
    sub.add_argument("--workdir", required=True)
    sub.add_argument("--max-student", type=float, default=0.2)
    sub = commands.add_parser("run")
    sub.add_argument("--workdir", required=True)
    sub.add_argument("--concurrency", type=int, default=3)
    sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        task_builder.judge_pass = R._REAL_JUDGE  # rescore_unscored's verdict-reuse wrapper is not used here
        return asyncio.run(run(args))
    tasks = [t for run_name in args.runs for t in selectable(run_name, args.max_student)]
    if args.command == "list":
        for t in tasks:
            print(json.dumps(t))
        print(json.dumps({"total": len(tasks), "key": probe_key(tasks) if tasks else None}))
        return 0
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    existing = read_json(workdir / "selection.json")
    if isinstance(existing, dict):
        if sorted(t["task_id"] for t in existing["tasks"]) != sorted(t["task_id"] for t in tasks):
            raise SystemExit(f"{workdir / 'selection.json'} holds another task set; it is frozen (move it aside first)")
        print(f"{workdir / 'selection.json'} already holds these {len(tasks)} task(s)")
        return 0
    write_json(workdir / "selection.json", {"created": R.now_iso(), "tool": TOOL, "max_student": args.max_student,
                                             "runs": args.runs, "key": probe_key(tasks), "tasks": tasks})
    print(f"wrote {workdir / 'selection.json'}: {len(tasks)} task(s), probe-{probe_key(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
