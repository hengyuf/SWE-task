"""Failure analyzer: why does the student fail a task the teacher solves? Codex reads the rollouts on the
controller and runs repository commands in a sandbox started from the task image -> failure_analysis.md/json.
Also hosts the two single-shot judges used by the task builder (cheated-pass audit, failure similarity).
"""
from common import (CHEAT_CATEGORIES, DEEPSWE, JUDGE, RUNS, STUDENT, TARGETS, TEACHER, log, read_json, render,
                    task_toml, write_json)

import argparse
import asyncio
import json
from pathlib import Path
import shutil

from llm_engine import agent, chat
from rollouts import SHOW_RULES, pass_stats, render as render_record, shown_records, target_records, write_rollouts
from sandbox import owned, write_wrappers


def _check(records, out_dir, shown=None):
    """result.json check; `shown` (default: every record) are the slots the analyst was given: student_slots must be
    graded slots among them, while "excluded" may name any slot of the task."""
    slots = {record["slot"] for record in records}
    usable = {record["slot"] for record in (records if shown is None else shown) if record["status"] in ("pass", "fail")}

    def check(result):
        problems = []
        patterns = result.get("patterns")
        if not isinstance(patterns, list) or not all(isinstance(p, dict) for p in patterns):
            return ["result.patterns must be a list of objects"]
        for pattern in patterns:
            missing = {"name", "category", "description", "student_slots", "teacher_contrast"} - set(pattern)
            if missing:
                problems.append(f"pattern {pattern.get('name')!r} lacks {sorted(missing)}")
            unknown = set(pattern.get("student_slots") or []) - usable
            if unknown:
                problems.append(f"pattern {pattern.get('name')!r} cites unusable slots {sorted(unknown)}")
        if not isinstance(result.get("transferable_signature"), str) or not result["transferable_signature"].strip():
            problems.append("transferable_signature missing")
        for item in result.get("excluded") or []:
            if item.get("slot") not in slots:
                problems.append(f"excluded slot {item.get('slot')!r} does not exist")
        report = Path(out_dir) / "failure_analysis.md"
        if not report.exists() or report.stat().st_size < 1000:
            problems.append("failure_analysis.md missing or too short")
        return problems
    return check


def write_inputs(out_dir: Path, task_dir: Path, records, shown=None, source=None) -> None:
    """inputs/ for the analyst. With a selection (`shown`, from target_source), rollouts/ holds the shown records only,
    while stats.json "pass_stats" still counts every record (task_builder reads the target's pass rates from it)."""
    inputs = out_dir / "inputs"
    (inputs / "tests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_dir / "instruction.md", inputs / "instruction.md")
    for name in ("test.patch", "config.json", "test.sh"):
        shutil.copy2(task_dir / "tests" / name, inputs / "tests" / name)
    shutil.copy2(task_dir / "solution/solution.patch", inputs / "solution.patch")
    if shown is not None:  # a selection: no transcript of an earlier, unfinished attempt may stay next to it
        shutil.rmtree(inputs / "rollouts", ignore_errors=True)
    write_rollouts(inputs / "rollouts", records if shown is None else shown)
    stats = {"pass_stats": pass_stats(records),
             "excluded_by_construction": [{"slot": r["slot"], "status": r["status"]}
                                          for r in records if r["status"] not in ("pass", "fail")]}
    if shown is not None:
        shown_slots = {r["slot"] for r in shown}
        # the dataset path stays out of the analyst's inputs (runs/<run>/target_source.json records it)
        stats.update(selection={"show": (source or {}).get("show")},
                     shown={"slots": [r["slot"] for r in shown], "pass_stats": pass_stats(shown)},
                     not_shown=[{"slot": r["slot"], "status": r["status"]} for r in records
                                if r["slot"] not in shown_slots and r["status"] in ("pass", "fail")],
                     origins={r["slot"]: r["origin"]["run"] for r in records if isinstance(r.get("origin"), dict)})
    write_json(inputs / "stats.json", stats)


def _slot_list(slots) -> str:
    return ", ".join(sorted(slots, key=lambda slot: (slot.rsplit("-r", 1)[0], int(slot.rsplit("-r", 1)[1])
                                                     if slot.rsplit("-r", 1)[-1].isdigit() else 0, slot)))


def selection_note(records, shown, show: str) -> str:
    """The prompt's account of a target selection; "" for "all", so that prompt is exactly the original one."""
    if show == "all":
        return ""
    total, seen = pass_stats(records), pass_stats(shown)
    s_total, t_total = total.get(STUDENT) or {}, total.get(TEACHER) or {}
    s_fails, t_passes = (seen.get(STUDENT) or {}).get("fail", 0), (seen.get(TEACHER) or {}).get("pass", 0)
    runs = {}
    for record in shown:
        origin = record["origin"] if isinstance(record.get("origin"), dict) else {}
        runs.setdefault((record["model"] != STUDENT, origin.get("run", "unknown"), origin.get("context_window"),
                         origin.get("per_call_output_cap")), []).append(record["slot"])
    sources = "\n".join(
        f"- {_slot_list(slots)}: run {run}" + (f", OpenCode context window {window:,} tokens" if window else "")
        + (f", per-call output cap {cap:,} tokens" if cap else "")
        for (_, run, window, cap), slots in sorted(runs.items(), key=lambda item: item[0][:2]))
    return (' They count every graded rollout of the task, shown or not (see "Rollout selection").\n\n'
            "## Rollout selection\n"
            f"`rollouts/` holds a selection, not every rollout: the {t_passes} {TEACHER} rollouts that PASSED "
            f"({t_total.get('pass', 0)} of its {t_total.get('pass', 0) + t_total.get('fail', 0)} graded rollouts "
            f"passed) and the {s_fails} {STUDENT} rollouts that FAILED ({s_total.get('fail', 0)} of its "
            f"{s_total.get('pass', 0) + s_total.get('fail', 0)} graded rollouts failed). The {STUDENT} passes, the "
            f"{TEACHER} failures and the ungraded slots are not shown (`stats.json`: \"not_shown\", "
            f"\"excluded_by_construction\"). Every {STUDENT} slot you read is therefore a failure: give a pattern's "
            f"frequency as k of the {s_fails} failing {STUDENT} slots. The shown slots come from these runs (same "
            f"OpenCode harness and task):\n{sources}\n"
            "An OpenCode context compaction shows in a transcript as an empty `### [n] user` item (OpenCode's request "
            "for a summary), then the model's own summary (a `### [n] thinking` item and usually an assistant item "
            "starting `## Goal`), then a `### [n] user` item that OpenCode adds (\"Continue if you have next steps, or "
            "stop and ask for clarification if you are unsure how to proceed.\"): the session had reached its context "
            "window, and from there on the model works from its summary instead of the full earlier history. A NUL "
            "byte in a tool output is shown as \u2400.\n"
            "Work only from ./inputs, your working directory and the sandbox: do not look elsewhere on the shared "
            "filesystem for the rollouts that are not shown.\n")


async def analyze(task_dir, image: str, records, out_dir, *, run: str, show: str = "all", source=None) -> dict:
    """failure_analysis.md + result.json for one task; skipped when a validated result already exists. `show` (one of
    rollouts.SHOW_RULES) selects the records the analyst sees; the pass counts always cover every record."""
    task_dir, out_dir = Path(task_dir), Path(out_dir)
    shown = shown_records(records, show)
    if show != "all" and not any(r["model"] == STUDENT for r in shown):
        raise ValueError(f"{task_dir.name}: no {STUDENT} rollout to analyse under {show!r} ({pass_stats(records)})")
    check = _check(records, out_dir, None if show == "all" else shown)
    # an analysis directory made under another selection is never reused or redone in place (e.g. target_source.json
    # added to a run that already has an analysis of every slot): a person moves it aside
    stats = read_json(out_dir / "inputs/stats.json")
    wanted = None if show == "all" else {"show": show}
    if isinstance(stats, dict) and stats.get("selection") != wanted:
        raise RuntimeError(f"{out_dir}: the existing analysis was made for selection {stats.get('selection')!r}, not "
                           f"{wanted!r}; move it aside first")
    if show != "all" and not isinstance(stats, dict) and any(
            (out_dir / name).exists() for name in ("result.json", "failure_analysis.md")):
        raise RuntimeError(f"{out_dir}: an analysis without inputs/stats.json exists; move it aside first")
    done = read_json(out_dir / "result.json")
    if isinstance(done, dict) and not check(done):
        return done
    write_inputs(out_dir, task_dir, records, None if show == "all" else shown, source)
    config = task_toml(task_dir)
    async with owned(image, run=run, purpose="failure-analysis") as box:
        write_wrappers(out_dir / "bin", box.sid)
        prompt = render("failure_analysis", task_id=config["metadata"]["task_id"], student=STUDENT, teacher=TEACHER,
                        base_commit=config["metadata"]["base_commit_hash"], cheat_categories=CHEAT_CATEGORIES,
                        sandbox=render("_sandbox", network_note="(no network)"),
                        rollout_selection=selection_note(records, shown, show))
        return await agent(prompt, out_dir, backend="codex", check=check)


async def judge_pass(record: dict, instruction: str, rubric: str) -> dict:
    """Audit one passing rollout for cheating and score its process against the task rubric."""
    transcript = render_record(record)
    if len(transcript) > 400_000:
        transcript = transcript[:200_000] + "\n…[middle of the transcript omitted]…\n" + transcript[-200_000:]
    reply = await chat(JUDGE, [{"role": "user", "content": render(
        "judge_pass", instruction=instruction, rubric=rubric, cheat_categories=CHEAT_CATEGORIES,
        transcript=transcript)}], json_mode=True)
    if not isinstance(reply, dict) or not isinstance(reply.get("cheated"), bool):
        raise ValueError(f"judge_pass returned an invalid verdict for {record['slot']}")
    return reply


async def similarity(target_dir, synthetic_dir) -> dict:
    """Score (0-10) how well the student's failures on a synthetic task match those on the target task."""
    target_dir, synthetic_dir = Path(target_dir), Path(synthetic_dir)
    reply = await chat(JUDGE, [{"role": "user", "content": render(
        "judge_similarity",
        target_md=(target_dir / "failure_analysis.md").read_text(),
        target_json=json.dumps(read_json(target_dir / "result.json")),
        synthetic_md=(synthetic_dir / "failure_analysis.md").read_text(),
        synthetic_json=json.dumps(read_json(synthetic_dir / "result.json")))}], json_mode=True)
    if not isinstance(reply, dict) or not isinstance(reply.get("score"), (int, float)):
        raise ValueError("similarity judge returned no numeric score")
    return reply


TARGET_SOURCE = "target_source.json"


def target_source(run: str) -> dict:
    """The target rollouts of a run's target analysis: runs/<run>/target_source.json {"dataset": <target dataset dir>,
    "show": one of rollouts.SHOW_RULES} (synth/queue4 writes it before a build starts; hengyuf, 2026-11-02), else the
    original setup: TARGETS, every slot shown. A file that is present but unusable is an error, never the default."""
    path = RUNS / run / TARGET_SOURCE
    if not path.exists():
        return {"dataset": str(TARGETS), "show": "all"}
    source = json.loads(path.read_text())
    if not isinstance(source, dict) or source.get("show") not in SHOW_RULES or not source.get("dataset") \
            or not (Path(source["dataset"]) / "COMPLETE.json").is_file():
        raise ValueError(f"{path}: not a usable target source: {source!r}")
    return source


async def analyze_target(task_id: str, run: str) -> dict:
    task_dir = DEEPSWE / task_id
    out_dir = RUNS / run / "analysis" / task_id
    source = target_source(run)
    records = target_records(task_id, dataset=source["dataset"])
    if source["show"] == "all":
        log(f"analyze {task_id}: {pass_stats(records)}")
    else:
        log(f"analyze {task_id}: {pass_stats(records)}; shown ({source['show']}, {source['dataset']}): "
            f"{pass_stats(shown_records(records, source['show']))}")
    return await analyze(task_dir, task_toml(task_dir)["environment"]["docker_image"], records, out_dir, run=run,
                         show=source["show"], source=source)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tasks", required=True, help="comma-separated DeepSWE task ids")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    gate = asyncio.Semaphore(args.concurrency)

    async def one(task_id):
        async with gate:
            try:
                result = await analyze_target(task_id, args.run)
                log(f"analyze {task_id}: {len(result['patterns'])} patterns")
                return True
            except Exception as error:
                log(f"analyze {task_id} FAILED: {type(error).__name__}: {error}")
                return False

    results = await asyncio.gather(*(one(task) for task in args.tasks.split(",")))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
