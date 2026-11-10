"""Failure follow-up (hengyuf, 2026-10-28): does a newer student checkpoint still show the failure patterns that the
earlier failure analysis (failure_analyzer.py) found on a DeepSWE target, and does it show new ones?

For each target, codex gets the task (instruction, hidden tests, reference solution), the earlier analysis
(failure_analysis.md, result.json, evidence/) with the exact rollouts it studied (5 sc14 + 5 kimi slots), and the new
checkpoint's 5 rollouts on the same task from a DeepSWE eval run (rendered like the target rollouts, slots
<label>-r<k>). It also gets a sandbox started from the task image. It writes followup.md + result.json: a status
for every earlier pattern (resolved | reduced | persists | not_observable), new patterns, and a per-slot primary
cause.

usage (CPU job, cwd synth/; see run_babysit.sbatch):
  followup_analyzer.py --campaign etp10-20261028 --label etp10 --eval-run <DeepSWE eval run dir>
                       [--tasks covered | --tasks a,b,...] [--concurrency 8] [--prepare-only]
  followup_analyzer.py --campaign etp10-20261028 --summarize
Outputs under synth/followup/<campaign>/:
  plan.json      target -> earlier analysis dir (the v2 SFT corpus run that analysed it)
  records/       <task>.jsonl, the new checkpoint's records
  <task_id>/     inputs/, bin/, .agent/, followup.md, result.json
  summary.json   from --summarize
Nothing outside synth/followup/<campaign>/ is written. failure_analyzer.py and the earlier analyses are only read.
"""
from common import CHEAT_CATEGORIES, DEEPSWE, RUNS, STUDENT, SYNTH, TEACHER, log, read_json, render, task_toml, write_json

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

from llm_engine import agent
from rollouts import pass_stats, render as render_record
from sandbox import owned, write_wrappers

FOLLOWUP = SYNTH / "followup"
BAIHE_RUNS = Path("/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu/synth/runs")  # read-only
CORPUS_META = SYNTH.parent / "sft/corpus/kimi-pass-v2-20261023/metadata.jsonl"
STATUSES = ("resolved", "reduced", "persists", "not_observable")
OUTCOMES = ("pass", "fail", "excluded")
TASK_ID = re.compile(rb'"task_id":\s*"([^"]+)"')
INDEX_HEADER = ["| slot | status | reward | F2P | P2P | tool calls | patch chars | agent error |",
                "|---|---|---|---|---|---|---|---|"]


def _reformat():
    """tools/reformat_rollouts.py (the target dataset builder) as a module: its eval-row -> record helpers."""
    spec = importlib.util.spec_from_file_location("synth_reformat_rollouts", SYNTH / "tools/reformat_rollouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_plan() -> dict:
    """target -> {source, run, analysis_dir, corpus_runs}: the DeepSWE targets of the v2 SFT corpus and, for each, the
    earlier analysis of the corpus run with the most kept trajectories (the run whose synthetic tasks came from it)."""
    runs = defaultdict(Counter)
    for line in CORPUS_META.read_text().splitlines():
        meta = json.loads(line)
        if (meta.get("corpus") or {}).get("dropped"):
            continue
        runs[meta["deepswe_target"], meta["source"]][meta["run"]] += 1
    plan = {}
    for (target, source), counter in sorted(runs.items()):
        if target in plan:
            raise ValueError(f"{target} has corpus runs from two sources")
        root = RUNS if source == "hengyuf" else BAIHE_RUNS
        ranked = [run for run, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
        chosen = next((run for run in ranked if (root / run / "analysis" / target / "result.json").exists()), None)
        if chosen is None:
            raise ValueError(f"no earlier analysis for {target} in {ranked}")
        plan[target] = {"source": source, "run": chosen, "analysis_dir": str(root / chosen / "analysis" / target),
                        "corpus_runs": dict(counter)}
    return plan


def eval_rows(eval_run: Path, tasks) -> dict:
    """{(task_id, rollout): (row, provenance)} for the wanted tasks; the last row of a key wins (resumes append)."""
    rows = {}
    for path in sorted(eval_run.glob("shards/shard_*/rollouts.jsonl")):
        with path.open("rb") as stream:
            offset = 0
            for raw in stream:
                start, offset = offset, offset + len(raw)
                match = TASK_ID.search(raw)
                if not match or match.group(1).decode() not in tasks:
                    continue
                row = json.loads(raw)
                if row.get("task_id") in tasks:
                    rows[row["task_id"], row["_ng_rollout_index"]] = (row, {
                        "file": str(path), "offset": start, "length": len(raw),
                        "line_sha256": hashlib.sha256(raw).hexdigest(),
                        "opencode_results_fpath": row.get("opencode_results_fpath")})
    return rows


def new_records(eval_run: Path, label: str, tasks) -> dict:
    """task -> the new checkpoint's records (target-record schema, model = label), built like sc14's."""
    reformat = _reformat()
    by_task = defaultdict(list)
    for (task_id, rollout), (row, source) in sorted(eval_rows(eval_run, tasks).items()):
        evaluated = row.get("evaluation_completed") is True
        status = ("pass" if row.get("reward") == 1 else "fail") if evaluated else "infra"
        failing, available = [], evaluated
        if evaluated:
            try:
                failing = reformat.failing_tests(row)
            except ValueError as error:  # keep the slot; its failing-test names are just unknown
                log(f"{task_id} r{rollout}: failing tests unavailable ({error})")
                available = False
        by_task[task_id].append(reformat._record(label, rollout, status, row, failing,
                                                 reformat.student_agent_error(row), None, row["_ng_task_index"],
                                                 available, source))
    return by_task


def write_inputs(out_dir: Path, task_id: str, analysis_dir: Path, records: list) -> dict:
    """inputs/: task files, the earlier analysis, its rollouts plus the new slots, a combined index and stats."""
    inputs = out_dir / "inputs"
    if inputs.exists():
        shutil.rmtree(inputs)
    (inputs / "tests").mkdir(parents=True)
    task_dir = DEEPSWE / task_id
    shutil.copy2(task_dir / "instruction.md", inputs / "instruction.md")
    for name in ("test.patch", "config.json", "test.sh"):
        shutil.copy2(task_dir / "tests" / name, inputs / "tests" / name)
    shutil.copy2(task_dir / "solution/solution.patch", inputs / "solution.patch")
    same = {name: ((analysis_dir / "inputs" / name).read_bytes() == (inputs / name).read_bytes()
                   if (analysis_dir / "inputs" / name).exists() else None)
            for name in ("instruction.md", "solution.patch", "tests/test.patch", "tests/config.json", "tests/test.sh")}
    previous = inputs / "previous"
    previous.mkdir()
    for name in ("failure_analysis.md", "result.json"):
        shutil.copy2(analysis_dir / name, previous / name)
    if (analysis_dir / "evidence").is_dir():
        shutil.copytree(analysis_dir / "evidence", previous / "evidence")
    rollouts = inputs / "rollouts"
    shutil.copytree(analysis_dir / "inputs/rollouts", rollouts)
    old_rows = [line for line in (rollouts / "index.md").read_text().splitlines()[2:] if line.startswith("| ")]
    new_rows = []
    for record in records:
        (rollouts / f"{record['slot']}.md").write_text(render_record(record))
        (rollouts / f"{record['slot']}.patch").write_text(record["patch"])
        calls = sum(item.get("type") == "function_call" for item in record["output"])
        new_rows.append(f"| {record['slot']} | {record['status']} | {record['reward']} | {record['f2p']} | "
                        f"{record['p2p']} | {calls} | {len(record['patch'])} | {record['agent_error'] or ''} |")
    (rollouts / "index.md").write_text("\n".join(INDEX_HEADER + old_rows + new_rows) + "\n")
    earlier = read_json(analysis_dir / "inputs/stats.json", {}) or {}
    write_json(inputs / "stats.json", {"pass_stats": {**(earlier.get("pass_stats") or {}), **pass_stats(records)},
                                       "excluded_by_construction": (earlier.get("excluded_by_construction") or [])
                                       + [{"slot": r["slot"], "status": r["status"]} for r in records
                                          if r["status"] not in ("pass", "fail")]})
    return {"task_files_match_earlier_inputs": same}


def _index_slots(out_dir: Path) -> set:
    index = out_dir / "inputs/rollouts/index.md"
    return {line.split("|")[1].strip() for line in index.read_text().splitlines()[2:] if line.startswith("| ")}


def _check(previous_names, new_slots, usable, out_dir: Path):
    def check(result):
        problems = []
        previous = result.get("previous_patterns")
        if not isinstance(previous, list) or not all(isinstance(p, dict) for p in previous):
            return ["previous_patterns must be a list of objects"]
        names = [p.get("name") for p in previous]
        if sorted(map(str, names)) != sorted(previous_names):
            problems.append(f"previous_patterns must list each earlier pattern exactly once: expected "
                            f"{sorted(previous_names)}, got {names}")
        excluded = {item.get("slot") for item in result.get("excluded") or [] if isinstance(item, dict)}
        graded = usable - excluded
        for pattern in previous:
            missing = {"name", "status", "new_slots", "frequency", "evidence"} - set(pattern)
            if missing:
                problems.append(f"previous pattern {pattern.get('name')!r} lacks {sorted(missing)}")
            if pattern.get("status") not in STATUSES:
                problems.append(f"previous pattern {pattern.get('name')!r}: status must be one of {STATUSES}")
            unknown = set(pattern.get("new_slots") or []) - graded
            if unknown:
                problems.append(f"previous pattern {pattern.get('name')!r} cites ungraded or unknown slots {sorted(unknown)}")
        new = result.get("new_patterns")
        if not isinstance(new, list) or not all(isinstance(p, dict) for p in new):
            problems.append("new_patterns must be a list of objects (possibly empty)")
            new = []
        for pattern in new:
            missing = {"name", "category", "description", "new_slots", "frequency", "decisive", "failing_tests",
                       "seen_in_earlier_student", "seen_in_teacher", "evidence"} - set(pattern)
            if missing:
                problems.append(f"new pattern {pattern.get('name')!r} lacks {sorted(missing)}")
            if "decisive" in pattern and not isinstance(pattern["decisive"], bool):
                problems.append(f"new pattern {pattern.get('name')!r}: decisive must be true or false")
            unknown = set(pattern.get("new_slots") or []) - graded
            if unknown:
                problems.append(f"new pattern {pattern.get('name')!r} cites ungraded or unknown slots {sorted(unknown)}")
        outcomes = result.get("slot_outcomes")
        if not isinstance(outcomes, list) or not all(isinstance(o, dict) for o in outcomes):
            problems.append("slot_outcomes must be a list of objects")
        else:
            slots = [o.get("slot") for o in outcomes]
            if sorted(map(str, slots)) != sorted(new_slots):
                problems.append(f"slot_outcomes must cover each new slot once: expected {sorted(new_slots)}, got {slots}")
            for outcome in outcomes:
                if outcome.get("status") not in OUTCOMES or not str(outcome.get("primary_cause") or "").strip():
                    problems.append(f"slot_outcomes {outcome.get('slot')!r}: status in {OUTCOMES} and a primary_cause")
        unknown = excluded - _index_slots(out_dir)
        if unknown:
            problems.append(f"excluded lists slots that are not in rollouts/index.md: {sorted(unknown)}")
        if not isinstance(result.get("summary"), str) or not result["summary"].strip():
            problems.append("summary missing")
        report = out_dir / "followup.md"
        if not report.exists() or report.stat().st_size < 1000:
            problems.append("followup.md missing or too short")
        return problems
    return check


async def followup(task_id: str, entry: dict, records: list, out_dir: Path, *, label: str, campaign: str) -> dict:
    analysis_dir = Path(entry["analysis_dir"])
    previous_names = [str(p["name"]) for p in read_json(analysis_dir / "result.json")["patterns"]]
    new_slots = [r["slot"] for r in records]
    usable = {r["slot"] for r in records if r["status"] in ("pass", "fail")}
    check = _check(previous_names, new_slots, usable, out_dir)
    done = read_json(out_dir / "result.json")
    if isinstance(done, dict) and not check(done):
        return done
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = write_inputs(out_dir, task_id, analysis_dir, records)
    write_json(out_dir / "provenance.json", {"entry": entry, **provenance, "new_slots": new_slots})
    config = task_toml(DEEPSWE / task_id)
    async with owned(config["environment"]["docker_image"], run=f"followup-{campaign}", purpose="failure-followup") as box:
        write_wrappers(out_dir / "bin", box.sid)
        prompt = render("failure_followup", task_id=task_id, student=STUDENT, teacher=TEACHER, new=label,
                        base_commit=config["metadata"]["base_commit_hash"], cheat_categories=CHEAT_CATEGORIES,
                        sandbox=render("_sandbox", network_note="(no network)"))
        return await agent(prompt, out_dir, backend="codex", check=check)


def summarize(campaign_dir: Path) -> dict:
    plan = read_json(campaign_dir / "plan.json")
    rows, totals, new_counter = [], Counter(), Counter()
    for task_id, entry in sorted(plan["targets"].items()):
        result = read_json(campaign_dir / task_id / "result.json")
        if not isinstance(result, dict):
            rows.append({"task_id": task_id, "missing": True})
            continue
        statuses = Counter(p.get("status") for p in result.get("previous_patterns") or [])
        totals.update(statuses)
        for pattern in result.get("new_patterns") or []:
            new_counter[str(pattern.get("category"))] += 1
        rows.append({"task_id": task_id, "source": entry["source"], "earlier_run": entry["run"],
                     "previous": {p.get("name"): p.get("status") for p in result.get("previous_patterns") or []},
                     "status_counts": dict(statuses),
                     "new_patterns": [{"name": p.get("name"), "category": p.get("category"),
                                       "frequency": p.get("frequency"), "decisive": p.get("decisive"),
                                       "seen_in_earlier_student": p.get("seen_in_earlier_student")}
                                      for p in result.get("new_patterns") or []],
                     "slot_outcomes": result.get("slot_outcomes"), "summary": result.get("summary")})
    summary = {"campaign": campaign_dir.name, "targets": len(plan["targets"]),
               "analysed": sum(1 for r in rows if not r.get("missing")), "previous_status_totals": dict(totals),
               "new_pattern_categories": dict(new_counter), "rows": rows}
    write_json(campaign_dir / "summary.json", summary)
    return summary


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--label")
    parser.add_argument("--eval-run", type=Path)
    parser.add_argument("--tasks", default="covered", help="'covered' (all v2-corpus targets) or comma-separated ids")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true", help="write plan, records and inputs; run no agent")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    campaign_dir = FOLLOWUP / args.campaign
    if args.summarize:
        summary = summarize(campaign_dir)
        log(f"summary: {summary['analysed']}/{summary['targets']} analysed; {summary['previous_status_totals']}")
        return
    if not (args.label and args.eval_run):
        parser.error("--label and --eval-run are required")
    plan = corpus_plan()
    tasks = sorted(plan) if args.tasks == "covered" else args.tasks.split(",")
    unknown = set(tasks) - set(plan)
    if unknown:
        parser.error(f"not v2-corpus targets: {sorted(unknown)}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    plan_file = {"label": args.label, "eval_run": str(args.eval_run.resolve()),
                 "targets": {task: plan[task] for task in sorted(plan) if task in tasks}}
    previous_plan = read_json(campaign_dir / "plan.json")
    if previous_plan and (previous_plan.get("label"), previous_plan.get("eval_run")) != (plan_file["label"], plan_file["eval_run"]):
        raise SystemExit(f"{campaign_dir} belongs to {previous_plan.get('label')} / {previous_plan.get('eval_run')}")
    if previous_plan:
        plan_file["targets"] = {**previous_plan.get("targets", {}), **plan_file["targets"]}
    write_json(campaign_dir / "plan.json", plan_file)
    records = new_records(args.eval_run.resolve(), args.label, set(tasks))
    (campaign_dir / "records").mkdir(exist_ok=True)
    for task_id in tasks:
        if len(records.get(task_id, [])) != 5:
            raise SystemExit(f"{task_id}: {len(records.get(task_id, []))} rollouts in {args.eval_run}, expected 5")
        lines = [json.dumps(record, ensure_ascii=True) for record in records[task_id]]
        (campaign_dir / "records" / f"{task_id}.jsonl").write_text("\n".join(lines) + "\n")
    if args.prepare_only:
        for task_id in tasks:
            out_dir = campaign_dir / task_id
            out_dir.mkdir(exist_ok=True)
            provenance = write_inputs(out_dir, task_id, Path(plan[task_id]["analysis_dir"]), records[task_id])
            log(f"prepared {task_id}: {provenance}")
        return
    gate = asyncio.Semaphore(args.concurrency)

    async def one(task_id):
        async with gate:
            try:
                result = await followup(task_id, plan[task_id], records[task_id], campaign_dir / task_id,
                                        label=args.label, campaign=args.campaign)
                log(f"followup {task_id}: " + ", ".join(f"{p['name'][:40]}={p['status']}"
                                                         for p in result["previous_patterns"])
                    + f"; {len(result['new_patterns'])} new")
                return True
            except Exception as error:
                log(f"followup {task_id} FAILED: {type(error).__name__}: {error}")
                return False

    results = await asyncio.gather(*(one(task) for task in tasks))
    summary = summarize(campaign_dir)
    log(f"summary: {summary['analysed']}/{summary['targets']} analysed; {summary['previous_status_totals']}")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
