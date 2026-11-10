"""Rollout records (target dataset and probe campaigns), transcript rendering, and probe launching.

A record is one rollout slot: {slot, model, status: pass|fail|infra|grader, reward, f2p, p2p, failing_tests,
agent_error, verifier_error, patch, output (Responses-API items), attempt_dir}. Target records (TARGETS) also carry
{task_id, task_index, failing_tests_available, source}.
"""
from common import ATP, SETTINGS, STUDENT, SYNTH, TARGETS, TEACHER, log, read_json, scratch, task_toml, write_json

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

ATP_STAGE_SCRIPTS = ("deepswe_stage.py", "deepswe_source.py", "deepswe_credentials.py")
TERMINAL_STATES = {"completed", "exhausted", "fatal", "needs_review"}


def task_ids() -> list:
    """DeepSWE task ids in task_index order (alphabetical, as used by the rollout campaigns)."""
    return json.loads((TARGETS / "task_ids.json").read_text())


def _record(model, repeat, status, generation, grade, attempt_dir) -> dict:
    failing = []
    reports = sorted(Path(attempt_dir).glob("grading/attempt-*/ctrf.json")) if attempt_dir else []
    if reports:
        tests = ((read_json(reports[-1], {}).get("results") or {}).get("tests")) or []
        failing = [test.get("name") for test in tests if test.get("status") != "passed"]
    return {"slot": f"{model}-r{repeat}", "model": model, "status": status, "reward": grade.get("reward"),
            "f2p": f"{grade.get('f2p_passed')}/{grade.get('f2p_total')}",
            "p2p": f"{grade.get('p2p_passed')}/{grade.get('p2p_total')}",
            "failing_tests": failing,
            "agent_error": generation.get("opencode_run_error_type") or generation.get("failure_reason"),
            "verifier_error": grade.get("verifier_error"),
            "patch": grade.get("model_patch") or generation.get("model_patch") or "",
            "output": (generation.get("response") or {}).get("output") or [],
            "attempt_dir": str(attempt_dir) if attempt_dir else None}


_INDEX_CHECKED = set()


def _check_index(dataset=None):
    """index.json must be the one COMPLETE.json recorded (checked once per process and dataset)."""
    dataset = Path(dataset or TARGETS)
    if dataset not in _INDEX_CHECKED:
        expected = read_json(dataset / "COMPLETE.json")["files"]["index.json"]["sha256"]
        if hashlib.sha256((dataset / "index.json").read_bytes()).hexdigest() != expected:
            raise ValueError(f"{dataset / 'index.json'} does not match COMPLETE.json")
        _INDEX_CHECKED.add(dataset)


def target_records(task_id: str, dataset=None) -> list:
    """The target slots of a DeepSWE task from a target dataset: TARGETS by default (5 sc14 rollouts, 5 selected kimi
    rollouts), or e.g. data/targets/sc14x10-kimix10-20261102 (10 + 10, tools/build_targets_x10.py); every line must
    match its sha256 in index.json. Refuses an unfinished dataset (no COMPLETE.json)."""
    dataset = Path(dataset or TARGETS)
    if not (dataset / "COMPLETE.json").exists():
        raise RuntimeError(f"{dataset} is incomplete (no COMPLETE.json)")
    _check_index(dataset)
    path = dataset / "records" / f"{task_id}.jsonl"
    lines = path.read_bytes().split(b"\n")
    if lines and not lines[-1]:
        lines.pop()
    expected = sorted(entry["record_sha256"] for entry in read_json(dataset / "index.json")[task_id]["records"])
    if sorted(hashlib.sha256(line).hexdigest() for line in lines) != expected:
        raise ValueError(f"{path} no longer matches {dataset / 'index.json'}")
    records = [json.loads(line) for line in lines]
    return sorted(records, key=lambda record: (record["model"] != STUDENT, record["slot"]))


# Which target rollouts a failure analyst sees (failure_analyzer.target_source). "all": every slot (the original
# setup). "teacher_pass_student_fail" (hengyuf, 2026-11-02: "放所有正确的kimi rollouts 和所有错误的base rollout"): every
# teacher rollout that passed and every student rollout that failed; the pass rates still count every graded slot.
SHOW_RULES = ("all", "teacher_pass_student_fail")


def shown_records(records, show="all") -> list:
    """The records of a SHOW_RULES selection, in their given order."""
    if show == "all":
        return list(records)
    if show == "teacher_pass_student_fail":
        return [record for record in records if (record["model"] == TEACHER and record["status"] == "pass")
                or (record["model"] == STUDENT and record["status"] == "fail")]
    raise ValueError(f"unknown rollout selection {show!r} (known: {', '.join(SHOW_RULES)})")


def _generation(attempt: Path) -> dict:
    """Regrade attempts carry no new response; use the latest attempt of the slot that has one."""
    for earlier in sorted(attempt.parent.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]), reverse=True):
        for name in ("terminal.json", "native-result.json"):
            data = read_json(earlier / name, {})
            if (data.get("response") or {}).get("output"):
                return data
    return {}


def probe_states(probe_dir) -> list:
    import deepswe_campaign as campaign
    manifest = campaign.load_manifest(probe_dir)
    return [(task, model, repeat, campaign.slot_state(probe_dir, model, task["_ng_task_index"], repeat))
            for task in manifest["tasks"] for model in manifest["settings"]["models"]
            for repeat in range(manifest["settings"]["repeats"])]


def probe_complete(probe_dir) -> bool:
    return all(state["status"] in TERMINAL_STATES for *_, state in probe_states(probe_dir))


def probe_records(probe_dir) -> dict:
    """{task_id: [records]} for a probe campaign; anything not 'completed' is excluded as infra."""
    records = {}
    for task, model, repeat, state in probe_states(probe_dir):
        attempt = Path(state["attempt_dir"]) if state["attempt_dir"] else None
        grade = read_json(attempt / "terminal.json", {}) if attempt else {}
        generation = grade if (grade.get("response") or {}).get("output") else (_generation(attempt) if attempt else {})
        status = ("pass" if grade.get("reward") == 1 else "fail") if state["status"] == "completed" else "infra"
        records.setdefault(task["task_id"], []).append(_record(model, repeat, status, generation, grade, attempt))
    return records


# Verifier infrastructure faults (hengyuf 2026-10-20, official1-wasmi c05 kimi-r3): the verifier sandbox hit I/O errors
# ("couldn't read .../registry/src/...: Input/output error (os error 5)", rustc "signal: 7, SIGBUS", even "rm: command
# not found"), so `[gate] build` failed and the slot graded F2P 0/17, P2P 0/394 -- a completed FAIL in ATP's terms, no
# retry. Such a grade says nothing about the patch. verifier_fault() names the fault when a FAIL has no F2P and no P2P
# test passing (or none reported) and its verifier output shows one of these signatures; score() (task_builder) then
# counts the slot as infra (a platform fault, not a grading failure) with that reason as the record's "verifier_fault",
# which render() shows on its own line. probe_records() itself stays ATP's view (the graded status): the tools that
# compare it with score()'s rendered rollouts (grading_repair, regrade) keep matching the tasks scored before.
VERIFIER_FAULT = re.compile(r"(?i:input/output error|no space left on device)|\(os error (?:5|28)\)|\bEIO\b|\bENOSPC\b"
                            r"|\bSIGBUS\b|\bBus error\b")
VERIFIER_LOGS = ("run.log", "stdout.log", "stderr.log")  # RUN_LOG of test.sh, then the verifier's own stdout/stderr


def _nothing_passed(fraction) -> bool:
    passed, _, _total = str(fraction).partition("/")
    return passed in ("0", "None", "")


def verifier_fault(record: dict, *, max_bytes: int = 20_000_000) -> str:
    """'' or the reason a graded FAIL is a verifier infrastructure fault (see VERIFIER_FAULT): the record's F2P and P2P
    have nothing passing, and the logs of its last grading attempt (<attempt_dir>/grading/attempt-<n>/: run.log,
    stdout.log, stderr.log) show an I/O, bus or disk-full error."""
    if record.get("status") != "fail" or not record.get("attempt_dir"):
        return ""
    if not (_nothing_passed(record.get("f2p")) and _nothing_passed(record.get("p2p"))):
        return ""
    gradings = [path for path in Path(record["attempt_dir"]).glob("grading/attempt-*")
                if path.name.split("-", 1)[1].isdigit()]
    if not gradings:
        return ""
    last = max(gradings, key=lambda path: int(path.name.split("-", 1)[1]))
    found, logs = set(), []
    for name in VERIFIER_LOGS:
        try:
            with open(last / name, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - max_bytes))
                text = stream.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        hits = {match.group(0) for match in VERIFIER_FAULT.finditer(text)}
        if hits:
            found |= hits
            logs.append(name)
    if not found:
        return ""
    return (f"verifier infrastructure fault, not a grade: {', '.join(sorted(found, key=str.lower))} in "
            f"{last.name}/{{{','.join(logs)}}} (F2P {record.get('f2p')}, P2P {record.get('p2p')})")


def _clip(text, head, tail=0) -> str:
    text = text or ""
    if len(text) <= head + tail:
        return text
    return text[:head] + f"\n…[{len(text) - head - tail} chars omitted]…\n" + (text[-tail:] if tail else "")


def render(record: dict) -> str:
    """Condensed markdown transcript of one rollout (tool outputs truncated head+tail)."""
    failing = ("unavailable" if record.get("failing_tests_available") is False  # e.g. kimi: grade counts only
               else ", ".join(record["failing_tests"][:40]) or "none")
    lines = [f"# {record['slot']} — {record['status']}",
             f"reward {record['reward']}; F2P {record['f2p']}; P2P {record['p2p']}; "
             f"agent_error {record['agent_error']}; verifier_error {record['verifier_error']}",
             "failing tests: " + failing]
    if record.get("verifier_fault"):  # score(): a graded FAIL counted as infra; other records render as before
        lines.append(f"counted as infra: {record['verifier_fault']}")
    lines += ["", "## Transcript"]
    for number, item in enumerate(record["output"], 1):
        kind = item.get("type")
        if kind == "reasoning":
            text = "\n".join(part.get("text", "") for part in item.get("summary") or [])
            lines += [f"### [{number}] thinking", _clip(text, 4000)]
        elif kind == "message":  # role "user": an OpenCode compaction summary inside the rollout
            text = "\n".join(part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict))
            lines += [f"### [{number}] {item.get('role', 'assistant')}", _clip(text, 4000)]
        elif kind == "function_call":
            lines += [f"### [{number}] call {item.get('name')}", "~~~~", _clip(item.get("arguments", ""), 3000), "~~~~"]
        elif kind == "function_call_output":
            lines += [f"### [{number}] output", "~~~~", _clip(str(item.get("output", "")), 1500, 1500), "~~~~"]
    lines += ["", f"## Final committed patch ({len(record['patch'])} chars; full copy: {record['slot']}.patch)", "~~~~diff",
              _clip(record["patch"], 30000, 5000), "~~~~"]
    # NUL bytes from tool outputs made grep/rg treat a transcript as binary and skip it (hengyuf 2026-11-02 review):
    # shown as U+2400 (a visible symbol that ordinary text does not contain)
    return ("\n".join(lines) + "\n").replace("\x00", "\u2400")


def write_rollouts(directory, records) -> None:
    """rollouts/<slot>.md for every record plus an index.md table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = ["| slot | status | reward | F2P | P2P | tool calls | patch chars | agent error |",
            "|---|---|---|---|---|---|---|---|"]
    for record in records:
        (directory / f"{record['slot']}.md").write_text(render(record))
        (directory / f"{record['slot']}.patch").write_text(record["patch"])
        calls = sum(item.get("type") == "function_call" for item in record["output"])
        rows.append(f"| {record['slot']} | {record['status']} | {record['reward']} | {record['f2p']} | "
                    f"{record['p2p']} | {calls} | {len(record['patch'])} | {record['agent_error'] or ''} |")
    (directory / "index.md").write_text("\n".join(rows) + "\n")


def pass_stats(records) -> dict:
    stats = {}
    for record in records:
        entry = stats.setdefault(record["model"], {"pass": 0, "fail": 0, "excluded": 0})
        entry["pass" if record["status"] == "pass" else "fail" if record["status"] == "fail" else "excluded"] += 1
    for entry in stats.values():
        valid = entry["pass"] + entry["fail"]
        entry["rate"] = entry["pass"] / valid if valid else None
    return stats


def dataset_row(task_dir) -> dict:
    """Gym DeepSWE JSONL row for a Harbor task directory (image must equal task.toml docker_image)."""
    config = task_toml(task_dir)
    task_id = config["metadata"]["task_id"]
    return {"image": config["environment"]["docker_image"],
            "responses_create_params": {"input": [{"role": "user",
                                                   "content": (Path(task_dir) / "instruction.md").read_text().lstrip("\n")}]},
            "split": "test", "subset": "deepswe-synth", "task_id": task_id, "verifier_metadata": {"task_id": task_id},
            "_atp_verifier_timeout": float(config["verifier"]["timeout_sec"])}


async def _run(argv, log_path, env) -> int:
    with open(log_path, "ab") as stream:
        process = await asyncio.create_subprocess_exec(*map(str, argv), stdout=stream, stderr=stream, env=env)
        return await process.wait()


PROBE_MAX_CONCURRENCY = 32  # rollouts per model per probe campaign, see probe_limits


def probe_limits(slots: int) -> tuple:
    """(max, initial) rollouts per model for one probe campaign. ATP admission needs shards (1) <= initial <= max.
    The max is capped at PROBE_MAX_CONCURRENCY (hengyuf, 2026-10-19; settings.json stays ATP's config): one ATP process per model
    drives every rollout of a probe, and at 125 concurrent rollouts it stalled, so every PTY socket in that process
    closed at once ("PTY connection closed unexpectedly", 50-70% of round-1 rollouts lost on 2026-10-19)."""
    maximum = max(1, min(slots, PROBE_MAX_CONCURRENCY))
    return maximum, min(SETTINGS["initial_concurrency_per_model"], maximum)


async def probe(probe_dir, task_dirs, *, models=(STUDENT, TEACHER), repeats=5) -> dict:
    """Roll out student/teacher on Harbor tasks through ATP's campaign code; returns probe_records.

    Needs a CPU Slurm job with >= ~3.6 h left (ATP admission). Resumable: re-running continues the campaign.
    A "local" model (sc14) needs this job's student router and >= 1 routable worker first (waits up to
    $SYNTH_STUDENT_WAIT_S, default 10800 s, but only until SLURM_JOB_END_TIME minus what ATP admission needs: agent
    timeout + largest verifier timeout + 600 s; raises at once when that point has passed); ATP's gym.yaml reads the
    router URL from SYNTH_STUDENT_BASE_URL.
    """
    probe_dir = Path(probe_dir)
    if not (probe_dir / "manifest.json").exists():
        (probe_dir / "tasks").mkdir(parents=True, exist_ok=True)
        rows = []
        for task_dir in task_dirs:
            shutil.copytree(task_dir, probe_dir / "tasks" / Path(task_dir).name, dirs_exist_ok=True)
            rows.append(dataset_row(task_dir))
        (probe_dir / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        maximum, initial = probe_limits(len(rows) * repeats)
        write_json(probe_dir / "settings.json", dict(
            SETTINGS, expected_tasks=len(rows), repeats=repeats, max_concurrency_per_model=maximum,
            initial_concurrency_per_model=initial, models={model: SETTINGS["models"][model] for model in models}))
    if (probe_dir / "manifest.json").exists():
        wanted = {task_toml(task_dir)["metadata"]["task_id"] for task_dir in task_dirs}
        if {task["task_id"] for task in read_json(probe_dir / "manifest.json")["tasks"]} != wanted:
            raise RuntimeError(f"{probe_dir} was created for a different task set")
        if probe_complete(probe_dir):
            return probe_records(probe_dir)
    local = [model for model, spec in read_json(probe_dir / "settings.json")["models"].items() if spec.get("local")]
    if local:
        from student.client import ensure_router, wait_for_workers
        # exports SYNTH_STUDENT_BASE_URL to the campaign below
        base_url = await asyncio.to_thread(ensure_router, log_dir=probe_dir)
        patience = float(os.environ.get("SYNTH_STUDENT_WAIT_S", 10800))
        end = float(os.environ.get("SLURM_JOB_END_TIME") or 0)
        if end:  # ATP admits a slot only with agent timeout + verifier timeout + 2 x 300 s grace left in the job
            rows = [json.loads(line) for line in (probe_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
            need = SETTINGS["agent_timeout_seconds"] + max(row["_atp_verifier_timeout"] for row in rows) + 600
            patience = min(patience, end - time.time() - need)
            if patience <= 0:
                raise RuntimeError(f"Probe {probe_dir}: less than {need:.0f} s left in this job; rerun in a fresh CPU job")
        log(f"probe: {'/'.join(local)} via {base_url}; waiting up to {patience:.0f} s for a routable worker")
        workers = await asyncio.to_thread(wait_for_workers, 1, timeout_s=patience)
        log(f"probe: {len(workers)} student worker(s) routable: " +
            ", ".join(f"{worker.get('job_id')} {worker.get('status')} {worker.get('base_url')}" for worker in workers))
    atp_scratch = scratch() / "atp"
    (atp_scratch / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ATP_STAGE_SCRIPTS:  # stage from a scratch copy (as deepswe.sbatch does): no writes under ATP
        shutil.copy2(ATP / "scripts" / name, atp_scratch / "scripts" / name)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ATP_", "PYTHON", "VIRTUAL_ENV"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log(f"probe: staging ATP Gym in {atp_scratch}")
    if await _run(["/usr/bin/python3", "-B", atp_scratch / "scripts/deepswe_stage.py", "--app", ATP,
                   "--scratch", atp_scratch], probe_dir / "stage.log", env):
        raise RuntimeError(f"ATP staging failed; see {probe_dir / 'stage.log'}")
    env.update(PYTHONPATH=str(atp_scratch / "scripts"), SYNTH_ATP_SCRATCH=str(atp_scratch))
    log(f"probe: running campaign {probe_dir}")
    code = await _run([atp_scratch / "controller/bin/python", "-B", SYNTH / "probe_worker.py", probe_dir],
                      probe_dir / "worker.log", env)
    if not probe_complete(probe_dir):
        raise RuntimeError(f"Probe {probe_dir} incomplete (worker exit {code}); rerun in a fresh CPU job")
    return probe_records(probe_dir)
