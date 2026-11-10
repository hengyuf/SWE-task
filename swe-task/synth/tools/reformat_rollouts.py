#!/usr/bin/env python3
"""Target rollouts for the synth pipeline: student sc14 (hengyuf's DeepSWE v1.1 eval of swonlysc-step0014, 131k output
cap, 5 rollouts per task) and teacher kimi (the 5 selected slots per task of Baihe's partial-selected super/kimi export)
-> data/targets/sc14-out131k-kimi/ (docs/port-contract.md, TARGET ROLLOUTS).

records/<task_id>.jsonl: 5 sc14 records (rollout 0-4) then 5 kimi records (ascending repeat), one JSON object per line
(ensure_ascii). A record is Baihe's rollouts._record schema {slot, model, status, reward, f2p, p2p, failing_tests,
agent_error, verifier_error, patch, output, attempt_dir} plus {task_id, task_index, failing_tests_available, source}.
Baihe's _record read ATP attempt dirs; neither source has readable ones, so fields come from the exported rows:
- sc14 status: pass/fail by reward when evaluation_completed, else infra.
- sc14 failing_tests: the "[verifier] ✗ [p2p|f2p] <node id>" lines after the last "===== grade =====" of test_output.
  grader.py prints exactly the names ("[p2p] "/"[f2p] " prefix included) and order (p2p, then f2p) of the non-passed
  tests in the ctrf.json that ATP copies to grading/attempt-*/ctrf.json, i.e. what _record listed.
- sc14 agent_error (the eval's Gym records no opencode_run_error_type): opencode_run_error_type or failure_reason; else
  "TimeoutError" when opencode did not finish and was killed ("signal: killed": the agent time cap, which ATP reports
  as TimeoutError); else "AgentSandboxError" when it did not finish and exported no output; else None (a rollout that
  ended on a reasoning item is an ordinary attempt; ATP has no such label).
- kimi: status from _snapshot as in rollouts.target_records; agent_error = opencode_run_error_type or failure_reason;
  failing_tests [] with failing_tests_available True for passes (nothing failed) and False otherwise (the export's
  test_output is empty and its attempt dirs are unreadable, so the failing names of a kimi fail are unknown).
- source: byte provenance of the input line; its length/line_sha256 include the trailing newline (like the
  verify-raw rows.jsonl reference), while index.json record_bytes/record_sha256 exclude it.

The dataset is built in a temporary sibling directory, verified, and renamed into place (COMPLETE.json written last);
an existing dataset is replaced only with --force. Stdlib only (python >= 3.9); the export is streamed through pigz
(else gzip) and abandoned after the last kimi line. `--verify DIR` re-checks a finished dataset.
"""
import argparse
import collections
import csv
import datetime
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import uuid

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
DEEPSWE = S / "synth/cache/deep-swe/tasks"
OUT = S / "data/targets/sc14-out131k-kimi"
EVAL_RUN = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs/"
                "base-step0-rep5-out131k")
STUDENT_ROWS = EVAL_RUN / "rollouts.all.jsonl"
STUDENT_ROWS_SHA256 = "3d078d3c6aa74288998fafe3ee3b3d403af6b0e786791d92ae984eacb8604cd7"  # analysed 2026-10-16
PARTIAL = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/"
               "agent-teacher-pipelines/results/partial-selected-super-kimi-20261015T1603Z")
ROW_HASHES = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/verify-raw/pass1/rows.jsonl")
STUDENT, TEACHER, REPEATS = "sc14", "kimi", 5
EXPECTED = {STUDENT: {"pass": 157, "fail": 401, "infra": 7}, TEACHER: {"pass": 377, "fail": 187, "grader": 1}}
STATUSES = {STUDENT: ("pass", "fail", "infra"), TEACHER: ("pass", "fail", "grader", "infra")}
GRADE_MARK = "===== grade ====="
FAIL_LINE = re.compile(r"^\[verifier\] ✗ (\[(?:p2p|f2p)\] .*)$")
FAIL_HEADER = re.compile(r"^\[verifier\] ===== FAILURES \((\d+)\) =====$")
RECORD_KEYS = ["slot", "model", "status", "reward", "f2p", "p2p", "failing_tests", "agent_error", "verifier_error",
               "patch", "output", "attempt_dir", "task_id", "task_index", "failing_tests_available", "source"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def peak_rss_gb(who=resource.RUSAGE_SELF):
    return round(resource.getrusage(who).ru_maxrss / 1024 ** 2, 2)  # ru_maxrss is KiB on Linux


def log(stage, **fields):
    print(json.dumps({"stage": stage, "at": utc(), "peak_rss_gb": peak_rss_gb(), **fields}), flush=True)


def signature(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_ids():
    """DeepSWE task ids in task_index order (alphabetical = the pinned task directories)."""
    return sorted(p.name for p in DEEPSWE.iterdir() if p.is_dir())


@functools.lru_cache(maxsize=None)
def whitelists(task_id):
    """grader.py load_ids: stripped, de-duplicated (p2p, f2p) node ids of a task."""
    config = json.loads((DEEPSWE / task_id / "tests/config.json").read_text())
    lists = []
    for key in ("p2p_node_ids", "f2p_node_ids"):
        ids, seen = [], set()
        for line in config.get(key, []):
            text = str(line).strip()
            if text and text not in seen:
                seen.add(text)
                ids.append(text)
        lists.append(tuple(ids))
    return tuple(lists)


def failing_tests(row):
    """Non-passed ctrf.json test names, recovered from the grader's '[verifier] ✗ [p2p|f2p] <id>' lines after the last
    '===== grade =====' of test_output; checked against the grade counts and the task's whitelists (ctrf order)."""
    text = row.get("test_output") or ""
    at = text.rfind(GRADE_MARK)
    if at < 0:
        raise ValueError("evaluated row without a grade section in test_output")
    lines = text[at + len(GRADE_MARK):].split("\n")
    names = [m.group(1) for m in map(FAIL_LINE.match, lines) if m]
    headers = [int(m.group(1)) for m in map(FAIL_HEADER.match, lines) if m]
    expected = row["f2p_total"] - row["f2p_passed"] + row["p2p_total"] - row["p2p_passed"]
    if len(names) != expected or headers != ([expected] if expected else []):
        raise ValueError(f"{len(names)} failing-test lines (headers {headers}) but the grade counts imply {expected}")
    p2p, f2p = whitelists(row["task_id"])
    if (row["p2p_total"], row["f2p_total"]) != (len(p2p), len(f2p)):
        raise ValueError("grade totals differ from the task's config.json whitelists")
    order = iter([f"[p2p] {name}" for name in p2p] + [f"[f2p] {name}" for name in f2p])
    if not all(name in order for name in names):  # an ordered subsequence of the ctrf test names
        raise ValueError("failing-test lines are not an ordered subset of the ctrf test names")
    return names


def student_agent_error(row):
    output = (row.get("response") or {}).get("output") or []
    if row.get("opencode_run_error_type") or row.get("failure_reason"):
        return row.get("opencode_run_error_type") or row.get("failure_reason")
    if not row.get("opencode_finished"):
        if "signal: killed" in (row.get("opencode_run_stderr") or ""):
            return "TimeoutError"
        if not output:
            return "AgentSandboxError"
    return None  # a rollout that merely ended on a reasoning item is an ordinary attempt (ATP has no such label)


def _record(model, repeat, status, row, failing, agent_error, attempt_dir, task_index, available, source):
    """rollouts._record with generation and grade both read from one exported row."""
    patch = row.get("model_patch") or ""
    if row.get("model_patch_sha256") and sha256(patch.encode("utf-8", "surrogatepass")) != row["model_patch_sha256"]:
        raise ValueError("model_patch does not match model_patch_sha256")
    return {"slot": f"{model}-r{repeat}", "model": model, "status": status, "reward": row.get("reward"),
            "f2p": f"{row.get('f2p_passed')}/{row.get('f2p_total')}",
            "p2p": f"{row.get('p2p_passed')}/{row.get('p2p_total')}",
            "failing_tests": failing, "agent_error": agent_error, "verifier_error": row.get("verifier_error"),
            "patch": patch, "output": (row.get("response") or {}).get("output") or [], "attempt_dir": attempt_dir,
            "task_id": row["task_id"], "task_index": task_index, "failing_tests_available": available,
            "source": source}


def student_record(row, task_index, source):
    evaluated = row.get("evaluation_completed") is True
    if evaluated and row.get("reward") not in (0, 1):
        raise ValueError(f"evaluated row with reward {row.get('reward')!r}")
    status = ("pass" if row["reward"] == 1 else "fail") if evaluated else "infra"
    return _record(STUDENT, row["_ng_rollout_index"], status, row, failing_tests(row) if evaluated else [],
                   student_agent_error(row), None, task_index, evaluated, source)


def teacher_record(row, slot, task_index, source):
    snapshot = row.get("_snapshot") or {}
    for key, value in slot.items():
        if key != "inputs" and snapshot.get(key) != value:
            raise ValueError(f"slots.jsonl {key}={value!r} but the row's _snapshot has {snapshot.get(key)!r}")
    if (row.get("model_alias"), row.get("_ng_task_index"), row.get("_ng_rollout_index")) != \
            (TEACHER, task_index, slot["repeat_index"]):
        raise ValueError("row identity differs from its slots.jsonl line")
    status = {"completed": "pass" if slot["score"] == 1 else "fail",
              "approved_audited_zero": "grader"}.get(slot["status"], "infra")
    if slot["status"] == "completed" and not (row.get("evaluation_completed") is True and
                                              row.get("reward") == slot["score"] and row["response"].get("output")):
        raise ValueError("completed slot without an evaluated grade matching its score, or without a response")
    return _record(TEACHER, slot["repeat_index"], status, row, [],
                   row.get("opencode_run_error_type") or row.get("failure_reason"), slot["attempt_dir"], task_index,
                   status == "pass", source)  # a passing slot has no failing tests; others are unknown


def encode(record):
    """(line bytes without newline, index entry without line/record_sha256/record_bytes)."""
    line = json.dumps(record, ensure_ascii=True, allow_nan=False).encode("ascii")
    entry = {key: record[key] for key in ("slot", "model", "status", "reward", "f2p", "p2p")}
    entry.update(n_items=len(record["output"]), patch_bytes=len(record["patch"].encode("utf-8", "surrogatepass")))
    return line, entry


def read_students(ids, keep, store, stats):
    """sc14 rows -> store[task_id][(0, rollout)]; every row is validated, records are kept for `keep` only."""
    file_sha256 = digest(STUDENT_ROWS)
    seen = set()
    hasher, offset = hashlib.sha256(), 0
    with STUDENT_ROWS.open("rb") as stream:
        before = signature(os.fstat(stream.fileno()))
        for number, raw in enumerate(stream):
            hasher.update(raw)
            row = json.loads(raw)
            task_id, rollout = row["task_id"], row["_ng_rollout_index"]
            if task_id not in ids or row["_ng_task_index"] != ids.index(task_id):
                raise ValueError(f"line {number}: task {task_id} / index {row['_ng_task_index']} not in task_ids")
            if rollout not in range(REPEATS) or (task_id, rollout) in seen:
                raise ValueError(f"line {number}: unexpected or duplicate rollout {task_id} r{rollout}")
            seen.add((task_id, rollout))
            source = {"file": str(STUDENT_ROWS), "file_sha256": file_sha256, "line": number, "offset": offset,
                      "length": len(raw), "line_sha256": sha256(raw),
                      "opencode_results_fpath": row.get("opencode_results_fpath")}
            try:
                record = student_record(row, row["_ng_task_index"], source)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"line {number} ({task_id} r{rollout}): {error}") from error
            stats["status", STUDENT, record["status"]] += 1
            stats["agent_error", STUDENT, str(record["agent_error"])] += 1
            if not row.get("opencode_finished") and record["agent_error"] is None:
                stats["unfinished_without_agent_error", STUDENT] += 1
            if task_id in keep:
                store[task_id][0, rollout] = encode(record)
            offset += len(raw)
            del row, record
            if (number + 1) % 100 == 0:
                log("sc14", rows=number + 1, gb=round(offset / 1e9, 2))
        after = signature(os.fstat(stream.fileno()))
    if before != after or after != signature(STUDENT_ROWS.stat()) or hasher.hexdigest() != file_sha256:
        raise ValueError(f"{STUDENT_ROWS} changed while it was read")
    if len(seen) != len(ids) * REPEATS:
        raise ValueError(f"{len(seen)} sc14 rows, expected {len(ids) * REPEATS}")
    return {"file": str(STUDENT_ROWS), "bytes": offset, "sha256": file_sha256, "expected_sha256": STUDENT_ROWS_SHA256,
            "sha256_as_expected": file_sha256 == STUDENT_ROWS_SHA256, "rows": len(seen),
            "mtime": datetime.datetime.fromtimestamp(before[3] / 1e9, datetime.timezone.utc).isoformat(),
            "run_config": str(EVAL_RUN / "run_config.json"),
            "model_path": json.loads((EVAL_RUN / "run_config.json").read_text()).get("model_path")}


def read_teachers(ids, keep, store, stats, verify_gz):
    """kimi lines of the export (its first 113*5 lines) -> store[task_id][(1, repeat)]; each line is checked against the
    verify-raw sha256/length reference and its slots.jsonl line; lines after the last needed one are never read."""
    sums = {name: value for value, name in
            (line.split("  ", 1) for line in (PARTIAL / "SHA256SUMS").read_text().splitlines() if line)}
    slots_raw = (PARTIAL / "slots.jsonl").read_bytes()
    if sha256(slots_raw) != sums["slots.jsonl"]:
        raise ValueError("slots.jsonl does not match SHA256SUMS")
    slots = [json.loads(line) for line in slots_raw.splitlines()]
    kimi = [number for number, slot in enumerate(slots) if slot["model_alias"] == TEACHER]
    if kimi != list(range(len(ids) * REPEATS)):
        raise ValueError("kimi rows are not the first 113*5 lines of the export")
    gz = PARTIAL / "rollouts.jsonl.gz"
    reference_gz = (ROW_HASHES.parent / "rollouts.jsonl.gz.sha256").read_text().split()[0]
    if reference_gz != sums["rollouts.jsonl.gz"]:
        raise ValueError("the verify-raw reference was computed on a different rollouts.jsonl.gz")
    references = []
    with ROW_HASHES.open() as stream:
        for line in stream:
            row = json.loads(line)
            references.append((row["line_no"], row["line_bytes"], row["line_sha256"]))
            if len(references) == len(kimi):
                break
    needed = {number for number in kimi if ids[slots[number]["task_index"]] in keep}
    stop = max(needed) + 1
    hasher = subprocess.Popen(["sha256sum", str(gz)], stdout=subprocess.PIPE) if verify_gz else None
    tool = shutil.which("pigz") or shutil.which("gzip")
    before = signature(gz.stat())
    process = subprocess.Popen([tool, "-dc", str(gz)], stdout=subprocess.PIPE, bufsize=16 << 20)
    raw_offset, started = 0, time.time()
    try:
        for number in range(stop):
            raw = process.stdout.readline()
            if not raw.endswith(b"\n"):
                raise ValueError(f"export ended inside line {number}")
            if references[number] != (number, len(raw), sha256(raw)):
                raise ValueError(f"export line {number} differs from the verify-raw reference")
            if number in needed:
                slot = slots[number]
                source = {"file": str(gz), "gz_sha256": sums["rollouts.jsonl.gz"], "line": number,
                          "raw_offset": raw_offset, "length": len(raw), "line_sha256": references[number][2],
                          "slots_line": number, "attempt_dir": slot["attempt_dir"],
                          "agent_artifacts_from": slot["agent_artifacts_from"]}
                row = json.loads(raw)
                del raw
                task_id = ids[slot["task_index"]]
                try:
                    if row.get("task_id") != task_id:
                        raise ValueError(f"task_id {row.get('task_id')!r} != {task_id!r}")
                    record = teacher_record(row, slot, slot["task_index"], source)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"export line {number} ({task_id} r{slot['repeat_index']}): {error}") from error
                stats["status", TEACHER, record["status"]] += 1
                stats["agent_error", TEACHER, str(record["agent_error"])] += 1
                store[task_id][1, slot["repeat_index"]] = encode(record)
                del row, record
                raw_offset += source["length"]
            else:
                raw_offset += len(raw)
            if (number + 1) % 25 == 0:
                log("kimi", lines=number + 1, raw_gb=round(raw_offset / 1e9, 2),
                    raw_mb_per_s=round(raw_offset / 1e6 / (time.time() - started), 1))
    except BaseException:
        if hasher:
            hasher.kill()
        raise
    finally:
        process.stdout.close()
        process.kill()
        process.wait()
    if signature(gz.stat()) != before:
        raise ValueError(f"{gz} changed while it was read")
    gz_info = {"file": str(gz), "bytes": before[2], "sha256": sums["rollouts.jsonl.gz"], "sha256_from": "SHA256SUMS",
               "sha256sums": {"file": str(PARTIAL / "SHA256SUMS"), "sha256": digest(PARTIAL / "SHA256SUMS")},
               "slots": {"file": str(PARTIAL / "slots.jsonl"), "sha256": sums["slots.jsonl"], "lines": len(slots)},
               "row_reference": {"file": str(ROW_HASHES), "sha256": digest(ROW_HASHES)},
               "decompressor": tool, "lines_read": stop, "raw_bytes_read": raw_offset,
               "decode_s": round(time.time() - started, 1)}
    if hasher:
        recomputed = hasher.communicate()[0].split()[0].decode()
        if recomputed != sums["rollouts.jsonl.gz"]:
            raise ValueError(f"{gz} sha256 {recomputed} does not match SHA256SUMS")
        gz_info["sha256_from"] = "SHA256SUMS, recomputed in this run"
    return gz_info


def write_dataset(directory, ids, keep, store):
    """records/, index.json, task_ids.json, per_task.csv -> per-model status counts."""
    (directory / "records").mkdir(parents=True)
    index, rows = {}, []
    counts = {model: collections.Counter() for model in (STUDENT, TEACHER)}
    for task_index, task_id in enumerate(ids):
        if task_id not in keep:
            continue
        group = store[task_id]
        students = sorted(key for key in group if key[0] == 0)
        teachers = sorted(key for key in group if key[0] == 1)
        if students != [(0, rollout) for rollout in range(REPEATS)] or len(teachers) != REPEATS:
            raise ValueError(f"{task_id}: slots {sorted(group)}")
        entries, blob = [], io.BytesIO()
        for number, key in enumerate(students + teachers):
            line, entry = group.pop(key)
            entries.append(dict(entry, record_sha256=sha256(line), record_bytes=len(line), line=number))
            blob.write(line + b"\n")
            counts[entry["model"]][entry["status"]] += 1
        (directory / "records" / f"{task_id}.jsonl").write_bytes(blob.getvalue())
        index[task_id] = {"task_index": task_index, "records": entries}
        tally = {model: collections.Counter(entry["status"] for entry in entries if entry["model"] == model)
                 for model in (STUDENT, TEACHER)}
        rows.append([task_id, task_index] + [tally[model][status] for model in (STUDENT, TEACHER)
                                             for status in STATUSES[model]])
    (directory / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (directory / "task_ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    with (directory / "per_task.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["task_id", "task_index"] + [f"{model}_{status}" for model in (STUDENT, TEACHER)
                                                     for status in STATUSES[model]])
        writer.writerows(rows)
    return {model: {status: counts[model][status] for status in STATUSES[model] if counts[model][status]}
            for model in counts}


def files_manifest(directory):
    return {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "COMPLETE.json"}


def verify(directory, full=True):
    """Re-read a dataset: COMPLETE.json file hashes, index.json line hashes, record identity/order, per_task.csv."""
    directory = Path(directory)
    complete = directory / "COMPLETE.json"
    if complete.exists():
        manifest = json.loads(complete.read_text())["files"]
        if manifest != files_manifest(directory):
            raise ValueError("files differ from COMPLETE.json")
        full = not json.loads(complete.read_text()).get("limited")
    index = json.loads((directory / "index.json").read_text())
    ids = json.loads((directory / "task_ids.json").read_text())
    if ids != sorted(ids) or (full and (list(index) != ids or len(ids) != len(task_ids()))):
        raise ValueError("task_ids.json / index.json do not cover the task set in order")
    counts = {model: collections.Counter() for model in (STUDENT, TEACHER)}
    csv_rows = {row["task_id"]: row for row in csv.DictReader((directory / "per_task.csv").open())}
    for task_id, info in index.items():
        if info["task_index"] != ids.index(task_id):
            raise ValueError(f"{task_id}: task_index {info['task_index']}")
        lines = (directory / "records" / f"{task_id}.jsonl").read_bytes().split(b"\n")
        if lines.pop() != b"" or len(lines) != len(info["records"]) or len(lines) != 2 * REPEATS:
            raise ValueError(f"{task_id}: {len(lines)} lines for {len(info['records'])} index entries")
        repeats = []
        for number, (line, entry) in enumerate(zip(lines, info["records"])):
            record = json.loads(line)
            model = STUDENT if number < REPEATS else TEACHER
            if (entry["line"], entry["record_bytes"], entry["record_sha256"]) != (number, len(line), sha256(line)):
                raise ValueError(f"{task_id} line {number}: does not match index.json")
            if list(record) != RECORD_KEYS or record["model"] != model or record["task_id"] != task_id or \
                    record["task_index"] != info["task_index"] or record["status"] not in STATUSES[model] or \
                    any(record[key] != entry[key] for key in ("slot", "model", "status", "reward", "f2p", "p2p")) or \
                    entry["n_items"] != len(record["output"]):
                raise ValueError(f"{task_id} line {number}: record fields")
            repeats.append(int(record["slot"].rsplit("-r", 1)[1]))
            counts[model][record["status"]] += 1
        if repeats[:REPEATS] != list(range(REPEATS)) or repeats[REPEATS:] != sorted(set(repeats[REPEATS:])):
            raise ValueError(f"{task_id}: slot order {repeats}")
        tally = [sum(entry["model"] == model and entry["status"] == status for entry in info["records"])
                 for model in (STUDENT, TEACHER) for status in STATUSES[model]]
        row = csv_rows.pop(task_id)
        if int(row["task_index"]) != info["task_index"] or \
                [int(value) for key, value in row.items() if key not in ("task_id", "task_index")] != tally:
            raise ValueError(f"{task_id}: per_task.csv")
    if csv_rows:
        raise ValueError(f"per_task.csv has extra tasks {sorted(csv_rows)}")
    return {model: {status: counts[model][status] for status in STATUSES[model] if counts[model][status]}
            for model in counts}


def build(args):
    started, clock = utc(), time.time()
    out = args.out.absolute()
    limited = sorted(set(args.limit_tasks.split(","))) if args.limit_tasks else []
    if limited and out == OUT:
        raise SystemExit("--limit-tasks builds a test dataset: give a different --out")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists ({'complete' if (out / 'COMPLETE.json').exists() else 'incomplete'}); "
                         "pass --force to replace it")
    ids = task_ids()
    if len(ids) != 113 or set(limited) - set(ids):
        raise SystemExit(f"{len(ids)} task dirs in {DEEPSWE}; unknown --limit-tasks {sorted(set(limited) - set(ids))}")
    keep = set(limited or ids)
    store, stats = collections.defaultdict(dict), collections.Counter()
    log("start", out=str(out), tasks=len(keep), host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"))
    student_info = read_students(ids, keep, store, stats)
    log("sc14_done", sha256=student_info["sha256"], as_expected=student_info["sha256_as_expected"])
    teacher_info = read_teachers(ids, keep, store, stats, args.verify_gz)
    log("kimi_done", lines_read=teacher_info["lines_read"], raw_gb=round(teacher_info["raw_bytes_read"] / 1e9, 2))
    counts = {model: {status: stats["status", model, status] for status in STATUSES[model]
                      if stats["status", model, status]} for model in (STUDENT, TEACHER)}
    if not limited:  # sc14 expectations hold for the analysed file only; kimi lines are hash-pinned
        for model in (STUDENT, TEACHER):
            if counts[model] != EXPECTED[model] and (model == TEACHER or student_info["sha256_as_expected"]):
                raise ValueError(f"{model} counts {counts[model]} != expected {EXPECTED[model]}")
            if counts[model] != EXPECTED[model]:
                log("WARNING", model=model, counts=counts[model], expected=EXPECTED[model])
    temporary = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:8]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = write_dataset(temporary, ids, keep, store)
        store.clear()
        script = Path(os.path.abspath(__file__))  # keep the canonical (unresolved) path of docs/port-contract.md
        (temporary / "sources.json").write_text(json.dumps({
            "dataset": out.name, "contract": str(S / "docs/port-contract.md"), "limited": limited,
            "created_by": {"script": str(script), "script_sha256": digest(script), "argv": sys.argv,
                           "python": sys.version.split()[0], "host": socket.gethostname(),
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
            "started_at": started, "finished_at": utc(), "elapsed_s": round(time.time() - clock, 1),
            "peak_rss_gb": peak_rss_gb(), "children_peak_rss_gb": peak_rss_gb(resource.RUSAGE_CHILDREN),
            "inputs": {STUDENT: student_info, TEACHER: teacher_info, "task_dirs": str(DEEPSWE)},
            "counts": counts, "expected_counts": EXPECTED,
            "agent_error_counts": {model: {key[2]: value for key, value in sorted(stats.items())
                                           if key[:2] == ("agent_error", model)} for model in (STUDENT, TEACHER)},
            "sc14_unfinished_without_agent_error": stats["unfinished_without_agent_error", STUDENT],
            "conventions": {
                "mapping": __doc__.split("\n\n")[1],
                "source_length_and_line_sha256": "bytes of the input line including its trailing newline",
                "index_record_bytes_and_record_sha256": "bytes of the records/<task_id>.jsonl line without newline",
                "slot_order": "sc14-r0..r4 then kimi selected repeats ascending",
            }}, indent=1, sort_keys=False) + "\n")
        checked = verify(temporary, full=not limited)
        if checked != written or (not limited and checked != counts):
            raise ValueError(f"verify counts {checked} != written {written} / streamed {counts}")
        (temporary / "COMPLETE.json").write_text(json.dumps({
            "completed_at": utc(), "tasks": len(keep), "records": 2 * REPEATS * len(keep), "limited": limited,
            "counts": checked, "files": files_manifest(temporary)}, indent=1) + "\n")
        if out.exists():  # --force: swap, then drop the replaced copy
            replaced = out.parent / f".{out.name}.replaced-{uuid.uuid4().hex[:8]}"
            os.rename(out, replaced)
            os.rename(temporary, out)
            shutil.rmtree(replaced)
        else:
            os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    log("complete", out=str(out), counts=checked, elapsed_s=round(time.time() - clock, 1),
        children_peak_rss_gb=peak_rss_gb(resource.RUSAGE_CHILDREN))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT, help=f"dataset directory (default {OUT})")
    parser.add_argument("--force", action="store_true", help="replace an existing dataset")
    parser.add_argument("--limit-tasks", help="comma-separated task ids: test build (needs a non-default --out)")
    parser.add_argument("--verify-gz", action="store_true", help="also recompute the export's sha256 (19.6 GB)")
    parser.add_argument("--verify", type=Path, metavar="DIR", help="only verify a finished dataset")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"verified": str(args.verify), "counts": verify(args.verify)}))
    else:
        build(args)


if __name__ == "__main__":
    main()
