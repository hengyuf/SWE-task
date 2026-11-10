#!/usr/bin/env python3
"""Target rollouts x10 (hengyuf, 2026-11-02): 10 sc14 and 10 Kimi K3 rollouts of every DeepSWE task
-> data/targets/sc14x10-kimix10-20261102/ (the layout and record schema of sc14-out131k-kimi, see reformat_rollouts.py,
plus an "origin" field per record).

Why (hengyuf, 2026-11-02): "在做failure analysis的时候每个不是放5个之前的kimi 的rollouts 和base的rollout, 而是放所有正确的
kimi rollouts 和所有错误的base rollout" -- the target failure analysis of the new builds shows every passing Kimi rollout
and every failing base (sc14) rollout of the task, out of the 10 + 10 rollouts that now exist per task.

records/<task_id>.jsonl, in this order:
- sc14-r0..r4: the 5 sc14 records of sc14-out131k-kimi, unchanged except "origin" (eval run base-step0-rep5-out131k).
- sc14-r5..r9: eval run base-step0-rep5-out131k-rerun1 (same checkpoint swonlysc-step0014, 131k output cap), rollout k
  -> sc14-r{5+k}. Its shard rows are merged as eval/deepswe-v1.1/tools/consolidate_rollouts.py merges them (the last
  row of each (task, rollout) wins) and converted by reformat_rollouts.student_record (status pass/fail by reward when
  evaluation_completed, else infra; failing tests from the grader lines of test_output).
- kimi-r<repeat> (repeat 0..15): the 5 selected Kimi K3 slots of sc14-out131k-kimi, unchanged except "origin" (Baihe's
  export partial-selected-super-kimi-20261015T1603Z, OpenCode context 1,048,576).
- kimi-r16..r20: the 5 Kimi K3 slots of sft/teacher-deepswe-20261030 (context 262,144, output cap 131,072, reasoning
  effort max, stock prompt), repeat k -> kimi-r{16+k}: each slot's final record in results/kimi-<campaign>.json,
  rebuilt from its attempt dir as rollouts.probe_records builds it (terminal.json; a regrade's generation from the
  slot's latest attempt with a response), with the campaign's GRADED_FAIL_OVERRIDES (collect.py) applied: kimi w2-g1
  scriggo-method-declarations r0 counts as a graded fail. reward/f2p/p2p/agent_error/verifier_error/patch length must
  equal the results record.
"origin": {"run", "slot" (the slot's name in that run), ...}. "source": byte/file provenance as in sc14-out131k-kimi.

The dataset is built in a temporary sibling directory, verified (verify()), and renamed into place with COMPLETE.json
written last; an existing dataset is replaced only with --force. `--verify DIR` re-checks a finished dataset.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SYNTH = S / "synth"
for _path in (SYNTH, SYNTH / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import reformat_rollouts as rr  # noqa: E402  (student_record, sha256, digest, signature, files_manifest, task_ids)
import rollouts  # noqa: E402  (synth: _record, _generation, read_json, target_records)

OUT = S / "data/targets/sc14x10-kimix10-20261102"
BASE = S / "data/targets/sc14-out131k-kimi"
EVAL_RUNS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs")
RERUN = EVAL_RUNS / "base-step0-rep5-out131k-rerun1"
TD = S / "sft/teacher-deepswe-20261030"
TD_CAMPAIGNS = ("w1-g0", "w1-g1", "w1-g2", "w2-g0", "w2-g1")
STUDENT, TEACHER, REPEATS = "sc14", "kimi", 5
RERUN_SLOT_BASE = 5    # rerun1 rollout k -> sc14-r{5+k}
TD_SLOT_BASE = 16      # teacher-deepswe repeat k -> kimi-r{16+k} (Baihe's export has repeats 0..15)
STATUSES = {STUDENT: ("pass", "fail", "infra"), TEACHER: ("pass", "fail", "grader", "infra")}
RECORD_KEYS = rr.RECORD_KEYS + ["origin"]
# context_window / per_call_output_cap: OpenCode's limits in each run (sc14: eval deepswe_eval.yaml max_model_len and
# per_call_output_cap; kimi-1m: the export's OpenCode context; kimi-256k: results/kimi-*.json)
ORIGINS = {
    "sc14-orig": {"run": "base-step0-rep5-out131k", "via": str(BASE), "context_window": 262144,
                  "per_call_output_cap": 131072},
    "sc14-rerun1": {"run": RERUN.name, "via": str(RERUN), "context_window": 262144, "per_call_output_cap": 131072},
    "kimi-1m": {"run": "partial-selected-super-kimi-20261015T1603Z", "via": str(BASE), "context_window": 1048576,
                "per_call_output_cap": 131072},
    "kimi-256k": {"run": "teacher-deepswe-20261030", "via": str(TD), "context_window": 262144,
                  "per_call_output_cap": 131072},
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def peak_rss_gb(who=resource.RUSAGE_SELF):
    return round(resource.getrusage(who).ru_maxrss / 1024 ** 2, 2)


def log(stage, **fields):
    print(json.dumps({"stage": stage, "at": utc(), "peak_rss_gb": peak_rss_gb(), **fields}), flush=True)


def td_collect():
    """teacher-deepswe's collect.py as a module (its GRADED_FAIL_OVERRIDES and graded_fail_override, read-only)."""
    spec = importlib.util.spec_from_file_location("td_collect_x10", TD / "collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(record):
    line = json.dumps(record, ensure_ascii=True, allow_nan=False).encode("ascii")
    entry = {key: record[key] for key in ("slot", "model", "status", "reward", "f2p", "p2p")}
    entry.update(n_items=len(record["output"]), patch_bytes=len(record["patch"].encode("utf-8", "surrogatepass")),
                 origin=record["origin"]["name"])
    return line, entry


# ------------------------------------------------------------------------------------------------ sources
def read_base(ids, store, stats):
    """sc14-out131k-kimi records (sha-checked by rollouts.target_records) -> store[task][(model, repeat)]."""
    complete = json.loads((BASE / "COMPLETE.json").read_text())
    for task_id in ids:
        for record in rollouts.target_records(task_id, dataset=BASE):
            if record["task_id"] != task_id or record["task_index"] != ids.index(task_id):
                raise ValueError(f"{BASE}: {task_id} {record['slot']} carries {record['task_id']}/{record['task_index']}")
            repeat = int(record["slot"].rsplit("-r", 1)[1])
            name = "sc14-orig" if record["model"] == STUDENT else "kimi-1m"
            if (record["model"] == STUDENT and repeat not in range(REPEATS)) or \
                    (record["model"] == TEACHER and repeat not in range(TD_SLOT_BASE)):
                raise ValueError(f"{BASE}: {task_id} unexpected slot {record['slot']}")
            record = dict(record, origin=dict(ORIGINS[name], name=name, slot=record["slot"]))
            key = (record["model"], repeat)
            if key in store[task_id]:
                raise ValueError(f"{task_id}: duplicate {record['slot']}")
            store[task_id][key] = encode(record)
            stats["status", name, record["status"]] += 1
    return {"dataset": str(BASE), "complete_sha256": rr.digest(BASE / "COMPLETE.json"),
            "index_sha256": complete["files"]["index.json"]["sha256"], "records": complete["records"],
            "counts": complete["counts"]}


def read_rerun(ids, store, stats):
    """rerun1's shard rows (last row per (task, rollout) wins, as consolidate_rollouts.py) -> sc14-r5..r9."""
    shards = sorted((p for p in (RERUN / "shards").glob("shard_*") if p.is_dir()), key=lambda p: int(p.name[6:]))
    last, files = {}, {}
    for shard in shards:
        path = shard / "rollouts.jsonl"
        if not path.exists():
            continue
        before = rr.signature(path.stat())
        hasher, offset = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for number, raw in enumerate(stream):
                hasher.update(raw)
                if raw.strip():
                    row = json.loads(raw)
                    key = (row["task_id"], row["_ng_rollout_index"])
                    last[key] = (path, number, offset, len(raw))
                    del row
                offset += len(raw)
        if rr.signature(path.stat()) != before:
            raise ValueError(f"{path} changed while it was read")
        files[str(path)] = {"bytes": offset, "sha256": hasher.hexdigest(), "signature": before}
    if set(last) != {(task_id, k) for task_id in ids for k in range(REPEATS)}:
        raise ValueError(f"rerun1 keys: {len(last)}, missing {sorted(set((t, k) for t in ids for k in range(REPEATS)) - set(last))[:5]}")
    for (task_id, rollout), (path, number, offset, length) in sorted(last.items()):
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"{path} line {number}: short read")
        row = json.loads(raw)
        if (row["task_id"], row["_ng_rollout_index"]) != (task_id, rollout) or \
                row["_ng_task_index"] != ids.index(task_id):
            raise ValueError(f"{path} line {number}: identity {row['task_id']}/{row['_ng_task_index']}")
        source = {"file": str(path), "file_sha256": files[str(path)]["sha256"], "line": number, "offset": offset,
                  "length": length, "line_sha256": rr.sha256(raw),
                  "opencode_results_fpath": row.get("opencode_results_fpath")}
        try:
            record = rr.student_record(row, row["_ng_task_index"], source)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} line {number} ({task_id} r{rollout}): {error}") from error
        slot = f"{STUDENT}-r{RERUN_SLOT_BASE + rollout}"
        record = dict(record, slot=slot, origin=dict(ORIGINS["sc14-rerun1"], name="sc14-rerun1",
                                                    slot=f"{STUDENT}-r{rollout}"))
        store[task_id][STUDENT, RERUN_SLOT_BASE + rollout] = encode(record)
        stats["status", "sc14-rerun1", record["status"]] += 1
        del row, record, raw
    config = json.loads((RERUN / "run_config.json").read_text())
    return {"run": str(RERUN), "run_config_sha256": rr.digest(RERUN / "run_config.json"),
            "model_path": config.get("model_path"), "reasoning_parser": config.get("reasoning_parser"),
            "files": {name: {k: v for k, v in info.items() if k != "signature"} for name, info in files.items()},
            "rows": len(last)}


def read_teacher(ids, store, stats):
    """teacher-deepswe Kimi K3 final records -> kimi-r16..r20, rebuilt from their attempt dirs."""
    collect = td_collect()
    seen, campaigns = set(), {}
    for campaign in TD_CAMPAIGNS:
        path = TD / "results" / f"{TEACHER}-{campaign}.json"
        raw = path.read_bytes()
        result = json.loads(raw)
        if not result.get("finished_at") or result.get("model") != TEACHER or result.get("campaign") != campaign:
            raise ValueError(f"{path}: not a finished {TEACHER} {campaign} campaign")
        campaigns[campaign] = {"file": str(path), "sha256": rr.sha256(raw), "finished_at": result["finished_at"],
                               "model_id": result.get("model_id"), "context_window": result.get("context_window"),
                               "max_output_tokens": result.get("max_output_tokens"), "prompt": result.get("prompt"),
                               "temperature": result.get("temperature"), "top_p": result.get("top_p"),
                               "tasks": len(result["records"])}
        for task_id, records in result["records"].items():
            if task_id not in ids or task_id in seen:
                raise ValueError(f"{path}: unexpected or repeated task {task_id}")
            seen.add(task_id)
            if sorted(r["slot"] for r in records) != [f"{TEACHER}-r{k}" for k in range(REPEATS)]:
                raise ValueError(f"{path}: {task_id} slots {[r['slot'] for r in records]}")
            for final in records:
                record = collect.graded_fail_override(TEACHER, campaign, task_id, final)
                overridden = record is not final
                attempt = Path(record["attempt_dir"]) if record.get("attempt_dir") else None
                if attempt is None:
                    raise ValueError(f"{path}: {task_id} {record['slot']} has no attempt dir")
                grade = rollouts.read_json(attempt / "terminal.json", {})
                generation = grade if (grade.get("response") or {}).get("output") else rollouts._generation(attempt)
                repeat = int(record["slot"].rsplit("-r", 1)[1])
                row = rollouts._record(TEACHER, repeat, record["status"], generation, grade, attempt)
                for key in ("reward", "f2p", "p2p", "agent_error", "verifier_error"):
                    if row[key] != record[key]:
                        raise ValueError(f"{task_id} {record['slot']}: rebuilt {key} {row[key]!r} != {record[key]!r}")
                if len(row["patch"]) != record["patch_chars"]:
                    raise ValueError(f"{task_id} {record['slot']}: patch {len(row['patch'])} != {record['patch_chars']}")
                if (record["status"] == "pass" and row["reward"] != 1) or (record["status"] == "fail" and row["reward"] == 1):
                    raise ValueError(f"{task_id} {record['slot']}: status {record['status']} with reward {row['reward']}")
                if record["status"] in ("pass", "fail") and not row["output"]:
                    raise ValueError(f"{task_id} {record['slot']}: graded slot without a transcript")
                reports = sorted(attempt.glob("grading/attempt-*/ctrf.json"))
                row.update(slot=f"{TEACHER}-r{TD_SLOT_BASE + repeat}", task_id=task_id, task_index=ids.index(task_id),
                           failing_tests_available=record["status"] == "pass" or bool(reports),
                           source={"file": str(path), "file_sha256": campaigns[campaign]["sha256"],
                                   "campaign": campaign, "attempt_dir": str(attempt),
                                   "status_override": record.get("status_override") if overridden else None},
                           origin=dict(ORIGINS["kimi-256k"], name="kimi-256k", slot=record["slot"], campaign=campaign,
                                       ended_by_broken_reply=record.get("ended_by_broken_reply")))
                store[task_id][TEACHER, TD_SLOT_BASE + repeat] = encode(row)
                stats["status", "kimi-256k", row["status"]] += 1
                del row, generation, grade
    if seen != set(ids):
        raise ValueError(f"teacher-deepswe Kimi covers {len(seen)} tasks, missing {sorted(set(ids) - seen)[:5]}")
    return {"campaigns": campaigns, "overrides_applied": {"|".join(map(str, key)): value for key, value in
                                                          collect.OVERRIDES_APPLIED.items()},
            "override_problems": {"|".join(map(str, key)): value for key, value in collect.OVERRIDE_PROBLEMS.items()}}


# ------------------------------------------------------------------------------------------------ dataset
def order(group):
    students = sorted(key for key in group if key[0] == STUDENT)
    teachers = sorted(key for key in group if key[0] == TEACHER)
    return students, teachers


def write_dataset(directory, ids, store):
    (directory / "records").mkdir(parents=True)
    index, rows = {}, []
    for task_index, task_id in enumerate(ids):
        group = store[task_id]
        students, teachers = order(group)
        if [k for _, k in students] != list(range(2 * REPEATS)) or len(teachers) != 2 * REPEATS or \
                [k for _, k in teachers if k >= TD_SLOT_BASE] != list(range(TD_SLOT_BASE, TD_SLOT_BASE + REPEATS)):
            raise ValueError(f"{task_id}: slots {sorted(group)}")
        entries, blob = [], io.BytesIO()
        for number, key in enumerate(students + teachers):
            line, entry = group.pop(key)
            entries.append(dict(entry, record_sha256=rr.sha256(line), record_bytes=len(line), line=number))
            blob.write(line + b"\n")
        (directory / "records" / f"{task_id}.jsonl").write_bytes(blob.getvalue())
        index[task_id] = {"task_index": task_index, "records": entries}
        rows.append(csv_row(task_id, task_index, entries))
    (directory / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (directory / "task_ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    with (directory / "per_task.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


CSV_HEADER = (["task_id", "task_index"] + [f"{model}_{status}" for model in (STUDENT, TEACHER)
                                          for status in STATUSES[model]]
              + [f"{name}_{status}" for name in ORIGINS for status in ("pass", "graded")])


def csv_row(task_id, task_index, entries):
    tally = collections.Counter((entry["model"], entry["status"]) for entry in entries)
    origin = collections.Counter((entry["origin"], entry["status"]) for entry in entries)
    return ([task_id, task_index] + [tally[model, status] for model in (STUDENT, TEACHER) for status in STATUSES[model]]
            + [value for name in ORIGINS for value in (origin[name, "pass"], origin[name, "pass"] + origin[name, "fail"])])


def verify(directory):
    """Re-read a dataset: COMPLETE.json file hashes, index.json line hashes, record identity/order, per_task.csv."""
    directory = Path(directory)
    complete = directory / "COMPLETE.json"
    if complete.exists() and json.loads(complete.read_text())["files"] != rr.files_manifest(directory):
        raise ValueError("files differ from COMPLETE.json")
    index = json.loads((directory / "index.json").read_text())
    ids = json.loads((directory / "task_ids.json").read_text())
    if ids != rr.task_ids() or list(index) != ids:
        raise ValueError("task_ids.json / index.json do not cover the DeepSWE task set in order")
    counts = collections.Counter()
    csv_rows = {row["task_id"]: row for row in csv.DictReader((directory / "per_task.csv").open())}
    for task_id, info in index.items():
        if info["task_index"] != ids.index(task_id):
            raise ValueError(f"{task_id}: task_index {info['task_index']}")
        lines = (directory / "records" / f"{task_id}.jsonl").read_bytes().split(b"\n")
        if lines.pop() != b"" or len(lines) != len(info["records"]) or len(lines) != 4 * REPEATS:
            raise ValueError(f"{task_id}: {len(lines)} lines for {len(info['records'])} index entries")
        slots = []
        for number, (line, entry) in enumerate(zip(lines, info["records"])):
            record = json.loads(line)
            model = STUDENT if number < 2 * REPEATS else TEACHER
            if (entry["line"], entry["record_bytes"], entry["record_sha256"]) != (number, len(line), rr.sha256(line)):
                raise ValueError(f"{task_id} line {number}: does not match index.json")
            if list(record) != RECORD_KEYS or record["model"] != model or record["task_id"] != task_id or \
                    record["task_index"] != info["task_index"] or record["status"] not in STATUSES[model] or \
                    any(record[key] != entry[key] for key in ("slot", "model", "status", "reward", "f2p", "p2p")) or \
                    entry["n_items"] != len(record["output"]) or record["origin"]["name"] != entry["origin"]:
                raise ValueError(f"{task_id} line {number}: record fields")
            if (record["status"] == "pass" and record["reward"] != 1) or \
                    (record["status"] == "fail" and record["reward"] == 1):
                raise ValueError(f"{task_id} line {number}: status {record['status']} with reward {record['reward']}")
            repeat = int(record["slot"].rsplit("-r", 1)[1])
            expected = ("sc14-orig" if repeat < RERUN_SLOT_BASE else "sc14-rerun1") if model == STUDENT else \
                ("kimi-1m" if repeat < TD_SLOT_BASE else "kimi-256k")
            if record["origin"]["name"] != expected:
                raise ValueError(f"{task_id} {record['slot']}: origin {record['origin']['name']} != {expected}")
            slots.append((model, repeat))
            counts[record["origin"]["name"], record["status"]] += 1
        if slots != sorted(slots, key=lambda s: (s[0] != STUDENT, s[1])) or len(set(slots)) != len(slots):
            raise ValueError(f"{task_id}: slot order {slots}")
        row = csv_rows.pop(task_id)
        if [str(value) for value in csv_row(task_id, info["task_index"], info["records"])] != \
                [row[key] for key in CSV_HEADER]:
            raise ValueError(f"{task_id}: per_task.csv")
    if csv_rows:
        raise ValueError(f"per_task.csv has extra tasks {sorted(csv_rows)}")
    return {name: {status: counts[name, status] for status in ("pass", "fail", "grader", "infra")
                   if counts[name, status]} for name in ORIGINS}


def build(args):
    started, clock = utc(), time.time()
    out = args.out.absolute()
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists ({'complete' if (out / 'COMPLETE.json').exists() else 'incomplete'}); "
                         "pass --force to replace it")
    ids = rr.task_ids()
    if len(ids) != 113 or ids != json.loads((BASE / "task_ids.json").read_text()):
        raise SystemExit(f"{len(ids)} task dirs, or they differ from {BASE / 'task_ids.json'}")
    store, stats = collections.defaultdict(dict), collections.Counter()
    log("start", out=str(out), host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"))
    base_info = read_base(ids, store, stats)
    log("base_done", records=sum(len(group) for group in store.values()))
    rerun_info = read_rerun(ids, store, stats)
    log("rerun1_done", rows=rerun_info["rows"])
    teacher_info = read_teacher(ids, store, stats)
    log("teacher_done", campaigns=len(teacher_info["campaigns"]), overrides=list(teacher_info["overrides_applied"]))
    if teacher_info["override_problems"]:
        raise ValueError(f"GRADED_FAIL_OVERRIDES did not match: {teacher_info['override_problems']}")
    streamed = {name: {status: stats["status", name, status] for status in ("pass", "fail", "grader", "infra")
                       if stats["status", name, status]} for name in ORIGINS}
    temporary = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:8]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_dataset(temporary, ids, store)
        store.clear()
        script = Path(os.path.abspath(__file__))
        (temporary / "sources.json").write_text(json.dumps({
            "dataset": out.name, "requested_by": __doc__.split("\n\n")[1],
            "created_by": {"script": str(script), "script_sha256": rr.digest(script), "argv": sys.argv,
                           "python": sys.version.split()[0], "host": socket.gethostname(),
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
            "started_at": started, "finished_at": utc(), "elapsed_s": round(time.time() - clock, 1),
            "peak_rss_gb": peak_rss_gb(),
            "inputs": {"sc14-orig+kimi-1m": base_info, "sc14-rerun1": rerun_info, "kimi-256k": teacher_info},
            "origins": ORIGINS, "counts": streamed,
            "conventions": {
                "slots": __doc__.split("\n\n")[2],
                "slot_order": "sc14-r0..r9, then kimi slots ascending by repeat (1M export repeats, then 16..20)",
                "index_record_bytes_and_record_sha256": "bytes of the records/<task_id>.jsonl line without newline",
            }}, indent=1, sort_keys=False) + "\n")
        checked = verify(temporary)
        if checked != streamed:
            raise ValueError(f"verify counts {checked} != streamed {streamed}")
        (temporary / "COMPLETE.json").write_text(json.dumps({
            "completed_at": utc(), "tasks": len(ids), "records": 4 * REPEATS * len(ids), "counts": checked,
            "files": rr.files_manifest(temporary)}, indent=1) + "\n")
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
    log("complete", out=str(out), counts=checked, elapsed_s=round(time.time() - clock, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT, help=f"dataset directory (default {OUT})")
    parser.add_argument("--force", action="store_true", help="replace an existing dataset")
    parser.add_argument("--verify", type=Path, metavar="DIR", help="only verify a finished dataset")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"verified": str(args.verify), "counts": verify(args.verify)}))
    else:
        build(args)


if __name__ == "__main__":
    main()
