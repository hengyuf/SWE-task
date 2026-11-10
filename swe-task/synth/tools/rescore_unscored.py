#!/usr/bin/env python3
"""Score the probed tasks whose score() failed (no metrics.json), from the probe that already holds their rollouts.

hengyuf 2026-11-04 ("给那 71 个没打上分的 task 重新打分"): in official4, build() logged "score <task> FAILED (rerun
resumes)" for 71 built and probed tasks and went on: 67 times chat(glm) failed after 6 retries with HTTP 429 (hub-wide
GLM capacity, 2026-11-03 11:40-18:49 PST; 65 in the teacher-pass audit, 2 in the similarity judgement), 4 times the
failure-analysis agent wrote no result.json. Those tasks have no metrics.json, so they were never accepted, evolved or
audited. Rerunning the build would re-probe them: probe_and_score() names its probe directory by the sha1 of the
still-unscored task set, which differs from the set that was probed, so every rollout would run again.

This tool gives task_builder.score() exactly what build() gave it: the task dict (cand/stage.json, as make_task()
returns it; an evolved task adds lineage = its parent's lineage or name, as evolve() does) and the records of the one
complete probe of that round that holds the task (rollouts.probe_records). Then accept() when the task is accepted, and
<round>/seeds.json and summary.json's accepted list are refreshed as a resumed build writes them (tools/regrade.py's
refresh_seeds / refresh_summary; backups *.pre-rescore). Teacher-pass verdicts already in the task's
teacher_audit.json (a failure after the audit) are kept for every slot whose transcript renders exactly as the
failed scoring wrote it (rollouts/<slot>.md); every other teacher pass is judged.

  list  --runs R [R ...] [--json]
        read-only: the unscored tasks (round, cand, task id, probe, raw sc14/kimi passes, reusable verdicts) and, as a
        check of the task-dict reconstruction, the same reconstruction for every scored task compared with its
        metrics.json; plus scored-and-accepted tasks of an earlier run of this tool whose accept() did not finish
  score --runs R [R ...] [--concurrency 3] [--passes 6] [--pause-s 900] [--max-wait-s 14400] [--dry-run]
        score them, at most --concurrency at a time; a task whose score() or accept() raises is retried in the next
        pass after --pause-s (at most --passes passes). Idempotent: rerun the same command to resume. Before each pass
        and each task the hub budget is checked (SpendGuard): NVAPI_KEY_NEW_3 until official4's $7,000 spend cap
        (queue4/state.json cap_spend, read-only), then NVAPI_KEY_NEW_2 (hub_key_active.txt + tools/make_secrets.py, as
        the queue4 runner switches) down to its $3,000 floor. The reading is a glm call, so an unreadable budget means
        the judge is down: nothing new starts until a reading succeeds (fail closed), and the run stops after
        --max-wait-s of unreadable budget in a row (2026-11-04: glm-5.3 answered HTTP 500 to every key from 14:41).
Every action is one JSON line in <build root>/rescore.jsonl. Run from S/synth with the pipeline venv, e.g.
  ../atp/runtime/controller/bin/python3 -B tools/rescore_unscored.py list --runs official4-yaegi-embed
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import datetime  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

TOOLS = Path(__file__).resolve().parent
SYNTH = TOOLS.parent
for _path in (SYNTH, TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import HUB_KEY_ACTIVE, RUNS, S, STUDENT, TEACHER, hub_key_name, read_json  # noqa: E402
import regrade  # noqa: E402  (tools/regrade.py: find_probe-like checks, make_build, refresh_seeds, refresh_summary)
import rollouts  # noqa: E402
import task_builder  # noqa: E402

TOOL = "tools/rescore_unscored.py"
LOG = "rescore.jsonl"                    # <build root>/rescore.jsonl: one JSON line per action
SUFFIX = ".pre-rescore"                  # first-version backups of seeds.json / summary.json
EVOLVED = re.compile(r"^(?P<parent>.+)-r(?P<round>\d+)b(?P<branch>\d+)$")  # evolve(): <parent>-r<round>b<branch>
QUEUE4_STATE = SYNTH / "queue4/state.json"
QUEUE4 = SYNTH / "queue4/queue.json"
HUB_BUDGET = S / "tools/hub_budget.py"
CAP_KEY, NEXT_KEY = "NVAPI_KEY_NEW_3", "NVAPI_KEY_NEW_2"
CAP_MARGIN_USD = 25.0                    # switch keys this far before the cap (in-flight tasks keep spending)
BUDGET_EVERY_S = 300                     # re-read the hub budget at most this often
BUDGET_UNKNOWN_MAX = 3                   # unreadable budget this many checks in a row: stop new tasks
regrade.LOG = LOG  # regrade.refresh_seeds / refresh_summary log through regrade.log_change into <root>/<regrade.LOG>


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def say(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def log_action(root: Path, entry: dict, dry_run=False) -> dict:
    entry = {"at": now_iso(), "tool": TOOL, **entry}
    if not dry_run:
        with (root / LOG).open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def logged(root: Path) -> list:
    path = root / LOG
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# ============================================================================================ discovery

def build_roots(run: str) -> list:
    """The finished builds of a run (summary.json written): runs/<run>/build/<target>/<owner__repo>."""
    return sorted(p for p in (RUNS / run / "build").glob("*/*") if (p / "summary.json").exists())


def stages(root: Path) -> dict:
    """{task_id: (cand name, stage)} of every built task (cand-*/stage.json with ok)."""
    found = {}
    for path in sorted(root.glob("cand-*/stage.json")):
        stage = read_json(path)
        if isinstance(stage, dict) and stage.get("ok") and stage.get("task_id"):
            found[stage["task_id"]] = (path.parent.name, stage)
    return found


def round_index(round_dir: Path) -> int:
    return int(round_dir.name.split("-", 1)[1])


def task_dict(root: Path, round_dir: Path, cand: str, stage: dict) -> dict:
    """The task dict build() passed to score() for this cand in this round: make_task()'s stage (read back from
    stage.json, as a resumed build does) and, for an evolved cand, evolve()'s lineage = the parent's lineage, else the
    parent's name, where the parent is the round-(r-1) metrics the child was evolved from."""
    if stage.get("name") != cand:
        raise regrade.FixError(f"{cand}/stage.json names {stage.get('name')!r}")
    index = round_index(round_dir)
    if index == 0:
        if stage.get("kind") != "mask" or EVOLVED.match(cand):
            raise regrade.FixError(f"{cand} in round 0 is not a round-0 candidate (kind {stage.get('kind')!r})")
        return dict(stage)
    match = EVOLVED.match(cand)
    if not match or int(match["round"]) != index or stage.get("kind") != "evolve":
        raise regrade.FixError(f"{cand} in {round_dir.name} is not a round-{index} child (kind {stage.get('kind')!r})")
    parent = read_json(root / f"round-{index - 1}" / match["parent"] / "metrics.json")
    if not isinstance(parent, dict) or parent.get("name") != match["parent"]:
        raise regrade.FixError(f"{cand}: parent {match['parent']} has no metrics.json in round-{index - 1}")
    return dict(stage, lineage=parent.get("lineage", parent["name"]))


_PROBES = {}


def probe_tasks(probe: Path) -> set:
    manifest = read_json(probe / "manifest.json") or {}
    return {t["task_id"] for t in manifest.get("tasks", [])}


def probe_records(probe: Path) -> dict:
    if probe not in _PROBES:
        if not rollouts.probe_complete(probe):
            raise regrade.FixError(f"{probe} is not complete (build() scores only after a complete probe)")
        _PROBES[probe] = rollouts.probe_records(probe)
    return _PROBES[probe]


def holding_probe(round_dir: Path, task_id: str) -> Path:
    """The one probe of this round that holds the task (a resumed build's second probe of a round holds only the
    tasks the first left unscored; two probes holding the same task need a person)."""
    found = [p for p in sorted(round_dir.glob("probe-*")) if task_id in probe_tasks(p)]
    if len(found) != 1:
        raise regrade.FixError(f"{len(found)} probes of {round_dir} hold {task_id}: {[p.name for p in found]}")
    return found[0]


def score_view(records: list) -> list:
    """The records as score() renders them into rollouts/: a FAIL with a verifier fault becomes infra."""
    faults = {r["slot"]: reason for r in records if (reason := rollouts.verifier_fault(r))}
    return [dict(r, status="infra", verifier_fault=faults[r["slot"]]) if r["slot"] in faults else r for r in records]


def same_render(md: Path, record: dict) -> bool:
    """rollouts/<slot>.md holds exactly render(record). Bytes, not read_text(): universal newlines would turn the
    \\r of a transcript into \\n (26 official4 slots looked different that way and are byte-identical)."""
    return md.exists() and md.read_bytes() == rollouts.render(record).encode("utf-8")


def reusable_verdicts(scored: Path, records: list) -> dict:
    """{slot: verdict} from the failed scoring's teacher_audit.json for every teacher pass whose rollouts/<slot>.md
    is exactly what score() renders for it now (the judge saw exactly this transcript, with the same instruction and
    rubric). A verdict without a boolean "cheated" is not kept."""
    audit = read_json(scored / "teacher_audit.json")
    if not isinstance(audit, dict):
        return {}
    kept = {}
    for record in score_view(records):
        verdict, md = audit.get(record["slot"]), scored / "rollouts" / f"{record['slot']}.md"
        if record["model"] == TEACHER and record["status"] == "pass" and isinstance(verdict, dict) \
                and isinstance(verdict.get("cheated"), bool) and same_render(md, record):
            kept[record["slot"]] = verdict
    return kept


def rendered_mismatches(scored: Path, records: list) -> list:
    """Slots whose rollouts/<slot>.md (written by the failed scoring) differs from what score() renders now."""
    if not (scored / "rollouts").is_dir():
        return []
    return [r["slot"] for r in score_view(records) if (scored / "rollouts" / f"{r['slot']}.md").exists()
            and not same_render(scored / "rollouts" / f"{r['slot']}.md", r)]


def passes(records: list, model: str) -> str:
    graded = [r for r in score_view(records) if r["model"] == model and r["status"] in ("pass", "fail")]
    return f"{sum(r['status'] == 'pass' for r in graded)}/{len(graded)}"


def accepted_ids(run: str) -> set:
    path = RUNS / run / "accepted/tasks.jsonl"
    return {json.loads(line)["task_id"] for line in path.read_text().splitlines() if line.strip()} \
        if path.exists() else set()


def discover(run: str, *, check_scored=True) -> dict:
    """{"todo": [...], "accept_todo": [...], "problems": [...], "reconstruction": {...}} for one run."""
    out = {"todo": [], "accept_todo": [], "problems": [], "reconstruction": {"checked": 0, "mismatches": []}}
    accepted = accepted_ids(run)
    for root in build_roots(run):
        built = stages(root)
        mine = {e["task_id"]: e for e in logged(root) if e.get("action") == "score"}
        for round_dir in sorted(root.glob("round-*"), key=round_index):
            probed = set().union(*(probe_tasks(p) for p in round_dir.glob("probe-*"))) or set()
            for task_id in sorted(probed):
                if task_id not in built:
                    out["problems"].append(f"{round_dir}: probed {task_id} has no ok cand stage")
                    continue
                cand, stage = built[task_id]
                scored = round_dir / cand
                reprobe = read_json(scored / "teacher_reprobe.json") or {}
                if reprobe.get("status") in ("moving", "aside", "scored"):  # tools/reprobe_teacher.py owns it now:
                    out["problems"].append(  # scoring it from this probe would undo the re-probe
                        f"{scored}: tools/reprobe_teacher.py is re-scoring it ({reprobe['status']})")
                    continue
                metrics = read_json(scored / "metrics.json")
                try:
                    task = task_dict(root, round_dir, cand, stage)
                except regrade.FixError as error:
                    out["problems"].append(str(error))
                    continue
                if isinstance(metrics, dict):
                    if check_scored:  # the reconstruction must give what build() gave for the scored tasks
                        out["reconstruction"]["checked"] += 1
                        wrong = sorted(k for k, v in task.items() if metrics.get(k) != v) + \
                            sorted(k for k in ("lineage",) if (k in metrics) != (k in task))
                        if wrong:
                            out["reconstruction"]["mismatches"].append(f"{scored}: {wrong}")
                    if metrics.get("accept") and task_id not in accepted and task_id in mine:
                        out["accept_todo"].append({"root": root, "round_dir": round_dir, "cand": cand,
                                                   "task_id": task_id, "metrics": metrics})
                    continue
                try:
                    probe = holding_probe(round_dir, task_id)
                    records = probe_records(probe).get(task_id) or []
                except regrade.FixError as error:
                    out["problems"].append(str(error))
                    continue
                if not records:
                    out["problems"].append(f"{probe}: no records for {task_id}")
                    continue
                out["todo"].append({
                    "run": run, "root": root, "round_dir": round_dir, "cand": cand, "task_id": task_id,
                    "task": task, "probe": probe, "records": records, "scored": scored,
                    "sc14": passes(records, STUDENT), "kimi": passes(records, TEACHER),
                    "reuse": sorted(reusable_verdicts(scored, records)),
                    "mismatch": rendered_mismatches(scored, records),
                    "analysis": (scored / "failure_analysis/result.json").exists()})
    return out


# ============================================================================================ budget

def hub_reading(key_name=None) -> dict:
    """S/tools/hub_budget.py's JSON line ({"key", "spend", "remaining", ...}; the key NAME only), or {} if unknown."""
    cmd = ["/usr/bin/python3", "-I", "-B", str(HUB_BUDGET), "--min-remaining", "0"]
    if key_name:
        cmd += ["--key-name", key_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        info = json.loads(line) if line else {}
    except Exception as error:  # noqa: BLE001 - an unknown reading fails closed in spend_guard
        say(f"budget: hub_budget.py failed: {type(error).__name__}: {error}")
        return {}
    return info if isinstance(info, dict) and info.get("spend") is not None else {}


def cap_left(spend: float) -> float:
    """official4's NEW_3 cap left at this hub spend reading: cap_usd - (cap_spend.spent + spend since its anchor).
    queue4/state.json is the stopped runner's (read here, never written); a reading below the anchor means the hub
    counter was reset since, so everything it reads now counts (fail-safe, as the runner re-anchors on a reset)."""
    state, queue = read_json(QUEUE4_STATE) or {}, read_json(QUEUE4) or {}
    cap = ((queue.get("settings") or {}).get("spend_caps") or {}).get(CAP_KEY) or {}
    acct = (state.get("cap_spend") or {}).get(CAP_KEY) or {}
    if not cap.get("cap_usd") or acct.get("spent") is None or acct.get("anchor") is None:
        raise RuntimeError(f"no {CAP_KEY} cap accounting in {QUEUE4} / {QUEUE4_STATE}")
    since = spend - acct["anchor"] if spend >= acct["anchor"] - 50 else spend
    return round(cap["cap_usd"] - (acct["spent"] + max(since, 0.0)), 2)


def next_key_floor() -> float:
    return float(((read_json(QUEUE4) or {}).get("settings") or {}).get("stop_launching_below_usd") or 3000.0)


def switch_to_next_key(why: str, roots: list, dry_run=False):
    say(f"budget: switching the active hub key {CAP_KEY} -> {NEXT_KEY} ({why})")
    if dry_run:
        return
    temporary = HUB_KEY_ACTIVE.with_name(f".{HUB_KEY_ACTIVE.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(NEXT_KEY + "\n")
    temporary.replace(HUB_KEY_ACTIVE)
    result = subprocess.run(["/usr/bin/python3", "-I", "-B", str(S / "tools/make_secrets.py")], cwd=S,
                            capture_output=True, text=True, timeout=300)
    say(f"budget: make_secrets.py rc {result.returncode}")
    for root in roots:
        log_action(root, {"action": "key_switch", "from": CAP_KEY, "to": NEXT_KEY, "why": why,
                          "make_secrets_rc": result.returncode})


class SpendGuard:
    """May a new task start? "ok", "wait" or "stop". NEW_3 until official4's cap (minus CAP_MARGIN_USD), then NEW_2
    down to its floor ("stop" below). The reading is one glm call (tools/hub_budget.py), so an unreadable budget is
    also "the judge is down" (2026-11-04 14:41: glm-5.3 answered HTTP 500/503 to every key): "wait" (no new task
    starts; fail closed) until a reading succeeds, "stop" once readings have failed for max_wait_s in a row."""

    def __init__(self, roots, dry_run=False, max_wait_s=14400):
        self.roots, self.dry_run, self.max_wait_s = roots, dry_run, max_wait_s
        self.checked, self.state, self.unknown_since, self.last = 0.0, "wait", None, {}

    def allow(self) -> str:
        if time.time() - self.checked < BUDGET_EVERY_S:
            return self.state
        self.checked = time.time()
        self.state = self._read()
        return self.state

    def _read(self) -> str:
        active = hub_key_name()
        info = hub_reading()
        if not info or info.get("key") != active:
            self.unknown_since = self.unknown_since or time.time()
            waited = time.time() - self.unknown_since
            say(f"budget: no reading for {active} (the hub's glm unreachable?) for {waited:.0f} s")
            return "stop" if waited >= self.max_wait_s else "wait"
        self.unknown_since = None
        spend, remaining = float(info["spend"]), float(info.get("remaining") or 0.0)
        self.last = {"key": active, "spend": spend, "remaining": remaining}
        if active == CAP_KEY:
            left = cap_left(spend)
            self.last["cap_left"] = left
            say(f"budget: {active} hub spend {spend:.2f}, remaining {remaining:.2f}; official4 cap left {left:.2f}")
            if left > CAP_MARGIN_USD:
                return "ok"
            other = hub_reading(NEXT_KEY)
            floor = next_key_floor()
            if not other or float(other.get("remaining") or 0.0) < floor:
                say(f"budget: {CAP_KEY} cap reached and {NEXT_KEY} not usable ({other.get('remaining')}, floor {floor})")
                return "stop"
            switch_to_next_key(f"official4 spend cap of {CAP_KEY} left {left:.2f} <= {CAP_MARGIN_USD}; {NEXT_KEY} "
                               f"remaining {other['remaining']}", self.roots, self.dry_run)
            return "ok"
        if active == NEXT_KEY:
            floor = next_key_floor()
            say(f"budget: {active} hub spend {spend:.2f}, remaining {remaining:.2f} (floor {floor})")
            return "ok" if remaining >= floor else "stop"
        say(f"budget: the active key {active} is neither {CAP_KEY} nor {NEXT_KEY}: no new tasks")
        return "stop"


# ============================================================================================ scoring

REUSE = {}  # id(record) -> verdict kept from the failed scoring; read by the judge_pass wrapper below
_REAL_JUDGE = task_builder.judge_pass


async def _judge_pass(record, instruction, rubric):
    kept = REUSE.get(id(record))
    return kept if kept is not None else await _REAL_JUDGE(record, instruction, rubric)


@contextlib.contextmanager
def locked_roots(roots, dry_run=False):
    """One rescore process per build root (and none while tools/regrade.py holds the root)."""
    if dry_run:
        yield
        return
    with contextlib.ExitStack() as stack:
        for root in roots:
            for name in (".rescore.lock", f".{regrade.FIX}.lock"):
                handle = stack.enter_context((root / name).open("a"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SystemExit(f"another process holds {root / name}") from None
        yield


def bookkeeping(item: dict, metrics: dict):
    """<round>/seeds.json and summary.json's accepted list, as a resumed build writes them (regrade's helpers, with
    this tool's log and backup suffix)."""
    b = regrade.make_build(item["root"])
    extra = {"tool": TOOL, "cand": item["cand"], "task_id": item["task_id"]}
    regrade.refresh_seeds(b, item["round_dir"], item["root"], SUFFIX, extra)
    regrade.refresh_summary(item["root"], metrics, SUFFIX, extra)


async def score_one(item: dict, dry_run=False) -> str:
    """'scored' / 'accepted' after score() (+ accept()), or raises (the caller retries in the next pass)."""
    root, scored = item["root"], item["scored"]
    if read_json(scored / "metrics.json"):  # scored meanwhile (an earlier pass got further than it reported)
        return "scored"
    records = item["records"]
    kept = reusable_verdicts(scored, records)
    teacher_passes = sum(r["model"] == TEACHER and r["status"] == "pass" for r in score_view(records))
    say(f"{item['run']} {item['cand']} ({item['round_dir'].name}): score() with {item['probe'].name}; sc14 "
        f"{item['sc14']} kimi {item['kimi']}; teacher passes {teacher_passes} ({len(kept)} verdict(s) kept)")
    if dry_run:
        return "dry-run"
    b = regrade.make_build(root)
    for record in records:
        if record["slot"] in kept:
            REUSE[id(record)] = kept[record["slot"]]
    try:
        metrics = await task_builder.score(b, item["task"], records, scored)
    finally:
        for record in records:
            REUSE.pop(id(record), None)
    log_action(root, {"action": "score", "cand": item["cand"], "task_id": item["task_id"], "round": item["round_dir"].name,
                      "probe": item["probe"].name, "verdicts_kept": sorted(kept),
                      "metrics": {k: metrics.get(k) for k in ("student_rate", "teacher_clean_rate", "valid", "accept",
                                                              "reason", "similarity", "cheated")}})
    if metrics.get("accept"):
        await task_builder.accept(b, metrics)
        log_action(root, {"action": "accept", "cand": item["cand"], "task_id": item["task_id"],
                          "path": str(RUNS / item["run"] / "accepted" / item["task_id"])})
    bookkeeping(item, metrics)
    return "accepted" if metrics.get("accept") else "scored"


async def accept_one(item: dict, dry_run=False) -> str:
    """Finish an accept() that an earlier run of this tool started after scoring (metrics accept, not in accepted/)."""
    say(f"{item['root'].parts[-4]} {item['cand']}: accept() (scored by an earlier run of this tool)")
    if dry_run:
        return "dry-run"
    b = regrade.make_build(item["root"])
    await task_builder.accept(b, item["metrics"])
    log_action(item["root"], {"action": "accept", "cand": item["cand"], "task_id": item["task_id"], "resumed": True})
    bookkeeping(item, item["metrics"])
    return "accepted"


async def run_score(args) -> int:
    task_builder.judge_pass = _judge_pass  # score() looks judge_pass up in task_builder at call time
    roots = [root for run in args.runs for root in build_roots(run)]
    guard = SpendGuard(roots, args.dry_run, args.max_wait_s)
    guard_lock = asyncio.Lock()

    async def budget_state() -> str:  # hub_budget.py makes a hub call: off the event loop, one reading at a time
        async with guard_lock:
            return await asyncio.to_thread(guard.allow)

    async def until_allowed() -> bool:
        """True once the guard says "ok", False on "stop"; while it says "wait" (no reading: the hub's glm is down),
        nothing starts and the guard is asked again (it re-reads every BUDGET_EVERY_S). A dry run never waits."""
        while True:
            state = await budget_state()
            if state != "wait" or args.dry_run:
                return state != "stop"
            await asyncio.sleep(60)

    results = {}
    stopped = False
    with locked_roots(roots, args.dry_run):
        for attempt in range(1, args.passes + 1):
            found = [discover(run, check_scored=False) for run in args.runs]
            todo = [("score", item) for f in found for item in f["todo"]] + \
                   [("accept", item) for f in found for item in f["accept_todo"]]
            problems = [p for f in found for p in f["problems"]]
            for problem in problems:
                say(f"NEEDS A PERSON: {problem}")
            if not todo:
                break
            if not await until_allowed():
                stopped = True
                for _, item in todo:
                    results[item["task_id"]] = "not started (budget)"
                break
            say(f"pass {attempt}/{args.passes}: {len(todo)} task(s)")
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(kind, item):
                nonlocal stopped
                async with semaphore:
                    if stopped or not await until_allowed():
                        stopped = True
                        return item["task_id"], "not started (budget)"
                    try:
                        if kind == "accept":
                            return item["task_id"], await accept_one(item, args.dry_run)
                        return item["task_id"], await score_one(item, args.dry_run)
                    except Exception as error:  # noqa: BLE001 - retried in the next pass; score() resumes
                        message = f"{type(error).__name__}: {str(error)[-400:]}"
                        say(f"{item['cand']} FAILED (next pass retries): {message}")
                        if not args.dry_run:
                            log_action(item["root"], {"action": "failed", "kind": kind, "cand": item["cand"],
                                                      "task_id": item["task_id"], "error": message})
                        return item["task_id"], f"failed: {message[:160]}"

            for task_id, outcome in await asyncio.gather(*(one(kind, item) for kind, item in todo)):
                results[task_id] = outcome
            if args.dry_run or stopped:
                break
            left = [t for t, o in results.items() if o.startswith("failed")]
            if not left or attempt == args.passes:
                break
            say(f"{len(left)} failed; next pass in {args.pause_s} s")
            await asyncio.sleep(args.pause_s)
    counts = {}
    for outcome in results.values():
        key = outcome.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    say(f"done: {json.dumps(counts, sort_keys=True)}; budget {json.dumps(guard.last)}")
    for task_id, outcome in sorted(results.items()):
        if not outcome.startswith(("scored", "accepted", "dry-run")):
            say(f"  {task_id}: {outcome}")
    return 0 if all(o.startswith(("scored", "accepted", "dry-run")) for o in results.values()) else 1


def run_list(args) -> int:
    rows, totals = [], {"todo": 0, "accept_todo": 0, "problems": 0, "checked": 0, "mismatches": 0}
    for run in args.runs:
        found = discover(run)
        totals["todo"] += len(found["todo"])
        totals["accept_todo"] += len(found["accept_todo"])
        totals["problems"] += len(found["problems"])
        totals["checked"] += found["reconstruction"]["checked"]
        totals["mismatches"] += len(found["reconstruction"]["mismatches"])
        for item in found["todo"]:
            rows.append({"run": run, "round": item["round_dir"].name, "cand": item["cand"], "task_id": item["task_id"],
                         "probe": item["probe"].name, "sc14": item["sc14"], "kimi": item["kimi"],
                         "lineage": item["task"].get("lineage"), "verdicts_kept": item["reuse"],
                         "rendered_mismatch": item["mismatch"], "analysis_result": item["analysis"]})
        for item in found["accept_todo"]:
            rows.append({"run": run, "cand": item["cand"], "task_id": item["task_id"], "accept_todo": True})
        for problem in found["problems"]:
            rows.append({"run": run, "problem": problem})
        for mismatch in found["reconstruction"]["mismatches"]:
            rows.append({"run": run, "reconstruction_mismatch": mismatch})
    if args.json:
        print(json.dumps({"totals": totals, "rows": rows}, indent=1, default=str))
    else:
        for row in rows:
            print(json.dumps(row, default=str))
        print(json.dumps({"totals": totals}))
    return 0 if not totals["problems"] and not totals["mismatches"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("list")
    sub.add_argument("--runs", nargs="+", required=True)
    sub.add_argument("--json", action="store_true")
    sub = commands.add_parser("score")
    sub.add_argument("--runs", nargs="+", required=True)
    sub.add_argument("--concurrency", type=int, default=3)
    sub.add_argument("--passes", type=int, default=6)
    sub.add_argument("--pause-s", type=int, default=900)
    sub.add_argument("--max-wait-s", type=int, default=14400,
                     help="stop once the hub budget has been unreadable (glm down) this long in a row")
    sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        return run_list(args)
    return asyncio.run(run_score(args))


if __name__ == "__main__":
    sys.exit(main())
