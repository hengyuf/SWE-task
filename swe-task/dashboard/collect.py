#!/usr/bin/env python3
"""Collector of the swe-task official-synthesis dashboard (stdlib only; /usr/bin/python3 -I -B, Python 3.10).

CONTRACT.md (same directory) is the source of truth for every file and schema written here. One cycle (run_cycle)
reads the pipeline's files READ-ONLY and writes, each atomically (temp file + rename):
  data/index.json             every DeepSWE task with the kimi / Super VL / sc14 accuracy, the official-queue rank,
                              and the official runs (status from squeue/sacct, current stage, failure-match audit
                              coverage of their eligible and of their accepted tasks)
  data/task/<task_id>.json    one per task under an official run: job + log tail, the target's failure patterns,
                              the plan, the rounds and the generation tree (one NODE per candidate dir) with the
                              decisions of synth/task_builder.py build() / seed_check() / valid_seeds(), whether each
                              task is in the failure-match audit's scope (built and probed, a graded sc14 rollout, not
                              discarded or withdrawn) and its audit (runs/<run>/audit/failure_match/<task_id>.json:
                              does sc14 repeat the target's failure patterns there?)
  data/fleet.json             student vLLM workers (fleet daemon status.json + registry + each worker's /metrics),
                              router stats from the official probes' student-router-requests.jsonl, 24 h history
  data/api.json               hub budget (tools/hub_budget.py, at most every 10 min), burn rate, per-run kimi /
                              codex / glm usage
  data/history/fleet.jsonl, data/history/hub.jsonl   append-only histories (trimmed to 7 / 30 days)
  data/run/<source>/<run>.json  (CONTRACT "v1.6") one per run of a batch from another tree, today Baihe's
                              (baiheh/swe-task-hengyu): the same document as data/task/<task_id>.json, built by the same
                              code from that run's own tree, never audited, each node flagged in_batch when its task is
                              one of the batch's outputs/tasks.jsonl
  data/explorer.json          (CONTRACT "v1.7") the Explorer's tables: one row per built synthetic task of the task and
                              run files above and one per rollout their probes list, made from those very nodes; rewritten
                              only when its content changes (its generated_at is the cycle that last changed it)
  data/run/hengyuf/<run>.json (CONTRACT "v1.8") one per official run that is not its DeepSWE task's latest (e.g.
                              official1-etree once official2-etree launched): the same document as a task file, rewritten
                              only when its content changes; index.json tasks[].official_runs lists every official run
  index.json "queues"         (CONTRACT "v1.8") each synthesis queue's runner state, from exactly four files of its
                              directory (S/synth/queue2: queue.json, state.json, heartbeat.json, the tail of runner.log)
Incremental state lives in state/ (dir 700, file 600): byte offsets of the append-only logs (keyed by inode, so a
moved file such as the narwhals budget-422-moved kimi attempts is never counted twice), the last /metrics scrape per
worker (tps = counter deltas), the router's last hour, the hub sample time, and finished probe slots.

Only official runs (synth/runs/official*) are read, never smoke1/val1; and (v1.6) a batch's outputs/ and the runs that
built its tasks, without their logs, Slurm jobs, usage logs or audits (Batch); and (v1.8) a synthesis queue's four runner
files, of which only the run list, statuses, legs and job ids, the heartbeat, the budget key NAME and number, two flags
and the redacted log tail are published (Queue). The collector never reads environment
variables, and never publishes anything from .env files, S/secrets or S/atp/.env: it loads their key VALUES in
memory only, so that redact() masks them wherever agent-written text repeats them (_secret_values). The hub budget
comes from a subprocess that reads the key itself. Files are read only inside the read roots (_open_regular).
Every text copied from a file goes through redact() and clip(). Slurm: read-only squeue/sacct only.

    /usr/bin/python3 -I -B collect.py once [--data DIR] [--state DIR] [--no-hub]
    /usr/bin/python3 -I -B collect.py loop [--interval 120] [--data DIR] [--state DIR] [--no-hub]
Library: run_cycle(data_dir, state_dir) -> summary dict (thread-safe; never raises for a broken source).
"""
import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import stat
import subprocess
import threading
import time
import urllib.parse
import urllib.request

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")  # hengyuf's swe-task tree (README.md)
D = Path(__file__).resolve().parent
THREE_WAY = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/three-way/"
                 "three_way_per_task.csv")  # Baihe's kimi K3 / Super VL exports, 5 rollouts per task each
SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
SLURM_CONF = "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"  # login nodes run no slurmd
SCHEMA = 1
STUDENT, TEACHER = "sc14", "kimi"  # synth/common.py STUDENT / TEACHER (the probes' model aliases)
DEFAULT_ROUNDS, DEFAULT_BRANCHES = 3, 5  # task_builder build defaults = the official setting (tools/launch_build.sh)
HUB_INTERVAL_S = 600  # hub_budget.py makes one (tiny) billed glm call: at most every 10 min
LOCK_WAIT_S = 90  # another collector process (e.g. a manual `once` next to the server) holds state/collector.lock
HUB_TIMEOUT_S = 45
SCRAPE_TIMEOUT_S = 3
SCAN_BUDGET_S = 35  # per cycle; a longer backlog of log bytes (first cycle: ~13 GB of kimi http.jsonl) continues
SCAN_MIN_S = 5  # ... but every cycle reads for at least this long
CHUNK = 8 << 20
FLEET_HISTORY_DAYS, HUB_HISTORY_DAYS = 7, 30
ROUTER_WINDOW_S = 600
ROUTER_SCOPE = "official-run probes"  # the router journals read: official runs' probes only (not smoke/val or evals)
HEARTBEAT_TIMEOUT_S = 120  # student/fleet.py: a worker is routable only with a heartbeat this fresh
SERVING = ("ready", "draining")
RUNNING_STATES = ("RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT", "SIGNALING", "RESIZING")
PENDING_STATES = ("PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SUSPENDED")
FAILED_STATES = ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE")
FLEET_BUCKETS = ("ready", "draining", "starting", "pending", "held", "stale", "unhealthy", "failed", "stopped",
                 "vanished", "other")
# Failure-match audits (CONTRACT "v1.3", "v1.4"): auditor agents judge, for each task in the audit's scope, whether
# sc14's rollouts there repeat the target's failure patterns; one file per task, runs/<run>/audit/failure_match/
# <task_id>.json. The scope (v1.4, hengyuf 2026-10-19: every task that is not discarded and on which the student has a
# rollout) is NODE.audit_eligible; v1.3 audited the accepted tasks only.
AUDIT_DIR = Path("audit/failure_match")
AUDIT_MAX_BYTES = 1 << 20  # an audit is a few KB; a bigger file is a runaway writer, never read whole
AUDIT_VERDICTS = ("strong", "partial", "weak", "none")
AUDIT_REPRODUCED = ("yes", "partial", "no")
ROLLOUT_STATUSES = ("pass", "fail", "infra")
ACCEPTED = ("accepted", "accepted_evolve")
WITHDRAWN_DIR = "withdrawn"  # runs/<run>/withdrawn/<task_id>/: an accepted task main withdrew (moved out of accepted/)
ABANDONED_ROOT = "abandoned-"  # a build root main abandoned (e.g. runs/official1-httpx/abandoned-psf__requests-...)
# Batches (CONTRACT "v1.6", hengyuf 2026-10-22: show Baihe's batch of tasks next to the official runs, without any
# audit). Baihe forked this pipeline into BAIHE (same layout, the same sc14 / kimi / glm models, the same launch.jsonl);
# his outputs/tasks.jsonl is the batch: one dataset row per task, outputs/<task_id>/provenance.json = its metrics.json,
# whose task_dir names the run and target that built it (synth/runs/<run>/build/<target>/<repo>/<cand>/task/<task_id>).
# Only outputs/ and the runs that built a task of the batch are read; never his secrets/, .env files, Slurm logs, jobs,
# usage logs or audits.
BAIHE = Path("/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu")
NOT_AUDITED = {"status": "not_audited"}  # NODE.audit / runs[].audit of a batch: no failure-match audit runs on it
# A batch's runs have ended: a run's document is rebuilt when its fingerprint (a few stats: build root and round dirs,
# summary.json, launch.jsonl, accepted/, the target's analysis, its batch tasks) changes, else reused, and rebuilt
# anyway once it is BATCH_REFRESH_S old (at most BATCH_REFRESHES per cycle, so they spread out). Its file is rewritten
# only when its content changed; its generated_at is the cycle that last changed it (index runs[].detail_generated_at).
BATCH_REFRESH_S = 1800
BATCH_REFRESHES = 2
# The Explorer (CONTRACT "v1.7", hengyuf 2026-10-22: filter the synthetic tasks and their trajectories by evolve round,
# similarity, sc14 / kimi pass rates, failure-match audit verdict and trajectory outcome, with coverage statistics):
# data/explorer.json holds one row per built synthetic task (NODE with build.task_id) of the documents the cycle writes,
# data/task/<task_id>.json (each official target's latest run) and data/run/<source>/<run>.json (a batch's runs), and
# one row per rollout their probes list (NODE.probe.rollouts), as column-headed arrays. It is computed every cycle from
# those very NODE dicts and rewritten only when its content (all but generated_at) changed.
EXPLORER = "explorer.json"
EXPLORER_TASK_COLUMNS = ("run", "node", "task_id", "round", "branch", "kind", "title", "decision", "accepted",
                         "withdrawn", "in_batch", "similarity", "student_rate", "teacher_clean_rate", "teacher_rate",
                         "valid", "audit", "reproduced_yes", "reproduced_partial", "f2p", "p2p", "scored",
                         "rates_differ")
EXPLORER_TRAJECTORY_COLUMNS = ("task", "slot", "model", "status", "reward", "f2p", "p2p", "tool_calls", "patch_chars",
                               "agent_error", "verifier_error", "clean")
EXPLORER_NOT_AUDITED, EXPLORER_NOT_ELIGIBLE = "not_audited", "not_eligible"  # task rows' audit besides the verdicts
OFFICIAL_SOURCE = {"source": "hengyuf", "name": "hengyuf", "label": "hengyuf · official runs", "audited": True}
# v1.8 (CONTRACT "Every official run, and the synthesis queue"; hengyuf 2026-10-22, the official2 queue): every official
# run has a document. A DeepSWE task's latest official run (the most recently launched) keeps data/task/<task_id>.json;
# each older official run of the same task writes data/run/hengyuf/<run>.json (the same document), rewritten only when
# its content changed, its generated_at the cycle that last changed it (index runs[].detail_generated_at).
OFFICIAL_RUNS_DIR = "run/hengyuf"
# v1.8: a synthesis queue (S/synth/queue2/run_queue.py): the statuses its runner writes to state.json, the age past which
# its heartbeat means no live runner (run_queue.py HEARTBEAT_STALE_S), the runner.log lines shown
QUEUE_STATUSES = ("queued", "pending", "running", "waiting_to_resume", "done", "failed", "stuck")
QUEUE_STALE_S = 900
QUEUE_LOG_LINES = 40
KEY_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}")  # a hub key's environment variable NAME (state.json budget.key)
QUEUE_PID = re.compile(r"\bpid \d+")  # the runner's process id in runner.log lines: masked ("pid ⋯")
JOB_ID = re.compile(r"\d{1,12}")

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # candidate/probe dir names; renamed `x.budget-422` dirs fail
SAFE_TASK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
STEP = re.compile(r"-r(\d+)b(\d+)$")
AUTHOR = re.compile(r"author-(\d+)")  # exact: author-0.budget-422 / author-1.stale-feedback are not attempts
GATES = re.compile(r"gates-(\d+)\.json")
ROUND = re.compile(r"round-(\d+)")
PROBE = re.compile(r"probe-[A-Za-z0-9_-]+")  # probe-<sha1[:8]> in practice; renamed (dotted) dirs never match
ATTEMPT = re.compile(r"attempt-(\d+)")
SLOT_DIR = re.compile(r"(\d+)-(\d+)")
BASE_URL = re.compile(r"http://[A-Za-z0-9.-]+:\d{2,5}(/v1)?/?")
PROM_LINE = re.compile(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")

MODELS = {  # display order: kimi, super_vl, sc14
    "kimi": {"label": "Kimi K3", "role": "teacher",
             "source": "three_way_per_task.csv kimi (Baihe's kimi K3 export, 5 rollouts per task)"},
    "super_vl": {"label": "Nemotron 3.5 Super VL", "role": "Baihe's student",
                 "source": "three_way_per_task.csv super (Baihe's Super VL export, 5 rollouts per task)"},
    "sc14": {"label": "Local Super ckpt sc14 (swonlysc-step0014, 131k cap)", "role": "student",
             "source": "data/targets/sc14-out131k-kimi index.json (hengyuf's DeepSWE eval, 5 rollouts per task)"},
}
NOTES = ["kimi cost is hub-reported per call (usage.cost of every streamed kimi call in the probes' http.jsonl, "
         "including attempts moved aside after a hub 422); codex and glm cost is not logged (tokens only)",
         "glm calls = teacher-pass audits (teacher_audit.json entries) + failure-similarity judgements; failed calls "
         "and tools/hub_budget.py's own tiny glm calls are not counted",
         "hub spend covers everything billed to the key (kimi, codex, glm; official and other runs)"]

# --------------------------------------------------------------------------------------------- text and file helpers

# What redact() masks (CONTRACT.md): key-like tokens anywhere, and long base64/hex values assigned to *key*/*token*/
# *secret* names. Each pattern is linear-time (anchored on a literal) so megabyte-long blobs stay cheap.
_SECRETS = [
    (re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{10,}"), "sk-<redacted>"),
    (re.compile(r"nvapi-[A-Za-z0-9_-]{10,}"), "nvapi-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{10,}"), "hf_<redacted>"),
    (re.compile(r"\b([Bb][Ee][Aa][Rr][Ee][Rr]\s+)[A-Za-z0-9._~+/=-]{6,}"), r"\1<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,})"), "gh_<redacted>"),
    (re.compile(r"glpat-[A-Za-z0-9_-]{10,}"), "glpat-<redacted>"),
    (re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])"), "AKIA<redacted>"),
    (re.compile(r"(?i)((?:key|token|secret)[A-Za-z0-9_.-]{0,40}[\"']?\s{0,3}[=:]\s{0,3}[\"']?)"
                r"[A-Za-z0-9+/_-]{32,}={0,2}"), r"\1<redacted>"),
]


# Every pattern above needs one of these substrings: strings without any (most of them) skip the full set.
_TRIGGER = re.compile(r"(?i)sk-|nvapi-|hf_|bearer|gh[pousr]_|github_pat_|glpat-|akia|asia|key|token|secret")


def redact(text):
    """Mask credentials in a string (non-strings pass through)."""
    if not isinstance(text, str) or not text:
        return text
    for value in _secret_values():
        if value in text:
            text = text.replace(value, "<redacted-secret>")
    if not _TRIGGER.search(text):
        return text
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def clip(text, limit):
    """Truncate to at most `limit` chars with a "…[N chars omitted]…" marker (CONTRACT.md)."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    marker = "\n…[{} chars omitted]…"
    keep = max(0, limit - len(marker.format(len(text))))
    return text[:keep] + marker.format(len(text) - keep)


def text_field(text, limit):
    """A text field copied from a file: redacted first, then clipped (so a cut never leaves half a key)."""
    return clip(redact(text), limit) if isinstance(text, str) else None


def sanitize(value):
    """Final pass over an output document: redact every string, drop non-finite floats (JSON has no NaN)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def iso(ts):
    """Local ISO time with offset, e.g. 2026-10-17T15:00:00-07:00."""
    if not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def parse_time(text):
    """Epoch seconds from an ISO string with an offset (launch.jsonl, hub_budget.py) or a local Slurm time."""
    if not isinstance(text, str) or not text or text in ("Unknown", "None", "N/A"):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


# Known secret VALUES, masked wherever they appear (2026-10-17 security review: several of hengyuf's real keys have
# no distinctive prefix, and agent-written text is published). They are loaded in memory only, never written,
# logged or returned; files that cannot be read are skipped.
_SECRET_FILES = (Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/.env"), S / "secrets/opensandbox.env",
                 S / "atp/.env")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")
_SECRET_VALUES = None


def _secret_values():
    global _SECRET_VALUES
    if _SECRET_VALUES is None:
        values = set()
        for path in _SECRET_FILES:
            try:
                lines = path.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                name, separator, value = line.strip().removeprefix("export ").partition("=")
                value = value.strip().strip("'\"")
                if separator and _SECRET_NAME.search(name.upper()) and len(value) >= 12:
                    values.add(value)
        _SECRET_VALUES = tuple(sorted(values, key=len, reverse=True))
    return _SECRET_VALUES


# Files are read only if their real path stays inside these roots (Cycle registers its sources' roots and its own
# data/state dirs) and they are regular files: agent-written files get published, and a codex agent could plant a
# symlink to e.g. ~/.ssh or the user's .env (2026-10-17 security review).
_READ_ROOTS = set()


def allow_reads(*roots):
    _READ_ROOTS.update(os.path.realpath(root) for root in roots)


_EXACT_FILES = set()  # v1.8: allow_files(): exact paths, never a symlink there


def _exact(path):
    """A file's path with its directory resolved, its own name kept (a symlink there is not followed)."""
    return os.path.join(os.path.realpath(os.path.dirname(str(path))), os.path.basename(str(path)))


def allow_files(*paths):
    """v1.8: exact files as read roots (a synthesis queue's runner files), their directory resolved but not the file
    itself: _open_regular(exact=True) opens such a path itself with O_NOFOLLOW, so a symlink at that path is refused
    wherever it points, also into another read root."""
    _EXACT_FILES.update(_exact(path) for path in paths)


def _open_regular(path, buffering=-1, exact=False):
    if exact:  # v1.8: one of the exact files: never through a symlink at its own path
        real = _exact(path)
        if real not in _EXACT_FILES:
            raise PermissionError(f"not one of the collector's exact files: {path}")
    else:
        real = os.path.realpath(path)
        if not any(real == root or real.startswith(root + os.sep) for root in _READ_ROOTS):
            raise PermissionError(f"outside the collector's read roots: {path}")
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PermissionError(f"not a regular file: {path}")
        return os.fdopen(fd, "rb", buffering=buffering)
    except BaseException:
        os.close(fd)
        raise


def read_json(path, default=None, exact=False):
    try:
        with _open_regular(path, exact=exact) as stream:
            return json.loads(stream.read())
    except (OSError, ValueError):
        return default


def read_text(path, cap=4 << 20):
    """A text file (at most `cap` bytes; undecodable bytes replaced), or None when missing."""
    try:
        with _open_regular(path) as stream:
            data = stream.read(cap + 1)
    except OSError:
        return None
    text = data[:cap].decode("utf-8", errors="replace")
    return text + f"\n…[file continues beyond {cap} bytes]…" if len(data) > cap else text


def read_head(path, size=4096):
    try:
        with _open_regular(path) as stream:
            return stream.read(size).decode("utf-8", errors="replace")
    except OSError:
        return None


def read_tail_lines(path, count=40, size=64 << 10, exact=False):
    try:
        with _open_regular(path, exact=exact) as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - size))
            data = stream.read()
    except OSError:
        return None
    lines = data.decode("utf-8", errors="replace").splitlines()
    if end > size and lines:
        lines = lines[1:]  # the first line was cut by the seek
    return lines[-count:]


def read_jsonl(path):
    rows = []
    try:
        with _open_regular(path) as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # e.g. a torn last line
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def listdir(path):
    try:
        return os.listdir(path)
    except OSError:
        return []


def exists(path):
    return os.path.exists(path)


def as_dict(value):
    return value if isinstance(value, dict) else {}


def as_list(value):
    return value if isinstance(value, list) else []


def natural(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def fraction(value):
    """'20/20' as is; 'None/None', '' and None -> None."""
    if not isinstance(value, str) or "/" not in value or "None" in value:
        return None
    return value


def ratio(value):
    try:
        passed, total = (int(part) for part in value.split("/"))
    except (AttributeError, ValueError):
        return None
    return passed / total if total > 0 else None


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (position - low), 3)


def encode_json(value, clean=True):
    """The bytes of an output file: compact ASCII JSON without NaN; clean: sanitize() first (every output document;
    the collector's own state holds no copied text)."""
    return json.dumps(sanitize(value) if clean else value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode()


def write_bytes_atomic(path, data, mode=0o600):
    """temp file in the same directory + rename: readers never see a partial file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "wb") as stream:
        stream.write(data)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def write_json_atomic(path, value, mode=0o600, clean=True):
    """encode_json() + write_bytes_atomic(); returns the size written."""
    data = encode_json(value, clean)
    write_bytes_atomic(path, data, mode)
    return len(data)


def file_signature(path):
    """(inode, size, mtime) of a file, or None: tells whether a file we wrote is still the one on disk."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_ino, st.st_size, st.st_mtime_ns


def append_jsonl(path, row):
    """One line, one write() on an O_APPEND descriptor (an append-only history file)."""
    data = (json.dumps(sanitize(row), ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def trim_jsonl(path, keep_after, slack_s=86400):
    """Rewrite a history file atomically without rows older than keep_after (only once it is a day over)."""
    rows = read_jsonl(path)
    if not rows or not isinstance(rows[0].get("t"), (int, float)) or rows[0]["t"] >= keep_after - slack_s:
        return rows
    rows = [row for row in rows if isinstance(row.get("t"), (int, float)) and row["t"] >= keep_after]
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:  # data/ files: 600
        stream.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return rows


class Memo:
    """Process-lifetime cache of parsed files keyed by (kind, path) and the file's (inode, mtime, size): in the server's
    long-lived collector thread a file is parsed again only after it changes. Entries not used in a cycle are
    dropped at its end (bounded memory)."""

    def __init__(self):
        self.entries, self.used = {}, set()

    def get(self, kind, path, loader):
        key = (kind, str(path))
        try:
            st = os.stat(path)
        except OSError:
            self.entries.pop(key, None)
            return None
        signature = (st.st_ino, st.st_mtime_ns, st.st_size)  # inode: atomic rewrites within Lustre's 1 s mtime
        self.used.add(key)
        hit = self.entries.get(key)
        if hit and hit[0] == signature:
            return hit[1]
        value = loader(path)
        self.entries[key] = (signature, value)
        return value

    def sweep(self):
        self.entries = {key: value for key, value in self.entries.items() if key in self.used}
        self.used = set()


_MEMO = Memo()
_LOCK = threading.Lock()
_BATCH_CACHE = {}  # v1.6, process lifetime: (data dir, run dir, target) -> the batch run's last build (Cycle.batch_docs)
_EXPLORER_CACHE = {}  # v1.7, process lifetime: data dir -> {"body", "stamp", "file"} of the explorer.json written last
_OFFICIAL_CACHE = {}  # v1.8, process lifetime: (data dir, detail) -> {"body", "stamp", "file"} of an older official run's file


class Batch:
    """A batch of tasks built by another tree's copy of this pipeline (CONTRACT "v1.6"): <root>/outputs/tasks.jsonl
    (the batch) and <root>/synth/runs/<run>/ (the runs that built it). `rounds` / `branches`: that tree's task_builder
    defaults, for launches whose "extra" does not set them."""

    def __init__(self, source, name, label, root, rounds=1, branches=5):
        self.source, self.name, self.label, self.root = source, name, label, Path(root)
        self.outputs = self.root / "outputs"
        self.runs = self.root / "synth/runs"
        self.defaults = {"rounds": rounds, "branches": branches}


# Baihe's batch: his synth/task_builder.py main() defaults to --rounds 1 ("baiheh: evolve once (no repeated evolution)
# in all runs") and --branches 5; his launch.jsonl rows carry no --rounds.
BATCHES = (Batch("baiheh", "Baihe", "Baihe · swe-task-hengyu", BAIHE, rounds=1, branches=5),)


class Queue:
    """A synthesis queue (CONTRACT "v1.8"): the runner's four files in its directory, the only ones read there:
    queue.json (the runs, in queue order), state.json (each run's status, leg and Slurm candidates; the budget; two
    flags), heartbeat.json (the live runner) and runner.log (its last lines). Any of them may not exist yet."""

    def __init__(self, name, root):
        self.name, self.root = name, Path(root)
        self.files = {kind: self.root / name for kind, name in (("queue", "queue.json"), ("state", "state.json"),
                                                                ("heartbeat", "heartbeat.json"), ("log", "runner.log"))}


# hengyuf's official2 queue (2026-10-22): 14 builds of reselected host repos, round 0 + round 1, at most 5 at a time;
# official3 (2026-10-26): the group-A DeepSWE targets (sc14 2-3/5, kimi >= 3/5), same runner (synth/queue3);
# official4 (2026-11-02): the 23 uncovered targets with Kimi >= 5/10 and sc14 <= 5/10 (synth/queue4; target analysis
# shows every passing Kimi and failing sc14 rollout of data/targets/sc14x10-kimix10-20261102)
QUEUES = (Queue("official4", S / "synth/queue4"), Queue("official3", S / "synth/queue3"),
          Queue("official2", S / "synth/queue2"))


class Sources:
    """The collector's read-only inputs (tests point `root` and `three_way` at fixture trees, and `batches` / `queues`
    at fixture batches and queues: a fixture tree has none unless given)."""

    def __init__(self, root=S, three_way=THREE_WAY, python="/usr/bin/python3", batches=None, queues=None):
        self.root = Path(root)
        self.runs = self.root / "synth/runs"
        self.slurm_logs = self.runs / "slurm"
        self.targets = self.root / "data/targets/sc14-out131k-kimi"
        self.three_way = Path(three_way)
        self.student = self.root / "student"
        self.hub_budget = self.root / "tools/hub_budget.py"
        self.common = self.root / "synth/common.py"
        self.python = python
        self.batches = tuple(batches) if batches is not None else (BATCHES if self.root == S else ())
        self.queues = tuple(queues) if queues is not None else (QUEUES if self.root == S else ())  # v1.8


# ------------------------------------------------------------------------------------------------- Slurm and the hub

def _scrubbed_env():
    """A minimal environment for subprocesses, built without reading ours (no proxies or keys to inherit)."""
    user = pwd.getpwuid(os.getuid())
    return {"PATH": f"{SLURM_BIN}:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": user.pw_dir, "USER": user.pw_name,
            "LOGNAME": user.pw_name, "SLURM_CONF": SLURM_CONF}


def slurm_jobs(job_ids):
    """{job_id: {state, partition, node, submit, start, end, elapsed, time_left, reason}} from squeue (live jobs)
    and sacct (the rest). Read-only commands only."""
    ids = sorted({str(job) for job in job_ids if str(job).isdigit()})
    out = {}
    if not ids:
        return out
    env = _scrubbed_env()
    result = subprocess.run([f"{SLURM_BIN}/squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%P|%N|%V|%S|%e|%M|%L|%r"],
                            capture_output=True, text=True, timeout=30, env=env)
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 10 and parts[0] in ids:
            out[parts[0]] = {"state": parts[1], "partition": parts[2], "node": parts[3] or None,
                             "submit": parse_time(parts[4]), "start": parse_time(parts[5]),
                             "end": parse_time(parts[6]), "elapsed": parts[7], "time_left": parts[8],
                             "reason": parts[9] if parts[1] in PENDING_STATES else None}
    missing = [job for job in ids if job not in out]
    if missing:
        result = subprocess.run([f"{SLURM_BIN}/sacct", "-n", "-P", "-X", "-j", ",".join(missing), "-o",
                                 "JobID,State,Partition,NodeList,Submit,Start,End,Elapsed"],
                                capture_output=True, text=True, timeout=30, env=env)
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) == 8 and parts[0] in missing:
                out[parts[0]] = {"state": parts[1].split()[0] if parts[1] else None, "partition": parts[2],
                                 "node": parts[3] if parts[3] not in ("", "None assigned") else None,
                                 "submit": parse_time(parts[4]), "start": parse_time(parts[5]),
                                 "end": parse_time(parts[6]), "elapsed": parts[7], "time_left": None, "reason": None}
    return out


def start_hub_sample(src):
    """Start tools/hub_budget.py (it reads the hub key itself; one tiny glm call). Returns the Popen."""
    return subprocess.Popen([src.python, "-I", "-B", str(src.hub_budget), "--min-remaining", "0"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            cwd=str(src.hub_budget.parent), env=_scrubbed_env(), text=True)


def finish_hub_sample(process, timeout):
    """(stdout, exit code) of a started hub sample; kills it after `timeout` s."""
    try:
        stdout, _ = process.communicate(timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("hub_budget.py timed out")
    return stdout, process.returncode


def scrape_url(url, timeout=SCRAPE_TIMEOUT_S):
    """GET a worker's Prometheus page without any proxy (workers are on the cluster network)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read(8 << 20).decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------------------------------- the cycle

class Cycle:
    """One collection pass. Every source is guarded: a broken one adds "<source>: <message>" to errors."""

    def __init__(self, data_dir, state_dir, sources=None, clock=time.time, slurm=None, scrape=None, hub=None):
        self.data_dir, self.state_dir = Path(data_dir), Path(state_dir)
        self.src = sources or Sources()
        self.clock = clock
        self.slurm = slurm or slurm_jobs
        self.scrape = scrape or scrape_url
        self.hub = (start_hub_sample, finish_hub_sample) if hub is None else hub  # False: no hub sample
        self.memo = _MEMO
        allow_reads(self.src.runs, self.src.targets, self.src.three_way.parent, self.src.student / "state",
                    self.src.student / "registry", self.src.common, self.data_dir, self.state_dir)
        for queue in self.src.queues:  # v1.8: exactly the runner's four files, never anything else of its directory
            allow_files(*queue.files.values())
        self.errors = []
        self.started = time.monotonic()
        self.now = clock()
        self.state = {}
        self.written = {}
        self.batches = []  # v1.6: what read_batch() found per batch (index.json "batches")
        self.batch_seconds = 0.0  # v1.6: the cycle's time spent on the batches (read, build, write)

    # -- plumbing
    def error(self, source, error):
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self.errors.append(clip(redact(f"{source}: {message}"), 600))

    def guard(self, source, function, *args, default=None):
        try:
            return function(*args)
        except Exception as error:  # one broken source must never stop the cycle
            self.error(source, error)
            return default

    def json(self, path):
        return self.memo.get("json", path, read_json)

    def text(self, path):
        return self.memo.get("text", path, read_text)

    def excerpt(self, path, limit):
        """text_field() of a file, cached until the file changes (no re-redaction of unchanged files)."""
        return self.memo.get(f"excerpt{limit}", path, lambda path: text_field(read_text(path), limit))

    def elapsed(self):
        return time.monotonic() - self.started

    def write(self, relative, value):
        self.written[relative] = write_json_atomic(self.data_dir / relative, value)

    def load_state(self):
        state = read_json(self.state_dir / "collector.json", {})
        self.state = state if isinstance(state, dict) and state.get("version") == 1 else {"version": 1}

    def save_state(self):
        write_json_atomic(self.state_dir / "collector.json", self.state, mode=0o600, clean=False)

    # -- the pass
    def run(self):
        for directory in (self.data_dir, self.data_dir / "task", self.data_dir / "history") + \
                ((self.data_dir / "run",) if self.src.batches else ()):
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        self.load_state()
        hub_process = self.guard("hub", self.start_hub)
        with ThreadPoolExecutor(2) as background:  # worker /metrics scrapes and squeue overlap the file work
            fleet_future = background.submit(self.fleet_snapshot)
            runs = self.guard("runs", self.discover_runs, default=[])
            # v1.8: an unresolved leg of the queue runner: every candidate job is polled
            slurm_future = background.submit(self.slurm, list(dict.fromkeys(
                job for run in runs for job in ([run["job_id"]] if run.get("job_id") else []) + run.get("candidates", []))))
            tasks = self.guard("targets", self.load_tasks, default=[])
            trees = {id(run): self.guard(f"run {run['run']} tree", Tree(self, run, run["root"]).build)
                     if run["root"] else None for run in runs}
            jobs = self.guard("slurm", slurm_future.result, default={})
            for run in runs:  # v1.8: an unresolved leg shows its most active candidate
                if run.get("candidates"):
                    self.pick_candidate(run, jobs)
            docs = []
            for run in runs:
                doc = self.guard(f"run {run['run']}", self.task_doc, run, jobs.get(str(run.get("job_id"))),
                                 trees[id(run)])
                if doc:
                    docs.append(doc)
            usage = self.guard("usage", self.scan_usage, docs, default={})
            # v1.6: the batches' runs, after the official work (their reads never shorten the usage scan's budget),
            # while the worker scrapes may still run; official docs, usage, fleet and api never include them
            batch_started = time.monotonic()
            batch_docs = self.guard("batches", self.batch_docs, default=[]) if self.src.batches else []
            self.batch_seconds += time.monotonic() - batch_started
            fleet = self.guard("fleet", fleet_future.result, default=None)
        self.write_task_docs(docs)
        # v1.8: every older official run's document (data/run/hengyuf/<run>.json), rewritten only when it changed
        self.guard("official runs", self.write_official_docs, docs)
        batch_started = time.monotonic()
        if self.src.batches:
            self.guard("batches", self.write_batch_docs, batch_docs)
        self.batch_seconds = round(self.batch_seconds + time.monotonic() - batch_started, 2)
        # v1.7: the Explorer's tables, from the nodes of the task and run files just written (never stops the cycle)
        explorer_started = time.monotonic()
        explorer = self.guard("explorer", self.write_explorer, docs, batch_docs)
        explorer_seconds = round(time.monotonic() - explorer_started, 2)
        self.write("fleet.json", self.guard("fleet", self.fleet_doc, fleet, usage.get("router"),
                                            default={"schema": SCHEMA, "generated_at": self.now}))
        self.write("api.json", self.guard("api", self.api_doc, hub_process, usage, docs,
                                          default={"schema": SCHEMA, "generated_at": self.now}))
        self.prune_slots(docs)
        self.guard("state", self.save_state)
        # v1.8: the synthesis queues' runner state (each guarded alone: a broken queue never stops the cycle)
        queues = [doc for doc in (self.guard(f"queue {queue.name}", self.queue_doc, queue) for queue in self.src.queues)
                  if doc is not None]
        # last: it carries every error and the cycle time
        self.write("index.json", self.index_doc(tasks, docs, batch_docs, explorer=explorer and explorer["index"],
                                                queues=queues))
        self.memo.sweep()
        seconds = round(self.elapsed(), 2)
        summary = {"ok": not self.errors, "seconds": seconds, "errors": self.errors, "runs": len(docs),
                   "tasks": len(tasks), "written_bytes": dict(sorted(self.written.items())),
                   "scan": usage.get("scan")}
        if self.src.batches:  # v1.6: per batch, its runs, nodes, in_batch nodes and accepted nodes, and the time taken
            summary["batches"] = {info["source"]: dict(
                runs=info["runs_count"], nodes=sum(len(item["doc"]["nodes"]) for item in batch_docs
                                                   if item["run"]["batch"].source == info["source"]),
                tasks=info["tasks"], in_batch=info["in_batch"], accepted=info["accepted"],
                unmatched=len(info["unmatched"]), seconds=self.batch_seconds) for info in self.batches
                if "runs_count" in info}
        # v1.7: the Explorer's rows by source, its size, whether this cycle rewrote it, and the time it took
        summary["explorer"] = dict(explorer["summary"], seconds=explorer_seconds) if explorer else None
        # v1.8: the older official runs (their own files) and each queue's runs and runner
        summary["official_runs"] = {"latest": sum(bool(item["run"]["latest"]) for item in docs),
                                    "older": sum(not item["run"]["latest"] for item in docs)}
        if self.src.queues:
            summary["queues"] = {doc["name"]: {"runs": len(doc["runs"]), "alive": doc["runner"]["alive"]}
                                 for doc in queues}
        return summary

    # ========================================================================================== DeepSWE tasks
    def load_tasks(self):
        """Every DeepSWE task (targets task_ids.json order) with the three models' accuracy and queue rank."""
        targets = self.src.targets
        ids = self.json(targets / "task_ids.json")
        index = self.json(targets / "index.json")
        for name, value, kind in (("task_ids.json", ids, list), ("index.json", index, dict)):
            if not isinstance(value, kind):
                self.error("targets", f"{targets / name} missing or unreadable")
        index = index if isinstance(index, dict) else {}
        per_task = self.memo.get("csv", targets / "per_task.csv", _read_csv) or {}
        three = self.memo.get("csv", self.src.three_way, _read_csv)
        if three is None:
            self.error("three-way", f"{self.src.three_way} unreadable")
            three = {}
        if not isinstance(ids, list):
            ids = sorted(set(index) | set(per_task) | set(three))
        tasks = []
        for position, task_id in enumerate(ids):
            entry = as_dict(index.get(task_id))
            records = as_list(entry.get("records"))
            row, way = per_task.get(task_id), three.get(task_id)
            sc14 = _sc14_card(records, row)
            kimi = _three_way_card(way, "kimi_rollouts(f2p|p2p|score)") if way else _record_card(records, TEACHER)
            super_vl = _three_way_card(way, "super_rollouts(f2p|p2p|score)") if way else None
            f2p_total = _int(way.get("f2p_total")) if way else None
            if f2p_total is None:
                f2p_total = max((int(r["f2p"].split("/")[1]) for r in records
                                 if fraction(r.get("f2p")) and r["f2p"].split("/")[1].isdigit()), default=None)
            index_position = entry.get("task_index")
            tasks.append({"task_id": task_id,
                          "task_index": index_position if isinstance(index_position, int) else position,
                          "f2p_total": f2p_total, "accuracy": {"kimi": kimi, "super_vl": super_vl, "sc14": sc14},
                          "eligible": bool(sc14 and kimi and sc14["passed"] <= 1 and kimi["passed"] == 5
                                           and kimi["total"] == 5),
                          "queue_rank": None, "run": None})
        eligible = sorted((task for task in tasks if task["eligible"]),
                          key=lambda task: (-(task["accuracy"]["sc14"].get("mean_f2p_ratio") or -1.0),
                                            task["task_index"]))
        for rank, task in enumerate(eligible, 1):
            task["queue_rank"] = rank
        return tasks

    # ========================================================================================== official runs
    def discover_runs(self):
        """One entry per (official run, target task): its build root, latest launch (job) and settings. When
        several official runs build the same task, the most recently launched one is "latest" (it owns
        data/task/<task_id>.json); index.json lists them all, and (v1.8) each older one writes its document to
        data/run/hengyuf/<run>.json ("detail"). A run dir with neither launch.jsonl nor a build (a queued run, e.g. an
        official2 run whose runner only copied its failure analysis) is no run."""
        runs = []
        for run_dir in sorted(self.src.runs.glob("official*")):
            if run_dir.is_dir() and SAFE_NAME.fullmatch(run_dir.name):
                runs += self.guard(f"run {run_dir.name}", self.read_run, run_dir, default=[])
        latest_run = {}
        for run in sorted(runs, key=lambda run: run["submitted_at"] or 0):
            latest_run[run["task_id"]] = run
        targets = {}
        for run in runs:
            targets[run["run"]] = targets.get(run["run"], 0) + 1
        for run in runs:
            run["latest"] = latest_run[run["task_id"]] is run
            # one file per run; a run that builds several targets (none so far) gets one per target, as a batch run
            run["detail"] = (f"task/{run['task_id']}.json" if run["latest"] else
                             f"{OFFICIAL_RUNS_DIR}/{run['run']}.json" if targets[run["run"]] == 1 else
                             f"{OFFICIAL_RUNS_DIR}/{run['run']}--{run['task_id']}.json")
        return runs

    def read_run(self, run_dir, batch=None):
        """One entry per target the run builds (or launched). batch (v1.6): a Batch's run, with that tree's build
        defaults and no log (its Slurm logs are outside the read roots)."""
        launches = read_jsonl(run_dir / "launch.jsonl")
        targets = {}
        for task_dir in sorted((run_dir / "build").glob("*")):
            # one repo per build; the most recent if ever more, an abandoned-* one only when nothing else is there
            repos = sorted((path for path in task_dir.glob("*") if path.is_dir()),
                           key=lambda path: (not path.name.startswith(ABANDONED_ROOT), path.stat().st_mtime))
            if repos and SAFE_TASK.fullmatch(task_dir.name):
                targets[task_dir.name] = repos[-1]
        for launch in launches:
            task = launch.get("task")
            if isinstance(task, str) and SAFE_TASK.fullmatch(task):
                targets.setdefault(task, None)  # launched, no build dir yet
        runs = []
        for task, root in targets.items():
            mine = [launch for launch in launches if launch.get("task") == task] or launches
            latest = mine[-1] if mine else {}
            # v1.8, the queue runner's launches (run_queue.py through the babysit helper): a leg first writes a line naming
            # its two candidate jobs (babysit.candidates; job_id the first), then, once one won, a line with
            # babysit.resolved and babysit.winner (job_id the winner). The leg's job is its winner when resolved; an
            # unresolved leg keeps its candidates, all polled, the most active one shown (pick_candidate)
            babysit = as_dict(latest.get("babysit"))
            won = next((info.get("winner") for info in (as_dict(launch.get("babysit")) for launch in reversed(mine))
                        if babysit and info.get("leg") == babysit.get("leg") and info.get("resolved") is True
                        and info.get("winner") is not None and JOB_ID.fullmatch(str(info.get("winner")))), None)
            job_id = str(won) if won is not None else (str(latest["job_id"]) if latest.get("job_id") else None)
            candidates = [] if won is not None or not babysit else list(dict.fromkeys(
                str(job) for job in as_list(babysit.get("candidates")) if job is not None and JOB_ID.fullmatch(str(job))))
            settings = _build_settings(latest.get("extra"), batch.defaults if batch else None)
            # task_builder.round_cap: <build root>/max_rounds.json lowers --rounds while a build runs (hengyuf,
            # 2026-10-17: the official1 builds stop after round 2)
            cap = (read_json(root / "max_rounds.json") or {}).get("max_rounds") if root else None
            if isinstance(cap, int) and not isinstance(cap, bool) and 0 <= cap < settings["rounds"]:
                settings["rounds"] = cap
            first = [when for when in (parse_time(launch.get("submitted_at")) for launch in launches) if when is not None]
            runs.append({"run": run_dir.name, "run_dir": run_dir, "task_id": task, "root": root,
                         "first_submitted_at": min(first) if first else None,  # v1.8: the run's first launch
                         "repo": root.name.replace("__", "/", 1) if root else None,
                         "job_id": job_id, "candidates": candidates,
                         # v1.8: each job once, in launch order (a resolved leg repeats its winner's line)
                         "jobs": list(dict.fromkeys(str(launch.get("job_id")) for launch in mine if launch.get("job_id"))),
                         "submitted_at": parse_time(latest.get("submitted_at")),
                         "partition": latest.get("partition"),
                         "log": None if batch else self.log_path(latest if str(latest.get("job_id")) == job_id
                                                                 else {"job_id": job_id}),
                         "max_rounds": settings["rounds"], "branches": settings["branches"], "batch": batch})
        return runs

    def pick_candidate(self, run, jobs):
        """v1.8: the job an unresolved leg of the queue runner shows, of its candidate jobs (the babysit helper submits
        two; the runner writes the winner to launch.jsonl a pass later): RUNNING, else CONFIGURING / COMPLETING, else
        PENDING, else the first; its log with it. So the run never reads "stopped" while its losing candidate has just
        exited."""
        rank = {"RUNNING": 0, "CONFIGURING": 1, "COMPLETING": 1, "PENDING": 2}
        order = run["candidates"]
        best = min(order, key=lambda job: (rank.get(as_dict(jobs.get(job)).get("state"), 3), order.index(job)))
        if best != run["job_id"]:
            run["job_id"], run["log"] = best, self.log_path({"job_id": best})
        if best not in run["jobs"]:
            run["jobs"].append(best)

    def log_path(self, launch):
        """The job log, only ever inside synth/runs/slurm/ (launch.jsonl paths are not trusted blindly)."""
        job = str(launch.get("job_id") or "")
        logs = self.src.slurm_logs
        candidates = []
        if isinstance(launch.get("log"), str):
            candidates.append(logs / Path(launch["log"]).name)
        if job.isdigit():
            candidates += sorted(logs.glob(f"*-{job}.out"))
        for path in candidates:
            if path.is_file():
                return path
        return None

    def task_doc(self, run, job, tree):
        """data/task/<task_id>.json for one official run (plus summary numbers for index.json)."""
        root = run["root"]
        analysis = self.guard(f"run {run['run']} failure analysis", self.failure_analysis,
                              run["run_dir"] / "analysis" / run["task_id"])
        plan = self.guard(f"run {run['run']} plan", self.plan, root, tree) if root else None
        audits = self.guard(f"run {run['run']} audit", self.failure_matches, run["run_dir"], default={}) if tree else {}
        for node in (tree or {}).get("nodes", []):  # every node: the audit of its built task, else null
            node["failure_match"] = audits.get(node["build"]["task_id"]) if node["build"]["task_id"] else None
        finished = bool(root and (root / "summary.json").is_file())
        status = _run_status(job, finished, run.get("log"), self.now)
        stage = _stage(status, tree, analysis, root)
        job = job or {}
        log = run.get("log")
        tail = read_tail_lines(log) if log else None
        doc = {"schema": SCHEMA, "generated_at": self.now, "task_id": run["task_id"], "run": run["run"],
               "repo": run["repo"], "build_root": str(root) if root else None, "max_rounds": run["max_rounds"],
               "branches": run["branches"],
               "job": {"job_id": run["job_id"], "slurm_state": job.get("state"),
                       "partition": job.get("partition") or run.get("partition"), "node": job.get("node"),
                       "elapsed": job.get("elapsed"), "time_left": job.get("time_left"),
                       "log_path": str(log) if log else None,
                       "log_tail": [text_field(line, 300) for line in tail] if tail else [],
                       "stage": stage, "status": status, "submitted_at": run["submitted_at"],
                       "started_at": job.get("start"), "ended_at": None if status in ("running", "pending") else
                       job.get("end"), "jobs": run["jobs"]},
               "failure_analysis": analysis, "plan": plan,
               "rounds": tree["rounds"] if tree else [], "nodes": tree["nodes"] if tree else []}
        nodes = [node for node in doc["nodes"] if node["status"] != "abandoned"]
        summary = {"run": run["run"], "task_id": run["task_id"], "repo": run["repo"], "job_id": run["job_id"],
                   "slurm_state": job.get("state"), "partition": doc["job"]["partition"],
                   "submitted_at": run["submitted_at"], "started_at": job.get("start"),
                   "ended_at": doc["job"]["ended_at"], "status": status, "stage": stage,
                   "rounds_started": sum(info["status"] != "pending" for info in doc["rounds"]),
                   "accepted": sum(node["decision"] in ACCEPTED for node in nodes),
                   "tasks_built": sum(as_dict(node["build"]).get("ok") is True for node in nodes),
                   "tasks_scored": sum(node["score"] is not None for node in nodes),
                   "audit": _audit_counts(nodes, doc["nodes"]),
                   # v1.8: whether it is its task's latest official run, and where its document is
                   "latest": run.get("latest", True), "detail": run.get("detail", f"task/{run['task_id']}.json")}
        return {"doc": doc, "summary": summary, "run": run, "tree": tree}

    def failure_matches(self, run_dir):
        """{task_id: NODE.failure_match} of one official run, from runs/<run>/audit/failure_match/<task_id>.json: the
        audits of its eligible tasks, written by auditor agents while the dashboard runs (CONTRACT "v1.3", "v1.4"). A
        file that cannot be read (yet) is skipped; one that breaks the format is skipped and reported in the cycle's
        errors."""
        directory = run_dir / AUDIT_DIR
        found = {}
        for name in sorted(listdir(directory)):
            if not name.endswith(".json") or not SAFE_TASK.fullmatch(name[:-len(".json")]):
                continue  # hidden temp files of an atomic write, backups, other formats
            entry = self.memo.get("failure_match", directory / name, _failure_match)
            if isinstance(entry, dict) and "error" in entry:
                self.error(f"run {run_dir.name} audit", f"{name}: {entry['error']}; ignored")
            elif isinstance(entry, dict):
                found[entry["task_id"]] = entry["match"]
        return found

    def failure_analysis(self, directory):
        """The target's failure patterns (failure_analyzer result.json + failure_analysis.md + inputs/stats.json)."""
        result = self.json(directory / "result.json")
        markdown = self.excerpt(directory / "failure_analysis.md", 60000)
        if not isinstance(result, dict) and markdown is None:
            return None
        result = as_dict(result)
        patterns = []
        for pattern in as_list(result.get("patterns")):
            if not isinstance(pattern, dict):
                continue
            known = ("name", "category", "description", "student_slots", "teacher_contrast")
            extra = [f"{key}: {value if isinstance(value, str) else json.dumps(value)}"
                     for key, value in pattern.items() if key not in known]
            patterns.append({"name": text_field(_str(pattern.get("name")), 300),
                             "category": text_field(_str(pattern.get("category")), 200),
                             "description": text_field(_str(pattern.get("description")), 4000),
                             "student_slots": [str(slot) for slot in as_list(pattern.get("student_slots"))],
                             "teacher_contrast": text_field(_str(pattern.get("teacher_contrast")), 4000),
                             "evidence": text_field("\n".join(extra), 4000) if extra else None})
        signature = result.get("transferable_signature")
        stats = as_dict(as_dict(self.json(directory / "inputs/stats.json")).get("pass_stats"))
        return {"patterns": patterns,
                "transferable_signature": text_field(signature if isinstance(signature, str) or signature is None
                                                     else json.dumps(signature), 6000),
                "excluded": sanitize(result.get("excluded") or []),
                "pass_stats": {model: {"pass": as_dict(value).get("pass"), "fail": as_dict(value).get("fail"),
                                       "excluded": as_dict(value).get("excluded"), "rate": as_dict(value).get("rate")}
                               for model, value in stats.items()} or None,
                "markdown": markdown}

    def plan(self, root, tree):
        """The mask plan's candidates (plan/stage.json, else candidates.json), the built ones marked."""
        data = self.json(root / "plan/stage.json") or self.json(root / "plan/candidates.json")
        candidates = as_list(as_dict(data).get("candidates"))
        excerpt = self.excerpt(root / "plan/mask_plan.md", 4000)
        if not candidates and excerpt is None:
            return None
        nodes = {node["id"]: node for node in (tree or {}).get("nodes", [])}
        out = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            node = nodes.get(f"cand-{candidate.get('id')}")
            tests, files = candidate.get("tests"), candidate.get("files")
            out.append({"id": _str(candidate.get("id")), "title": text_field(_str(candidate.get("title")), 300),
                        "summary": text_field(_str(candidate.get("summary")), 1000),
                        "files": [str(item) for item in as_list(files)][:40],
                        "tests": len(tests) if isinstance(tests, list) else _int(tests),
                        "target_patterns": [str(item) for item in as_list(candidate.get("target_patterns"))][:10],
                        "built": bool(node and as_dict(node["build"]).get("ok") is True),
                        "status": node["status"] if node else None, "decision": node["decision"] if node else None})
        return {"candidates": out, "mask_plan_excerpt": excerpt}

    def prune_slots(self, docs):
        live = {probe["path"] for item in docs for probe in (item["tree"] or {}).get("probes", []) if probe["live"]}
        slots = as_dict(self.state.get("slots"))
        self.state["slots"] = {key: value for key, value in slots.items() if key.split("|", 1)[0] in live}

    def write_task_docs(self, docs):
        keep = set()
        for item in docs:
            if not item["run"]["latest"]:
                continue
            name = f"{item['doc']['task_id']}.json"
            keep.add(name)
            self.write(f"task/{name}", item["doc"])
        for name in listdir(self.data_dir / "task"):  # a run that disappeared leaves no stale task file
            if name.endswith(".json") and name not in keep and SAFE_TASK.fullmatch(name[:-5]):
                try:
                    os.unlink(self.data_dir / "task" / name)
                except OSError:
                    pass

    def write_official_docs(self, docs):
        """v1.8: data/run/hengyuf/<run>.json for every official run that is not its task's latest: the same document as
        a task file (task_doc()), written when its content (all but generated_at) changed or the file is not the one
        written last; generated_at (= runs[].detail_generated_at) is the cycle that last changed it. Files of runs that
        are no older official run any more (e.g. the latest again) are removed."""
        directory, keep = self.data_dir / OFFICIAL_RUNS_DIR, set()
        for item in docs:
            run = item["run"]
            if run["latest"]:
                continue
            if not directory.is_dir():
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory.parent, 0o700)
                os.chmod(directory, 0o700)
            detail, doc = run["detail"], item["doc"]
            keep.add(detail)
            path = self.data_dir / detail
            body = encode_json(dict(doc, generated_at=None))
            key, signature = (str(self.data_dir), detail), file_signature(path)
            cached = _OFFICIAL_CACHE.get(key)
            if not cached or signature is None or cached["file"] != signature:  # not the file written last: read it
                old = read_json(path) if signature is not None else None
                stamp = old.get("generated_at") if isinstance(old, dict) else None
                cached = ({"body": encode_json(dict(old, generated_at=None)), "stamp": stamp, "file": signature}
                          if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) else None)
            changed = not cached or cached["body"] != body
            doc["generated_at"] = self.now if changed else cached["stamp"]
            item["summary"]["detail_generated_at"] = doc["generated_at"]
            if changed:
                data = encode_json(doc)
                write_bytes_atomic(path, data)
                self.written[detail] = len(data)
            _OFFICIAL_CACHE[key] = {"body": body, "stamp": doc["generated_at"], "file": file_signature(path)}
        for name in listdir(directory):
            if name.endswith(".json") and f"{OFFICIAL_RUNS_DIR}/{name}" not in keep and SAFE_TASK.fullmatch(name[:-5]):
                try:
                    os.unlink(directory / name)
                except OSError:
                    pass

    # ========================================================================================== batches (v1.6)
    def batch_docs(self):
        """Every batch's runs as documents: read_batch() finds them, Tree builds each run's tree exactly as for an
        official run (its probes are never live: the run has ended), batch_doc() makes the document, match_batch()
        checks that every task of the batch is an accepted node. A run whose fingerprint did not change reuses its
        last build (BATCH_REFRESH_S). One broken batch or run never stops the cycle."""
        items, refreshes = [], 0
        for batch in self.src.batches:
            info = self.guard(f"batch {batch.source}", self.read_batch, batch)
            if info is None:
                continue
            self.batches.append(info)
            mine = []
            for run in info["runs"]:
                where = f"batch {batch.source} run {run['run']}"
                key = (str(self.data_dir), str(run["run_dir"]), run["task_id"])
                fingerprint = self.guard(f"{where} fingerprint", self.batch_fingerprint, run)
                cached = _BATCH_CACHE.get(key)
                if cached and fingerprint is not None and cached["fingerprint"] == fingerprint and (
                        self.now - cached["built_at"] < BATCH_REFRESH_S or refreshes >= BATCH_REFRESHES):
                    self.memo.used |= cached["keys"]  # its files stay cached for the next rebuild
                    mine.append(dict(cached["item"], run=run))
                    continue
                refreshes += bool(cached and cached["fingerprint"] == fingerprint)  # rebuilt for its age alone
                before, errors = set(self.memo.used), len(self.errors)
                tree = self.guard(f"{where} tree", Tree(self, run, run["root"]).build) if run["root"] else None
                item = self.guard(where, self.batch_doc, run, tree)
                if not item:
                    _BATCH_CACHE.pop(key, None)
                    continue
                # the file's bytes; unchanged content keeps the stamp (and the file) of the cycle that last changed it
                item["doc"]["generated_at"] = cached["stamp"] if cached else self.now
                data = encode_json(item["doc"])
                if not cached or data != cached["data"]:
                    item["doc"]["generated_at"] = self.now
                    data = encode_json(item["doc"])
                item["summary"]["detail_generated_at"] = item["doc"]["generated_at"]
                item["data"] = data
                if len(self.errors) == errors:  # a run built while one of its sources failed is built again next cycle
                    item["cache_key"] = key
                    _BATCH_CACHE[key] = {"fingerprint": fingerprint, "built_at": self.now, "item": item, "data": data,
                                         "stamp": item["doc"]["generated_at"], "keys": self.memo.used - before,
                                         "file": cached["file"] if cached and data == cached["data"] else None}
                else:
                    _BATCH_CACHE.pop(key, None)
                mine.append(item)
            self.guard(f"batch {batch.source}", self.match_batch, info, mine)
            items += mine
        return items

    def batch_fingerprint(self, run):
        """A few stats that change when an ended batch run changes after all (resumed, a task scored, accepted or
        withdrawn): the build root and its round dirs (new candidates, probes, scored dirs and seeds.json land in
        them), summary.json, plan/stage.json, launch.jsonl, accepted/ and accepted/tasks.jsonl, the target's
        result.json; plus the run's settings and its batch tasks (outputs/tasks.jsonl and provenance.json)."""
        run_dir, root = run["run_dir"], run["root"]
        paths = [run_dir / "launch.jsonl", run_dir / "accepted", run_dir / "accepted/tasks.jsonl",
                 run_dir / "analysis" / run["task_id"] / "result.json"]
        if root:
            paths += [root, root / "summary.json", root / "plan/stage.json"]
            paths += sorted(root / name for name in listdir(root) if ROUND.fullmatch(name))
        return (tuple((str(path), file_signature(path)) for path in paths), tuple(sorted(run["batch_tasks"])),
                run["max_rounds"], run["branches"], run["job_id"], run["submitted_at"], tuple(run["jobs"]))

    def read_batch(self, batch):
        """One batch: its tasks (the task ids of outputs/tasks.jsonl, in file order, once each), the run and target
        that built each (outputs/<task_id>/provenance.json task_dir) and read_run() of every such run. Only outputs/
        and those runs' directories join the read roots: other runs of that tree are never opened."""
        if os.path.realpath(batch.outputs) != os.path.join(os.path.realpath(batch.root), "outputs"):
            raise RuntimeError(f"{batch.outputs} leaves {batch.root} (a symlink?); not read")
        allow_reads(batch.outputs)
        rows = self.memo.get("batch_tasks", batch.outputs / "tasks.jsonl", _batch_task_ids)
        if rows is None:
            raise RuntimeError(f"{batch.outputs / 'tasks.jsonl'} missing")
        task_ids = list(dict.fromkeys(rows))
        if len(task_ids) < len(rows):
            self.error(f"batch {batch.source}", f"outputs/tasks.jsonl lists {len(rows) - len(task_ids)} task(s) twice")
        info = {"source": batch.source, "name": batch.name, "label": batch.label, "root": str(batch.root),
                "outputs": str(batch.outputs),
                "tasks": len(task_ids), "task_ids": task_ids, "origin": {}, "runs": [], "unmatched": [],
                "not_in_batch": [], "in_batch": 0, "accepted": 0}
        by_run = {}
        for task_id in task_ids:
            origin = _batch_origin(batch, task_id, self.json(batch.outputs / task_id / "provenance.json")
                                   if SAFE_TASK.fullmatch(task_id) else None)
            if isinstance(origin, str):
                info["unmatched"].append(f"{task_id}: {origin}")
                continue
            info["origin"][task_id] = origin
            by_run.setdefault(origin[0], {}).setdefault(origin[1], set()).add(task_id)
        runs_real = os.path.realpath(batch.runs)
        for name, targets in sorted(by_run.items()):
            run_dir = batch.runs / name
            # a run dir that is no directory of synth/runs itself (e.g. a symlink out of it) never joins the read roots
            if not run_dir.is_dir() or os.path.realpath(run_dir) != os.path.join(runs_real, name):
                for target in sorted(targets):
                    for task_id in sorted(targets[target]):
                        info["unmatched"].append(f"{task_id}: no run dir synth/runs/{name}")
                        info["origin"].pop(task_id, None)  # reported here, not again by match_batch()
                continue
            allow_reads(run_dir)
            runs = self.guard(f"batch {batch.source} run {name}", self.read_run, run_dir, batch, default=[])
            for run in runs:
                run["batch_tasks"] = targets.get(run["task_id"], set())
                # one file per run; a run that builds several targets (none so far) gets one per target
                run["detail"] = (f"run/{batch.source}/{name}.json" if len(runs) == 1
                                 else f"run/{batch.source}/{name}--{run['task_id']}.json")
            built = {run["task_id"] for run in runs}
            for target in sorted(set(targets) - built):
                for task_id in sorted(targets[target]):
                    info["unmatched"].append(f"{task_id}: run {name} has no build of target {target}")
                    info["origin"].pop(task_id, None)
            info["runs"] += runs
        info["runs_count"] = len(info["runs"])
        info["targets"] = len({run["task_id"] for run in info["runs"]})
        return info

    def batch_doc(self, run, tree):
        """data/run/<source>/<run>.json for one run of a batch: task_doc()'s document of that run's own tree, plus the
        batch fields. Never audited: no audit file is read, every node has failure_match null, audit_eligible false and
        audit {"status": "not_audited"}, as has the run. A node is in_batch when its task is one of the batch's tasks
        that provenance.json attributes to this run. The batch's runs have ended, so their Slurm jobs are not polled and
        their logs not read: status finished (summary.json), else stopped; the job fields come from launch.jsonl."""
        batch, root = run["batch"], run["root"]
        where = f"batch {batch.source} run {run['run']}"
        analysis = self.guard(f"{where} failure analysis", self.failure_analysis,
                              run["run_dir"] / "analysis" / run["task_id"])
        plan = self.guard(f"{where} plan", self.plan, root, tree) if root else None
        nodes = (tree or {}).get("nodes", [])
        for node in nodes:
            task_id = node["build"]["task_id"]
            node.update(failure_match=None, audit_eligible=False, audit=dict(NOT_AUDITED),
                        in_batch=bool(task_id) and task_id in run["batch_tasks"])
        status = "finished" if root and (root / "summary.json").is_file() else "stopped"
        stage = _batch_stage(status, tree, analysis, root)
        doc = {"schema": SCHEMA, "generated_at": self.now, "task_id": run["task_id"], "run": run["run"],
               "repo": run["repo"], "build_root": str(root) if root else None, "max_rounds": run["max_rounds"],
               "branches": run["branches"],
               "job": {"job_id": run["job_id"], "slurm_state": None, "partition": run.get("partition"), "node": None,
                       "elapsed": None, "time_left": None, "log_path": None, "log_tail": [], "stage": stage,
                       "status": status, "submitted_at": run["submitted_at"], "started_at": None, "ended_at": None,
                       "jobs": run["jobs"]},
               "failure_analysis": analysis, "plan": plan,
               "rounds": tree["rounds"] if tree else [], "nodes": nodes,
               "source": batch.source, "source_name": batch.name, "source_label": batch.label,
               "root": str(run["run_dir"]), "outputs": str(batch.outputs), "batch": len(run["batch_tasks"]),
               "audit": dict(NOT_AUDITED)}
        shown = [node for node in nodes if node["status"] != "abandoned"]
        summary = {"run": run["run"], "task_id": run["task_id"], "repo": run["repo"], "job_id": run["job_id"],
                   "slurm_state": None, "partition": doc["job"]["partition"], "submitted_at": run["submitted_at"],
                   "started_at": None, "ended_at": None, "status": status, "stage": stage,
                   "rounds_started": sum(info["status"] != "pending" for info in doc["rounds"]),
                   "accepted": sum(node["decision"] in ACCEPTED for node in shown),
                   "tasks_built": sum(as_dict(node["build"]).get("ok") is True for node in shown),
                   "tasks_scored": sum(node["score"] is not None for node in shown),
                   "audit": dict(NOT_AUDITED), "source": batch.source, "source_name": batch.name,
                   "source_label": batch.label, "root": str(run["run_dir"]), "batch": len(run["batch_tasks"]),
                   "detail": run["detail"]}
        return {"doc": doc, "summary": summary, "run": run, "tree": tree}

    def match_batch(self, info, items):
        """Every task of the batch must be an accepted node of the run that built it; mismatches go to the cycle's
        errors and index.json batches[].unmatched (and accepted nodes outside the batch to not_in_batch)."""
        nodes = {}
        for item in items:
            for node in item["doc"]["nodes"]:
                if node["build"]["task_id"]:
                    nodes[(item["run"]["run"], node["build"]["task_id"])] = node
                info["in_batch"] += bool(node["in_batch"])
                accepted = node["decision"] in ACCEPTED and node["status"] != "abandoned"
                info["accepted"] += accepted
                if accepted and not node["in_batch"]:
                    info["not_in_batch"].append(f"{item['run']['run']} {node['id']} ({node['build']['task_id']})")
        for task_id, (run, _target) in info["origin"].items():
            node = nodes.get((run, task_id))
            if node is None:
                info["unmatched"].append(f"{task_id}: no node of run {run} builds it")
            elif node["decision"] not in ACCEPTED or not node["in_batch"]:
                info["unmatched"].append(f"{task_id}: {run} {node['id']} is {node['decision']}, not accepted")
        for problem in info["unmatched"]:
            self.error(f"batch {info['source']}", f"outputs task {problem}")
        for name in info["not_in_batch"]:
            self.error(f"batch {info['source']}", f"accepted node {name} is not in outputs/tasks.jsonl")

    def write_batch_docs(self, items):
        """data/run/<source>/<run>.json for every batch run, written when its content changed (or the file is not the
        one written last); files of runs no longer listed are removed."""
        keep = set()
        for item in items:
            directory = self.data_dir / "run" / item["run"]["batch"].source
            if not directory.is_dir():
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory, 0o700)
            detail = item["run"]["detail"]
            keep.add(detail)
            entry = _BATCH_CACHE.get(item.get("cache_key"))
            if entry is None:  # not cached (built with errors): written every cycle
                write_bytes_atomic(self.data_dir / detail, item["data"])
                self.written[detail] = len(item["data"])
            elif entry["file"] is None or entry["file"] != file_signature(self.data_dir / detail):
                write_bytes_atomic(self.data_dir / detail, entry["data"])
                entry["file"] = file_signature(self.data_dir / detail)
                self.written[detail] = len(entry["data"])
        for source in listdir(self.data_dir / "run"):
            if f"run/{source}" == OFFICIAL_RUNS_DIR:  # v1.8: hengyuf's older official runs (write_official_docs)
                continue
            for name in listdir(self.data_dir / "run" / source):
                if name.endswith(".json") and f"run/{source}/{name}" not in keep and SAFE_TASK.fullmatch(name[:-5]):
                    try:
                        os.unlink(self.data_dir / "run" / source / name)
                    except OSError:
                        pass

    def index_doc(self, tasks, docs, batch_docs=(), explorer=None, queues=None):
        by_task = {item["doc"]["task_id"]: item["summary"]["run"] for item in docs if item["run"]["latest"]}
        # v1.8: every official run of a task, its latest first, then the older ones (newest launch first)
        official = {}
        for item in sorted(docs, key=lambda item: (not item["run"]["latest"], -(item["run"]["submitted_at"] or 0),
                                                   item["run"]["run"])):
            official.setdefault(item["doc"]["task_id"], []).append(
                {"run": item["run"]["run"], "latest": item["run"]["latest"], "detail": item["run"]["detail"]})
        for task in tasks:
            task["run"] = by_task.get(task["task_id"])
        known = {task["task_id"] for task in tasks}
        for task_id in sorted(set(by_task) - known):  # a run on a task the targets do not list (or cannot be read
            # this cycle): still a task entry, without accuracy, so the page lists the conversion
            tasks.append({"task_id": task_id, "task_index": len(tasks), "f2p_total": None,
                          "accuracy": {"kimi": None, "super_vl": None, "sc14": None}, "eligible": False,
                          "queue_rank": None, "run": by_task[task_id]})
        for task in tasks:
            if task["task_id"] in official:
                task["official_runs"] = official[task["task_id"]]
        batch = self.guard("batches", self.batch_index, tasks, batch_docs, default=None) if self.src.batches else None
        index = {"schema": SCHEMA, "generated_at": self.now, "generated_at_iso": iso(self.now),
                 "collect_seconds": round(self.elapsed(), 2), "errors": list(self.errors), "models": MODELS,
                 "runs": [item["summary"] for item in docs] + (batch["runs"] if batch else []), "tasks": tasks}
        if self.src.batches:  # v1.6: each batch's own totals (never added to the official runs')
            index["batches"] = batch["batches"] if batch else []
        # v1.7: where the Explorer's tables are and the cycle that last changed them (null: not built this cycle, see
        # "errors"; an earlier explorer.json may still be on disk)
        index["explorer"] = explorer
        if self.src.queues:  # v1.8: each synthesis queue's runner state (a queue whose files cannot be read: none)
            index["queues"] = list(queues or [])
        return index

    def batch_index(self, tasks, batch_docs):
        """v1.6, index.json's batch parts: a DeepSWE task that batch runs converted lists them in "batch_runs" (oldest
        launch first; "run" stays its official run, or null; a target the targets do not list gets a stub entry, like a
        run's), the batch runs' runs[] entries, and each batch's own totals ("batches")."""
        batch_runs = {}
        for item in sorted(batch_docs, key=lambda item: (item["summary"]["submitted_at"] or 0, item["summary"]["run"])):
            batch_runs.setdefault(item["doc"]["task_id"], []).append(
                {"source": item["summary"]["source"], "run": item["summary"]["run"]})
        known = {task["task_id"] for task in tasks}
        for task_id in sorted(set(batch_runs) - known):
            tasks.append({"task_id": task_id, "task_index": len(tasks), "f2p_total": None,
                          "accuracy": {"kimi": None, "super_vl": None, "sc14": None}, "eligible": False,
                          "queue_rank": None, "run": None})
        for task in tasks:
            if task["task_id"] in batch_runs:
                task["batch_runs"] = batch_runs[task["task_id"]]
        return {"runs": [item["summary"] for item in batch_docs], "batches": [{
            "source": info["source"], "name": info["name"], "label": info["label"], "root": info["root"],
            "outputs": info["outputs"], "tasks": info["tasks"], "runs": info["runs_count"], "targets": info["targets"],
            "in_batch": info["in_batch"], "accepted": info["accepted"], "unmatched": info["unmatched"][:50],
            "not_in_batch": info["not_in_batch"][:50], "audit": dict(NOT_AUDITED)} for info in self.batches]}

    # ========================================================================================== queues (v1.8)
    def queue_doc(self, queue):
        """index.json queues[] entry of one synthesis queue (CONTRACT "v1.8"), from its runner's four files and nothing
        else: queue.json gives the runs in queue order (run, DeepSWE task, repo, the batch whose failure analysis it
        reuses) and max_active; state.json each run's status (queued until the runner records one), leg, winner and
        Slurm candidates, the budget (the active hub key's NAME and its remaining dollars, never anything else of it),
        serving and stop_launching; heartbeat.json the runner (age_s and alive as of read_at, the time the files were
        read: a heartbeat at most QUEUE_STALE_S old whose note is not "stopped"); runner.log its last QUEUE_LOG_LINES
        lines, redacted, the runner's pid masked. Each file is opened at its own path without following a symlink there
        (one is refused, and named in the errors). None when the queue's directory does not exist; an unreadable
        queue.json is an error (the cycle goes on)."""
        if not queue.root.is_dir():
            return None
        for kind, path in queue.files.items():  # v1.8: a symlink at a runner file's path is never followed
            try:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    self.error(f"queue {queue.name}", f"{path.name} is a symlink: not read")
            except OSError:
                pass  # not there (yet)
        spec = read_json(queue.files["queue"], exact=True)
        if not isinstance(spec, dict) or not isinstance(spec.get("runs"), list):
            raise RuntimeError("queue.json missing, unreadable or without a run list")
        state = as_dict(read_json(queue.files["state"], exact=True))
        beat = as_dict(read_json(queue.files["heartbeat"], exact=True))
        read_at = self.clock()  # v1.8: the heartbeat's age is measured when the files were read, not at the cycle's start
        recorded = as_dict(state.get("runs"))
        runs, seen = [], set()
        for position, entry in enumerate(spec["runs"]):
            entry = as_dict(entry)
            name, task = entry.get("run"), entry.get("task")
            if not (isinstance(name, str) and SAFE_NAME.fullmatch(name) and isinstance(task, str)
                    and SAFE_TASK.fullmatch(task)) or name in seen:
                self.error(f"queue {queue.name}", f"queue.json runs[{position}]: no valid run and task (or twice); skipped")
                continue
            seen.add(name)
            info = as_dict(recorded.get(name))
            status = info.get("status", "queued")
            if status not in QUEUE_STATUSES:
                self.error(f"queue {queue.name}", f"{name}: unknown status {clip(redact(str(status)), 40)!r}")
                status = None
            leg = info.get("leg")
            winner = info.get("winner")
            runs.append({"run": name, "task": task, "repo": text_field(_str(entry.get("repo")), 200),
                         "batch": text_field(_str(entry.get("batch")), 60), "status": status,
                         "leg": leg if isinstance(leg, int) and not isinstance(leg, bool) and leg >= 0 else 0,
                         "winner": str(winner) if winner is not None and JOB_ID.fullmatch(str(winner)) else None,
                         "candidates": [str(job) for job in as_list(info.get("candidates"))
                                        if job is not None and JOB_ID.fullmatch(str(job))][:8]})
        at = number(beat.get("at")) if not isinstance(beat.get("at"), bool) else None
        age = round(max(0.0, read_at - at), 1) if at is not None else None
        note = text_field(_str(beat.get("note")), 40)
        job = beat.get("slurm_job_id")
        runner = {"heartbeat_at": at, "age_s": age, "host": text_field(_str(beat.get("host")), 200),
                  "slurm_job_id": str(job) if job is not None and JOB_ID.fullmatch(str(job)) else None, "note": note,
                  "alive": at is not None and age <= QUEUE_STALE_S and note != "stopped"}
        budget = state.get("budget")
        if isinstance(budget, dict):  # the key's NAME and a number: nothing else (never the error text)
            key, remaining = budget.get("key"), budget.get("remaining")
            budget = {"key": key if isinstance(key, str) and KEY_NAME.fullmatch(key) else None,
                      "remaining": round(number(remaining), 2) if number(remaining) is not None
                      and not isinstance(remaining, bool) else None,
                      "at": parse_time(budget.get("at"))}
        else:
            budget = None
        settings = as_dict(spec.get("settings"))
        active = settings.get("max_active")
        tail = read_tail_lines(queue.files["log"], QUEUE_LOG_LINES, exact=True) or []
        return {"name": queue.name, "root": str(queue.root), "read_at": round(read_at, 3),
                "max_active": active if isinstance(active, int) and not isinstance(active, bool) and active > 0 else None,
                "runner": runner, "budget": budget, "serving": state.get("serving") is True,
                "stop_launching": state.get("stop_launching") is True,
                "updated_at": parse_time(state.get("updated_at")), "finished_at": parse_time(state.get("finished_at")),
                "runs": runs, "log_tail": [text_field(QUEUE_PID.sub("pid ⋯", line), 300) for line in tail]}

    # ========================================================================================== explorer (v1.7)
    def explorer_doc(self, docs, batch_docs):
        """data/explorer.json (CONTRACT "v1.7"): the runs whose documents this cycle writes (v1.8: every official run,
        data/task/<task_id>.json for a target's latest run and data/run/hengyuf/<run>.json for its older ones;
        data/run/<source>/<run>.json for a batch's runs), the DeepSWE targets with their failure patterns, one row per
        built synthetic task of those documents (a NODE with build.task_id, in document order) and one row per rollout
        its probe lists (NODE.probe.rollouts, in order). Every value is copied or derived from those very NODE dicts
        (CONTRACT "v1.7" names the source of each column), so the Explorer's counts are the files' by construction. A
        batch's tasks are never audited ("not_audited")."""
        items = [(OFFICIAL_SOURCE, item) for item in docs]
        sources = [dict(OFFICIAL_SOURCE)]
        for batch in self.src.batches:
            sources.append({"source": batch.source, "name": batch.name, "label": batch.label, "audited": False})
        by_source = {source["source"]: source for source in sources}
        items += [(by_source[item["run"]["batch"].source], item) for item in batch_docs]
        totals = {source["source"]: {"runs": 0, "tasks": 0, "trajectories": 0, "accepted": 0, "audited": 0,
                                     "unmatched_patterns": 0, "rates_differ": 0} for source in sources}
        runs, tasks, trajectories, targets = [], [], [], {}
        for source, item in items:
            doc, run = item["doc"], item["run"]
            batch = run.get("batch")
            name, target = doc["run"], doc["task_id"]
            # app.js taskHash(task, null, run): a batch run and (v1.8) an older official run need run=
            latest = None if batch else bool(run.get("latest", True))
            link = "#task=" + _uri(target) + (f"&run={_uri(name)}" if batch or latest is False else "")
            runs.append({"run": name, "source": source["source"], "deepswe_task": target,
                         "build_root": doc.get("build_root"), "max_rounds": doc.get("max_rounds"),
                         "detail": run.get("detail") or f"task/{target}.json", "link": link, "latest": latest})
            totals[source["source"]]["runs"] += 1
            info = targets.setdefault(target, {"deepswe_task": target, "runs": [], "patterns": [], "exact": {},
                                               "loose": {}})
            info["runs"].append(len(runs) - 1)
            # the target's failure patterns: its first run's list as is (P1..Pn of that run's task view, the official
            # run's when there is one: official runs come first), then any other name a later run's analysis adds
            for pattern in as_list(as_dict(doc.get("failure_analysis")).get("patterns")):
                pattern = as_dict(pattern)
                pattern_name = pattern.get("name") if isinstance(pattern.get("name"), str) else None
                if len(info["runs"]) > 1 and (pattern_name is None or pattern_name in info["exact"]):
                    continue
                if pattern_name is not None:
                    info["exact"].setdefault(pattern_name, len(info["patterns"]))
                    info["loose"].setdefault(_loose(pattern_name), len(info["patterns"]))
                info["patterns"].append({"name": pattern_name, "category": pattern.get("category")
                                         if isinstance(pattern.get("category"), str) else None})
            for node in doc["nodes"]:
                build = as_dict(node.get("build"))
                if not build.get("task_id"):
                    continue
                score = node["score"] if isinstance(node.get("score"), dict) else None
                probe = node["probe"] if isinstance(node.get("probe"), dict) else {}
                audit = _explorer_audit(node, source["audited"])
                yes = partial = None
                if audit in AUDIT_VERDICTS:  # the target patterns the counted audit marks reproduced
                    yes, partial = set(), set()
                    for pattern in as_list(node["failure_match"].get("patterns")):
                        pattern = as_dict(pattern)
                        pattern_name = pattern.get("name")
                        index = (info["exact"].get(pattern_name, info["loose"].get(_loose(pattern_name)))
                                 if isinstance(pattern_name, str) else None)
                        if index is None:
                            totals[source["source"]]["unmatched_patterns"] += 1
                        elif pattern.get("reproduced") == "yes":
                            yes.add(index)
                        elif pattern.get("reproduced") == "partial":
                            partial.add(index)
                    yes, partial = sorted(yes), sorted(partial - yes)
                accepted = node.get("decision") in ACCEPTED
                # the teacher's cheat audit (score step, teacher_audit.json): which kimi passes were clean
                cheats = {entry.get("slot"): entry.get("cheated") for entry in
                          as_list(as_dict(node.get("details")).get("teacher_audit")) if isinstance(entry, dict)}
                differ = _rates_differ(score, as_list(probe.get("rollouts")), cheats) if score else False
                tasks.append([len(runs) - 1, node["id"], build["task_id"], node.get("round"), node.get("branch"),
                              node.get("kind"), node.get("title"), node.get("decision"), accepted,
                              bool(node.get("withdrawn")), node.get("in_batch") is True,
                              _finite(score.get("similarity")) if score else None,
                              _finite(score.get("student_rate")) if score else None,
                              _finite(score.get("teacher_clean_rate")) if score else None,
                              _finite(as_dict(probe.get("teacher")).get("rate")) if score else None,
                              _valid_pair(score.get("valid")) if score else None,
                              audit, yes, partial, _finite(build.get("f2p")), _finite(build.get("p2p")),
                              score is not None, differ])
                total = totals[source["source"]]
                total["tasks"] += 1
                total["accepted"] += accepted
                total["audited"] += audit in AUDIT_VERDICTS
                total["rates_differ"] += differ
                for rollout in as_list(probe.get("rollouts")):
                    if not isinstance(rollout, dict):
                        continue
                    clean = None
                    if rollout.get("model") == TEACHER and rollout.get("status") == "pass":
                        cheated = cheats.get(rollout.get("slot"))
                        clean = (not cheated) if isinstance(cheated, bool) else None
                    trajectories.append([len(tasks) - 1, rollout.get("slot"), rollout.get("model"),
                                         rollout.get("status"), _finite(rollout.get("reward")), rollout.get("f2p"),
                                         rollout.get("p2p"), _finite(rollout.get("tool_calls")),
                                         _finite(rollout.get("patch_chars")), bool(rollout.get("agent_error")),
                                         bool(rollout.get("verifier_error")), clean])
                    total["trajectories"] += 1
        return {"schema": SCHEMA, "generated_at": self.now, "sources": sources, "runs": runs,
                "targets": [{key: info[key] for key in ("deepswe_task", "runs", "patterns")}
                            for _, info in sorted(targets.items())],
                "task_columns": list(EXPLORER_TASK_COLUMNS), "tasks": tasks,
                "trajectory_columns": list(EXPLORER_TRAJECTORY_COLUMNS), "trajectories": trajectories,
                "totals": totals}

    def write_explorer(self, docs, batch_docs):
        """explorer_doc() to data/explorer.json, rewritten only when its content (everything but generated_at) differs
        from the file's: then generated_at is this cycle's, else it keeps the stamp of the cycle that last changed it.
        Returns {"index": index.json "explorer", "summary": the cycle summary's}."""
        doc = sanitize(self.explorer_doc(docs, batch_docs))
        path = self.data_dir / EXPLORER
        body = encode_json(dict(doc, generated_at=None), clean=False)
        signature = file_signature(path)
        cached = _EXPLORER_CACHE.get(str(self.data_dir))
        if not cached or signature is None or cached["file"] != signature:  # not the file written last: read it
            old = read_json(path) if signature is not None else None
            stamp = old.get("generated_at") if isinstance(old, dict) else None
            cached = ({"body": encode_json(dict(old, generated_at=None), clean=False), "stamp": stamp,
                       "file": signature} if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) else None)
        written = not cached or cached["body"] != body
        stamp = self.now if written else cached["stamp"]
        data = encode_json(dict(doc, generated_at=stamp), clean=False)
        if written:
            write_bytes_atomic(path, data)
            self.written[EXPLORER] = len(data)
        _EXPLORER_CACHE[str(self.data_dir)] = {"body": body, "stamp": stamp, "file": file_signature(path)}
        count = lambda key: {source: total[key] for source, total in doc["totals"].items()}  # noqa: E731
        return {"index": {"path": EXPLORER, "generated_at": stamp, "bytes": len(data), "tasks": len(doc["tasks"]),
                          "trajectories": len(doc["trajectories"])},
                "summary": {"bytes": len(data), "written": written, "tasks": count("tasks"),
                            "trajectories": count("trajectories"), "runs": count("runs")}}

    # ========================================================================================== fleet
    def fleet_snapshot(self):
        """Fleet daemon status + registry, plus one parallel /metrics scrape of every running worker (runs in a
        background thread: it only reads files and the network, and returns new scrape counters for the state)."""
        student = self.src.student
        status = read_json(student / "state/status.json")
        if not isinstance(status, dict):
            raise RuntimeError(f"{student / 'state/status.json'} unreadable")
        daemon = read_json(student / "state/daemon.json")
        registry = {}
        for path in (student / "registry").glob("*.json"):
            record = read_json(path)
            if path.stem.isdigit() and isinstance(record, dict) and str(record.get("job_id")) == path.stem:
                registry[path.stem] = record
        rows = [row for row in as_list(status.get("jobs")) if isinstance(row, dict) and row.get("job_id")]
        targets = {}
        for row in rows:
            record = registry.get(str(row["job_id"])) or {}
            url = record.get("base_url") or row.get("base_url")
            if row.get("state") in RUNNING_STATES and isinstance(url, str) and BASE_URL.fullmatch(url):
                targets[str(row["job_id"])] = re.sub(r"/v1/?$|/$", "", url) + "/metrics"
        scraped = {}
        if targets:
            with ThreadPoolExecutor(min(16, len(targets))) as pool:
                futures = {job: pool.submit(self._scrape_one, url) for job, url in targets.items()}
                scraped = {job: future.result() for job, future in futures.items()}
        return {"status": status, "daemon": daemon if isinstance(daemon, dict) else status.get("daemon"),
                "registry": registry, "rows": rows, "scraped": scraped}

    def _scrape_one(self, url):
        at = self.clock()
        try:
            return dict(parse_metrics(self.scrape(url)), scraped_at=at, error=None)
        except Exception as error:  # same keys as a good scrape (CONTRACT fleet.json workers[].metrics), values null
            return {"running": None, "waiting": None, "kv_usage": None, "prompt_tps_incl_cache": None,
                    "prompt_tps": None, "gen_tps": None, "scraped_at": at,
                    "error": clip(redact(f"{type(error).__name__}: {error}"), 200)}

    def fleet_doc(self, snapshot, router):
        now = self.now
        doc = {"schema": SCHEMA, "generated_at": now, "daemon": None, "config": None, "coverage": None,
               "counts": None, "workers": [], "totals": None, "router": router, "history": []}
        previous = as_dict(self.state.get("scrapes"))
        scrapes = {job: value for job, value in previous.items()
                   if isinstance(value, dict) and now - (value.get("t") or 0) < 3600}
        if snapshot:
            status, registry = snapshot["status"], snapshot["registry"]
            daemon = as_dict(snapshot["daemon"])
            breaker = as_dict(status.get("breaker"))
            if breaker.get("open"):
                half = breaker.get("probe_job") or now >= (number(breaker.get("probe_after")) or float("inf"))
                breaker_state = "half-open" if half else "open"
            else:
                breaker_state = "closed" if breaker else None
            heartbeat = number(daemon.get("heartbeat_at"))
            doc["daemon"] = {"job_id": daemon.get("job_id"), "host": daemon.get("host"),
                             "heartbeat_age_s": round(now - heartbeat) if heartbeat else None,
                             "status_age_s": round(now - status["ts"]) if number(status.get("ts")) else None,
                             "breaker": breaker_state, "stopped_at": daemon.get("stopped_at"),
                             "stop_flag": status.get("stop_flag")}
            cfg = as_dict(status.get("config"))
            doc["config"] = {key: cfg.get(key) for key in ("target", "horizon_effective_min", "horizon_min",
                                                           "stagger_min", "max_jobs", "time", "coverage_partitions")}
            counts = as_dict(status.get("counts"))
            doc["coverage"] = {"value": counts.get("coverage"), "target": cfg.get("target")}
            doc["counts"] = {bucket: int(counts.get(bucket) or 0) for bucket in FLEET_BUCKETS}
            workers = []
            for row in snapshot["rows"]:
                job = str(row["job_id"])
                record = registry.get(job) or {}
                running = row.get("state") in RUNNING_STATES
                metrics = snapshot["scraped"].get(job)
                if metrics is not None and metrics.get("error") is None:
                    previous_scrape = as_dict(scrapes.get(job))
                    span = metrics["scraped_at"] - (previous_scrape.get("t") or 0)
                    # vLLM's prompt_tokens_total counts prefix-cache hits too: prompt tok/s is not prefill work
                    for name, counter, key in (("prompt_tps_incl_cache", "prompt_tokens_total", "prompt"),
                                               ("gen_tps", "generation_tokens_total", "gen")):
                        value, before = metrics.get(counter), previous_scrape.get(key)
                        metrics[name] = (round((value - before) / span, 1) if value is not None and before is not None
                                         and value >= before and span >= 1 else None)
                    metrics["prompt_tps"] = metrics["prompt_tps_incl_cache"]  # the old name, kept for one release
                    scrapes[job] = {"t": metrics["scraped_at"], "prompt": metrics.get("prompt_tokens_total"),
                                    "gen": metrics.get("generation_tokens_total")}
                heartbeat = number(record.get("heartbeat_at"))
                end = number(row.get("end"))
                registry_status = record.get("status") if running else None
                workers.append({
                    "job_id": job, "slot": row.get("slot"), "partition": row.get("partition"),
                    "node": row.get("node") or None, "state": row.get("state"), "registry_status": registry_status,
                    "routing": _routing(record, now) if running else None,
                    "ready_at": record.get("ready_at") if running else None, "start": row.get("start"),
                    "end": row.get("end"),
                    "left_s": round(end - now) if running and end else row.get("left_s"),
                    "heartbeat_age_s": round(now - heartbeat) if running and heartbeat else None,
                    "base_url": (record.get("base_url") or row.get("base_url")) if running else None,
                    "pending_reason": row.get("reason") or None if row.get("state") in PENDING_STATES else None,
                    "est_start": row.get("start") if row.get("state") in PENDING_STATES else None,
                    "parity": row.get("parity"),
                    "registry_reason": text_field(_str(record.get("reason")), 300) if running else None,
                    "metrics": metrics})
            doc["workers"] = workers
            scraped = [w["metrics"] for w in workers if w["metrics"] and w["metrics"].get("error") is None]
            kv = [m["kv_usage"] for m in scraped if m.get("kv_usage") is not None]
            prompt_tps = _sum(m.get("prompt_tps_incl_cache") for m in scraped)
            doc["totals"] = {"routable": sum(bool(w["routing"]) for w in workers),
                             "busy_workers": sum(1 for w in workers if w["routing"] and w["metrics"]
                                                 and w["metrics"].get("error") is None
                                                 and (w["metrics"].get("running") or 0) >= 1),
                             "scraped": len(scraped), "scrape_errors": sum(bool(w["metrics"] and w["metrics"].get(
                                 "error")) for w in workers),
                             "running": _sum(m.get("running") for m in scraped),
                             "waiting": _sum(m.get("waiting") for m in scraped),
                             "kv_usage_mean": round(sum(kv) / len(kv), 4) if kv else None,
                             "prompt_tps_incl_cache": prompt_tps, "prompt_tps": prompt_tps,  # old name, one release
                             "gen_tps": _sum(m.get("gen_tps") for m in scraped)}
        self.state["scrapes"] = scrapes
        totals, router = as_dict(doc["totals"]), as_dict(router)
        point = {"t": round(now, 1), "ready": as_dict(doc["counts"]).get("ready"), "routable": totals.get("routable"),
                 "running": totals.get("running"), "waiting": totals.get("waiting"), "kv": totals.get("kv_usage_mean"),
                 "gen_tps": totals.get("gen_tps"), "prompt_tps_incl_cache": totals.get("prompt_tps_incl_cache"),
                 "req": router.get("requests"), "err": router.get("errors"), "p90_s": router.get("p90_s")}
        path = self.data_dir / "history/fleet.jsonl"
        append_jsonl(path, point)
        rows = trim_jsonl(path, now - FLEET_HISTORY_DAYS * 86400)
        doc["history"] = [row for row in rows if isinstance(row.get("t"), (int, float)) and row["t"] >= now - 86400]
        return doc

    # ========================================================================================== API usage
    def start_hub(self):
        hub = as_dict(self.state.get("hub"))
        if not self.hub or self.now - (number(hub.get("last_attempt_at")) or 0) < HUB_INTERVAL_S:
            return None
        hub["last_attempt_at"] = self.now  # an attempt counts even if it fails: never more than one per 10 min
        self.state["hub"] = hub
        return self.hub[0](self.src)

    def finish_hub(self, process):
        hub = as_dict(self.state.get("hub"))
        try:
            stdout, code = self.hub[1](process, HUB_TIMEOUT_S - self.elapsed())
            line = next((line for line in reversed((stdout or "").splitlines()) if line.strip()), "")
            try:
                out = json.loads(line)
            except ValueError:
                out = None
            if not isinstance(out, dict):
                raise RuntimeError(f"hub_budget.py printed no JSON line (exit {code})")
            sample = {"spend": number(out.get("spend")), "max_budget": number(out.get("max_budget")),
                      "remaining": number(out.get("remaining")), "status": out.get("status"), "exit_code": code,
                      "sampled_at": self.clock(), "at": out.get("at"),
                      "key": out.get("key") if isinstance(out.get("key"), str) and SAFE_NAME.fullmatch(out["key"]) else None,
                      "error": text_field(_str(out.get("error")), 300) if out.get("error") else None}
        except Exception as error:
            self.error("hub", error)
            sample = {"spend": None, "max_budget": None, "remaining": None, "status": None, "exit_code": None,
                      "sampled_at": self.clock(), "at": None, "error": clip(redact(str(error)), 300)}
        hub["sample"] = sample
        if sample["spend"] is not None and sample["max_budget"] is not None:
            hub["last_good"] = sample
            append_jsonl(self.data_dir / "history/hub.jsonl",
                         {"t": round(sample["sampled_at"], 1), "spend": sample["spend"],
                          "max_budget": sample["max_budget"]})
        self.state["hub"] = hub

    def api_doc(self, hub_process, usage, docs):
        if hub_process is not None:
            self.finish_hub(hub_process)
        now = self.now
        hub = as_dict(self.state.get("hub"))
        sample = as_dict(hub.get("sample"))
        good = as_dict(hub.get("last_good")) or sample
        # the NAME of the key the sample checked (hub_budget.py reports it since the 2026-10-18 key rotation)
        key_name = good.get("key") or self.memo.get("hubkey", self.src.common, _hub_key_name)
        path = self.data_dir / "history/hub.jsonl"
        history = trim_jsonl(path, now - HUB_HISTORY_DAYS * 86400) if path.exists() else []
        history = [row for row in history if isinstance(row.get("t"), (int, float))]
        burn = _burn(history, good.get("remaining"), good.get("sampled_at"))
        glm = {item["run"]["run"]: (item["tree"] or {}).get("glm_calls", 0) for item in docs}
        by_run = []
        for run in sorted(set(usage.get("runs", {})) | set(glm)):
            entry = as_dict(usage.get("runs", {}).get(run))
            by_run.append({"run": run, "kimi": entry.get("kimi"), "codex": entry.get("codex"),
                           "glm": {"calls": glm.get(run, 0)}})
        return {"schema": SCHEMA, "generated_at": now,
                "hub": {"spend": good.get("spend"), "max_budget": good.get("max_budget"),
                        "remaining": good.get("remaining"), "status": sample.get("status"),
                        "sampled_at": good.get("sampled_at"), "key": key_name, "error": sample.get("error"),
                        "exit_code": sample.get("exit_code"), "last_attempt_at": hub.get("last_attempt_at"),
                        "age_s": round(max(0.0, now - good["sampled_at"])) if number(good.get("sampled_at")) else None},
                "burn": burn,
                "history": [{"t": row["t"], "spend": row.get("spend"), "max_budget": row.get("max_budget")}
                            for row in history],
                "by_run": by_run, "scan": usage.get("scan"), "notes": NOTES}

    def scan_usage(self, docs):
        """Incremental scans of the append-only logs of every official run: kimi http.jsonl (cost and tokens of
        every hub call), codex .agent/events.jsonl (turn.completed usage), student-router-requests.jsonl. v1.8: a failure
        analyzer's session (analysis/<task>/.agent/events.jsonl) counts only when written after the run's first launch:
        run_queue.py prep() copies an earlier run's analysis/<task>/ into a queued run with shutil.copy2 (old mtimes
        kept) and task_builder reuses it, so a copy is the earlier run's usage (or another tree's), while an analysis
        redone in the run is written after its launch; a run dir without a launch has no session of its own. A copy
        counted before this rule is dropped from the state (by its inode)."""
        files = as_dict(self.state.get("files"))
        launched = {item["run"]["run"]: item["run"].get("first_submitted_at") for item in docs}
        sealed = as_dict(self.state.get("sealed"))
        jobs = []
        complete_probes = {}
        for item in docs:
            for probe in (item["tree"] or {}).get("probes", []):
                complete_probes[probe["path"]] = probe["complete"] or probe["abandoned"]
        for run_dir in sorted(self.src.runs.glob("official*")):
            if not run_dir.is_dir() or not SAFE_NAME.fullmatch(run_dir.name):
                continue
            run = run_dir.name
            for probe_dir in sorted(run_dir.glob("build/*/*/round-*/probe-*")):
                # a job's router journals into the FIRST probe dir of that job, possibly a finished (sealed) one
                jobs += [("router", run, path, None) for path in probe_dir.glob("student-router-requests.jsonl")]
                if str(probe_dir) in sealed:
                    continue
                for pattern in ("models/kimi/attempts/*/attempt-*/http.jsonl", "*/kimi-attempts/*/attempt-*/http.jsonl"):
                    jobs += [("kimi", run, path, str(probe_dir)) for path in probe_dir.glob(pattern)]
            # a run dir with neither launch.jsonl nor a build is queued: prep() copied its analysis, never its own
            has_own = (run_dir / "launch.jsonl").is_file() or (run_dir / "build").is_dir()
            for path in run_dir.glob("analysis/*/.agent/events.jsonl"):
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                key = f"{st.st_dev}:{st.st_ino}"
                if not has_own:
                    files.pop(key, None)
                    continue
                if run not in launched:  # its document failed this cycle: keep its state, decide next cycle
                    continue
                first = launched[run]
                if first is not None and st.st_mtime < first:  # v1.8: a copied analysis (counted where it ran)
                    files.pop(key, None)
                    continue
                jobs.append(("codex", run, path, None))
            for pattern in ("build/*/*/*/.agent/events.jsonl", "build/*/*/cand-*/*/.agent/events.jsonl",
                            "build/*/*/round-*/cand-*/*/.agent/events.jsonl"):
                jobs += [("codex", run, path, None) for path in run_dir.glob(pattern)]
        todo, unfinished_probes = [], set()
        for kind, run, path, probe in jobs:
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = f"{st.st_dev}:{st.st_ino}"
            entry = files.get(key)
            if not isinstance(entry, dict) or entry.get("kind") != kind:
                entry = {"kind": kind, "run": run, "offset": 0, "size": 0, "mtime_ns": 0, "fp": None, "acc": {}}
                files[key] = entry
            entry["path"] = str(path)
            if entry["size"] == st.st_size and entry["mtime_ns"] == st.st_mtime_ns:
                continue  # unchanged since it was read to its end
            todo.append((key, entry, path, st))
            if probe:
                unfinished_probes.add(probe)
        deadline = time.monotonic() + max(SCAN_MIN_S, SCAN_BUDGET_S - self.elapsed())
        recent = [row for row in as_list(self.state.get("router_recent"))
                  if isinstance(row, list) and len(row) == 7 and row[0] >= self.now - 3600]
        finished_all = True
        todo.sort(key=lambda item: item[3].st_size - item[1]["offset"])  # small backlogs first
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda item: _scan_safely(item[1], item[2], item[3], deadline), todo))
        for (key, entry, path, st), (rewritten, offset, eof, fingerprint, delta, rows) in zip(todo, results):
            # rewritten: codex reruns a stage by rewriting its events.jsonl; the earlier usage was real, so the
            # accumulated totals stay and the new content is added from offset 0
            entry["offset"], entry["fp"] = offset, fingerprint
            entry["size"], entry["mtime_ns"] = (st.st_size, st.st_mtime_ns) if eof else (0, 0)
            acc = entry.setdefault("acc", {})
            for name, value in delta.items():
                acc[name] = round(acc.get(name, 0) + value, 6)
            if rows:
                recent += [row for row in rows if row[0] >= self.now - 3600]
            if not eof:
                finished_all = False
        for probe, complete in complete_probes.items():  # nothing new will ever appear in a finished probe
            if complete and probe not in unfinished_probes and finished_all:
                sealed[probe] = self.now
        recent.sort(key=lambda row: row[0])
        self.state.update(files=files, sealed=sealed, router_recent=recent)
        runs = {}
        for entry in files.values():
            if not isinstance(entry, dict) or entry.get("kind") not in ("kimi", "codex"):
                continue
            bucket = runs.setdefault(entry.get("run"), {"kimi": None, "codex": None})
            acc = as_dict(entry.get("acc"))
            if entry["kind"] == "kimi":
                kimi = bucket["kimi"] = bucket["kimi"] or {"calls": 0, "cost": 0.0, "input_tokens": 0,
                                                           "cached_tokens": 0, "output_tokens": 0,
                                                           "reasoning_tokens": 0, "http_errors": 0, "files": 0}
                kimi["files"] += 1
                for name in ("calls", "cost", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
                             "http_errors"):
                    kimi[name] += acc.get(name, 0)
                kimi["cost"] = round(kimi["cost"], 4)
            else:
                codex = bucket["codex"] = bucket["codex"] or {"sessions": 0, "turns_completed": 0,
                                                              "turns_failed": 0, "input_tokens": 0,
                                                              "cached_tokens": 0, "output_tokens": 0,
                                                              "reasoning_tokens": 0}
                for name in codex:
                    codex[name] += acc.get(name, 0)
        pending = sum(max(0, st.st_size - entry["offset"]) for _, entry, _, st in todo if not entry["size"])
        scan = {"files": len(jobs), "read": len(todo), "complete": finished_all, "pending_bytes": pending,
                "sealed_probes": len(sealed)}
        return {"runs": runs, "router": _router_stats(recent, self.now), "scan": scan}


# ------------------------------------------------------------------------------------------------- the tree

class Tree:
    """The generation tree of one build root, read the way synth/task_builder.py writes it:
      cand-<id>/                      round-0 (mask) task; cand-<parent>-r<k>b<j>/ = round-k evolve branch j
        stage.json {ok, task_id, f2p, p2p, ...} | {ok: false, problems}; built.json; gates-<n>.json; author-<n>/;
        evolve/{stage.json, evolve_plan.md}; instruction/{instruction.md, rubric.md}
      round-<k>/<cand>/{metrics.json, rollouts/, teacher_audit.json, failure_analysis/}; round-<k>/seeds.json;
      round-<k>/probe-<key>/ (ATP campaign: manifest.json, models/<m>/attempts/<task>-<repeat>/attempt-<n>/)
    Renamed files (stage.json.budget-422, author-0.budget-422, ...) are never stage data. Evolve branch j of every
    round aims at the target's failure pattern j mod n (runs/<run>/analysis/<task>/result.json, n patterns). Main may
    withdraw an accepted task later: its metrics.json gets "withdrawn" and its task dir moves from runs/<run>/accepted/
    to runs/<run>/withdrawn/."""

    def __init__(self, cycle, run, root):
        self.cycle, self.run, self.root = cycle, run, Path(root)
        self.max_rounds, self.branches = run["max_rounds"], run["branches"]
        # runs/<run>/build/<task>/<repo>: a root main abandoned (abandoned-*) holds no task of the audit's scope
        self.abandoned_root = any(part.startswith(ABANDONED_ROOT) for part in (self.root.name, self.root.parent.name))
        self.withdrawn_dirs = set()
        # v1.6: a batch's run has ended (finished: summary.json; else stopped): none of its probes is live, and an
        # incomplete one is unfinished, never superseded (Baihe's fork probes every task alone, keyed by that task)
        self.batch = run.get("batch")
        self.ended = ("finished" if (self.root / "summary.json").is_file() else "stopped") if self.batch else None

    def build(self):
        cycle, root = self.cycle, self.root
        names = sorted((name for name in listdir(root) if name.startswith("cand-") and SAFE_NAME.fullmatch(name)
                        and (root / name).is_dir()), key=natural)
        plan = {str(c.get("id")): c for c in as_list(as_dict(cycle.json(root / "plan/stage.json")
                                                             or cycle.json(root / "plan/candidates.json"))
                                                     .get("candidates")) if isinstance(c, dict)}
        # the target's failure analysis, as task_builder evolve() reads it (each branch aims at one of its patterns)
        self.analysis = cycle.json(self.run["run_dir"] / "analysis" / self.run["task_id"] / "result.json")
        self.nodes = {}
        for name in names:
            parsed = _parse_node(name)
            if parsed:
                self.nodes[name] = self.read_node(name, *parsed, plan)
        self.rounds = self.read_rounds()
        # tasks main withdrew after accepting them (the decisions need it: a withdrawn task is never accepted)
        withdrawn = self.run["run_dir"] / WITHDRAWN_DIR
        self.withdrawn_dirs = {name for name in listdir(withdrawn)
                               if SAFE_TASK.fullmatch(name) and (withdrawn / name).is_dir()}
        for node in self.nodes.values():
            node["withdrawn"] = self.withdrawal(node)
        self.attach_probes()
        self.decide()
        for node in self.nodes.values():  # v1.4: the failure-match audit's scope
            node["audit_eligible"] = self.eligible(node)
        nodes = sorted(self.nodes.values(), key=lambda node: (node["round"], natural(node["id"])))
        out = {"nodes": nodes, "rounds": self.rounds_out(), "probes": self.probe_list(),
               "glm_calls": sum(info["glm_calls"] for info in self.rounds.values())}
        for node in nodes:
            node.pop("_", None)  # private working data, not part of the NODE schema
        return out

    # -- candidate dirs
    def read_node(self, name, lineage, steps, plan):
        cycle, cand = self.cycle, self.root / name
        round_index = steps[-1][0] if steps else 0
        entries = listdir(cand)
        stage = cycle.json(cand / "stage.json")
        stage = stage if isinstance(stage, dict) else None
        built = cycle.memo.get("built", cand / "built.json", _gate_summary)
        gates = sorted((int(m.group(1)), m.group(0)) for m in map(GATES.fullmatch, entries) if m)
        authors = sorted(int(m.group(1)) for m in map(AUTHOR.fullmatch, entries) if m)
        evolve = cycle.json(cand / "evolve/stage.json") if steps else None
        evolve = evolve if isinstance(evolve, dict) else None
        candidate = plan.get(lineage[len("cand-"):]) if not steps else None
        title = (as_dict(evolve).get("title") or as_dict(candidate).get("title") or as_dict(stage).get("title")
                 or None)  # evolve plan / mask plan title first (stage.json repeats it once built)
        summary = as_dict(evolve).get("summary") if steps else as_dict(candidate).get("summary")
        # build facts: stage.json when finished, else built.json (gates passed) or the latest gates-<n>.json
        gate_summaries = {n: cycle.memo.get("gates", cand / file, _gate_summary) for n, file in gates}
        best = built or (gate_summaries[gates[-1][0]] if gates else None)
        problems = []
        for n in authors:
            exit_info = as_dict(cycle.json(cand / f"author-{n}/.agent/exit.json"))
            if exit_info.get("problems"):
                problems.append(f"author-{n}: " + "; ".join(map(str, exit_info["problems"])))
        for n, summary_n in gate_summaries.items():
            if summary_n and summary_n.get("problems"):
                problems.append(f"gates attempt {n}: " + "; ".join(summary_n["problems"]))
        if stage and stage.get("ok") is False and stage.get("problems"):
            problems.append(str(stage["problems"]))
        ok = stage.get("ok") if stage and isinstance(stage.get("ok"), bool) else None
        last_author = cand / f"author-{authors[-1]}" if authors else None
        node = {
            "id": name, "parent": name[:STEP.search(name).start()] if steps else None, "lineage": lineage,
            "round": round_index, "branch": steps[-1][1] if steps else None, "kind": "evolve" if steps else "mask",
            "focus_pattern": _focus_pattern(self.analysis, steps[-1][1]) if steps else None,
            "title": text_field(_str(title), 300), "summary": text_field(_str(summary), 600), "status": None,
            "build": {"ok": ok, "attempts": len(authors),
                      "f2p": _int(as_dict(stage).get("f2p")) if ok else as_dict(best).get("f2p_n"),
                      "p2p": _int(as_dict(stage).get("p2p")) if ok else as_dict(best).get("p2p_n"),
                      "f2p_tests": [text_field(t, 300) for t in as_dict(best).get("f2p", [])][:60],
                      "p2p_tests_sample": [text_field(t, 300) for t in as_dict(best).get("p2p", [])][:20],
                      "problems": text_field("\n".join(problems), 3000) if problems else None,
                      "task_id": as_dict(stage).get("task_id") if ok else None},
            "probe": None, "score": None, "decision": None, "decision_reason": None, "children": [],
            "details": None, "failure_match": None,  # v1.3: set by Cycle.task_doc() from the run's audit files
            "audit_eligible": False, "withdrawn": None,  # v1.4: set by build()
            "seed_invalid": None,  # v1.5: set by fill_decision()
            "_": {"stage": stage, "built": built is not None, "planned": evolve is not None,
                  "author_running": bool(last_author and not exists(last_author / ".agent/exit.json")),
                  "steps": steps},
        }
        return node

    # -- round dirs
    def read_rounds(self):
        cycle, rounds = self.cycle, {}
        for entry in listdir(self.root):
            match = ROUND.fullmatch(entry)
            if not match or not (self.root / entry).is_dir():
                continue
            k = int(match.group(1))
            directory = self.root / entry
            seeds = cycle.json(directory / "seeds.json")
            seeds = ({"valid": [str(v) for v in as_list(seeds.get("valid"))],
                      "invalid": {str(k2): str(v) for k2, v in as_dict(seeds.get("invalid")).items()}}
                     if isinstance(seeds, dict) else None)
            probes, glm = [], 0
            for name in sorted(listdir(directory)):
                if PROBE.fullmatch(name) and (directory / name).is_dir():
                    probes.append(self.read_probe(k, directory / name))
            for name in listdir(directory):
                node = self.nodes.get(name)
                metrics = cycle.json(directory / name / "metrics.json") if node else None
                if node and isinstance(metrics, dict) and node["round"] == k:
                    node["_"]["metrics"] = metrics
                    node["_"]["scored_dir"] = directory / name
                audit = cycle.json(directory / name / "teacher_audit.json") if name.startswith("cand-") else None
                glm += len(audit) if isinstance(audit, (dict, list)) else 0
                glm += bool(isinstance(metrics, dict) and metrics.get("similarity_detail"))
            probes.sort(key=lambda probe: (probe["created_at"] or 0, probe["name"]))
            rounds[k] = {"dir": directory, "seeds": seeds, "probes": probes, "glm_calls": glm}
        return rounds

    def read_probe(self, k, directory):
        manifest = self.cycle.memo.get("manifest", directory / "manifest.json", _manifest_summary)
        if manifest is None:  # being created: rollouts.probe() copies tasks/ before the campaign writes manifest
            manifest = {"tasks": sorted(listdir(directory / "tasks")), "models": [STUDENT, TEACHER], "repeats": 5,
                        "max_attempts": 3, "created_at": None}
            try:
                manifest["created_at"] = directory.stat().st_mtime
            except OSError:
                pass
        return dict(manifest, name=directory.name, path=str(directory), rel=f"round-{k}/{directory.name}",
                    round=k, complete=False, abandoned=False, live=False, slots={})

    def attach_probes(self):
        """Mark each probe complete (all its tasks scored), abandoned (incomplete while a newer probe of the same
        round exists: probe_and_score() keys a probe by its todo set, so a newer key means the old campaign was
        superseded, e.g. narwhals round-1/probe-7f97a213 after the 2026-10-17 seed-rule change) or live; scan the
        live probes' slots."""
        by_task = {}
        for node in self.nodes.values():
            if node["build"]["task_id"]:
                by_task[node["build"]["task_id"]] = node
        for k, info in self.rounds.items():
            probes = info["probes"]
            for position, probe in enumerate(probes):
                tasks = [task for task in probe["tasks"] if task]
                scored = [task for task in tasks if by_task.get(task) and by_task[task]["_"].get("metrics")]
                probe["complete"] = bool(tasks) and len(scored) == len(tasks)
                if self.batch:  # v1.6: an ended run's incomplete probe is unfinished (nothing scans it); the slots of
                    # its scored tasks count as done
                    probe["abandoned"] = probe["live"] = False
                    probe["unfinished"] = not probe["complete"]
                    probe["scored_tasks"] = len(scored)
                else:
                    probe["abandoned"] = not probe["complete"] and position < len(probes) - 1
                    probe["live"] = not probe["complete"] and not probe["abandoned"]
                if probe["live"]:
                    self.scan_live(probe)
                for index, task in enumerate(tasks):
                    node = by_task.get(task)
                    if node:
                        node["_"].setdefault("probes", []).append((probe, index))

    def scan_live(self, probe):
        """Per-slot progress of a live campaign from cheap files: a slot is done once an attempt has terminal.json
        and a grading/attempt-*/reward.json (pass/fail), or its last allowed attempt ended without a grade
        (infra). Finished slots are cached in the state (they never change unless ATP adds an attempt)."""
        cache = self.cycle.state.setdefault("slots", {})
        directory = Path(probe["path"])
        for model in probe["models"]:
            attempts_dir = directory / "models" / model / "attempts"
            for slot_name in listdir(attempts_dir):
                match = SLOT_DIR.fullmatch(slot_name)
                if not match:
                    continue
                slot_dir = attempts_dir / slot_name
                names = sorted((int(m.group(1)), m.group(0)) for m in map(ATTEMPT.fullmatch, listdir(slot_dir)) if m)
                key = f"{directory}|{model}|{slot_name}"
                cached = cache.get(key)
                if isinstance(cached, dict) and cached.get("done") and cached.get("n") == len(names):
                    result = cached
                else:
                    result = _slot_result(slot_dir, names, probe["max_attempts"])
                    if result.get("done"):
                        cache[key] = dict(result, t=self.cycle.now)
                probe["slots"][(model, int(match.group(1)), int(match.group(2)))] = result

    # -- decisions
    def decide(self):
        """Status, probe, score and decision of every node, round by round (parents before children); the rules
        are CONTRACT.md's, mirroring task_builder build() / seed_check() / valid_seeds()."""
        for node in self.nodes.values():
            if node["parent"] in self.nodes:
                self.nodes[node["parent"]]["children"].append(node["id"])
        for node in self.nodes.values():
            node["children"].sort(key=natural)
        selected = self.selection()
        ordered = sorted(self.nodes.values(), key=lambda node: (node["round"], natural(node["id"])))
        for node in ordered:  # parents before children: a child's abandonment reads its parent's status
            self.fill_status(node, selected)
        for node in ordered:  # then decisions, which read the children's and the next round's statuses
            self.fill_decision(node, selected)

    def validity(self, node):
        """'' when the scored node may seed evolution, else why not: <round>/seeds.json when it lists the node,
        else seed_check() recomputed (grading failures, uniform failure, >= 3 valid rollouts per model)."""
        seeds = (self.rounds.get(node["round"]) or {}).get("seeds")
        if seeds and node["id"] in seeds["invalid"]:
            return seeds["invalid"][node["id"]]
        if seeds and node["id"] in seeds["valid"]:
            return ""
        metrics = node["_"]["metrics"]
        failed = self.grading_failures(node)
        if failed:
            return f"verifier produced no grade for {len(failed)} rollout(s): {', '.join(failed)}"
        if metrics.get("uniform_failure"):  # task_builder score(): every patched rollout scores the same short P2P
            return f"tests cannot run normally: {metrics['uniform_failure']}"
        valid = metrics.get("valid")
        counts = [number(v) for v in valid] if isinstance(valid, list) else []
        if not counts or None in counts or min(counts) < 3:
            return f"fewer than 3 valid rollouts per model (valid {valid})"
        return ""

    def grading_failures(self, node):
        metrics = node["_"]["metrics"]
        if isinstance(metrics.get("grading_failures"), list):
            return [str(slot) for slot in metrics["grading_failures"]]
        return [row["slot"] for row in self.rollouts(node) if row["status"] == "infra" and row["verifier_error"]]

    def rollouts(self, node):
        """The scored probe's rollouts: rollouts/index.md rows + each <slot>.md header for the verifier error."""
        if "rollouts" in node["_"]:
            return node["_"]["rollouts"]
        directory = node["_"]["scored_dir"] / "rollouts"
        rows = self.cycle.memo.get("index_md", directory / "index.md", _parse_index_md) or []
        out = []
        for row in rows:
            header = self.cycle.memo.get("slot_md", directory / f"{row['slot']}.md", _parse_slot_header) or {}
            out.append(dict(row, verifier_error=header.get("verifier_error") or "",
                            agent_error=row.get("agent_error") or header.get("agent_error") or ""))
        node["_"]["rollouts"] = out
        return out

    def withdrawal(self, node):
        """NODE.withdrawn (CONTRACT "v1.4"): {at, by, why} when main withdrew the task after accepting it, from
        metrics.json "withdrawn" ({at, by, why, was}; another non-null value counts too), or when its task dir sits in
        runs/<run>/withdrawn/ (official1-bandit, -fastapi and -testem, 2026-10-18); else None."""
        mark = as_dict(node["_"].get("metrics")).get("withdrawn")
        task_id = node["build"]["task_id"]
        moved = bool(task_id) and task_id in self.withdrawn_dirs
        if mark in (None, False) and not moved:
            return None
        mark = mark if isinstance(mark, dict) else {"why": mark} if isinstance(mark, str) else {}
        return {"at": _text(mark.get("at"), 64), "by": _text(mark.get("by"), 100), "why": _text(mark.get("why"), 600)}

    def accepted_then(self, node):
        """Whether task_builder saw the scored task as accepted when it chose its lineage's next seed: metrics.json
        "accept", or for a task main withdrew later, its accept from before (withdrawn.was.accept; true when the record
        does not say: only accepted tasks are withdrawn). official1-bandit cand-c05-r1b1, accepted in round 1 and
        withdrawn on 2026-10-18, was skipped when cand-c05-r1b2 was picked to seed round 2."""
        metrics = as_dict(node["_"].get("metrics"))
        if metrics.get("accept"):
            return True
        if not node["withdrawn"]:
            return False
        return bool(as_dict(as_dict(metrics.get("withdrawn")).get("was")).get("accept", True))

    def eligible(self, node):
        """NODE.audit_eligible (CONTRACT "v1.4"; hengyuf 2026-10-19: every task that is not discarded and on which the
        student has a rollout): built and probed (round-<k>/<cand>/metrics.json with its rollouts/), at least one graded
        sc14 rollout (pass or fail), and not discarded: accepted, or valid in its round's seeds.json (seed_check()
        recomputed until that file is written), not withdrawn, not under an abandoned-* build root. Accepted tasks
        (v1.5: also when seeds.json lists them invalid, which only bars them from seeding evolution), tasks rejected on
        pass rates and tasks rejected on similarity all qualify."""
        metrics = node["_"].get("metrics")
        if self.abandoned_root or node["withdrawn"] or not metrics or node["build"]["ok"] is False:
            return False
        if not any(row["model"] == STUDENT and row["status"] in ("pass", "fail") for row in self.rollouts(node)):
            return False
        return bool(metrics.get("accept")) or not self.validity(node)

    def selection(self):
        """{round k: {lineage: selected node id}} for rounds >= 1 whose lineage is final (every node of that lineage
        in round k scored or failed to build): task_builder's best valid non-accepted child per lineage, ranked by
        (similarity or 0, -|student_rate - target_student_rate|), ties to the lowest branch. "Accepted" as task_builder
        saw it then: a task withdrawn since still counts as accepted (accepted_then)."""
        chosen = {}
        for k in sorted({node["round"] for node in self.nodes.values() if node["round"] >= 1}):
            lineages = {}
            for node in sorted(self.nodes.values(), key=lambda node: (node["branch"] or 0, node["id"])):
                if node["round"] == k and not self.superseded(node):  # never part of the pipeline's pool
                    lineages.setdefault(node["lineage"], []).append(node)
            for lineage, members in lineages.items():
                if not all(node["_"].get("metrics") or as_dict(node["_"]["stage"]).get("ok") is False
                           for node in members):
                    continue
                best = None
                for node in members:
                    metrics = node["_"].get("metrics")
                    if not metrics or self.accepted_then(node) or self.validity(node):
                        continue
                    student = number(metrics.get("student_rate"))
                    student = 1.0 if student is None else student
                    rank = (number(metrics.get("similarity")) or 0,
                            -abs(student - (number(metrics.get("target_student_rate")) or 0.0)))
                    if best is None or rank > best[0]:
                        best = (rank, node["id"])
                chosen.setdefault(k, {})[lineage] = best[1] if best else None
        return chosen

    def fill_status(self, node, selected):
        private = node["_"]
        metrics = private.get("metrics")
        probes = private.get("probes", [])
        live = [(probe, index) for probe, index in probes if probe["live"]]
        node["details"] = self.details(node)
        # -- probe facts
        if metrics:
            rows = self.rollouts(node)
            node["probe"] = {"dir": next((p["rel"] for p, _ in reversed(probes) if p["complete"]), None),
                             "live": False, "progress": _progress(rows),
                             "student": _counts(rows, STUDENT, rate=metrics.get("student_rate")),
                             "teacher": dict(_counts(rows, TEACHER), clean_rate=metrics.get("teacher_clean_rate"),
                                             cheated=metrics.get("cheated")),
                             "rollouts": [{key: row.get(key) for key in ("slot", "model", "status", "reward", "f2p",
                                                                         "p2p", "tool_calls", "patch_chars",
                                                                         "agent_error", "verifier_error")}
                                          for row in rows]}
            detail = metrics.get("similarity_detail")
            node["score"] = {"accept": metrics.get("accept"), "reason": text_field(_str(metrics.get("reason")), 500),
                             "similarity": metrics.get("similarity"),
                             "similarity_detail": text_field(json.dumps(detail, indent=1), 4000) if detail else None,
                             "valid": metrics.get("valid"), "grading_failures": self.grading_failures(node),
                             "answer_retrieval": metrics.get("answer_retrieval"),
                             "target_student_rate": metrics.get("target_student_rate"),
                             "target_teacher_rate": metrics.get("target_teacher_rate"),
                             "student_rate": metrics.get("student_rate"),
                             "teacher_clean_rate": metrics.get("teacher_clean_rate"),
                             "cheated": metrics.get("cheated")}
        elif live:
            probe, index = live[-1]
            rows, progress, partial = [], {}, {}
            for model in probe["models"]:
                counts = {"done": 0, "total": probe["repeats"], "pass": 0, "fail": 0, "infra": 0}
                for repeat in range(probe["repeats"]):
                    result = probe["slots"].get((model, index, repeat))
                    if result and result.get("done"):  # graded (pass / fail by its reward) or out of attempts
                        status = result.get("status") if result.get("status") in ("pass", "fail") else "infra"
                        counts["done"] += 1
                        counts[status] += 1
                        rows.append({"slot": f"{model}-r{repeat}", "model": model, "status": status,
                                     "reward": result.get("reward"), "f2p": result.get("f2p"),
                                     "p2p": result.get("p2p"), "tool_calls": None, "patch_chars": None,
                                     "agent_error": "", "verifier_error": ""})
                progress[model] = [counts["done"], counts["total"]]
                partial[model] = counts
            # own_done: all of the node's own slots are final (its metrics.json comes when the whole campaign is scored)
            node["probe"] = {"dir": probe["rel"], "live": True, "progress": progress, "partial": partial,
                             "own_done": bool(partial) and all(c["done"] >= c["total"] for c in partial.values()),
                             "student": _counts(rows, STUDENT), "teacher": dict(_counts(rows, TEACHER),
                                                                                 clean_rate=None, cheated=None),
                             "rollouts": rows}
        # -- status
        stage, ok = private["stage"], node["build"]["ok"]
        abandoned = self.abandoned(node, selected)
        if abandoned:
            node["status"] = "abandoned"
        elif metrics:
            node["status"] = "scored"
        elif live:
            node["status"] = "probing"
        elif ok is True:
            node["status"] = "built"
        elif ok is False:
            node["status"] = "build_failed"
        elif private["built"]:
            node["status"] = "building"
        elif private["steps"] and not private["planned"]:
            node["status"] = "planning"
        elif node["build"]["attempts"] and not private["author_running"]:
            node["status"] = "building"
        else:
            node["status"] = "authoring"
        private["abandoned"] = abandoned

    def fill_decision(self, node, selected):
        private, k = node["_"], node["round"]
        metrics, stage, ok, abandoned = private.get("metrics"), private["stage"], node["build"]["ok"], \
            private["abandoned"]
        live_children = [child for child in node["children"] if child in self.nodes
                         and self.nodes[child]["status"] != "abandoned"]
        # v1.5: why the scored task may not seed evolution (its round's seeds.json, else seed_check() recomputed), one
        # line; None for a valid seed and for a task never scored. It discards a task only when it was not accepted.
        invalid = " ".join(redact(self.validity(node)).split()) if metrics and ok is not False else ""
        node["seed_invalid"] = invalid or None
        if abandoned:
            node["decision"], node["decision_reason"] = "abandoned", abandoned
        elif ok is False:
            first = next((line.strip() for line in str(as_dict(stage).get("problems") or "").splitlines()
                          if line.strip()), "no problem recorded")
            node["decision"], node["decision_reason"] = "discard", f"build failed: {clip(first, 300)}"
        elif not metrics:
            node["decision"] = "pending"
            node["decision_reason"] = {"probing": "probe in progress", "built": "built; waiting for its probe",
                                       "building": "gates / instruction / packaging", "planning": "evolve plan",
                                       "authoring": "authoring"}.get(node["status"], node["status"])
            if self.batch:  # v1.6: the batch run has ended, so this task stays unscored
                unfinished = [probe for probe, _ in private.get("probes", []) if probe.get("unfinished")]
                node["decision_reason"] = (
                    f"its probe {unfinished[-1]['rel']} never finished: the run {self.ended} before scoring it"
                    if unfinished else f"built, never probed: the run {self.ended} before probing it"
                    if node["status"] == "built" else f"{node['decision_reason']}, unfinished: the run {self.ended}")
        else:
            reason = _str(metrics.get("reason")) or "no reason recorded"
            withdrawn = node["withdrawn"]
            if withdrawn and not reason.strip().lower().startswith("withdrawn"):  # main rewrites it when it withdraws
                reason = "withdrawn" + (f": {withdrawn['why']}" if withdrawn["why"] else
                                        f" (its task dir is in runs/{self.run['run']}/{WITHDRAWN_DIR}/)")
            accepted = bool(metrics.get("accept")) and not withdrawn  # CONTRACT "v1.4": never once withdrawn
            if invalid and not accepted:  # v1.5: acceptance beats an invalid seed (task_builder accepts on the score)
                node["decision"], node["decision_reason"] = "discard", invalid
            elif k == 0 and self.max_rounds >= 1 and not invalid:
                node["decision"] = "accepted_evolve" if accepted else "evolve"
                node["decision_reason"] = (f"{'accepted' if accepted else 'not accepted'} ({reason}); every valid "
                                           f"round-0 task seeds {self.branches} branches")
            elif accepted:  # an invalid seed too (NODE.seed_invalid): accepted, never evolved
                node["decision"], node["decision_reason"] = "accepted", f"accepted ({reason})"
                if invalid and k == 0 and self.max_rounds >= 1:  # where a valid one would have seeded branches
                    node["decision_reason"] += "; an invalid seed, so not evolved"
            elif live_children:
                node["decision"] = "evolve"
                node["decision_reason"] = (f"best valid non-accepted task of lineage {node['lineage']} ({reason}); "
                                           f"{len(live_children)} branches in round {k + 1}")
            elif any(other["round"] == k + 1 and other["lineage"] == node["lineage"]
                     and other["status"] != "abandoned" for other in self.nodes.values()):
                pick = as_dict(selected.get(k)).get(node["lineage"])
                node["decision"] = "not_selected"
                node["decision_reason"] = (f"not accepted ({reason}); round {k + 1} continues lineage "
                                           f"{node['lineage']}" + (f" from {pick}" if pick else ""))
            elif k >= self.max_rounds:
                node["decision"], node["decision_reason"] = "end", f"not accepted ({reason}); last round ({k})"
            else:
                node["decision"] = "pending"
                node["decision_reason"] = f"not accepted ({reason}); waiting for round {k + 1}"
        # one line (CONTRACT): copied reasons may hold line breaks, and clip() puts its marker on a line of its own
        node["decision_reason"] = " ".join(str(node["decision_reason"]).split()) or "no reason recorded"

    def abandoned(self, node, selected):
        """Why the node is no longer part of the build, or None. Its parent was invalid (seeds.json) or abandoned,
        or (rounds >= 1, lineage final) not its lineage's selected seed; or its task sits only in superseded
        probes of its round."""
        parent = self.nodes.get(node["parent"]) if node["parent"] else None
        if parent:
            if parent["status"] == "abandoned":
                return f"parent {parent['id']} was abandoned"
            if parent["_"].get("metrics"):
                invalid = self.validity(parent)
                if invalid:
                    return f"parent {parent['id']} is not a valid seed ({invalid})"
                pick = as_dict(selected.get(parent["round"]))
                if parent["round"] >= 1 and parent["lineage"] in pick and pick[parent["lineage"]] != parent["id"]:
                    chosen = pick[parent["lineage"]]
                    return (f"parent {parent['id']} was not selected as lineage {parent['lineage']}'s seed"
                            + (f" ({chosen} was)" if chosen else " (no valid non-accepted task)"))
        probe = self.superseded(node)
        if probe:
            return f"its probe {probe['rel']} was superseded by a newer probe of round {node['round']}"
        return None

    def superseded(self, node):
        """The abandoned probe holding the node's unscored task when no other probe of its round does, else None."""
        probes = node["_"].get("probes", [])
        if probes and not node["_"].get("metrics") and all(probe["abandoned"] for probe, _ in probes):
            return probes[-1][0]
        return None

    def details(self, node):
        cycle, cand, private = self.cycle, self.root / node["id"], node["_"]
        scored = private.get("scored_dir")
        instruction = cycle.excerpt(cand / "instruction/instruction.md", 20000)
        if instruction is None and node["build"]["task_id"]:
            instruction = cycle.excerpt(cand / "task" / node["build"]["task_id"] / "instruction.md", 20000)
        audit = cycle.memo.get("audit", scored / "teacher_audit.json", _audit_entries) if scored else None
        return {"instruction_md": instruction,
                "evolve_plan_md": cycle.excerpt(cand / "evolve/evolve_plan.md", 20000),
                "synthetic_failure_analysis_md": cycle.excerpt(scored / "failure_analysis/failure_analysis.md",
                                                               20000) if scored else None,
                "rubric_md": cycle.excerpt(cand / "instruction/rubric.md", 8000),
                "teacher_audit": audit}

    # -- outputs
    def probe_summary(self, probes):
        """One round's probe progress over its non-abandoned campaigns (the newest is "dir")."""
        if not probes:
            return None
        by_model, total, done = {}, 0, 0
        for probe in probes:
            for model in probe["models"]:
                slots = len(probe["tasks"]) * probe["repeats"]
                finished = slots if probe["complete"] else probe["scored_tasks"] * probe["repeats"] if probe.get(
                    "unfinished") else sum(1 for (m, _, _), result in probe["slots"].items() if m == model
                                           and result.get("done"))
                pair = by_model.setdefault(model, [0, 0])
                pair[0] += finished
                pair[1] += slots
                total += slots
                done += finished
        return {"dir": probes[-1]["rel"], "dirs": [probe["rel"] for probe in probes],
                "tasks": sum(len(probe["tasks"]) for probe in probes), "slots_total": total, "slots_done": done,
                "by_model": by_model, "live": any(probe["live"] for probe in probes)}

    def rounds_out(self):
        out, previous_done = [], True
        last = max([self.max_rounds] + list(self.rounds) + [node["round"] for node in self.nodes.values()])
        for k in range(0, last + 1):
            info = self.rounds.get(k) or {"seeds": None, "probes": []}
            probes = [probe for probe in info["probes"] if not probe["abandoned"]]
            members = [node for node in self.nodes.values() if node["round"] == k and node["status"] != "abandoned"]
            final = members and all(node["status"] in ("scored", "build_failed") for node in members)
            if self.batch:  # v1.6: an ended run (each lineage went on alone): done, unfinished for good, or never begun
                status = "pending" if not members else "done" if final else "unfinished"
            elif any(probe["live"] for probe in probes):
                status = "probing"
            elif not members or not previous_done:
                status = "pending"
            elif final:
                status = "done"
            else:
                status = "building"
            previous_done = status == "done"
            out.append({"index": k, "status": status, "probe": self.probe_summary(probes),
                        "abandoned_probes": [probe["rel"] for probe in info["probes"] if probe["abandoned"]],
                        "seeds": info["seeds"], "nodes": len(members),
                        "built": sum(node["build"]["ok"] is True for node in members),
                        "scored": sum(node["status"] == "scored" for node in members),
                        "accepted": sum(node["decision"] in ("accepted", "accepted_evolve") for node in members)})
            if self.batch:  # v1.6: the probes of an ended batch run that never finished
                out[-1]["unfinished_probes"] = [probe["rel"] for probe in info["probes"] if probe.get("unfinished")]
        return out

    def probe_list(self):
        """Every probe of the build; a live one with its own slot / scoring progress (for the run's stage)."""
        out = []
        for info in self.rounds.values():
            for probe in info["probes"]:
                entry = {"path": probe["path"], "rel": probe["rel"], "round": probe["round"],
                         "complete": probe["complete"], "abandoned": probe["abandoned"], "live": probe["live"]}
                if probe["live"]:
                    entry["slots_total"] = len(probe["tasks"]) * probe["repeats"] * len(probe["models"])
                    entry["slots_done"] = sum(1 for result in probe["slots"].values() if result.get("done"))
                    entry["tasks"] = len(probe["tasks"])
                    entry["scored"] = sum(1 for node in self.nodes.values() if node["_"].get("metrics")
                                          and node["build"]["task_id"] in probe["tasks"])
                out.append(entry)
        return out


# ----------------------------------------------------------------------------------------- pure parsing helpers

def _focus_pattern(result, branch):
    """The target failure pattern evolve branch `branch` aims at, as task_builder evolve() picks it from
    analysis/<task>/result.json: patterns[branch % len(patterns)]["name"], "overall" when it lists none. None when
    evolve() could not have read it (no such file, no "patterns" key, a pattern without a name)."""
    if not isinstance(result, dict) or "patterns" not in result or not isinstance(branch, int):
        return None
    patterns = result["patterns"] or [{"name": "overall"}]
    pattern = patterns[branch % len(patterns)] if isinstance(patterns, list) else None
    return text_field(_str(pattern.get("name")), 300) if isinstance(pattern, dict) else None


def _audit_entries(path):
    """teacher_audit.json (task_builder score(): {slot: judge_pass verdict} over the teacher's passing rollouts; a
    list of verdicts carrying "slot" is read too) -> [{slot, cheated, categories, summary, rubric_score}] in slot
    order. The summary is the judge's notes (else its evidence quotes), at most 400 chars."""
    data = read_json(path)
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(as_dict(verdict).get("slot"), verdict) for verdict in data]
    else:
        return None
    out = []
    for slot, verdict in items:
        verdict = verdict if isinstance(verdict, dict) else {"notes": verdict}
        cheated, notes = verdict.get("cheated"), verdict.get("notes")
        if not (isinstance(notes, str) and notes.strip()):
            notes = "; ".join(str(item) for item in as_list(verdict.get("evidence"))) or _str(notes)
        out.append({"slot": text_field(_str(slot), 100),
                    "cheated": cheated if isinstance(cheated, bool) or cheated is None else bool(cheated),
                    "categories": [text_field(str(item), 200) for item in as_list(verdict.get("categories"))][:20],
                    "summary": text_field(notes, 400), "rubric_score": number(verdict.get("rubric_score"))})
    return sorted(out, key=lambda entry: _slot_order(entry["slot"]))


def _failure_match(path):
    """An auditor agent's verdict on one accepted task (does sc14 repeat the target's failure patterns there?), from
    runs/<run>/audit/failure_match/<task_id>.json -> {"task_id", "match": NODE.failure_match} (CONTRACT "v1.3");
    {"error": why} for a file that reads as JSON but breaks the format (not an object, schema != 1, a task_id other
    than the file's name, no verdict strong / partial / weak / none); None when it cannot be read (missing, outside
    the read roots, not a regular file, not JSON: perhaps still being written). Enums match case-insensitively; every
    text is redacted and clipped to the format's limits, every list capped; other fields are dropped."""
    try:
        if os.stat(path).st_size > AUDIT_MAX_BYTES:
            return {"error": f"larger than {AUDIT_MAX_BYTES >> 10} KB"}
    except OSError:
        return None
    data = read_json(path)
    if data is None:
        return None
    if not isinstance(data, dict):
        return {"error": "not a JSON object"}
    stem = Path(path).name[:-len(".json")]
    task_id = data.get("task_id") or stem
    if data.get("schema", 1) != 1:
        return {"error": f"schema {_brief(data.get('schema'))} is not 1"}
    if task_id != stem:
        return {"error": f"task_id {_brief(task_id)} is not the file's name"}
    verdict = _choice(data.get("verdict"), AUDIT_VERDICTS)
    if verdict is None:
        return {"error": f"verdict {_brief(data.get('verdict'))} is not one of {' / '.join(AUDIT_VERDICTS)}"}
    patterns = [{"name": _text(item.get("name"), 300), "reproduced": _choice(item.get("reproduced"), AUDIT_REPRODUCED),
                 "slots": _names(item.get("slots"), 20, 100), "evidence": _text(item.get("evidence"), 600)}
                for item in as_list(data.get("patterns")) if isinstance(item, dict)][:20]
    rollouts = [{"slot": _text(item.get("slot"), 100), "status": _choice(item.get("status"), ROLLOUT_STATUSES),
                 "failing_tests": _names(item.get("failing_tests"), 8, 300), "cause": _text(item.get("cause"), 300),
                 "pattern": _text(item.get("pattern"), 300)}
                for item in as_list(data.get("student_rollouts")) if isinstance(item, dict)][:10]
    judge = data.get("similarity_judge")
    if isinstance(judge, dict):
        score, agrees = judge.get("score"), judge.get("agrees")
        judge = {"score": None if isinstance(score, bool) else number(score),
                 "agrees": agrees if isinstance(agrees, bool) else None, "comment": _text(judge.get("comment"), 300)}
    else:
        judge = None
    return {"task_id": task_id,
            "match": {"verdict": verdict, "summary": _text(data.get("summary"), 1000), "patterns": patterns,
                      "student_rollouts": rollouts, "other_causes": _text(data.get("other_causes"), 400),
                      "similarity_judge": judge, "audited_at": _text(data.get("audited_at"), 64)}}


def _audit_counts(nodes, every=()):
    """index.json runs[].audit: how many of the run's accepted tasks (`nodes`: abandoned ones left out) carry a
    failure-match audit, and its verdicts (v1.3); the same over the audit's scope, the eligible tasks among `every`
    node of the run (v1.4: eligible, audited_eligible, eligible_verdicts)."""
    accepted = [node for node in nodes if node["decision"] in ACCEPTED]
    verdicts = [node["failure_match"]["verdict"] for node in accepted if isinstance(node.get("failure_match"), dict)]
    eligible = [node for node in every if node.get("audit_eligible") is True]
    found = [node["failure_match"]["verdict"] for node in eligible if isinstance(node.get("failure_match"), dict)]
    return dict({"accepted": len(accepted), "audited": len(verdicts)},
                **{verdict: verdicts.count(verdict) for verdict in AUDIT_VERDICTS},
                eligible=len(eligible), audited_eligible=len(found),
                eligible_verdicts={verdict: found.count(verdict) for verdict in AUDIT_VERDICTS})


def _explorer_audit(node, audited):
    """An Explorer task row's "audit" (CONTRACT "v1.7"): the failure-match verdict of a task in the audit's scope
    (NODE.audit_eligible, v1.4) that has an audit; "not_audited" for an eligible task without one, and for every task of
    a source that is never audited (a batch: NODE.audit {"status": "not_audited"}, v1.6); "not_eligible" for a task
    outside the scope (discarded, withdrawn, not scored yet, no graded sc14 rollout, an abandoned-* build root), also
    when an audit names it: as in runs[].audit, only eligible tasks count."""
    if not audited or as_dict(node.get("audit")).get("status") == "not_audited":
        return EXPLORER_NOT_AUDITED
    if node.get("audit_eligible") is not True:
        return EXPLORER_NOT_ELIGIBLE
    match = node.get("failure_match")
    verdict = match.get("verdict") if isinstance(match, dict) else None
    return verdict if verdict in AUDIT_VERDICTS else EXPLORER_NOT_AUDITED


def _rates_differ(score, rollouts, cheats):
    """Whether a scored task's metrics.json rates disagree with its own rollout rows (CONTRACT "v1.7"): task_builder
    score() computes student_rate (sc14 passes / graded sc14 rollouts), teacher_clean_rate (kimi passes the teacher audit
    did not flag / graded kimi rollouts) and valid ([graded sc14, graded kimi]) from the very records it writes to
    rollouts/index.md, so they agree when scored; they differ when metrics.json was rewritten afterwards (a regrade or
    recovery re-scored the task, or main withdrew it) while index.md kept the probe as recorded."""
    student = [row for row in rollouts if isinstance(row, dict) and row.get("model") == STUDENT
               and row.get("status") in ("pass", "fail")]
    teacher = [row for row in rollouts if isinstance(row, dict) and row.get("model") == TEACHER
               and row.get("status") in ("pass", "fail")]
    rows_student = sum(row["status"] == "pass" for row in student) / len(student) if student else None
    rows_clean = (sum(row["status"] == "pass" and cheats.get(row.get("slot")) is not True for row in teacher)
                  / len(teacher) if teacher else None)

    def same(stored, computed):
        stored = _finite(stored)
        return stored is None and computed is None or (stored is not None and computed is not None
                                                      and abs(stored - computed) < 1e-9)
    valid = _valid_pair(score.get("valid"))
    return not (same(score.get("student_rate"), rows_student) and same(score.get("teacher_clean_rate"), rows_clean)
                and (valid is None or valid == [len(student), len(teacher)]))


def _finite(value):
    """A JSON number as is (int or finite float, never a bool), else None: an Explorer cell copied from a NODE field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _valid_pair(value):
    """score.valid ([sc14, kimi] valid rollouts, metrics.json) when it is two numbers, else None."""
    if isinstance(value, list) and len(value) == 2 and all(_finite(item) is not None for item in value):
        return list(value)
    return None


def _uri(text):
    """encodeURIComponent() of app.js (its taskHash builds the page's deep links)."""
    return urllib.parse.quote(str(text), safe="-_.!~*'()")


def _loose(name):
    """A failure-pattern name for a lenient match (an audit naming a target pattern with other spacing or case)."""
    return " ".join(name.split()).casefold() if isinstance(name, str) else None


def _choice(value, choices):
    """One of `choices` (a string, matched after strip / lower-casing), else None."""
    value = value.strip().lower() if isinstance(value, str) else None
    return value if value in choices else None


def _text(value, limit):
    """A free-text field of an audit file: stripped, redacted, clipped to `limit`; None when empty or not text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    return text_field(value.strip(), limit) if isinstance(value, str) and value.strip() else None


def _names(value, count, limit):
    """At most `count` names (slots, test ids), each clipped to `limit`; a single string is split at commas/blanks."""
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    return [text_field(item.strip(), limit) for item in as_list(value) if isinstance(item, str) and item.strip()][:count]


def _brief(value):
    """A short one-line rendering of a rejected value for an error message."""
    try:
        shown = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        shown = "?"
    return shown if len(shown) <= 60 else shown[:59] + "…"


def _str(value):
    return value if isinstance(value, str) else (None if value is None else str(value))


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return round(sum(values), 3) if values else None


def _read_csv(path):
    with _open_regular(path) as raw, io.TextIOWrapper(raw, newline="") as stream:
        return {row["task_id"]: row for row in csv.DictReader(stream) if row.get("task_id")}


def _slot_order(slot):
    match = re.search(r"-r(\d+)$", str(slot))
    return (str(slot).split("-r")[0], int(match.group(1)) if match else 0)


def _sc14_card(records, row):
    """sc14's 5 target rollouts (index.json: status pass/fail/infra, F2P and P2P fractions)."""
    mine = sorted((r for r in records if isinstance(r, dict) and r.get("model") == STUDENT),
                  key=lambda r: _slot_order(r.get("slot")))
    if mine:
        rollouts = [{"slot": r.get("slot"), "f2p": fraction(r.get("f2p")), "p2p": fraction(r.get("p2p")),
                     "score": 1 if r.get("status") == "pass" else 0 if r.get("status") == "fail" else None,
                     "status": r.get("status")} for r in mine]
        ratios = [ratio(r["f2p"]) for r in rollouts if r["score"] is not None and ratio(r["f2p"]) is not None]
        return {"passed": sum(r["score"] == 1 for r in rollouts), "total": len(rollouts),
                "infra": sum(r["score"] is None for r in rollouts), "rollouts": rollouts,
                "mean_f2p_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None}
    if row:  # per_task.csv counts only
        passed, failed, infra = (_int(row.get(f"sc14_{key}")) or 0 for key in ("pass", "fail", "infra"))
        return {"passed": passed, "total": passed + failed + infra, "infra": infra, "rollouts": [],
                "mean_f2p_ratio": None}
    return None


def _record_card(records, model):
    mine = sorted((r for r in records if isinstance(r, dict) and r.get("model") == model),
                  key=lambda r: _slot_order(r.get("slot")))
    if not mine:
        return None
    rollouts = [{"f2p": fraction(r.get("f2p")), "p2p": fraction(r.get("p2p")),
                 "score": 1 if r.get("status") == "pass" else 0 if r.get("status") == "fail" else None} for r in mine]
    return {"passed": sum(r["score"] == 1 for r in rollouts), "total": len(rollouts),
            "unscored": sum(r["score"] is None for r in rollouts), "rollouts": rollouts}


def _three_way_card(row, column):
    """three_way_per_task.csv rollouts: space-separated "f2p|p2p|score" (score 1, 0, or U = unscored)."""
    items = (row.get(column) or "").split()
    if not items:
        return None
    rollouts = []
    for item in items:
        parts = item.split("|")
        score = parts[2] if len(parts) == 3 else "U"
        rollouts.append({"f2p": fraction(parts[0]) if parts else None,
                         "p2p": fraction(parts[1]) if len(parts) > 1 else None,
                         "score": 1 if score == "1" else 0 if score == "0" else None})
    return {"passed": sum(r["score"] == 1 for r in rollouts), "total": len(rollouts),
            "unscored": sum(r["score"] is None for r in rollouts), "rollouts": rollouts}


def _build_settings(extra, defaults=None):
    """--rounds / --branches from launch.jsonl "extra" (the build args after --); task_builder defaults (`defaults`: a
    batch tree's own, v1.6)."""
    settings = dict(defaults) if defaults else {"rounds": DEFAULT_ROUNDS, "branches": DEFAULT_BRANCHES}
    try:
        words = shlex.split(extra) if isinstance(extra, str) else []
    except ValueError:
        words = []
    for position, word in enumerate(words):
        for name in settings:
            value = None
            if word == f"--{name}" and position + 1 < len(words):
                value = words[position + 1]
            elif word.startswith(f"--{name}="):
                value = word.split("=", 1)[1]
            if value is not None and value.isdigit():
                settings[name] = int(value)
    return settings


def _batch_task_ids(path):
    """A batch's outputs/tasks.jsonl (one Gym dataset row per task) -> its task ids in file order (repeats kept)."""
    return [row["task_id"] for row in read_jsonl(path) if isinstance(row.get("task_id"), str) and row["task_id"]]


def _batch_origin(batch, task_id, provenance):
    """(run, target) that built one task of a batch, from outputs/<task_id>/provenance.json "task_dir":
    <batch root>/synth/runs/<run>/build/<target>/<repo>/<cand>/task/<task_id> (the root as configured or its real
    path); else why not, as text."""
    if not SAFE_TASK.fullmatch(task_id):
        return "not a task id"
    if not isinstance(provenance, dict):
        return "outputs/<task_id>/provenance.json missing or unreadable"
    task_dir = provenance.get("task_dir")
    if not isinstance(task_dir, str):
        return "provenance.json has no task_dir"
    for prefix in (str(batch.runs), os.path.realpath(batch.runs)):
        if task_dir.startswith(prefix + "/"):
            parts = task_dir[len(prefix) + 1:].split("/")
            break
    else:
        return f"task_dir {_brief(task_dir)} is outside {batch.runs}"
    if (len(parts) != 7 or parts[1] != "build" or parts[5] != "task" or parts[6] != task_id
            or not SAFE_NAME.fullmatch(parts[0]) or not SAFE_TASK.fullmatch(parts[2])
            or not SAFE_TASK.fullmatch(parts[3]) or not SAFE_NAME.fullmatch(parts[4])):
        return f"task_dir {_brief(task_dir)} is not synth/runs/<run>/build/<target>/<repo>/<cand>/task/{task_id}"
    return parts[0], parts[2]


def _parse_node(name):
    """cand-c01-r1b4-r2b0 -> ("cand-c01", [(1, 4), (2, 0)]); None for anything else."""
    steps, base = [], name
    while (match := STEP.search(base)):
        steps.insert(0, (int(match.group(1)), int(match.group(2))))
        base = base[:match.start()]
    if not base.startswith("cand-") or len(base) <= len("cand-"):
        return None
    if [k for k, _ in steps] != list(range(1, len(steps) + 1)):
        return None  # evolve() names round-k children <parent>-r<k>b<j>: rounds must count up from 1
    return base, steps


def _gate_summary(path):
    """gates-<n>.json / built.json (its "gate"): F2P/P2P counts, the first test names, the gate problems."""
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    gate = as_dict(data.get("gate")) if "gate" in data else data
    f2p, p2p = as_list(gate.get("f2p")), as_list(gate.get("p2p"))
    return {"f2p_n": len(f2p), "p2p_n": len(p2p), "f2p": [str(t) for t in f2p[:60]], "p2p": [str(t) for t in p2p[:20]],
            "problems": [str(p) for p in as_list(gate.get("problems"))], "attempt": data.get("attempt")}


def _manifest_summary(path):
    """An ATP campaign manifest (100+ KB of instructions) reduced to what the dashboard needs."""
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    settings = as_dict(data.get("settings"))
    tasks = sorted((task for task in as_list(data.get("tasks")) if isinstance(task, dict)),
                   key=lambda task: task.get("_ng_task_index", 0))
    return {"tasks": [str(task.get("task_id")) for task in tasks], "models": list(as_dict(settings.get("models"))),
            "repeats": _int(settings.get("repeats")) or 5, "max_attempts": _int(settings.get("max_attempts")) or 3,
            "created_at": number(data.get("created_at"))}


def _slot_result(slot_dir, names, max_attempts):
    """One probe slot from its attempt dirs (see Tree.scan_live)."""
    if not names:
        return {"n": 0, "done": False, "status": "pending"}
    for _, name in reversed(names):
        attempt = slot_dir / name
        if not exists(attempt / "terminal.json"):
            continue
        grades = sorted((int(m.group(1)), m.group(0)) for m in map(ATTEMPT.fullmatch, listdir(attempt / "grading")) if m)
        for _, grade in reversed(grades):
            reward = read_json(attempt / "grading" / grade / "reward.json")
            if isinstance(reward, dict) and "reward" in reward:
                return {"n": len(names), "done": True, "status": "pass" if reward.get("reward") == 1 else "fail",
                        "reward": number(reward.get("reward")),
                        "f2p": f"{reward.get('f2p_passed')}/{reward.get('f2p_total')}",
                        "p2p": f"{reward.get('p2p_passed')}/{reward.get('p2p_total')}"}
    latest = slot_dir / names[-1][1]
    if len(names) >= max_attempts and any(exists(latest / name) for name in
                                          ("terminal.json", "failure.json", "dispatch-failure.json")):
        return {"n": len(names), "done": True, "status": "infra", "reward": None, "f2p": None, "p2p": None}
    return {"n": len(names), "done": False, "status": "running"}


def _parse_index_md(path):
    """rollouts/index.md (rollouts.write_rollouts): | slot | status | reward | F2P | P2P | tool calls | patch chars |
    agent error |"""
    text = read_text(path) or ""
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("slot", "") or set(cells[0]) <= set("-"):
            continue
        slot = cells[0]
        rows.append({"slot": slot, "model": slot.rsplit("-r", 1)[0], "status": cells[1],
                     "reward": number(cells[2]), "f2p": fraction(cells[3]), "p2p": fraction(cells[4]),
                     "tool_calls": _int(cells[5]), "patch_chars": _int(cells[6]),
                     "agent_error": text_field("|".join(cells[7:]), 300) if len(cells) > 7 else ""})
    return rows


def _parse_slot_header(path):
    """The first lines of rollouts/<slot>.md: "# slot — status" then "reward ...; agent_error X; verifier_error Y"."""
    head = read_head(path, 8192) or ""
    lines = head.split("\n")
    out = {}
    if len(lines) > 1:
        match = re.search(r"; agent_error (.*?); verifier_error (.*)$", lines[1])
        if match:
            out["agent_error"] = "" if match.group(1) == "None" else text_field(match.group(1), 300)
            out["verifier_error"] = "" if match.group(2) == "None" else text_field(match.group(2), 300)
    return out


def _counts(rows, model, rate=None):
    mine = [row for row in rows if row.get("model") == model]
    passed = sum(row["status"] == "pass" for row in mine)
    failed = sum(row["status"] == "fail" for row in mine)
    infra = len(mine) - passed - failed
    if rate is None and passed + failed:
        rate = round(passed / (passed + failed), 4)
    return {"pass": passed, "fail": failed, "infra": infra, "rate": rate}


def _progress(rows):
    progress = {}
    for row in rows:
        pair = progress.setdefault(row.get("model"), [0, 0])
        pair[0] += 1
        pair[1] += 1
    return progress


def _routing(record, now):
    """student/fleet.py worker_routing(): 'ready' / 'draining' when the registry record is routable, else None."""
    if not isinstance(record, dict) or record.get("status") not in SERVING:
        return None
    heartbeat, end, drain = record.get("heartbeat_at"), record.get("end_epoch"), record.get("drain_at")
    if not isinstance(heartbeat, (int, float)) or now - heartbeat > HEARTBEAT_TIMEOUT_S:
        return None
    if isinstance(end, (int, float)) and now >= end:
        return None
    if record["status"] == "ready" and isinstance(drain, (int, float)) and now >= drain:
        return "draining"
    return record["status"]


def parse_metrics(text):
    """vLLM's Prometheus page -> running / waiting requests, KV-cache usage, token counters (summed over engines;
    KV usage averaged)."""
    sums, counts = {}, {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = PROM_LINE.match(line)
        if not match:
            continue
        value = number(match.group(2))
        if value is None:
            continue
        sums[match.group(1)] = sums.get(match.group(1), 0.0) + value
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1

    def first(*names, mean=False):
        for name in names:
            if name in sums:
                return sums[name] / counts[name] if mean else sums[name]
        return None

    return {"running": first("vllm:num_requests_running"), "waiting": first("vllm:num_requests_waiting"),
            "kv_usage": first("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", mean=True),
            "prompt_tokens_total": first("vllm:prompt_tokens_total"),
            "generation_tokens_total": first("vllm:generation_tokens_total")}


def _router_stats(recent, now):
    """Router stats over the last 10 min (by request completion time) and the last hour. A row is
    [t_end, worker, code (0 ok, 1 error, 2 client gone), duration_s, prompt_tokens, completion_tokens, clamps]."""
    window = [row for row in recent if row[0] >= now - ROUTER_WINDOW_S]
    hour = [row for row in recent if row[0] >= now - 3600]
    durations = [row[3] for row in window if row[2] == 0 and isinstance(row[3], (int, float))]
    by_worker = {}
    for row in window:
        entry = by_worker.setdefault(str(row[1]), {"requests": 0, "errors": 0, "durations": []})
        entry["requests"] += 1
        entry["errors"] += row[2] == 1
        if row[2] == 0 and isinstance(row[3], (int, float)):
            entry["durations"].append(row[3])
    errors = sum(row[2] == 1 for row in window)
    return {"window_s": ROUTER_WINDOW_S, "scope": ROUTER_SCOPE, "requests": len(window), "errors": errors,
            "disconnects": sum(row[2] == 2 for row in window),
            "error_rate": round(errors / len(window), 4) if window else None,
            "p50_s": quantile(durations, 0.5), "p90_s": quantile(durations, 0.9),
            "prompt_tokens": sum(row[4] or 0 for row in window), "completion_tokens": sum(row[5] or 0 for row in window),
            "clamps": sum(bool(row[6]) for row in window),
            "by_worker": {worker: {"requests": entry["requests"], "errors": entry["errors"],
                                   "p90_s": quantile(entry["durations"], 0.9)}
                          for worker, entry in sorted(by_worker.items())},
            "last_hour": {"requests": len(hour), "errors": sum(row[2] == 1 for row in hour)}}


def _router_row(record):
    """A student-router-requests.jsonl record (student/router.py _handle) -> compact router row."""
    started, duration = number(record.get("ts")), number(record.get("duration_s"))
    if started is None:
        return None
    status = record.get("status")
    code = 0 if status == 200 else 2 if status in ("client_disconnected", "router_shutdown") else 1
    return [round(started + (duration or 0.0), 3), str(record.get("worker")), code, duration,
            _int(record.get("prompt_tokens")), _int(record.get("completion_tokens")),
            len(record["clamps"]) if isinstance(record.get("clamps"), list) else 0]


# Markers of the log lines each scan needs (compact JSON as written by ATP / codex --json); every matched line is
# parsed and checked, a marker only avoids parsing the 100+ MB of request/stream lines.
_MARKERS = {"kimi": (b'"event":"upstream_stream_completed"', b'"event":"response"'),
            "codex": (b'"type":"turn.completed"', b'"type":"thread.started"', b'"type":"turn.failed"'),
            "router": None}


def _lines_with(body, markers):
    """The complete lines of `body` (ending in a newline) that contain any marker."""
    spans = set()
    for marker in markers:
        start = 0
        while (position := body.find(marker, start)) >= 0:
            line_start = body.rfind(b"\n", 0, position) + 1
            line_end = body.find(b"\n", position)
            spans.add((line_start, line_end))
            start = line_end + 1
    return [body[a:b] for a, b in sorted(spans)]


def _scan_safely(entry, path, st, deadline):
    try:
        return _scan_file(entry, path, st, deadline)
    except OSError:  # vanished or unreadable: no progress, retried next cycle
        return False, entry.get("offset") or 0, False, entry.get("fp"), {}, []


def _scan_file(entry, path, st, deadline):
    """Read the complete lines appended since entry["offset"]; from 0 when the file was rewritten (now smaller than
    the offset, or its first bytes differ from the fingerprint [length, sha1] taken earlier). Returns (rewritten,
    new_offset, reached_eof, fingerprint, usage_delta, router_rows); a partial last line is left for next time."""
    kind = entry["kind"]
    with _open_regular(path, buffering=0) as stream:
        head = stream.read(256)
        offset = entry.get("offset") or 0
        old = entry.get("fp")
        changed = (isinstance(old, list) and len(old) == 2 and len(head) >= old[0]
                   and hashlib.sha1(head[:old[0]]).hexdigest() != old[1])
        rewritten = st.st_size < offset or changed
        if rewritten:
            offset = 0
        fingerprint = [len(head), hashlib.sha1(head).hexdigest()]
        stream.seek(offset)
        delta, rows, carry, eof = {}, [], b"", False
        while True:
            if time.monotonic() > deadline:
                break
            block = stream.read(CHUNK)
            if not block:
                eof = True
                break
            data = carry + block if carry else block
            end = data.rfind(b"\n") + 1
            if not end:
                carry = data
                continue
            body, carry = data[:end], data[end:]
            offset += len(body)
            lines = body.splitlines() if _MARKERS[kind] is None else _lines_with(body, _MARKERS[kind])
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    _account(kind, event, delta, rows)
    return rewritten, offset, eof, fingerprint, delta, rows


def _account(kind, event, delta, rows):
    def add(name, value):
        if isinstance(value, (int, float)) and math.isfinite(value):
            delta[name] = delta.get(name, 0) + value

    if kind == "kimi":
        if event.get("event") == "upstream_stream_completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
            add("calls", 1)
            add("cost", usage.get("cost"))
            add("input_tokens", usage.get("prompt_tokens"))
            add("output_tokens", usage.get("completion_tokens"))
            add("cached_tokens", as_dict(usage.get("prompt_tokens_details")).get("cached_tokens"))
            add("reasoning_tokens", as_dict(usage.get("completion_tokens_details")).get("reasoning_tokens"))
        elif event.get("event") == "response" and isinstance(event.get("status"), int) and event["status"] >= 400:
            add("http_errors", 1)
    elif kind == "codex":
        kind_of = event.get("type")
        if kind_of == "thread.started":
            add("sessions", 1)
        elif kind_of == "turn.failed":
            add("turns_failed", 1)
        elif kind_of == "turn.completed":
            usage = as_dict(event.get("usage"))
            add("turns_completed", 1)
            add("input_tokens", usage.get("input_tokens"))
            add("cached_tokens", usage.get("cached_input_tokens"))
            add("output_tokens", usage.get("output_tokens"))
            add("reasoning_tokens", usage.get("reasoning_output_tokens"))
    elif kind == "router":
        row = _router_row(event)
        if row:
            rows.append(row)
            add("requests", 1)


def _hub_key_name(path):
    """Fallback NAME of the hub key (first of synth/common.py HUB_KEYS, or its older HUB_KEY), never its value."""
    text = read_text(path) or ""
    match = (re.search(r'^HUB_KEYS\s*=\s*\(\s*"([A-Za-z0-9_]+)"', text, re.M)
             or re.search(r'^HUB_KEY\s*=\s*"([A-Za-z0-9_]+)"', text, re.M))
    return match.group(1) if match else None


def _burn(history, remaining, sampled_at):
    """$/h over the last 1 h and 6 h of hub samples (within the newest stretch with one budget and rising spend):
    from the oldest sample inside each window to the newest, over the span window_min_* actually measured; a window
    holding fewer than 2 samples has no rate (never extrapolated). The remaining budget (of the sample taken at
    `sampled_at`) runs out at the HIGHER of the two rates (conservative while probes burst), projected from that
    sample's own time."""
    points = [row for row in history if number(row.get("spend")) is not None and number(row.get("t")) is not None]
    segment = points[-1:]
    for row in reversed(points[:-1]):
        # a new key shows a new budget or a big drop; the hub's own counter can dip by a few dollars between samples
        # (2026-10-18: 1788 -> 1785 cut the windows to 10 min and projected exhaustion from one 10-min jump)
        drop = row["spend"] - segment[0]["spend"]
        if row.get("max_budget") != segment[0].get("max_budget") or drop > max(5.0, 0.02 * row["spend"]):
            break
        segment.insert(0, row)
    out = {"per_hour_1h": None, "per_hour_6h": None, "window_min_1h": None, "window_min_6h": None,
           "eta_exhaust_iso": None, "eta_exhaust": None}
    if not segment:
        return out
    newest = segment[-1]["t"]
    for label, window in (("1h", 3600), ("6h", 6 * 3600)):
        inside = [row for row in segment if row["t"] >= newest - window]
        span = inside[-1]["t"] - inside[0]["t"] if len(inside) >= 2 else 0
        if span > 0:
            out[f"per_hour_{label}"] = round((inside[-1]["spend"] - inside[0]["spend"]) / span * 3600, 2)
            out[f"window_min_{label}"] = round(span / 60, 1)
    rate = max([value for value in (out["per_hour_1h"], out["per_hour_6h"]) if value] or [0])
    at = number(sampled_at) if number(sampled_at) is not None else newest
    if rate and rate > 0 and number(remaining) is not None:
        eta = at + max(0.0, remaining) / rate * 3600
        out["eta_exhaust"], out["eta_exhaust_iso"] = round(eta), iso(eta)
    return out


def _run_status(job, finished, log, now):
    """running | pending | finished | stopped | failed (CONTRACT.md)."""
    if finished:
        return "finished"
    state = (job or {}).get("state")
    if state in RUNNING_STATES:
        return "running"
    if state in PENDING_STATES:
        return "pending"
    if state in FAILED_STATES:
        return "failed"
    if state is None and log is not None:  # Slurm did not answer: a log written in the last 10 min means running
        try:
            return "running" if now - os.stat(log).st_mtime < 600 else "stopped"
        except OSError:
            pass
    return "stopped"


def _stage(status, tree, analysis, root):
    """A one-line account of where the build is, e.g. "round-1 probe (37/200 slots)"."""
    if status == "finished":
        accepted = sum(node["decision"] in ("accepted", "accepted_evolve") for node in (tree or {}).get("nodes", []))
        return f"finished ({accepted} accepted)"
    prefix = "" if status == "running" else f"{status}: "
    live = [probe for probe in (tree or {}).get("probes", []) if probe["live"]]
    if live:
        probe = max(live, key=lambda probe: probe["round"])
        if probe["slots_done"] < probe["slots_total"]:
            return prefix + f"round-{probe['round']} probe ({probe['slots_done']}/{probe['slots_total']} slots)"
        return prefix + f"round-{probe['round']} scoring ({probe['scored']}/{probe['tasks']} tasks scored)"
    if analysis is None:
        return prefix + "target failure analysis"
    for stage, label in (("env/stage.json", "image A (build_env)"), ("understanding/stage.json", "understanding"),
                         ("plan/stage.json", "mask plan")):
        if root is None or not (root / stage).is_file():
            return prefix + label
    if tree:
        for info in tree["rounds"]:
            if info["status"] == "building":
                members = [node for node in tree["nodes"] if node["round"] == info["index"]
                           and node["status"] != "abandoned"]
                counts = {}
                for node in members:
                    counts[node["status"]] = counts.get(node["status"], 0) + 1
                detail = ", ".join(f"{counts[key]} {key}" for key in ("planning", "authoring", "building",
                                                                       "build_failed") if counts.get(key))
                return prefix + f"round-{info['index']} building ({info['built']}/{len(members)} built" + \
                    (f"; {detail})" if detail else ")")
        done = [info for info in tree["rounds"] if info["status"] == "done"]
        if done:
            return prefix + f"round-{done[-1]['index']} done"
    return prefix + "round-0 building"


def _batch_stage(status, tree, analysis, root):
    """v1.6, the stage line of a batch run, which has ended: "finished (N accepted)" as _stage() says it; a stopped run
    names its first unfinished round with how many of its tasks were scored and how many probes never finished."""
    if status == "stopped" and tree:
        for info in tree["rounds"]:
            if info["status"] != "done" and info.get("nodes"):
                unfinished = len(info.get("unfinished_probes") or [])
                return (f"stopped: round-{info['index']} unfinished ({info['scored']}/{info['nodes']} tasks scored"
                        + (f", {unfinished} probe{'' if unfinished == 1 else 's'} unfinished)" if unfinished else ")"))
    return _stage(status, tree, analysis, root)


# ------------------------------------------------------------------------------------------------------ entry points

def run_cycle(data_dir=D / "data", state_dir=D / "state", **options):
    """One collection pass -> summary dict. Serialized by a thread lock (the server calls it from a background
    thread) and by a flock on state/collector.lock (another collector process); never raises for a broken source
    (see the returned "errors" and index.json "errors")."""
    with _LOCK:
        cycle = Cycle(data_dir, state_dir, **options)
        try:
            Path(state_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
            with open(os.open(Path(state_dir) / "collector.lock", os.O_WRONLY | os.O_CREAT, 0o600)) as lock:
                deadline = time.monotonic() + LOCK_WAIT_S
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            raise RuntimeError(f"another collector holds {state_dir}/collector.lock; cycle skipped")
                        time.sleep(1)
                return cycle.run()  # the lock is released when the file closes
        except Exception as error:  # a bug outside the guarded sources: report it, keep the server alive
            cycle.error("collector", error)
            return {"ok": False, "seconds": round(cycle.elapsed(), 2), "errors": cycle.errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("once", "loop"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default=str(D / "data"))
        sub.add_argument("--state", default=str(D / "state"))
        sub.add_argument("--no-hub", action="store_true", help="skip the hub budget sample (one tiny billed call)")
        if name == "loop":
            sub.add_argument("--interval", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.command == "once":
        summary = run_cycle(args.data, args.state, hub=False if args.no_hub else None)
        print(json.dumps(summary, indent=1))
        return 0 if summary.get("ok") else 1
    while True:
        started = time.monotonic()
        summary = run_cycle(args.data, args.state, hub=False if args.no_hub else None)
        print(time.strftime("%H:%M:%S"), f"cycle {summary.get('seconds')} s, {len(summary.get('errors') or [])} "
              f"errors" + "".join(f"\n  {error}" for error in summary.get("errors") or []), flush=True)
        time.sleep(max(5.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    sys.exit(main())
