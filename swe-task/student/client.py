"""Student fleet client: this job's student router, and the worker-registry routing rules.

    sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
    from student.client import ensure_router, wait_for_workers
    base_url = ensure_router()      # "http://127.0.0.1:<port>/v1", also exported as $SYNTH_STUDENT_BASE_URL
    wait_for_workers(n=1)           # blocks until >= 1 vLLM worker is routable and answers GET /health

ensure_router() starts student/router.py once per process tree: children inherit SYNTH_STUDENT_BASE_URL and reuse
it; the router is stopped at exit only by the process that started it.

Registry rules (the single implementation; router.py uses them too): registry/<job_id>.json is written atomically
by worker jobs only. A worker is routable iff status is "ready" or "draining", heartbeat_at is within 120 s, the job
is not past end_epoch and base_url is an http(s) URL. Ready is preferred over draining; a "ready" worker past its
drain_at counts as draining. Missing, partial and stale files are tolerated.
Load-aware routing (2026-10-20; the router applies it, see router.py): a routable worker serves at most
max_num_seqs(entry) sequences (64, or the record's "max_num_seqs"), and a routable worker is "leaving" (leaving())
when its job ends within END_MARGIN_S (600 s) or an operator created registry/<job_id>.drain (drain_requested()):
such a worker gets calls only when every other routable worker is leaving too.
Stale heartbeats (2026-10-20): worker_status(entry, now, healthy_at) also accepts a heartbeat_at older than 120 s
when the caller's own GET /health of the worker passed within HEALTH_GRACE_S (60 s), because a Lustre stall can hang
the workers' registry writes while vLLM keeps serving (15:22-15:26 all 4 bridges went stale at once). The router
keeps checking such a worker only while its checks keep passing, so a dead worker still drops out.

Stdlib only, so any Python can import this module; the router itself needs aiohttp (Gym controller venv).
"""
import sys

sys.dont_write_bytecode = True

import atexit
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

STUDENT = Path(__file__).absolute().parent
REGISTRY = STUDENT / "registry"
ROUTER = STUDENT / "router.py"
CONTROLLER_PYTHON = STUDENT.parent / "atp/runtime/controller/bin/python"  # has aiohttp (Gym 2f11dec3 deps)
BASE_URL_ENV = "SYNTH_STUDENT_BASE_URL"
ROUTABLE = ("ready", "draining")
HEARTBEAT_TIMEOUT_S = 120
HEALTH_GRACE_S = 60  # the caller's own passing GET /health this recent stands in for a stale heartbeat_at
MAX_NUM_SEQS = 64  # vLLM --max-num-seqs of the workers (serve_student.sh); a record's "max_num_seqs" overrides it
# A worker stops 30 s before end_epoch and aborts what is still running; 99.96% of the 576k unqueued official1 student
# calls up to 2026-10-20 took < 600 s (router journals).
END_MARGIN_S = 600
DRAIN_FLAG = ".drain"  # registry/<job_id>.drain: drain on demand before an early scancel (README "Drain on demand")
SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")  # matched against upper-cased variable names
# Loopback and worker traffic must never go through a site HTTP proxy from the environment.
_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_router = None  # (Popen, base_url, owner pid) of the router this process started


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_registry(registry=REGISTRY) -> dict:
    """{job_id: entry} for the registry files; entry is None while a file is unreadable (e.g. partial)."""
    try:
        paths = sorted(Path(registry).glob("*.json"))
    except OSError:
        return {}
    entries = {}
    for path in paths:
        if path.name.startswith("."):
            continue
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            entry = None
        if not isinstance(entry, dict):
            entries[path.stem] = None
        elif str(entry.get("job_id", path.stem)) == path.stem:  # skip copies that are not <job_id>.json
            entries[path.stem] = entry
    return entries


def worker_status(entry, now=None, healthy_at=None):
    """Routing status of a registry entry: "ready", "draining", or None when it is not routable.

    healthy_at: when the caller's own GET /health of this worker last passed (time.time() clock). A pass within
    HEALTH_GRACE_S stands in for a heartbeat_at older than HEARTBEAT_TIMEOUT_S (a stalled filesystem, or a clock
    problem); the record must still say ready/draining, and the job must not be past end_epoch."""
    if not isinstance(entry, dict) or entry.get("status") not in ROUTABLE:
        return None
    now = time.time() if now is None else now
    heartbeat, end, drain = entry.get("heartbeat_at"), entry.get("end_epoch"), entry.get("drain_at")
    if not _number(heartbeat):
        return None
    if now - heartbeat > HEARTBEAT_TIMEOUT_S and not (_number(healthy_at) and now - healthy_at <= HEALTH_GRACE_S):
        return None
    if _number(end) and now >= end:
        return None
    if not isinstance(entry.get("base_url"), str) or not entry["base_url"].startswith(("http://", "https://")):
        return None
    if entry["status"] == "ready" and _number(drain) and now >= drain:
        return "draining"
    return entry["status"]


def max_num_seqs(entry) -> int:
    """Sequences the worker's vLLM runs at once: the record's "max_num_seqs" if it is a positive integer, else 64."""
    value = entry.get("max_num_seqs") if isinstance(entry, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else MAX_NUM_SEQS


def drain_requested(registry=REGISTRY) -> set:
    """Job ids that have an operator drain flag registry/<job_id>.drain."""
    try:
        return {path.name[:-len(DRAIN_FLAG)] for path in Path(registry).glob("*" + DRAIN_FLAG)
                if not path.name.startswith(".")}
    except OSError:
        return set()


def leaving(entry, now=None, drain_flag=False):
    """Why a routable worker should get calls only when every other routable worker is leaving too, else None:
    "drain_flag" (an operator created registry/<job_id>.drain to empty it before an early scancel; this takes effect
    before the worker publishes "draining") or "ending" (the job ends within END_MARGIN_S; the worker stops 30 s
    before end_epoch and aborts the calls still running)."""
    if drain_flag:
        return "drain_flag"
    now = time.time() if now is None else now
    end = entry.get("end_epoch") if isinstance(entry, dict) else None
    if _number(end) and end - now < END_MARGIN_S:
        return "ending"
    return None


def routable_workers(registry=REGISTRY, now=None) -> list:
    """Routable registry entries (with their "routing_status"), ready before draining, then by job id."""
    now = time.time() if now is None else now
    workers = [dict(entry, routing_status=status) for entry in read_registry(registry).values()
               if (status := worker_status(entry, now))]
    return sorted(workers, key=lambda entry: (entry["routing_status"] != "ready", str(entry.get("job_id"))))


def registry_summary(registry=REGISTRY, now=None) -> str:
    """Counts by routing state, e.g. "ready=2 starting=1 stale=1"."""
    counts = {}
    for entry in read_registry(registry).values():
        state = worker_status(entry, now)
        if state is None:
            state = "unreadable" if entry is None else (
                "stale" if entry.get("status") in ROUTABLE else str(entry.get("status")))
        counts[state] = counts.get(state, 0) + 1
    return " ".join(f"{state}={count}" for state, count in sorted(counts.items())) or "empty"


def root_url(base_url: str) -> str:
    """http://<ip>:<port> of an OpenAI base URL (vLLM serves /health and /tokenize there)."""
    return base_url.rstrip("/").removesuffix("/v1")


def http_status(url: str, timeout: float = 10.0):
    """Status of GET url, or None if it cannot be reached."""
    try:
        with _HTTP.open(url, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, ValueError, http.client.HTTPException):
        return None


def scratch_dir() -> Path:
    """Node-local scratch of this job (same path as synth/common.scratch())."""
    base = Path("/raid/scratch") if os.access("/raid/scratch", os.W_OK) else Path("/tmp")
    return base / f"synth-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getuid()}"


def _router_python() -> str:
    if importlib.util.find_spec("aiohttp") is None and CONTROLLER_PYTHON.exists():
        return str(CONTROLLER_PYTHON)
    return sys.executable


def _stop(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def stop_router() -> None:
    """Stop the router this process started (also the atexit hook); a reused router is left alone."""
    global _router
    if _router is None:
        return
    (process, url, owner), _router = _router, None
    if owner != os.getpid():  # a forked child inherited this state: the router belongs to the parent
        return
    if os.environ.get(BASE_URL_ENV) == url:
        del os.environ[BASE_URL_ENV]
    _stop(process)


def ensure_router(*, registry=REGISTRY, log_dir=None, python=None, timeout_s=60.0) -> str:
    """This process tree's student router base URL ("http://127.0.0.1:<port>/v1"), starting the router if needed.

    Reuses $SYNTH_STUDENT_BASE_URL when it answers GET /models with 200. Otherwise starts `python -B router.py`
    detached (own session; python defaults to sys.executable, or the ATP controller's when that lacks aiohttp) with
    its logs in log_dir (default: node-local <scratch>/student-router) and an environment without
    SYNTH_STUDENT_BASE_URL and without any variable whose name contains KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL; waits
    until it answers, exports SYNTH_STUDENT_BASE_URL and stops it at exit.
    """
    global _router
    url = os.environ.get(BASE_URL_ENV)
    if url and http_status(url.rstrip("/") + "/models", 5) == 200:
        return url
    if not log_dir:  # keep the job scratch private (mkdir -p would create it with default permissions)
        scratch_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir else scratch_dir() / "student-router"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    port_file = log_dir / f"router-{os.getpid()}.port"
    port_file.unlink(missing_ok=True)
    argv = [python or _router_python(), "-B", str(ROUTER), "--registry", str(registry), "--host", "127.0.0.1",
            "--port", "0", "--port-file", str(port_file), "--log-dir", str(log_dir), "--parent-pid", str(os.getpid())]
    env = {key: value for key, value in os.environ.items()
           if not SECRET_NAME.search(key.upper()) and key != BASE_URL_ENV}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    output = log_dir / "student-router.stderr"
    with output.open("ab") as stream:
        process = subprocess.Popen(argv, cwd=log_dir, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Student router exited with code {process.returncode}; see {output}")
        try:
            url = f"http://127.0.0.1:{int(port_file.read_text())}/v1"
        except (OSError, ValueError):
            url = None
        if url and http_status(url + "/models", 5) == 200:
            break
        if time.monotonic() > deadline:
            _stop(process)
            raise TimeoutError(f"Student router did not answer within {timeout_s:.0f} s; see {output}")
        time.sleep(0.2)
    if _router is None:
        atexit.register(stop_router)
    elif _router[2] == os.getpid():  # a router this process started earlier stopped answering: replace it
        _stop(_router[0])
    _router = (process, url, os.getpid())
    os.environ[BASE_URL_ENV] = url
    return url


def wait_for_workers(n=1, timeout_s=10800, registry=REGISTRY, poll_s=30, log=print) -> list:
    """Block until >= n workers are routable (registry rules) and answer GET /health; returns their entries."""
    deadline = time.monotonic() + timeout_s
    reported, reported_at = None, 0.0
    while True:
        workers = [worker for worker in routable_workers(registry)
                   if http_status(root_url(worker["base_url"]) + "/health", 10) == 200]
        if len(workers) >= n:
            log(f"student workers: {len(workers)} routable ({', '.join(str(w.get('job_id')) for w in workers)})")
            return workers
        now, summary = time.monotonic(), registry_summary(registry)
        if now >= deadline:
            raise TimeoutError(f"{len(workers)}/{n} student workers routable after {timeout_s:.0f} s "
                               f"(registry {registry}: {summary})")
        if (len(workers), summary) != reported or now - reported_at >= 300:
            log(f"waiting for {n} student worker(s): {len(workers)} routable; registry: {summary}")
            reported, reported_at = (len(workers), summary), now
        time.sleep(max(0.0, min(poll_s, deadline - now)))
