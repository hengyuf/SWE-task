"""Durable DeepSWE attempt artifacts; imported by the isolated Gym overlay."""

import asyncio
from concurrent.futures import ThreadPoolExecutor  # hengyuf: _STREAM_IO
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import time
import fcntl
from urllib.parse import quote

from deepswe_http import redact


_ATTEMPT = ContextVar("deepswe_attempt", default=None)
REMOTE_DATA = "/tmp/atp-opencode-data"
REMOTE_CAPTURE = "/tmp/atp-capture"
STREAM_DRAIN_TIMEOUT_S = 15
# hengyuf: run_opencode writes + fsyncs each OpenCode output chunk in these threads, never on the event loop: a
# stalled Lustre fsync there starved aiohttp's PTY heartbeats and closed every PTY of the agent process at once
# (SandboxPtyError bursts, 2026-11-07; upstreams glm-rollouts-sftv3 capture_patch.py). Own pool: the loop's default
# executor (8 threads in a 4-CPU job) also resolves DNS for PTY and sandbox connections. 32 = 2 streams x 16
# sessions per process; threads start on demand.
_STREAM_IO = ThreadPoolExecutor(max_workers=32, thread_name_prefix="atp-stream-io")


async def _in_stream_io(function, *args):
    """hengyuf: function(*args) in _STREAM_IO; returns its result or raises its error, as a direct call would.

    A cancellation first waits until the call has finished (a file must not close under its write), then propagates.
    """
    call = asyncio.get_running_loop().run_in_executor(_STREAM_IO, function, *args)
    cancelled = False
    while not call.done():
        try:
            await asyncio.wait((call,))
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not call.cancelled():
            call.exception()  # retrieved: a cancelled caller sees only the cancellation
        raise asyncio.CancelledError
    return call.result()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with Path(source).open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    _sync_directory(destination.parent)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(redact(value), stream, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _event(directory, event, **details):
    with (Path(directory) / "capture-events.jsonl").open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write((json.dumps(redact({"time": time.time(), "event": event, **details}), default=str) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    _sync_directory(directory)


def _validated_directory(value):
    campaign = Path(os.environ["ATP_DEEPSWE_CAMPAIGN"]).resolve()
    directory = Path(value).resolve()
    relative = directory.relative_to(campaign)
    parts = relative.parts
    if not (len(parts) == 5 and parts[0] == "models" and parts[2] == "attempts"
            and re.fullmatch(r"[0-9]+-[0-9]+", parts[3])
            and re.fullmatch(r"attempt-[0-9]+", parts[4])):
        raise ValueError("Invalid DeepSWE attempt directory")
    if not directory.is_dir():
        raise ValueError("The dispatcher must create the attempt directory before dispatch")
    return directory


def _body_dict(body):
    return body.model_dump() if hasattr(body, "model_dump") else dict(body)


def _context(body):
    data = _body_dict(body)
    directory = _validated_directory(data["_atp_attempt_dir"])
    deadline = float(data["_atp_deadline"])
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("An absolute attempt deadline is required")
    grade = data.get("_atp_grade_attempt", 0)
    if type(grade) is not int or grade < 0:
        raise ValueError("_atp_grade_attempt must be a nonnegative integer")
    return {"directory": directory, "deadline": deadline, "body": data, "grade": grade}


def capture_attempt(function):
    """Keep terminal responses durable even when the collector disconnects."""
    @wraps(function)
    async def wrapped(self, request, body, *args, **kwargs):
        context = _context(body)
        token = _ATTEMPT.set(context)
        directory = context["directory"]
        kind = function.__name__
        destination = directory / "terminal.json" if kind == "run" else _grade_directory() / "result.json"
        _event(directory, kind + "_started")
        try:
            result = await function(self, request, body, *args, **kwargs)
            payload = _body_dict(result)
            if kind != "seed_session":
                payload["_atp_attempt_dir"] = str(directory)
                payload["_atp_capture"] = artifact_status(directory)
                _write_json(destination, payload)
                if kind == "verify" and context["body"].get("_atp_regrade"):
                    _write_json(directory / "terminal.json", payload)
                if hasattr(result, "model_copy"):
                    result = result.model_copy(update={"_atp_capture": payload["_atp_capture"]})
            _event(directory, kind + "_returned")
            return result
        except BaseException as error:
            _event(directory, kind + "_failed", error_type=type(error).__name__, error=str(error))
            if kind != "seed_session":
                _write_json(destination.with_name("failure.json"), {
                    "error_type": type(error).__name__, "error": str(error),
                    "artifact_complete": False, "_atp_attempt_dir": str(directory),
                })
            raise
        finally:
            _ATTEMPT.reset(token)
    wrapped.__annotations__ = inspect.get_annotations(function, eval_str=True)
    return wrapped


def _grade_directory():
    context = _ATTEMPT.get()
    directory = context["directory"] / "grading" / f"attempt-{context['grade']}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def verifier_directory():
    return _grade_directory()


def attempt_metadata():
    context = _ATTEMPT.get()
    if context is None:
        return {}
    directory = context["directory"]
    return {"atp-model": directory.parents[2].name,
            "atp-attempt": directory.parent.name + "-" + directory.name}


async def capture_model_headers(request):
    context = _ATTEMPT.get() or _context(await request.json())
    directory = context["directory"]
    return {"x-atp-attempt-dir": str(directory), "x-atp-deadline": str(context["deadline"]),
            "x-atp-rollout-id": directory.parent.name + "-" + directory.name}


async def register_sandbox(sandbox, phase, collect_command=None, collect_timeout=None):
    context = _ATTEMPT.get()
    if context is None:
        return
    directory = context["directory"] if phase == "agent" else _grade_directory()
    record = {"descriptor": await sandbox.serialize(), "phase": phase,
              "owner": attempt_metadata(), "collect_command": collect_command,
              "collect_timeout": collect_timeout, "recorded_at": time.time()}
    _write_json(directory / f"{phase}-sandbox.json", record)
    _event(context["directory"], "sandbox_registered", **record)
    from deepswe_admission import note_health
    await asyncio.to_thread(note_health, context["directory"].parents[2].name, "sandbox_started")


def mark_sandbox_stopped(descriptor, phase, directory=None):
    """Call only after stop succeeds, using the descriptor saved before stopping."""
    phase = "agent" if "agent" in phase else "verifier"
    if directory is None:
        context = _ATTEMPT.get()
        if context is None:
            return
        directory = context["directory"] if phase == "agent" else _grade_directory()
    sandbox_id = descriptor.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id:
        raise ValueError("Cannot confirm sandbox deletion without its sandbox_id")
    _write_json(Path(directory) / f"{phase}-sandbox-stopped.json", {
        "sandbox_id": sandbox_id, "descriptor": descriptor, "stopped_at": time.time(),
    })


def _stop_confirmed(directory, descriptor, phase):
    normalized = "agent" if "agent" in phase else "verifier"
    sandbox_id = descriptor.get("sandbox_id")
    if not sandbox_id:
        return False
    for name in (f"{normalized}-sandbox-stopped.json", "callback-stopped.json"):
        path = Path(directory) / name
        if path.is_file() and json.loads(path.read_text()).get("sandbox_id") == sandbox_id:
            return True
    return False


async def install_opencode(sandbox):
    binary = Path(os.environ["ATP_DEEPSWE_OPENCODE_BINARY"])
    rg_binary = Path(os.environ["ATP_DEEPSWE_RG_BINARY"])
    for path in (binary, rg_binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = await sandbox.exec(f"mkdir -p {REMOTE_CAPTURE}", timeout_s=60)
    if result.return_code:
        raise RuntimeError("Could not create OpenCode staging directory")
    await sandbox.upload(binary, f"{REMOTE_CAPTURE}/opencode")
    await sandbox.upload(rg_binary, f"{REMOTE_CAPTURE}/rg")
    wrapper = ('#!/bin/sh\n'
               'export OPENCODE_DISABLE_DEFAULT_PLUGINS=true OPENCODE_DISABLE_MODELS_FETCH=true\n'
               'export OPENCODE_DISABLE_AUTOUPDATE=true npm_config_offline=true\n'
               'exec "$(dirname "$0")/opencode.bin" "$@"\n')
    return (f'mkdir -p "$HOME/.opencode/bin" && '
            f'install -m 0755 {REMOTE_CAPTURE}/opencode "$HOME/.opencode/bin/opencode.bin" && '
            f'install -m 0755 {REMOTE_CAPTURE}/rg "$HOME/.opencode/bin/rg" && '
            '''test "$("$HOME/.opencode/bin/rg" --version | head -n 1 | cut -d ' ' -f 1,2)" = "ripgrep 15.1.0" && '''
            f'printf %s {shlex.quote(wrapper)} > "$HOME/.opencode/bin/opencode" && '
            'chmod 0755 "$HOME/.opencode/bin/opencode" && '
            'test "$("$HOME/.opencode/bin/opencode" --version)" = "1.17.11"')


def save_export(path):
    context = _ATTEMPT.get()
    if context is not None:
        _atomic_copy(path, context["directory"] / "export.json")


def save_verify_request(body, run_result=None):
    context = _ATTEMPT.get()
    if context is not None:
        payload = _body_dict(body)
        payload.update({key: value for key, value in (run_result or {}).items() if not key.startswith("_ng_")})
        _write_json(context["directory"] / "verify-request.json", payload)


def save_model_patch(patch):
    context = _ATTEMPT.get()
    if context is None:
        return
    quiescence = context["directory"] / "generation-quiescence.json"
    if quiescence.exists() and not json.loads(quiescence.read_text())["confirmed"]:
        raise RuntimeError("Cannot finalize a candidate before the agent process has stopped")
    path = context["directory"] / "model.patch"
    if path.exists():
        if path.read_bytes() != patch:
            raise ValueError("A saved candidate patch is immutable")
        return
    with path.open("xb") as stream:
        stream.write(patch)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    _event(context["directory"], "candidate_saved", size=len(patch))


def saved_patch_for_regrade(body):
    data = _body_dict(body)
    if not data.get("_atp_regrade", False):
        return None
    return _validated_directory(data["_atp_attempt_dir"]).joinpath("model.patch").read_bytes()


async def _download(sandbox, remote, destination):
    with tempfile.TemporaryDirectory(prefix="deepswe-artifact-") as temporary:
        staged = Path(temporary) / "artifact"
        await sandbox.download(remote, staged)
        await asyncio.to_thread(_atomic_copy, staged, destination)


async def snapshot_agent(sandbox, directory, *, final=False):
    directory = Path(directory)
    backup = (
        "import sqlite3,sys;"
        "source=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True);"
        "dest=sqlite3.connect(sys.argv[2]);source.backup(dest);dest.close();source.close()"
    )
    command = (
        f"mkdir -p {REMOTE_CAPTURE}/snapshot && "
        f"python3 -c {shlex.quote(backup)} {REMOTE_DATA}/opencode/opencode.db "
        f"{REMOTE_CAPTURE}/snapshot/opencode.db && "
        f"for name in agent-process.json wrapper-process.json replay-prevented.log stream-writer-error.json stream-writers-complete.json; do "
        f"if test -f {REMOTE_CAPTURE}/$name; then cp {REMOTE_CAPTURE}/$name {REMOTE_CAPTURE}/snapshot/$name; fi; done && "
        f"tar --exclude='./opencode/opencode.db' --exclude='./opencode/opencode.db-wal' "
        f"--exclude='./opencode/opencode.db-shm' -czf {REMOTE_CAPTURE}/opencode-data.tar.gz "
        f"-C {REMOTE_CAPTURE} snapshot -C {REMOTE_DATA} ."
    )
    try:
        result = await sandbox.exec(command, timeout_s=180)
        if result.return_code != 0:
            raise RuntimeError(f"OpenCode snapshot failed: stdout={(result.stdout or '')[-2000:]!r}; "
                               f"stderr={(result.stderr or '')[-2000:]!r}")
        name = "opencode-data.tar.gz" if final else "opencode-checkpoint.tar.gz"
        await _download(sandbox, f"{REMOTE_CAPTURE}/opencode-data.tar.gz", directory / name)
        for name in ("stdout", "stderr"):
            await _download(sandbox, f"{REMOTE_CAPTURE}/{name}.log", directory / f"remote-{name}.log")
        replay = await sandbox.exec(f"test ! -s {REMOTE_CAPTURE}/replay-prevented.log", timeout_s=15)
        if replay.return_code != 0:
            await _download(sandbox, f"{REMOTE_CAPTURE}/replay-prevented.log", directory / "replay-prevented.log")
            _event(directory, "agent_command_replay_prevented")
        for marker in ("stream-writer-error.json", "stream-writers-complete.json"):
            present = await sandbox.exec(f"test -f {REMOTE_CAPTURE}/{marker}", timeout_s=15)
            if present.return_code == 0:
                await _download(sandbox, f"{REMOTE_CAPTURE}/{marker}", directory / marker)
                if marker == "stream-writer-error.json":
                    _event(directory, "agent_stream_writer_error", **json.loads((directory / marker).read_text()))
        if final:
            await asyncio.to_thread(_record_remote_stream_completion, directory)
        _write_json(directory / ("agent-final-snapshot.json" if final else "agent-checkpoint.json"), {
            "captured_at": time.time(), "consistent_database": True, "all_sessions": True,
            "includes_tool_files": True, "final": final,
        })
        _event(directory, "agent_snapshot_saved", final=final)
        return True
    except Exception as error:
        _event(directory, "agent_snapshot_failed", final=final, error=str(error))
        return False


async def _snapshot_agent_without_database(sandbox, directory):
    """Preserve partial startup evidence only after positively proving DB absence."""
    directory = Path(directory)
    program = f"""import io,json,tarfile,time
from pathlib import Path
data=Path({REMOTE_DATA!r})
capture=Path({REMOTE_CAPTURE!r})
database=data/'opencode/opencode.db'
try:
    database.stat()
except FileNotFoundError:
    pass
else:
    raise RuntimeError('OpenCode database exists; partial startup fallback refused')
expected=[data,database,capture,capture/'stdout.log',capture/'stderr.log',capture/'agent-process.json',
          capture/'wrapper-process.json',capture/'snapshot',capture/'stream-writers-complete.json']
inventory={{'captured_at':time.time(),'artifact_complete':False,'database_absent':True,
           'present_paths':[],'missing_paths':[],
           'excluded_capture_files':['opencode','rg','agent-partial-data.tar.gz','agent-partial-inventory.json']}}
for path in expected:
    try:
        path.stat()
    except FileNotFoundError:
        inventory['missing_paths'].append(str(path))
    else:
        inventory['present_paths'].append(str(path))
capture.mkdir(parents=True,exist_ok=True)
def keep(member):
    relative=Path(member.name).parts
    if len(relative)==2 and relative[0]=='capture':
        name=relative[1]
        if name in inventory['excluded_capture_files']:
            return None
    return member
with tarfile.open(capture/'agent-partial-data.tar.gz','w:gz') as archive:
    if str(data) in inventory['present_paths']:
        archive.add(data,arcname='data')
    archive.add(capture,arcname='capture',filter=keep)
    inventory['archived_paths']=archive.getnames()
    payload=(json.dumps(inventory,indent=2)+'\\n').encode()
    member=tarfile.TarInfo('inventory.json')
    member.size=len(payload)
    archive.addfile(member,io.BytesIO(payload))
(capture/'agent-partial-inventory.json').write_bytes(payload)
"""
    try:
        result = await sandbox.exec("python3 -c " + shlex.quote(program), timeout_s=180)
        if result.return_code != 0:
            raise RuntimeError(f"Partial startup snapshot failed: stdout={(result.stdout or '')[-2000:]!r}; "
                               f"stderr={(result.stderr or '')[-2000:]!r}")
        await _download(sandbox, f"{REMOTE_CAPTURE}/agent-partial-data.tar.gz", directory / "agent-partial-data.tar.gz")
        await _download(sandbox, f"{REMOTE_CAPTURE}/agent-partial-inventory.json", directory / "agent-partial-inventory.json")
        inventory = json.loads((directory / "agent-partial-inventory.json").read_text())
        if inventory.get("database_absent") is not True:
            raise RuntimeError("Partial startup snapshot did not confirm database absence")
        _write_json(directory / "agent-partial-snapshot.json", {
            "captured_at": time.time(), "artifact_complete": False, "database_absent": True,
            "archive": "agent-partial-data.tar.gz", "inventory": "agent-partial-inventory.json",
            "note": "Available startup evidence preserved; no finalized candidate or complete rollout claimed.",
        })
        _event(directory, "agent_partial_startup_snapshot_saved")
        return True
    except Exception as error:
        _event(directory, "agent_partial_startup_snapshot_failed", error=str(error))
        return False


def _record_remote_stream_completion(directory):
    directory = Path(directory)
    try:
        control = json.loads((directory / "process-control.json").read_text())
        writers = json.loads((directory / "stream-writers-complete.json").read_text())
    except FileNotFoundError:
        return False
    if (control.get("confirmed") is not True or control.get("wrapper", {}).get("confirmed") is not True or
            control.get("wrapper_forced_stop") is not False or
            any(writers.get(key) != 0 for key in ("stdout_exit_code", "stderr_exit_code")) or
            (directory / "stream-writer-error.json").exists()):
        return False
    files = {}
    for name in ("stdout", "stderr"):
        path = directory / f"remote-{name}.log"
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files[name] = {"path": path.name, "size_bytes": size, "sha256": digest.hexdigest()}
    _write_json(directory / "remote-streams-complete.json", {
        "capture_protocol_version": 2, "completed_at": time.time(), "files": files,
        "producer_and_wrapper_stopped": True, "writers_flushed": True,
    })
    return True


def _process_record_code(name, *, isolated_producer=False):
    return (
        "import json,os,time;from pathlib import Path;"
        + ("os.setsid();pid=os.getpid();" if isolated_producer else "pid=os.getppid();") +
        "pgid=os.getpgid(pid);"
        "fields=Path('/proc/'+str(pgid)+'/stat').read_text().rsplit(')',1)[1].split();"
        "assert pgid>1 and int(fields[2])==pgid;"
        f"path=Path({REMOTE_CAPTURE + '/' + name!r});"
        "temporary=path.with_name(path.name+'.tmp');stream=temporary.open('x');"
        "json.dump({'pid':pid,'pgid':pgid,'start_time':int(fields[19]),"
        "'recorded_at':time.time()},stream);stream.flush();os.fsync(stream.fileno());stream.close();"
        "os.link(temporary,path);temporary.unlink();"
        "fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)"
    )


def _process_record_command(name="agent-process.json"):
    code = _process_record_code(name)
    return "python3 -c " + shlex.quote(code)


def _producer_command(command):
    code = (_process_record_code("agent-process.json", isolated_producer=True) +
            ";import sys;os.execv('/bin/bash',['bash','-c',sys.argv[1]])")
    return "python3 -c " + shlex.quote(code) + " " + shlex.quote(command)


def _process_control_command(*, include_wrapper=True):
    # Run through /command, never a new PTY websocket: execd may restart an
    # exited command when attaching to its old session id.
    code = """import json,os,signal,time
from pathlib import Path
def process(pid):
    try:
        fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return None
    return (int(fields[2]),int(fields[19]),fields[0])
def members(group):
    found={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item=process(int(path.name))
            if item and item[0]==group and item[2] not in ('Z','X'):
                found[int(path.name)]=item[1]
    return found
def control(path,terminate,wait_seconds=0):
    record=json.loads(Path(path).read_text())
    group=int(record['pgid'])
    if group<=1 or group==os.getpgrp():
        raise RuntimeError('Refusing unsafe process group')
    known=members(group)
    leader=process(group)
    if leader and leader[1]!=record['start_time']:
        raise RuntimeError('Recorded process group identity changed')
    if known and leader is None:
        raise RuntimeError('Cannot establish ownership of a leaderless process group')
    def refresh(known):
        current=members(group)
        leader=process(group)
        if leader and leader[1]!=record['start_time']:
            raise RuntimeError('Recorded process group identity changed')
        if current and not any(current.get(pid)==start for pid,start in known.items()):
            raise RuntimeError('Process group ownership could not be followed safely')
        return current
    sent=[]
    if not terminate:
        until=time.monotonic()+wait_seconds
        while known and time.monotonic()<until:
            time.sleep(.05)
            known=refresh(known)
    else:
        for sig in (signal.SIGTERM,signal.SIGKILL):
            known=refresh(known)
            if not known:
                break
            # Pin each member; PID reuse cannot redirect a signal.
            for pid,start in known.items():
                try:
                    fd=os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    item=process(pid)
                    if item is None:
                        continue
                    if item[:2]!=(group,start):
                        raise RuntimeError('Process identity changed before signaling')
                    signal.pidfd_send_signal(fd,sig)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
            sent.append(sig.name)
            until=time.monotonic()+5
            while time.monotonic()<until:
                known=refresh(known)
                if not known:
                    break
                time.sleep(.05)
    known=refresh(known)
    if known and terminate:
        raise RuntimeError('Owned process group did not stop')
    return {'confirmed':not bool(known),'process':record,'signals':sent}
proof=control(RECORD,True)
proof['control_channel']='sandbox_exec'
if INCLUDE_WRAPPER:
    # The producer has stopped; its pipes can now drain without signaling tees.
    wrapper=control(WRAPPER_RECORD,False,DRAIN_TIMEOUT)
    proof['wrapper_forced_stop']=not wrapper['confirmed']
    if not wrapper['confirmed']:
        wrapper=control(WRAPPER_RECORD,True)
    proof['wrapper']=wrapper
print(json.dumps(proof))
""".replace("WRAPPER_RECORD", repr(REMOTE_CAPTURE + "/wrapper-process.json"))
    code = code.replace("RECORD", repr(REMOTE_CAPTURE + "/agent-process.json"))
    code = code.replace("INCLUDE_WRAPPER", repr(include_wrapper)).replace("DRAIN_TIMEOUT", repr(STREAM_DRAIN_TIMEOUT_S))
    return "python3 -c " + shlex.quote(code)


async def _stop_process_group(sandbox, directory):
    result = await sandbox.exec(_process_control_command(), timeout_s=60)
    if result.return_code != 0:
        raise RuntimeError("Independent process control failed: " + (result.stderr or result.stdout or "")[-2000:])
    proof = json.loads(result.stdout)
    if (proof.get("confirmed") is not True or proof.get("control_channel") != "sandbox_exec" or
            proof.get("wrapper", {}).get("confirmed") is not True):
        raise RuntimeError("Independent process control did not confirm termination")
    _write_json(Path(directory) / "process-control.json", proof)
    _event(directory, "agent_control_recovered", **proof)
    return proof


async def _stop_session(session, *, sandbox, directory):
    await _stop_process_group(sandbox, directory)
    if session is not None:
        try:
            return await asyncio.wait_for(session.wait_exit(), timeout=5)
        except Exception:
            pass  # /proc proved termination; the old transport may have no exit frame.
    return None


def _stream_writer_record_command():
    code = (
        "import json,os,sys,time;from pathlib import Path;"
        "stdout,stderr=map(int,sys.argv[1:]);"
        f"directory=Path({REMOTE_CAPTURE!r});"
        "name='stream-writer-error.json' if stdout or stderr else 'stream-writers-complete.json';"
        "temporary=directory/(name+'.tmp');stream=temporary.open('w');"
        "json.dump({'stdout_exit_code':stdout,'stderr_exit_code':stderr,'recorded_at':time.time()},stream);"
        "stream.flush();os.fsync(stream.fileno());stream.close();"
        "os.replace(temporary,directory/name);fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)"
    )
    return "python3 -c " + shlex.quote(code) + ' "$ATP_STDOUT_EXIT" "$ATP_STDERR_EXIT"'


async def run_opencode(sandbox, command, request, timeout_s):
    from nemo_gym.sandbox.providers.base import SandboxExecResult

    context = _ATTEMPT.get() or _context(await request.json())
    directory = context["directory"]
    timeout = min(float(timeout_s), context["deadline"] - time.time())
    if timeout <= 0:
        raise TimeoutError("Attempt deadline expired before OpenCode execution")
    # Remote files survive a controller disconnect; pipe output is copied continuously.
    wrapped = (f"mkdir -p {REMOTE_CAPTURE} && "
               f"if ! mkdir {REMOTE_CAPTURE}/command-once 2>/dev/null; then "
               f"printf 'Command replay prevented\\n' >> {REMOTE_CAPTURE}/replay-prevented.log; exit 125; fi; "
               f"{_process_record_command('wrapper-process.json')} || exit $?; set -o pipefail; "
               f"exec 3> >(tee --output-error=warn-nopipe -a {REMOTE_CAPTURE}/stdout.log); ATP_STDOUT_PID=$!; "
               f"exec 4> >(exec 3>&-; tee --output-error=warn-nopipe -a {REMOTE_CAPTURE}/stderr.log >&2); ATP_STDERR_PID=$!; "
               f"{_producer_command(command)} < /dev/null >&3 2>&4 & ATP_PRODUCER_PID=$!; "
               "wait \"$ATP_PRODUCER_PID\" 2>/dev/null; ATP_COMMAND_EXIT=$?; "
               "exec 3>&- 4>&-; wait \"$ATP_STDOUT_PID\"; ATP_STDOUT_EXIT=$?; "
               "wait \"$ATP_STDERR_PID\"; ATP_STDERR_EXIT=$?; "
               f"{_stream_writer_record_command()} || exit 125; "
               "if ((ATP_STDOUT_EXIT || ATP_STDERR_EXIT)); then exit 125; fi; exit \"$ATP_COMMAND_EXIT\"")
    _write_json(directory / "agent-command.json", {
        "capture_protocol_version": 2,
        "command": command, "wrapped_command": wrapped, "pty": False,
        "timeout_seconds": timeout, "deadline": context["deadline"], "recorded_at": time.time(),
    })
    session = await sandbox.pty.create(command="bash -c " + shlex.quote(wrapped), pty=False)
    _write_json(directory / "pty-session.json", {"session_id": session.session_id, "started_at": time.time()})

    async def drain(reader, name):
        with (directory / name).open("ab", buffering=0) as stream:
            # hengyuf: the same write + fsync per chunk, in _STREAM_IO; the next chunk is read, and the file is
            # closed, only after it has finished (also when cancelled).
            def persist(data):
                stream.write(data)
                os.fsync(stream.fileno())
            while chunk := await reader():
                await _in_stream_io(persist, chunk)

    stop_checkpoints = asyncio.Event()

    async def checkpoints():
        while not stop_checkpoints.is_set():
            try:
                await asyncio.wait_for(stop_checkpoints.wait(), timeout=60)
                return
            except TimeoutError:
                pass
            await snapshot_agent(sandbox, directory)

    drains = [asyncio.create_task(drain(session.read, "stdout.log")),
              asyncio.create_task(drain(session.read_stderr, "stderr.log"))]
    checkpoint = asyncio.create_task(checkpoints())
    completion = asyncio.gather(session.wait_exit(), *drains)
    return_code = None
    error_type = None
    quiesced = False
    try:
        # A deadline must not cancel readers before the terminated command's final output arrives.
        completed = await asyncio.wait_for(asyncio.shield(completion), timeout=timeout)
        return_code = completed[0]
        await _stop_process_group(sandbox, directory)
        quiesced = True
        _write_json(directory / "streams-complete.json", {"completed_at": time.time(), "exit_code": return_code})
    except (Exception, asyncio.CancelledError) as error:
        error_type = ("timeout" if isinstance(error, TimeoutError) else
                      "cancelled" if isinstance(error, asyncio.CancelledError) else type(error).__name__)
        _event(directory, "agent_interrupted", error_type=error_type)
        try:
            return_code = await _stop_session(session, sandbox=sandbox, directory=directory)
            quiesced = True
            _write_json(directory / "generation-quiescence.json", {
                "confirmed": True, "exit_code": return_code, "confirmed_at": time.time(),
                "streams_complete": False,
            })
        except Exception as stop_error:
            _write_json(directory / "generation-quiescence.json", {
                "confirmed": False, "recorded_at": time.time(), "streams_complete": False,
                "error_type": type(stop_error).__name__, "error": str(stop_error),
            })
            _event(directory, "generation_not_quiesced", error_type=type(stop_error).__name__, error=str(stop_error))
        if quiesced:
            try:
                drained = await asyncio.wait_for(asyncio.gather(*drains, return_exceptions=True),
                                                 timeout=STREAM_DRAIN_TIMEOUT_S)
                if any(isinstance(value, BaseException) for value in drained):
                    raise RuntimeError("An OpenCode stream reader failed before EOF")
            except Exception as drain_error:
                _event(directory, "agent_streams_incomplete", error_type=type(drain_error).__name__,
                       error=str(drain_error))
            else:
                _write_json(directory / "streams-complete.json", {
                    "completed_at": time.time(), "exit_code": return_code, "after_interruption": True,
                })
                _write_json(directory / "generation-quiescence.json", {
                    "confirmed": True, "exit_code": return_code, "confirmed_at": time.time(),
                    "streams_complete": True,
                })
        raise
    finally:
        stop_checkpoints.set()
        await asyncio.gather(checkpoint, return_exceptions=True)
        for task in drains:
            if not task.done():
                task.cancel()
        if not completion.done():
            completion.cancel()
        await asyncio.gather(completion, *drains, return_exceptions=True)
        await asyncio.shield(snapshot_agent(sandbox, directory, final=quiesced))
        if quiesced:
            await session.close()
        else:
            # Keep the remote control handle recoverable until a later harvest
            # and owned-sandbox shutdown; release this broken local connection.
            await session.close(preserve_session=True)
    # Large raw streams remain in files; only the original completion marker is needed by Gym.
    return SandboxExecResult(stdout="OpenCode run finished" if return_code == 0 else "",
                             stderr=None, return_code=return_code,
                             error_type=("CommandReplayPrevented" if (directory / "replay-prevented.log").exists() else
                                         "StreamWriterError" if (directory / "stream-writer-error.json").exists() else error_type))


async def capture_verifier_logs(sandbox, directory=None):
    directory = Path(directory) if directory is not None else _grade_directory()
    context = _ATTEMPT.get()
    present = []
    errors = []
    for name in ("reward.json", "ctrf.json", "run.log", "stdout.log", "stderr.log"):
        remote = (f"{REMOTE_CAPTURE}/verifier-{name}" if name in {"stdout.log", "stderr.log"}
                  else f"/logs/verifier/{name}")
        try:
            result = await sandbox.exec("test -f " + shlex.quote(remote), timeout_s=30)
            if result.return_code == 0:
                await _download(sandbox, remote, directory / name)
                present.append(name)
        except Exception as error:
            errors.append({"artifact": name, "error": str(error)})
    try:
        result = await sandbox.exec("test -d /logs/verifier", timeout_s=30)
        if result.return_code == 0:
            archive = f"{REMOTE_CAPTURE}/verifier-logs.tar.gz"
            result = await sandbox.exec(f"mkdir -p {REMOTE_CAPTURE} && tar -czf {archive} -C /logs/verifier .",
                                        timeout_s=180)
            if result.return_code != 0:
                raise RuntimeError("Could not archive the full verifier artifact directory")
            await _download(sandbox, archive, directory / "verifier-logs.tar.gz")
            present.append("verifier-logs.tar.gz")
    except Exception as error:
        errors.append({"artifact": "verifier-logs.tar.gz", "error": str(error)})
    _write_json(directory / "capture.json", {"captured_at": time.time(), "present": present,
                                            "errors": errors, "artifact_complete": not errors and
                                            all(name in present for name in (
                                                "reward.json", "ctrf.json", "run.log", "stdout.log", "stderr.log",
                                                "verifier-logs.tar.gz"))})
    if context is not None:
        _event(context["directory"], "verifier_artifacts_saved", present=present, errors=errors)
    return not errors


async def run_verifier(sandbox, command, timeout_s):
    command = (f"mkdir -p {REMOTE_CAPTURE}; " + command +
               f" > {REMOTE_CAPTURE}/verifier-stdout.log 2> {REMOTE_CAPTURE}/verifier-stderr.log")
    try:
        result = await sandbox.exec(command, cwd="/app", timeout_s=timeout_s)
        return result
    finally:
        await asyncio.shield(capture_verifier_logs(sandbox))


async def capture_before_stop(sandbox, phase):
    context = _ATTEMPT.get()
    if context is None:
        return True
    if "agent" in phase:
        quiescence = context["directory"] / "generation-quiescence.json"
        if quiescence.exists() and not json.loads(quiescence.read_text())["confirmed"]:
            captured = await snapshot_agent(sandbox, context["directory"], final=False)
            if captured:
                _event(context["directory"], "agent_partial_harvest_before_stop")
            return captured
        if (context["directory"] / "agent-final-snapshot.json").exists():
            return True
        return await snapshot_agent(sandbox, context["directory"], final=True)
    return await capture_verifier_logs(sandbox)


def artifact_status(directory):
    directory = Path(directory)
    return {"artifact_complete": False,
            "agent_snapshot_saved": (directory / "agent-final-snapshot.json").is_file(),
            "streams_complete": (directory / "streams-complete.json").is_file(),
            "stream_writers_complete": (directory / "stream-writers-complete.json").is_file(),
            "stream_writer_error": (directory / "stream-writer-error.json").is_file(),
            "export_saved": (directory / "export.json").is_file(),
            "exact_patch_saved": (directory / "model.patch").is_file(),
            "note": "Campaign completion also requires hosted-request and verifier artifact validation."}


async def _sandbox_information(descriptor, sandbox_config):
    """Query management state before reconnecting; deleted sandboxes have no ready backend."""
    import httpx

    connection = sandbox_config["sandbox"]["opensandbox"]["connection"]
    base_url = connection.get("protocol", "http") + "://" + connection["domain"].rstrip("/")
    async with httpx.AsyncClient(timeout=30, trust_env=False, verify=connection.get("tls_verify", True)) as client:
        response = await client.get(base_url + "/v1/sandboxes/" + quote(descriptor["sandbox_id"], safe=""),
                                    headers={"OPEN-SANDBOX-API-KEY": connection["api_key"]})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


async def recover_attempt(attempt_dir, sandbox_config):
    """Harvest surviving owned sandboxes; interrupted attempts remain explicitly partial."""
    from nemo_gym.sandbox import AsyncSandbox, create_provider
    from nemo_gym.sandbox.config import resolve_provider_config

    directory = _validated_directory(attempt_dir)
    report = {"artifact_complete": False, "recovered_patch": (directory / "model.patch").exists(),
              "remaining_sandboxes": [], "gone_sandboxes": [], "confirmed_stopped": [], "errors": []}
    records = [directory / "agent-sandbox.json", *sorted(directory.glob("grading/attempt-*/*-sandbox.json"))]
    for path in records:
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        descriptor = record["descriptor"]
        if _stop_confirmed(path.parent, descriptor, record["phase"]):
            report["confirmed_stopped"].append(descriptor)
            continue
        sandbox = None
        operation = "get_info"
        try:
            info = await _sandbox_information(descriptor, sandbox_config)
            if info is None:
                report["gone_sandboxes"].append(descriptor)
                mark_sandbox_stopped(descriptor, record["phase"], path.parent)
                continue
            metadata = info.get("metadata") or {}
            if not record.get("owner") or any(metadata.get(key) != value for key, value in record["owner"].items()):
                raise ValueError("Refusing recovery of a sandbox with different attempt ownership")
            # Failed is a terminal provider state (e.g. an OOM-killed pod).
            # Its exec/file endpoints cannot recover the lost ephemeral disk.
            # Retain the provider evidence and all earlier local checkpoints.
            if (info.get("status") or {}).get("state") == "Failed":
                _write_json(path.parent / (record["phase"] + "-remote-unavailable.json"), {
                    "observed_at": time.time(), "descriptor": descriptor,
                    "provider_status": info["status"], "artifact_complete": False,
                    "note": "Provider reports terminal failure; remote artifacts unavailable; local evidence retained.",
                })
                report["gone_sandboxes"].append(descriptor)
                mark_sandbox_stopped(descriptor, record["phase"], path.parent)
                continue
            operation = "connect"
            sandbox = await AsyncSandbox.connect(descriptor,
                      provider=create_provider(resolve_provider_config("sandbox", sandbox_config)))
            operation = "harvest"
            if record["phase"] == "agent":
                started = directory / "started.json"
                is_callback_probe = (started.is_file() and json.loads(started.read_text()).get("purpose")
                                     == "golden_verifier_and_callback_check")
                if is_callback_probe:
                    _write_json(directory / "callback-recovery.json", {
                        "recovered_at": time.time(), "artifact_complete": False,
                        "note": "Owned callback probe has no OpenCode database; probe records were preserved.",
                    })
                    await sandbox.stop()
                    mark_sandbox_stopped(descriptor, "agent", directory)
                    continue
                # Freeze before final artifacts. A missing/unusable PTY must not
                # prevent harvesting a partial snapshot and stopping its owned
                # sandbox, but can never authorize candidate finalization.
                session_path = directory / "pty-session.json"
                quiescence_path = directory / "generation-quiescence.json"
                quiesced = ((directory / "streams-complete.json").is_file() or
                            (quiescence_path.is_file() and json.loads(quiescence_path.read_text()).get("confirmed") is True))
                if session_path.is_file() and not quiesced:
                    session_id = json.loads(session_path.read_text())["session_id"]
                    try:
                        exit_code = await _stop_session(None, sandbox=sandbox, directory=directory)
                        quiesced = True
                        _write_json(quiescence_path, {"confirmed": True, "exit_code": exit_code,
                                    "confirmed_at": time.time(), "streams_complete": False, "recovered": True})
                        _event(directory, "recovery_process_stopped", session_id=session_id, exit_code=exit_code)
                    except Exception as error:
                        _event(directory, "recovery_process_unconfirmed", session_id=session_id,
                               error_type=type(error).__name__, error=str(error))
                captured = full_snapshot = await snapshot_agent(sandbox, directory, final=quiesced)
                if not captured:
                    captured = await _snapshot_agent_without_database(sandbox, directory)
                if captured and not quiesced:
                    _event(directory, "agent_partial_harvest_before_stop")
                if full_snapshot and quiesced and record.get("collect_command") and not report["recovered_patch"]:
                    result = await sandbox.exec(record["collect_command"], timeout_s=record.get("collect_timeout") or 60)
                    if result.return_code != 0:
                        raise RuntimeError("Interrupted candidate collection failed")
                    await _download(sandbox, "/logs/artifacts/model.patch", directory / "model.patch")
                    report["recovered_patch"] = True
            else:
                captured = await capture_verifier_logs(sandbox, path.parent)
            if captured:
                await sandbox.stop()
                mark_sandbox_stopped(descriptor, record["phase"], path.parent)
            else:
                report["remaining_sandboxes"].append(descriptor)
        except Exception as error:
            report["errors"].append({"sandbox": descriptor, "error_type": type(error).__name__, "error": str(error)})
            gone = False
            if operation != "get_info":
                try:
                    gone = await _sandbox_information(descriptor, sandbox_config) is None
                except Exception:
                    pass
            report["gone_sandboxes" if gone else "remaining_sandboxes"].append(descriptor)
            if gone:
                mark_sandbox_stopped(descriptor, record["phase"], path.parent)
    _write_json(directory / "recovery.json", report)
    return report
