#!/usr/bin/env python3
"""Queue-manager state helper (S/synth/queue/state.json). Atomic writes (temp file + rename).

  python3 -I -B qstate.py show
  python3 -I -B qstate.py set RUN key=json_value ...        # e.g. status='"running"' repo='"tobymao/saq"'
  python3 -I -B qstate.py add-job RUN JOBID [kind]          # kind: launch|resume|relaunch-abandon
  python3 -I -B qstate.py note RUN "text"                   # per-run note (timestamped)
  python3 -I -B qstate.py gnote "text"                      # global note (timestamped)
  python3 -I -B qstate.py gset key=json_value ...           # global field
"""
import json, os, sys, tempfile, time
from pathlib import Path

STATE = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue/state.json")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load():
    return json.loads(STATE.read_text())


def save(state):
    state["updated_at"] = now()
    fd, tmp = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(STATE.parent))
    with os.fdopen(fd, "w") as handle:
        json.dump(state, handle, indent=1, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, STATE)


def parse_pairs(pairs):
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv):
    cmd = argv[0]
    state = load()
    if cmd == "show":
        print(json.dumps(state, indent=1))
        return 0
    if cmd == "set":
        run = state["runs"][argv[1]]
        run.update(parse_pairs(argv[2:]))
    elif cmd == "add-job":
        run = state["runs"][argv[1]]
        kind = argv[3] if len(argv) > 3 else "launch"
        run.setdefault("jobs", []).append({"job_id": argv[2], "kind": kind, "at": now()})
    elif cmd == "note":
        state["runs"][argv[1]].setdefault("notes", []).append(f"{now()} {argv[2]}")
    elif cmd == "gnote":
        state["global"].setdefault("notes", []).append(f"{now()} {argv[1]}")
    elif cmd == "gset":
        state["global"].update(parse_pairs(argv[1:]))
    else:
        raise SystemExit(f"unknown command {cmd}")
    save(state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
