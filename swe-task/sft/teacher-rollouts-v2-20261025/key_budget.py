#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): budget of a hub key by NAME (default NVAPI_KEY_NEW_1), via the unchanged
tools/hub_budget.py (its --key-name choices come from common.HUB_KEYS, so the name is appended in this process only;
main's recipe of 2026-10-25). Prints hub_budget.py's one JSON line (key NAME, spend, max_budget, remaining; never
the key) and appends it to budget-log.jsonl here. Exit codes as hub_budget.py (0 fine, 3 below --min-remaining, 2 unknown).

    /usr/bin/python3 -I -B key_budget.py [--key-name NVAPI_KEY_NEW_1] [--min-remaining 0] [--tag TEXT]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import runpy  # noqa: E402

HERE = Path(__file__).resolve().parent
S = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-name", default="NVAPI_KEY_NEW_1")
    parser.add_argument("--min-remaining", default="0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
    sys.path.insert(0, S + "/synth")
    import common
    if args.key_name not in common.HUB_KEYS:
        common.HUB_KEYS = tuple(common.HUB_KEYS) + (args.key_name,)
    sys.argv = ["hub_budget.py", "--key-name", args.key_name, "--min-remaining", str(args.min_remaining)]
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            runpy.run_path(S + "/tools/hub_budget.py", run_name="__main__")
        except SystemExit as stop:
            code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 2)
    text = buffer.getvalue().strip()
    print(text)
    try:
        record = json.loads(text.splitlines()[-1])
        record.update(tag=args.tag, exit=code)
        with open(HERE / "budget-log.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
    except (ValueError, IndexError):
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
