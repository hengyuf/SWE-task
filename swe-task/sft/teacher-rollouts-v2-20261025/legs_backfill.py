#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-26): put the legs run by the pre-knob driver (key hard-coded NVAPI_KEY_NEW_1) into
legs.jsonl, from logs/tr-*.out. A pre-knob leg is a log whose job line has no "key-names" (driver.sbatch before
2026-10-26 01:00); new-driver legs write their own start/end records. Idempotent: a (job, event) already in legs.jsonl
is skipped, so rerunning after an old leg ends adds its end record. Losing babysit candidates (no job line) are skipped.

    /usr/bin/python3 -I -B legs_backfill.py [--dry-run]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "legs.jsonl"
JOB = re.compile(r"^\[(?P<at>[^\]]+)\] job (?P<job>\d+) on (?P<host>\S+): model (?P<model>\w+) taskset (?P<taskset>\w+) "
                 r"group (?P<group>\d+)(?P<rest>.*)$")
KEY = re.compile(r"teacher-rollouts: \w+ \(.*\) (?P<campaign>[\w-]+): .* key (?P<key>NVAPI_KEY_\w+),")
EXIT = re.compile(r"^\[(?P<at>[^\]]+)\] driver exit (?P<rc>-?\d+)")


def legs_from_logs() -> list:
    out = []
    for path in sorted((HERE / "logs").glob("tr-*.out")):
        text = path.read_text(errors="replace").splitlines()
        job = next((m for m in map(JOB.match, text) if m), None)
        if not job or "key-names" in job["rest"]:
            continue
        campaign = "pilot" if job["taskset"] == "pilot" else f"group-{job['group']}"
        key = next((m for m in map(KEY.search, text) if m), None)
        leg = {"model": job["model"], "campaign": campaign, "job": job["job"],
               "key": key["key"] if key else "NVAPI_KEY_NEW_1", "source": "backfill-log", "log": path.name}
        start = dict(leg, event="start", at=datetime.fromisoformat(job["at"]).strftime("%Y-%m-%dT%H:%M:%S%z"),
                     campaign_started=bool(key))
        out.append(start)
        end = next((m for m in map(EXIT.match, text) if m), None)
        if end:
            out.append(dict(leg, event="end", at=datetime.fromisoformat(end["at"]).strftime("%Y-%m-%dT%H:%M:%S%z"),
                            exit=int(end["rc"]), complete=int(end["rc"]) == 0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seen = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            seen.add((str(entry.get("job")), entry.get("event")))
    new = [entry for entry in legs_from_logs() if (entry["job"], entry["event"]) not in seen]
    for entry in new:
        print(json.dumps(entry))
    if not args.dry_run and new:
        with open(LEDGER, "a") as stream:
            stream.write("".join(json.dumps(entry) + "\n" for entry in new))
    print(f"{'would add' if args.dry_run else 'added'} {len(new)} records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
