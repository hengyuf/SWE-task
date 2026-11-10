"""blocking-io audit probe 3: wall time of ATP's real sync capture helpers on Lustre (private scratch only).

Measures, in the calling thread, what each helper would block an event loop for:
  chunk      : drain's per-chunk  stream.write(chunk) + os.fsync()  (unbuffered append, 120-byte chunks)
  _event     : deepswe_capture._event      (open ab + flock + write + fsync + unlock + close + dir fsync)
  _write_json: deepswe_capture._write_json (mkdir + tmp write + fsync + rename + dir fsync), ~200 B payload
  _atomic_copy_2MB: deepswe_capture._atomic_copy of a 2 MB file (save_export's export.json size class)
Stops each series early when the probe's time budget runs out.

Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir> [budget_seconds]
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/scripts")  # read-only, -B
import deepswe_capture  # noqa: E402


def summarize(name, samples):
    if not samples:
        return {"name": name, "n": 0}
    ordered = sorted(samples)

    def pick(q):
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 4)

    return {"name": name, "n": len(samples), "p50_s": pick(0.5), "p90_s": pick(0.9), "p99_s": pick(0.99),
            "max_s": round(ordered[-1], 4), "mean_s": round(statistics.fmean(samples), 4),
            "over_1s": sum(value > 1 for value in samples), "over_0_1s": sum(value > 0.1 for value in samples)}


def main():
    scratch = Path(sys.argv[1]).resolve() / f"latency-{os.getpid()}"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    scratch.mkdir(parents=True)
    deadline = time.monotonic() + budget
    plans = (("chunk", 120), ("_event", 40), ("_write_json", 40), ("_atomic_copy_2MB", 6))
    share = budget / len(plans)
    source = scratch / "export-source.json"
    source.write_bytes(os.urandom(2 * 1024 * 1024))
    results = []
    stream = (scratch / "stdout.log").open("ab", buffering=0)
    try:
        for name, count in plans:
            samples, series_end = [], min(deadline, time.monotonic() + share)
            for index in range(count):
                if time.monotonic() > series_end:
                    break
                started = time.monotonic()
                if name == "chunk":
                    stream.write(b"x" * 119 + b"\n")
                    os.fsync(stream.fileno())
                elif name == "_event":
                    deepswe_capture._event(scratch, "agent_snapshot_saved", final=False)
                elif name == "_write_json":
                    deepswe_capture._write_json(scratch / "agent-checkpoint.json", {
                        "captured_at": time.time(), "consistent_database": True, "all_sessions": True,
                        "includes_tool_files": True, "final": False})
                else:
                    deepswe_capture._atomic_copy(source, scratch / "export.json")
                samples.append(time.monotonic() - started)
            results.append(summarize(name, samples))
    finally:
        stream.close()
    for row in results:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
