#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): per-task size of the existing kimi build-probe rollouts and sc14 train-acc
rollouts (main-session assistant turns and step-finish tokens from export.json, wall time from started.json to
terminal.json), to pick 2 representative pilot tasks (median-sized, two different languages/runs). Read-only.

    /usr/bin/python3 -I -B pick_pilot.py   -> pilot_pick.json
"""
import sys

sys.dont_write_bytecode = True

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(HERE))
import sources  # noqa: E402


def size(slot):
    attempt = slot.get("attempt_dir")
    if not attempt:
        return None
    try:
        export = json.loads((Path(attempt) / "export.json").read_text())
    except (OSError, ValueError):
        return None
    turns = tokens = output = 0
    for message in export.get("messages", []):
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        turns += 1
        t = info.get("tokens") or {}
        tokens += (t.get("input") or 0) + ((t.get("cache") or {}).get("read") or 0)
        output += (t.get("output") or 0) + (t.get("reasoning") or 0)
    started = Path(attempt) / "started.json"
    terminal = Path(attempt) / "terminal.json"
    wall = (os.path.getmtime(terminal) - os.path.getmtime(started)) if started.exists() and terminal.exists() else None
    return {"turns": turns, "prompt_tokens": tokens, "output_tokens": output, "wall_s": wall}


def main():
    rows = {}
    for source in ("kimi", "sc14"):
        slots = sources.slots(source)
        with ThreadPoolExecutor(16) as pool:
            sizes = list(pool.map(size, slots))
        for slot, measured in zip(slots, sizes):
            entry = rows.setdefault(slot["task_id"], {"run": slot["run"]})
            entry.setdefault(source, []).append(dict(measured or {}, status=slot["status"]))
    table = []
    for task_id, entry in rows.items():
        line = {"task_id": task_id, "run": entry["run"]}
        for source in ("kimi", "sc14"):
            valid = [m for m in entry.get(source, []) if m.get("turns")]
            line[source] = {
                "passes": sum(m["status"] == "pass" for m in entry.get(source, [])),
                "turns_med": statistics.median([m["turns"] for m in valid]) if valid else None,
                "prompt_Mtok_mean": round(statistics.fmean([m["prompt_tokens"] for m in valid]) / 1e6, 2) if valid else None,
                "output_ktok_mean": round(statistics.fmean([m["output_tokens"] for m in valid]) / 1e3, 1) if valid else None,
                "wall_min_med": round(statistics.median([m["wall_s"] for m in valid if m["wall_s"]]) / 60, 1)
                if any(m["wall_s"] for m in valid) else None}
        table.append(line)
    kimi_prompt = [line["kimi"]["prompt_Mtok_mean"] for line in table if line["kimi"]["prompt_Mtok_mean"] is not None]
    summary = {"tasks": len(table), "kimi_prompt_Mtok_mean_over_tasks": round(statistics.fmean(kimi_prompt), 3),
               "kimi_prompt_Mtok_median_over_tasks": statistics.median(kimi_prompt)}
    (HERE / "pilot_pick.json").write_text(json.dumps({"summary": summary, "tasks": table}, indent=1) + "\n")
    print(json.dumps(summary))
    for line in sorted(table, key=lambda l: l["kimi"]["prompt_Mtok_mean"] or 0):
        print(f"{line['task_id']:45s} {line['run']:24s} kimi {line['kimi']} | sc14 {line['sc14']}")


if __name__ == "__main__":
    main()
