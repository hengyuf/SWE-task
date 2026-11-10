#!/usr/bin/env python3
"""teacher-deepswe collect phase (2026-10-31): ATP's own state of every slot of this run's campaigns, read-only.

The run stopped at the NVAPI_KEY_NEW_3 budget floor before any campaign finished, so no campaign has a results file.
This script asks ATP's own code which slots are graded ("completed") and what ATP would do with the others:
deepswe_campaign.slot_state() from S/atp/scripts (unchanged, imported from that directory) on every slot of
probes/<model>/<campaign> (default: every campaign dir with a manifest.json).

Read-only: slot_state() can write a generation-validity cache into an attempt dir (deepswe_campaign.atomic_json).
Here atomic_json is replaced by a no-op that only records the path it would have written, so nothing under probes/
changes. The mtime / size of every file under every attempt dir is compared before and after (the result says
"probes_unchanged"). ATP_DEEPSWE_RECOVERY_ALLOWANCE is unset, as in the legs (no recovery allowance). ATP's
tool-delivery audit may extract an OpenCode snapshot into a TemporaryDirectory under /raid/scratch or /tmp, which ATP's
code removes itself. No hub, sandbox or secret is touched.

Output (--out, default analysis/atp-slot-states.json):
  campaigns.<model>/<campaign>: {repeats, max_attempts, tasks, status_counts, slots: [{task_id, task_index, repeat,
  status, retryable, attempts, attempt_dir, regrade}]}; would_write: the validity caches slot_state() would have
  written (path, failure); probes_unchanged / changed_files.

    /usr/bin/python3 -I -B final/atp_states.py [--campaigns kimi/w1-g0 ...] [--out PATH]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

TD = Path(__file__).resolve().parent.parent
ATP_SCRIPTS = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/scripts"
MODELS = ("kimi", "glm", "deepseek")


def snapshot(probe: Path, model: str) -> dict:
    """{path: (size, mtime_ns)} of every file under every attempt dir of a campaign."""
    out = {}
    for slot_dir in (probe / "models" / model / "attempts").glob("*-*"):
        for root, _, files in os.walk(slot_dir):
            for name in files:
                path = os.path.join(root, name)
                try:
                    info = os.lstat(path)
                except OSError:
                    continue
                out[path] = (info.st_size, info.st_mtime_ns)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="*", help="model/campaign (default: every probe dir with a manifest)")
    parser.add_argument("--out", type=Path, default=TD / "analysis" / "atp-slot-states.json")
    args = parser.parse_args()
    os.environ.pop("ATP_DEEPSWE_RECOVERY_ALLOWANCE", None)
    sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, ATP_SCRIPTS)
    import deepswe_campaign as campaign  # noqa: E402  (ATP's own code, unchanged)

    would_write = []

    def no_write(path, value):
        would_write.append({"path": str(path), "failure": (value or {}).get("failure")})

    campaign.atomic_json = no_write
    if args.campaigns:
        targets = [tuple(item.split("/", 1)) for item in args.campaigns]
    else:
        targets = [(model, probe.name) for model in MODELS for probe in sorted((TD / "probes" / model).glob("*"))
                   if (probe / "manifest.json").exists()]
    out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "atp_scripts": ATP_SCRIPTS,
           "function": "deepswe_campaign.slot_state (atomic_json replaced by a no-op recorder)", "campaigns": {}}
    before = {}
    for model, name in targets:
        before[(model, name)] = snapshot(TD / "probes" / model / name, model)
    for model, name in targets:
        probe = TD / "probes" / model / name
        manifest = campaign.load_manifest(probe)
        settings = manifest["settings"]
        slots = []
        for task in manifest["tasks"]:
            for repeat in range(settings["repeats"]):
                state = campaign.slot_state(probe, model, task["_ng_task_index"], repeat)
                slots.append({"task_id": task["task_id"], "task_index": task["_ng_task_index"], "repeat": repeat,
                              "status": state["status"], "retryable": state["retryable"],
                              "attempts": state["attempts"], "attempt_dir": state["attempt_dir"],
                              "regrade": state.get("regrade")})
        out["campaigns"][f"{model}/{name}"] = {
            "repeats": settings["repeats"], "max_attempts": settings.get("max_attempts"),
            "tasks": len(manifest["tasks"]), "status_counts": dict(collections.Counter(s["status"] for s in slots)),
            "slots": slots}
        print(json.dumps({"campaign": f"{model}/{name}", "status_counts": out["campaigns"][f"{model}/{name}"]["status_counts"]}),
              flush=True)
    changed = []
    for (model, name), files in before.items():
        after = snapshot(TD / "probes" / model / name, model)
        for path in sorted(set(files) | set(after)):
            if files.get(path) != after.get(path):
                changed.append({"path": path, "before": files.get(path), "after": after.get(path)})
    out["would_write"] = would_write
    out["files_checked"] = sum(len(files) for files in before.values())
    out["probes_unchanged"] = not changed
    out["changed_files"] = changed[:200]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name("." + args.out.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps({"out": str(args.out), "files_checked": out["files_checked"], "probes_unchanged": not changed,
                      "changed": len(changed), "would_write": len(would_write)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
