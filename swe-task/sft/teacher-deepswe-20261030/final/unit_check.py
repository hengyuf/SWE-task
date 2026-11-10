#!/usr/bin/env python3
"""teacher-deepswe collect phase (2026-10-31): a small read-only check of collect.py's slot view and row builder on a
few real slots (login-node sized: small journals only). Prints JSON; writes nothing.

    /usr/bin/python3 -I -B final/unit_check.py [COLLECT_PY]
"""
import sys

sys.dont_write_bytecode = True

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else TD / "collect.py"
loader = importlib.machinery.SourceFileLoader("tdw_collect_check", str(path))  # also a draft without .py
spec = importlib.util.spec_from_loader("tdw_collect_check", loader)
collect = importlib.util.module_from_spec(spec)
loader.exec_module(collect)
probe = TD / "probes"
slots = [("deepseek", "w1-g2", "deepswe-task-0-1", probe / "deepseek/w1-g2/models/deepseek/attempts/0-1"),
         ("deepseek", "w1-g2", "cancelled 5-2", probe / "deepseek/w1-g2/models/deepseek/attempts/5-2"),
         ("kimi", "w1-g0", "cancelled 2-2", probe / "kimi/w1-g0/models/kimi/attempts/2-2"),
         ("kimi", "w1-g0", "not started 26-4", probe / "kimi/w1-g0/models/kimi/attempts/26-4")]
for model, name, label, slot_dir in slots:
    view = collect.own_view({"model": model, "slot_dir": str(slot_dir), "journal": True})
    print(json.dumps({"label": label, "view": view}, default=str)[:1500])
view = collect.own_view({"model": "deepseek", "slot_dir": str(slots[0][3]), "journal": True})
record = {"slot": "deepseek-r1", "status": view["status"], "attempt_dir": view["attempt_dir"]}
job = {"model": "deepseek", "campaign": "w1-g2", "complete": False, "record": record, "first": record,
       "task": {"task_id": "?", "task_index": None, "language": None, "category": None, "wave": 1,
                "covered_by_accepted_synth": False, "base_k10": None},
       "journal": True, "model_id": collect.MODEL_IDS["deepseek"]}
row = collect.build_row(job)
print(json.dumps({key: value for key, value in row.items() if key not in ("output", "patch", "failing_tests")},
                 default=str)[:4000])
print(json.dumps({"output_items": len(row["output"]), "patch_chars": len(row["patch"])}))
