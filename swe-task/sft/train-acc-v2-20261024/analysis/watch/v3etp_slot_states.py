#!/usr/bin/env python3
"""Read-only per-group slot states of the v3etp train-acc campaigns (2026-10-31, continuation watcher c1w1).

    /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -B \
        analysis/watch/v3etp_slot_states.py [MODEL ...]   # default: v3etp5 v3etp10

Per group: ATP slot-state counts, the agent-deadline range of the active slots (started.json "deadline"), and every
non-completed terminal / retryable slot with terminal.json's opencode_run_error_type and verifier_error.
"""
import sys

sys.dont_write_bytecode = True
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

SYNTH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402,F401  (puts ATP's scripts on sys.path for rollouts)
import rollouts  # noqa: E402


def hm(epoch):
    return time.strftime("%H:%M", time.localtime(epoch)) if epoch else "-"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


for model in sys.argv[1:] or ["v3etp5", "v3etp10"]:
    total = collections.Counter()
    print(f"== {model}")
    for group in range(8):
        probe = TA / "probes" / model / f"group-{group}"
        if not (probe / "manifest.json").exists():
            print(f"  g{group}: no campaign yet")
            continue
        counts, deadlines = collections.Counter(), []
        for task, _model, repeat, state in rollouts.probe_states(probe):
            counts[state["status"]] += 1
            attempt = Path(state["attempt_dir"]) if state.get("attempt_dir") else None
            if state["status"] == "active" and attempt:
                deadline = read_json(attempt / "started.json").get("deadline")
                if deadline:
                    deadlines.append(deadline)
            elif state["status"] not in ("completed", "active", "pending"):
                terminal = read_json(attempt / "terminal.json") if attempt else {}
                print(f"  g{group} {task['task_id']} r{repeat} {state['status']} attempts={state['attempts']} "
                      f"agent_error={terminal.get('opencode_run_error_type')} "
                      f"verifier_error={str(terminal.get('verifier_error'))[:120]} "
                      f"patch_bytes={terminal.get('model_patch_bytes')}")
        total.update(counts)
        span = f"{hm(min(deadlines))}-{hm(max(deadlines))}" if deadlines else "-"
        results = (TA / "results" / f"{model}-group-{group}.json").exists()
        print(f"  g{group}: {dict(counts)} active deadlines {span} results_file={results}")
    print(f"  total {dict(total)}")
