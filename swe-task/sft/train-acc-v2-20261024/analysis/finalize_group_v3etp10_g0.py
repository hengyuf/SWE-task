"""2026-10-31 (main agent): write results/v3etp10-group-0.json for a campaign that cannot finish.

The v3etp10 group-0 campaign has one non-terminal slot (syn-ibis-cand-c03-r1b4-3d3b86b5 rep 3: attempt 0 died at 16 min
when its sandbox was OOMKilled; state 'retryable'). A retry would need >= ~70-180 min of serving, but the last workers
stopped at ~09:04 and new batch workers were estimated to start after 12:40, so the retry would end in a serving gap.
Decision: close the campaign as it stands. Records come from rollouts.probe_records(), the synth pipeline's own view,
in which every slot that is not 'completed' is 'infra' (excluded from pass@1), compacted exactly as driver.py does.
Run with the ATP controller python from swe-task/synth (as driver.sbatch does). Read-only on the campaign.

Result (2026-10-31 07:4x): 33 completed (21 pass, 12 fail), 2 excluded as infra (1 needs_review, 1 retryable).
"""
import json, os, sys, time
from pathlib import Path
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
SYNTH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402,F401
import rollouts  # noqa: E402
model, group, groups, repeats = "v3etp10", 0, 8, 5
tasks = json.loads((TA / "tasks.json").read_text())["tasks"][group::groups]
probe_dir = TA / "probes" / model / f"group-{group}"
states = rollouts.probe_states(probe_dir)
print("slot states:", {s: sum(1 for *_, st in states if st["status"] == s) for s in {st["status"] for *_, st in states}})
records = rollouts.probe_records(probe_dir)
assert sorted(records) == sorted(t["task_id"] for t in tasks), "task set mismatch"
compact = {task_id: [{key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error",
                                                        "verifier_error", "attempt_dir")} | {"patch_chars": len(record.get("patch") or "")}
                     for record in slot_records] for task_id, slot_records in records.items()}
out = TA / "results" / f"{model}-group-{group}.json"
assert not out.exists()
ckpt = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf/"
        "super-kimi-sft-v3-empty-think-prefix-wsd-10ep-16n-ep32-20261029/exports/iter_0000142/hf")
tmp = out.with_name("." + out.name + ".tmp")
tmp.write_text(json.dumps({"model": model, "checkpoint": ckpt, "group": group, "tasks": [t["task_id"] for t in tasks],
                           "repeats": repeats, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": compact,
                           "finalized_by": "main agent 2026-10-31 via analysis/finalize_group_v3etp10_g0.py: campaign "
                                           "closed with 1 non-terminal (retryable, sandbox OOM) slot counted as infra "
                                           "(rollouts.probe_records rule); no retry (no serving capacity before ~12:40)"},
                          indent=1) + "\n")
os.replace(tmp, out)
print("wrote", out, rollouts.pass_stats([r for rs in records.values() for r in rs]))
