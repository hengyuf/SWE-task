#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): the live monitor's metrics (analysis/watch/live_etp.py, main's live_v3etp20.py) at the
end of the run, recomputed from the final data without ATP's slot-state code (stdlib only; read-only on the inputs):
  pass@1             passes / graded slots (TA/results; infra excluded)
  pass@5             over the tasks whose 5 slots are all graded: >= 1 pass
  empty patch        graded slots with patch_chars 0, % of graded
  timeouts           graded slots whose agent_error is "TimeoutError" (the live flag only)
  every action reasoned  rollouts with all_actions_reasoned (teacher-rollout extractor rows, ../features/v3etp20.jsonl)
  missing </think>   (recovered + unclosed_no_tool) / calls from the counting parser's counters: the last
                     TA_THINK_COUNTS line per vLLM process in fleets/v3etp20/logs (logged every 200 calls), as the live
                     table reads them; next to it the final-attempt join of analysis/v3etp20/think/split_table.json
    python3 -I -B live_metrics_final.py -> live_metrics_final.{json,txt}
"""
import collections
import json
from pathlib import Path
import re

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
COUNTS = re.compile(rb"TA_THINK_COUNTS pid=(\d+) (.*)")


def main():
    per_task = collections.defaultdict(list)
    for path in sorted(TA.glob("results/v3etp20-group-*.json")):
        for task_id, recs in json.loads(path.read_text())["records"].items():
            per_task[task_id] += recs
    graded = [r for v in per_task.values() for r in v if r["status"] in ("pass", "fail")]
    full = [v for v in per_task.values() if len(v) == 5 and all(r["status"] in ("pass", "fail") for r in v)]
    feats = [json.loads(line) for line in open(HERE.parent / "features/v3etp20.jsonl")]
    ex = [f for f in feats if f.get("export")]
    tot = collections.Counter()
    for log in sorted((TA / "fleets/v3etp20/logs").glob("vllm-*.log")):
        last = {}
        with open(log, "rb") as handle:
            for line in handle:
                if b"TA_THINK_COUNTS" in line:
                    m = COUNTS.search(line)
                    if m:
                        last[m.group(1)] = m.group(2).decode(errors="replace")
        for text in last.values():
            for k, v in re.findall(r"(\w+)=(\d+)", text):
                tot[k] += int(v)
    split = json.loads((HERE.parent / "think/split_table.json").read_text())["v3etp20"]
    joined = split["recovered"]["calls"] + split["unclosed_no_tool"]["calls"]
    out = {
        "pass_at_1": [sum(r["status"] == "pass" for r in graded), len(graded)],
        "pass_at_5_full_tasks": [sum(any(r["status"] == "pass" for r in v) for v in full), len(full)],
        "empty_patch_graded": [sum(not r.get("patch_chars") for r in graded), len(graded)],
        "timeout_flag_graded": [sum(r.get("agent_error") == "TimeoutError" for r in graded), len(graded)],
        "every_action_reasoned": [sum(bool(f.get("all_actions_reasoned")) for f in ex), len(ex)],
        "parser_counters": dict(tot),
        "missing_close_counters_pct": round(100.0 * (tot["recovered"] + tot["unclosed_no_tool"]) / tot["calls"], 3),
        "missing_close_final_join": [joined, split["calls"], round(100.0 * joined / split["calls"], 3)],
    }
    lines = [f"{k}: {v}" + (f" = {100.0 * v[0] / v[1]:.1f}%" if isinstance(v, list) and len(v) == 2 else "") for k, v in out.items()]
    (HERE / "live_metrics_final.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "live_metrics_final.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
