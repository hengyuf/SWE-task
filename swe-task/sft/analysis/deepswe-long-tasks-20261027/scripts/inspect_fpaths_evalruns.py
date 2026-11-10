#!/usr/bin/env python3
"""Print small identifying fields of every row of one rollouts.jsonl (read-only)."""
import json, sys, os
p = sys.argv[1]
with open(p) as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        fp = r.get("opencode_results_fpath")
        print(i, r.get("task_id"), r.get("_ng_task_index"), r.get("_ng_rollout_index"), r.get("reward"),
              r.get("model_patch_bytes"), repr(r.get("failure_kind")), repr((r.get("failure_reason") or "")[:120]),
              r.get("opencode_finished"), r.get("opencode_export_found"), repr(r.get("log_dir")), fp, os.path.exists(fp) if fp else None)
