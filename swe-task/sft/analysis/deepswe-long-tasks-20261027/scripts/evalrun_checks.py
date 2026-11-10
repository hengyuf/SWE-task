#!/usr/bin/env python3
"""Sanity checks over <OUT>/<run>.jsonl rows (read-only)."""
import json, statistics, sys
from collections import Counter
OUT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027"
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{OUT}/{run}.jsonl")]
    ex = [x for x in rows if x.get("export_ok")]
    print("==", run)
    print(" other-session msgs:", sum(x.get("n_other_session_msgs") or 0 for x in ex),
          "| peak from compaction msg:", sum(bool(x.get("peak_is_compaction")) for x in ex),
          "| peak_ctx != peak_ctx_calls_only:", sum(x.get("peak_ctx") != x.get("peak_ctx_calls_only") for x in ex))
    print(" assistant error msgs: rollouts with any =", sum(1 for x in ex if x.get("n_error_msgs")),
          Counter(e.split(" x")[0][:90] for x in ex for e in (x.get("errors") or [])).most_common(6))
    nf = [x for x in ex if not x.get("opencode_finished")]
    fin = [x for x in ex if x.get("opencode_finished")]
    w = [x["session_wall_s"] for x in nf if x.get("session_wall_s") is not None]
    print(f" not finished (with export): {len(nf)}; wall_s min/median/max = {min(w) if w else None}/{statistics.median(w) if w else None}/{max(w) if w else None}; with patch>0: {sum((x.get('model_patch_bytes') or 0) > 0 for x in nf)}; resolved: {sum(x['resolved'] for x in nf)}")
    wf = [x["session_wall_s"] for x in fin if x.get("session_wall_s") is not None]
    print(f" finished: {len(fin)}; wall_s median/max = {statistics.median(wf)}/{max(wf)}; empty patch: {sum((x.get('model_patch_bytes') or 0) == 0 for x in fin)}")
    print(" last_finish among not finished:", Counter(str(x.get('last_finish')) for x in nf))
    noex = [x for x in rows if not x.get("export_ok")]
    print(" no export:", len(noex), Counter((str(x.get("verifier_error") or "")[:40], x.get("n_output_items")) for x in noex))
    print(" failure_reason non-empty:", sum(1 for x in rows if x.get("failure_reason")), Counter(str(x.get("failure_kind")) for x in rows))
    print(" zero-token assistant msgs per rollout:", Counter(x.get("n_zero_token_assistant_msgs") for x in ex))
    # finish=length detail
    L = [x for x in ex if (x.get("finish_counts") or {}).get("length")]
    print(" rollouts with finish=length:", len(L), "max_call_output_tokens median among them:", statistics.median([x["max_call_output_tokens"] for x in L]) if L else None)
