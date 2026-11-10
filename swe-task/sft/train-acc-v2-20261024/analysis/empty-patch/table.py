#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): empty-patch table for all models from the classifier's all.jsonl
(analysis/empty-patch/<model>/all.jsonl). Definitions (each reproduces the REPORT.md 01:30 numbers for sftv2 / sc14
except where noted):
  resolved = reward 1; empty patch = model_patch_bytes 0; committed = n_commits > 0 (export: a successful commit);
  1-token stop = the main session's last assistant turn has 1 output token and no text (EOS right after the <think>
    generation prefix) [REPORT said 26 for sftv2; this definition gives 24];
  stopped after the compaction "continue" = empty patch, >= 1 compaction, session finished (the stop came in the
    segment after the injected "Continue if you have next steps..." turn) [79 / 10 as in REPORT];
  causes = the classifier's code/subcode: E1 edited but uncommitted, E3b tool call in think, E3c reasoning-only, ...

2026-10-27: + etp5 / etp10 (empty-think-prefix SFT 5 / 10 ep) and rows that split the not-finished and cap-cut codes:
  deadline_kill_3h = not finished, not N4, and agent_error TimeoutError in TA/results or session wall >= 10,500 s
    (the attempt's 3 h deadline; 2 such slots in base / raw 5 ep have agent_error None but end at 10,751 / 10,764 s;
    the classifier files them all under N9_OTHER_NF because this harness leaves stderr empty); sandbox_lost = N4_*;
  E3a_context_overflow_400 = E3a_cap_cut whose last turn has 0 output tokens: vLLM rejected the call with HTTP 400
    (prompt + max_tokens > 262144) and the session ended on an empty "length" turn (ATP's generation-validity marks
    such a slot "fatal" = infra; one in raw 5 ep, one in etp 10 ep); E3a_runaway_cap = the other E3a (a call
    generated the full 131,072-token cap).

    python3 -I -B table.py [MODEL ...]   (default: every model in LABEL) -> table.md, table.json
"""
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
LABEL = {"sc14": "base (sc14)", "sftv2": "5 ep (iter38)", "sftv2e10": "10 ep (iter76)", "sftv2e20": "20 ep (iter152)",
         "etp5": "etp 5 ep (iter38)",
         "etp10": "etp 10 ep (iter76)",
         "etp20": "etp 20 ep (iter152)",
         "v3etp5": "v3 etp 5 ep (iter71)",  # 2026-10-31: SFT-v3 empty-think-prefix
         "v3etp10": "v3 etp 10 ep (iter142)",  # 2026-10-31: SFT-v3 empty-think-prefix
         }  # the one model registry (default model list and column order)


def agent_errors(model):
    """{(group, task_id, repeat): agent_error} from TA/results (2026-10-27)."""
    out = {}
    for path in TA.glob(f"results/{model}-group-*.json"):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for record in records:
                out[(data["group"], task_id, int(record["slot"].rsplit("-r", 1)[1]))] = record.get("agent_error")
    return out


def stats(model):
    rs = [json.loads(line) for line in (HERE / model / "all.jsonl").read_text().splitlines() if line.strip()]
    f = lambda r: r["f"]  # noqa: E731
    n = len(rs)
    empty = [r for r in rs if r["empty_patch"]]
    patched = [r for r in rs if not r["empty_patch"]]
    compacted = [r for r in rs if f(r).get("n_compactions")]
    plain = [r for r in rs if not f(r).get("n_compactions")]
    codes = collections.Counter((r["code"], r["subcode"] or "") for r in empty if True)
    errors = agent_errors(model)
    error_of = lambda r: errors.get((r["ti"] // 100, r["task_id"], r["ri"]))  # noqa: E731
    deadline = lambda r: r["not_finished"] and not r["code"].startswith("N4") and (  # noqa: E731
        error_of(r) == "TimeoutError" or (r.get("wall_s") or 0) >= 10500)
    return {
        "rollouts": n,
        "resolved": sum(r.get("reward") == 1 for r in rs),
        "empty_patch": len(empty), "empty_patch_pct": round(100 * len(empty) / n, 1) if n else None,
        "resolved_with_patch": sum(r.get("reward") == 1 for r in patched), "with_patch": len(patched),
        "committed": sum((f(r).get("n_commits") or 0) > 0 for r in rs),
        "one_token_stops": sum(f(r).get("last_out_tokens") == 1 and f(r).get("text_len") == 0 for r in rs),
        "stopped_after_compaction_continue": sum(1 for r in empty if f(r).get("n_compactions") and not r["not_finished"]),
        "compacted": len(compacted),
        "empty_no_compaction": f"{sum(r['empty_patch'] for r in plain)}/{len(plain)}",
        "empty_compacted": f"{sum(r['empty_patch'] for r in compacted)}/{len(compacted)}",
        "not_finished": sum(r["not_finished"] for r in rs),
        "E1_edited_uncommitted": sum(1 for r in empty if r["code"] == "E1_UNCOMMITTED"),
        "E1a_defers_commit": codes[("E1_UNCOMMITTED", "E1a_defers_commit")],
        "E1b_midwork_status": codes[("E1_UNCOMMITTED", "E1b_midwork_status_summary")],
        "E1b_other": codes[("E1_UNCOMMITTED", "E1b_other")],
        "E3_premature_end": sum(1 for r in empty if r["code"] == "E3_PREMATURE_END"),
        "E3b_toolcall_in_think": codes[("E3_PREMATURE_END", "E3b_toolcall_in_think")],
        "E3c_reasoning_only": codes[("E3_PREMATURE_END", "E3c_reasoning_only")],
        "E3d_malformed_toolcall": codes[("E3_PREMATURE_END", "E3d_malformed_toolcall")],
        "E3a_cap_cut": codes[("E3_PREMATURE_END", "E3a_cap_cut")],
        "E2_no_edit": sum(1 for r in empty if r["code"] == "E2_NO_EDIT"),
        "not_finished_empty": sum(1 for r in empty if r["code"].startswith("N")),
        "deadline_kill_3h": sum(1 for r in rs if deadline(r)),
        "deadline_kill_3h_empty": sum(1 for r in empty if deadline(r)),
        "sandbox_lost": sum(1 for r in rs if r["code"].startswith("N4")),
        "not_finished_other": sum(1 for r in rs if r["not_finished"] and not deadline(r) and not r["code"].startswith("N4")),
        "E3a_context_overflow_400": sum(1 for r in empty if r["subcode"] == "E3a_cap_cut"
                                        and not f(r).get("last_out_tokens")),
        "E3a_runaway_cap": sum(1 for r in empty if r["subcode"] == "E3a_cap_cut" and f(r).get("last_out_tokens")),
        "all_codes_empty": {f"{k[0]}/{k[1]}": v for k, v in sorted(codes.items())},
    }


def main():
    models = sys.argv[1:] or list(LABEL)
    out = {m: stats(m) for m in models if (HERE / m / "all.jsonl").exists()}
    (HERE / "table.json").write_text(json.dumps(out, indent=1) + "\n")
    rows = [k for k in next(iter(out.values())) if k != "all_codes_empty"]
    lines = ["| | " + " | ".join(LABEL.get(m, m) for m in out) + " |", "|---|" + "---|" * len(out)]
    for k in rows:
        lines.append(f"| {k} | " + " | ".join(str(out[m][k]) for m in out) + " |")
    (HERE / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
