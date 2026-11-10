#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30, pilot phase): one summary of the three 1-rollout pilots (kimi / glm / deepseek on
geo-shapeindex-serialization) -> analysis/pilot-verification.json (and printed).

* runs verify_pilot.py on each pilot attempt (probes/<model>/pilot/models/<model>/attempts/0-0/attempt-<last>) and keeps
  its checks, cost and key facts (verify_pilot writes analysis/verify-<model>-pilot-0-0-attempt-<n>.json);
* joint spend check: the pilot legs ran at the same time on NVAPI_KEY_NEW_2 (started 21:34, before the 21:45 switch to
  NEW_3), so each leg's own spend delta also holds the other pilots' calls. Here: NEW_2's spend at the earliest pilot leg
  start (legs.jsonl start budget) vs a fresh key_budget.py NEW_2 check (one 4-token GLM call; --no-budget: the latest
  budget-log.jsonl NEW_2 line instead) against the sum of the three attempts' journal costs (kimi: LiteLLM usage.cost;
  glm / deepseek: prices.json x journal usage). ok when the delta covers the sum (+$0.01); the ratio is reported (NEW_2
  may also carry other users' calls, so a ratio > 1 alone is no error);
* the ATP golden-patch check of the pilot task (golden/<task>-<job>/golden-summary.json, golden_pilot.py) and, as
  context, the 2026-10-14 Gym-stock golden run of the DeepSWE eval (read-only).
Exit 0 iff all three pilots are graded and every check passed.

    /usr/bin/python3 -I -B setup/pilot_summary.py [--no-budget]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

TD = Path(__file__).resolve().parents[1]
MODELS = ("kimi", "glm", "deepseek")
TASK = "geo-shapeindex-serialization"
PILOT_KEY = "NVAPI_KEY_NEW_2"
EVAL_GOLDEN = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs/dev/"
                   "golden-all/golden.jsonl")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def attempt_of(model):
    attempts = sorted((TD / "probes" / model / "pilot" / "models" / model / "attempts" / "0-0").glob("attempt-*"),
                      key=lambda p: int(p.name.split("-")[1]))
    return attempts[-1] if attempts else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-budget", action="store_true")
    args = parser.parse_args()
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "task_id": TASK, "pilots": {}}
    costs = {}
    for model in MODELS:
        attempt = attempt_of(model)
        results = read_json(TD / "results" / f"{model}-pilot.json", {}) or {}
        entry = {"attempt": str(attempt) if attempt else None, "results": bool(results)}
        if attempt and (attempt / "terminal.json").exists():
            run = subprocess.run(["/usr/bin/python3", "-I", "-B", str(TD / "verify_pilot.py"), str(attempt)],
                                 capture_output=True, text=True, timeout=1800)
            report = json.loads(run.stdout) if run.returncode == 0 else {}
            if not report:
                entry["verify_error"] = run.stderr[-1500:]
            record = ((report.get("grading") or {}).get("results_record") or {})
            entry.update({
                "status": record.get("status"), "reward": record.get("reward"), "f2p": record.get("f2p"),
                "p2p": record.get("p2p"), "ended_by_broken_reply": record.get("ended_by_broken_reply"),
                "ended_by_output_cap_runaway": record.get("ended_by_output_cap_runaway"),
                "last_reply_stop_class": record.get("last_reply_stop_class"),
                "all_ok": report.get("all_ok"), "checks": report.get("checks"),
                "failed_checks": sorted(k for k, v in (report.get("checks") or {}).items() if not v),
                "cost_usd": report.get("cost_usd"), "cost_source": report.get("cost_source"),
                "calls": (report.get("replies") or {}).get("calls"),
                "reply_statuses": (report.get("replies") or {}).get("statuses"),
                "window_min": (report.get("replies") or {}).get("window_min"),
                "max_prompt_tokens": (report.get("replies") or {}).get("max_prompt_tokens"),
                "params": report.get("params"), "opencode_limit": report.get("opencode_limit"),
                "image": ((report.get("image_and_grading") or {}).get("image") or {}).get("task_toml"),
                "leg_key": (report.get("keys_and_sandbox") or {}).get("expected_key"),
                "leg_key_spend": (report.get("keys_and_sandbox") or {}).get("leg_key_spend"),
                "verify_report": f"analysis/verify-{model}-pilot-{attempt.parent.name}-{attempt.name}.json"})
            costs[model] = report.get("cost_usd")
        out["pilots"][model] = entry

    starts = [leg for leg in jsonl(TD / "legs.jsonl") if leg.get("event") == "start" and leg.get("campaign") == "pilot"]
    baseline = min(((leg.get("budget") or {}).get("spend") for leg in starts
                    if leg.get("hub_key") == PILOT_KEY and (leg.get("budget") or {}).get("spend") is not None),
                   default=None)
    if args.no_budget:
        lines = [line for line in jsonl(TD / "budget-log.jsonl") if line.get("key") == PILOT_KEY]
        now = lines[-1] if lines else {}
    else:
        run = subprocess.run(["/usr/bin/python3", "-I", "-B", str(TD / "key_budget.py"), "--key-name", PILOT_KEY,
                              "--tag", "pilot summary"], capture_output=True, text=True, timeout=300)
        try:
            now = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            now = {}
    total = round(sum(c for c in costs.values() if isinstance(c, (int, float))), 4) if costs else None
    delta = round(now["spend"] - baseline, 4) if now.get("spend") is not None and baseline is not None else None
    out["joint_spend"] = {
        "key": PILOT_KEY, "spend_at_first_pilot_start": baseline, "spend_now": now.get("spend"),
        "remaining_now": now.get("remaining"), "checked_at": now.get("at"), "delta": delta,
        "sum_of_pilot_journal_costs": total, "per_model": costs,
        "ratio_delta_over_costs": round(delta / total, 3) if delta is not None and total else None,
        "note": "kimi = LiteLLM usage.cost from the replies; glm / deepseek = prices.json x journal usage; the delta "
                "also holds key_budget.py's 4-token checks (guards every 300 s) and any other use of NEW_2"}
    out["joint_spend"]["ok"] = delta is not None and total is not None and len(costs) == 3 and delta + 0.01 >= total

    golden = sorted(TD.glob(f"golden/{TASK}-*/golden-summary.json"))
    summary = read_json(golden[-1], {}) if golden else {}
    out["golden_atp"] = {"summary": str(golden[-1].relative_to(TD)) if golden else None, "ok": summary.get("ok"),
                         "verify": summary.get("verify"), "check_result": summary.get("check_result"),
                         "task_identity_ok": summary.get("task_identity_ok"),
                         "atp_scripts_ok": summary.get("atp_scripts_ok"),
                         "hub_key_in_process": summary.get("hub_key_in_process"), "job": summary.get("job")}
    eval_row = {}
    if EVAL_GOLDEN.exists():
        with open(EVAL_GOLDEN) as stream:
            for line in stream:
                if f'"{TASK}"' in line:
                    row = json.loads(line)
                    if row.get("task_id") == TASK:
                        eval_row = {k: row.get(k) for k in ("reward", "f2p_passed", "f2p_total", "p2p_passed",
                                                            "p2p_total", "evaluation_completed", "image")}
                        break
    out["golden_eval_gym_stock_20261014"] = {"file": str(EVAL_GOLDEN), "row": eval_row}
    graded = all((out["pilots"][m].get("status") in ("pass", "fail")) for m in MODELS)
    out["all_graded"] = graded
    out["all_checks_ok"] = all(out["pilots"][m].get("all_ok") is True for m in MODELS)
    out["ok"] = graded and out["all_checks_ok"] and out["joint_spend"]["ok"] and summary.get("ok") is True
    path = TD / "analysis" / "pilot-verification.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
