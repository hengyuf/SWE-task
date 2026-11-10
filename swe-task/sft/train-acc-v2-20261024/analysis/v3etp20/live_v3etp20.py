#!/usr/bin/env python3
"""train-acc-v2 live report for the empty-think-prefix SFT checkpoints (2026-10-27; user: "在rollout过程中实时汇报
accuracy, 空patch比例和缺失think token的比例, 与之前的run对比"). Read-only on the campaigns; caches per-slot results.

    /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -B \
        analysis/watch/live_etp.py [--models etp5 etp10] [--json OUT.json]

Per model, over the slots ATP has finished in probes/<model>/group-*/ (state "completed" = graded; fatal / exhausted /
needs_review = infra, excluded as in pass_stats):
- pass@1 = passes / graded slots; pass@5 over the tasks whose 5 slots are all graded (>= 1 pass);
- empty patch = graded slots whose final attempt's model.patch is empty (as patch_chars / model_patch_bytes 0 before);
- thinking, per finished slot with the teacher-rollout extractor (TR/features.py, the source of the earlier
  "% action turns with reasoning" rows): action turns without reasoning (pooled), bare action turns (no reasoning and
  no visible text), rollouts with every action reasoned;
- missing </think>, per model call so far, from the counting parser's vLLM counters (fleets/<model>/logs/vllm-*.log,
  last TA_THINK_COUNTS line per server process; logged every 200 calls): recovered (tool call inside an unclosed think,
  rescued by the lenient parser) + unclosed without a tool call; empty outputs (EOS right after <think>).
- timeouts (2026-10-30): graded slots whose agent was killed at the 3 h deadline (terminal.json opencode_run_error_type
  "TimeoutError", what TR/features.py reports as agent_error); read when a slot is first cached, so slots cached before
  2026-10-30 have no value and the row shows "-" for them.
Reference rows: the finished runs of the same 50 tasks x 5 (REPORT.md "SFT v2 at 10 and 20 epochs",
analysis/features/comparison.md, analysis/think/split_table.json).
2026-10-30 (user): v3etp5 / v3etp10 = the SFT-v3 corpus with the empty think prefix, 5 ep (iter71) and 10 ep (iter142);
same-stage references: raw v2 of the same epoch count (as etp5 / etp10) and the v2 empty-think-prefix run of the same
epoch count (etp5 / etp10).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
SYNTH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
CACHE = TA / "analysis/watch/live_etp_cache.json"
ACK = TA / "analysis/watch/live_etp_ack.json"  # handled failures: a JSON list of job ids (and driver log names)
LABEL = {"etp5": "空think前缀 5ep", "etp10": "空think前缀 10ep", "etp20": "空think前缀 20ep",
         "v3etp5": "v3 空think前缀 5ep", "v3etp10": "v3 空think前缀 10ep", "v3etp20": "v3 空think前缀 20ep"}
GROUPS, REPEATS, TASKS = 8, 5, 50
COUNTS = re.compile(rb"TA_THINK_COUNTS pid=(\d+) (.*)")
SHORT_CHARS = 40  # an action call whose (stripped) reasoning is shorter than this counts as "token reasoning"

# Finished runs, same 50 tasks x 5 and harness (pct). missing_close = recovered + unclosed-no-tool per call; for base and
# raw 5 ep (plain lenient parser, nothing logged) the recovered part is the bare-rescue lower bound, hence ">=".
REF = {
    "base (sc14)": {"pass1": 17.2, "pass5": 46.0, "empty": 6.4, "noreason": 0.0, "bare": 0.0, "noreason_live": 0.0, "bare_live": 0.0, "short_live": 2.4, "all_reasoned": 90.8,
                    "missing_close": ">=0.020", "empty_out": 0.0, "timeout": 1.6},
    "raw v2 5ep": {"pass1": 18.1, "pass5": 52.0, "empty": 51.2, "noreason": 35.3, "bare": 35.0, "noreason_live": 35.2, "bare_live": 34.9, "short_live": 0.5, "all_reasoned": 0.4,
                   "missing_close": ">=0.342", "empty_out": 0.034, "timeout": 3.2},
    "raw v2 10ep": {"pass1": 36.0, "pass5": 74.0, "empty": 38.0, "noreason": 40.7, "bare": 40.3, "noreason_live": 40.6, "bare_live": 40.2, "short_live": 0.1, "all_reasoned": 0.4,
                    "missing_close": 0.107, "empty_out": 0.006, "timeout": 2.8},
    "raw v2 20ep": {"pass1": 44.0, "pass5": 74.0, "empty": 26.4, "noreason": 36.9, "bare": 36.7, "noreason_live": 36.8, "bare_live": 36.6, "short_live": 0.1, "all_reasoned": 0.0,
                    "missing_close": 0.100, "empty_out": 0.0, "timeout": 1.6},
    # final runs of 2026-10-27 (aggregate.py pass@1/pass@5; live_etp_final_20261027.txt for the rest)
    "空think前缀 5ep (终)": {"pass1": 26.9, "pass5": 68.0, "empty": 24.1, "noreason": 0.0, "bare": 0.0,
                             "noreason_live": 0.0, "bare_live": 0.0, "short_live": 0.7, "all_reasoned": 89.2,
                             "missing_close": 0.055, "empty_out": 0.0, "timeout": 6.4},
    "空think前缀 10ep (终)": {"pass1": 39.4, "pass5": 80.0, "empty": 17.7, "noreason": 0.0, "bare": 0.0,
                              "noreason_live": 0.0, "bare_live": 0.0, "short_live": 0.7, "all_reasoned": 99.2,
                              "missing_close": 0.036, "empty_out": 0.0, "timeout": 4.8},
    # final run of 2026-10-28 (live_etp20_final_20261028.txt)
    "空think前缀 20ep (终)": {"pass1": 52.8, "pass5": 84.0, "empty": 8.4, "noreason": 0.0, "bare": 0.0,
                              "noreason_live": 0.0, "bare_live": 0.0, "short_live": 0.5, "all_reasoned": 99.2,
                              "missing_close": 0.058, "empty_out": 0.0, "timeout": 1.2},
}
# "timeout" (added 2026-10-30) = % of graded slots with agent_error "TimeoutError" in the finished runs' feature rows
# (TR/features/{sc14,sftv2}.jsonl, analysis/features/{sftv2e10,sftv2e20,etp5,etp10,etp20}.jsonl):
# 4/250, 8/249, 7/250, 4/250, 16/249, 12/249, 3/250.


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def parser_counts(model):
    """Summed last TA_THINK_COUNTS per (log, pid) of the fleet's vLLM logs."""
    total = collections.Counter()
    for log in sorted((TA / "fleets" / model / "logs").glob("vllm-*.log")):
        last = {}
        try:
            with open(log, "rb") as handle:
                for line in handle:
                    if b"TA_THINK_COUNTS" in line:
                        match = COUNTS.search(line)
                        if match:
                            last[match.group(1)] = match.group(2).decode(errors="replace")
        except OSError:
            continue
        for text in last.values():
            for key, value in re.findall(r"(\w+)=(\d+)", text):
                total[key] += int(value)
    return dict(total)


def registry(model):
    now, ready, starting = time.time(), 0, 0
    for path in (TA / "fleets" / model / "registry").glob("*.json"):
        entry = read_json(path, {}) or {}
        fresh = now - (entry.get("heartbeat_at") or 0) <= 180 and (entry.get("end_epoch") or 0) > now
        if fresh and entry.get("status") in ("ready", "draining"):
            ready += 1
        elif fresh and entry.get("status") == "starting":
            starting += 1
    return ready, starting


def health(model):
    """ATTENTION lines: failed / unhealthy workers (registry), and driver logs that ended badly."""
    out = []
    acked = set(map(str, read_json(ACK, []) or []))  # job ids main already handled (node excluded, slot replaced)
    for path in sorted((TA / "fleets" / model / "registry").glob("*.json")):
        entry = read_json(path, {}) or {}
        if entry.get("status") in ("failed", "unhealthy") and str(entry.get("job_id")) not in acked:
            out.append(f"{model} worker {entry.get('job_id')} {entry.get('status')} on {entry.get('node')} "
                       f"(exclude_node {entry.get('exclude_node')}): {str(entry.get('reason'))[:160]}")
    winners = set()  # only a pair's winner counts: the helper cancels the losing candidate (its log says CANCELLED)
    for path in (TA / "babysit").glob(f"drv-{model}-g*/babysit/legs/*/winner"):
        try:
            winners.add(path.read_text().strip())
        except OSError:
            pass
    for log in sorted((TA / "logs").glob(f"ta-drv-{model}-g*-*.out")):
        if log.stem.rsplit("-", 1)[-1] not in winners or log.name in acked:
            continue
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        if "sibling candidate won" in text:
            continue
        bad = [m for m in re.findall(r"driver exit (\d+)", text) if m != "0"]
        if bad or "Traceback" in text or "CANCELLED" in text or "babysit_enter failed" in text:
            tail = [line for line in text.splitlines() if line.strip()][-1:] or [""]
            out.append(f"{model} driver log {log.name}: exit {bad or '-'}, traceback {'Traceback' in text}, "
                       f"cancelled {'CANCELLED' in text}: {tail[0][:200]}")
    return out


def slot_info(attempt, cache):
    """Cached per finished attempt: patch size and the extractor's thinking counts."""
    key = str(attempt)
    try:
        stamp = max((attempt / name).stat().st_mtime for name in ("terminal.json", "export.json")
                    if (attempt / name).exists())
    except ValueError:
        stamp = 0
    hit = cache.get(key)
    if hit and hit.get("stamp") == stamp:
        if "outcome" not in hit:  # entries cached before 2026-10-27 19:30 lack it: read terminal.json once
            grade = read_json(attempt / "terminal.json", {}) or {}
            hit["outcome"] = "pass" if grade.get("reward") == 1 else "fail"
        return hit
    import features as tr_features  # TR/features.py (needs tokenizers: run with the vLLM env python)
    row = tr_features.features({"source": "live", "attempt_dir": key, "origin": "live"})
    grade = read_json(attempt / "terminal.json", {}) or {}
    info = {"stamp": stamp, "outcome": "pass" if grade.get("reward") == 1 else "fail",
            "timeout": (grade.get("opencode_run_error_type") or grade.get("failure_reason")) == "TimeoutError",
            "patch_chars": row.get("patch_chars", 0), "export": row.get("export", False),
            "action_turns": row.get("action_turns", 0), "action_with_reasoning": row.get("action_with_reasoning", 0),
            "action_bare": row.get("action_bare", 0), "all_actions_reasoned": row.get("all_actions_reasoned", False),
            "compactions": row.get("compactions", 0)}
    cache[key] = info
    return info


def live_calls(model, cache):
    """Every model call so far (running and finished attempts) from the attempts' http.jsonl, read incrementally (byte
    offsets cached; only complete lines): 200 responses with >= 1 tool call = action calls; without reasoning
    (reasoning / reasoning_content empty); bare (no reasoning, no content). The same per-call definition as
    analysis/think/calls-*.jsonl, which gives 0.0 / 35.2 / 40.6 / 36.8 % no-reasoning action calls for base / raw
    5 / 10 / 20 ep (the extractor's per-turn numbers: 0.0 / 35.3 / 40.7 / 36.9)."""
    store = cache.setdefault("_http", {})
    total = collections.Counter()
    for path in (TA / "probes" / model).glob(f"group-*/models/{model}/attempts/*/attempt-*/http.jsonl"):
        key = str(path)
        entry = store.get(key) or {"offset": 0}
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < entry["offset"]:
            entry = {"offset": 0}
        if size > entry["offset"]:
            with open(path, "rb") as handle:
                handle.seek(entry["offset"])
                data = handle.read(size - entry["offset"])
            end = data.rfind(b"\n") + 1
            for raw in data[:end].split(b"\n"):
                head = raw[:2000]
                if b'"event":"response"' not in head and b'"event": "response"' not in head:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                body = event.get("body")
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except ValueError:
                        body = None
                if event.get("event") != "response" or event.get("status") != 200 or not isinstance(body, dict) \
                        or not body.get("choices"):
                    continue
                message = body["choices"][0].get("message") or {}
                reasoning = message.get("reasoning")
                if reasoning is None:
                    reasoning = message.get("reasoning_content")
                entry["calls"] = entry.get("calls", 0) + 1
                if message.get("tool_calls"):
                    entry["action"] = entry.get("action", 0) + 1
                    text = reasoning.strip() if isinstance(reasoning, str) else ""
                    if not text:  # whitespace-only reasoning counts as none (the extractor strips too)
                        entry["noreason"] = entry.get("noreason", 0) + 1
                        content = message.get("content")
                        if not (content.strip() if isinstance(content, str) else content):
                            entry["bare"] = entry.get("bare", 0) + 1
                    elif len(text) < SHORT_CHARS:
                        entry["short"] = entry.get("short", 0) + 1
            entry["offset"] += end
        store[key] = entry
        for name in ("calls", "action", "noreason", "bare", "short"):
            total[name] += entry.get(name, 0)
    return dict(total)


SAME_STAGE = {"etp5": ("sftv2", TR / "features/sftv2.jsonl"), "etp10": ("sftv2e10", TA / "analysis/features/sftv2e10.jsonl"),
              "etp20": ("sftv2e20", TA / "analysis/features/sftv2e20.jsonl"),
              "v3etp5": ("sftv2", TR / "features/sftv2.jsonl"), "v3etp10": ("sftv2e10", TA / "analysis/features/sftv2e10.jsonl"),
              "v3etp20": ("sftv2e20", TA / "analysis/features/sftv2e20.jsonl")}
# 2026-10-30: the v2 empty-think-prefix run of the same epoch count, for the SFT-v3 checkpoints
SAME_STAGE_ETP = {"v3etp5": ("etp5", TA / "analysis/features/etp5.jsonl"),
                  "v3etp10": ("etp10", TA / "analysis/features/etp10.jsonl"),
                  "v3etp20": ("etp20", TA / "analysis/features/etp20.jsonl")}


def same_stage(model, n, cache, table=None, key=None):
    """The raw v2 run of the same epoch count at the same stage: its first n graded slots to finish (terminal.json time
    after the group's first start), as (pass %, empty patch %). Early finishers are biased (short rollouts, early
    stops), so live numbers are compared with these as well as with the final ones. Cached (the old runs are final)."""
    table = SAME_STAGE if table is None else table
    key = key or model  # cache key: the model for SAME_STAGE (as before 2026-10-30), "<model>|etp" for SAME_STAGE_ETP
    if model not in table or not n:
        return None, None
    store = cache.setdefault("_stage", {})
    if key not in store:
        raw, path = table[model]
        starts, out = {}, []
        for line in open(path):
            row = json.loads(line)
            attempt = Path(row["attempt_dir"]) if row.get("attempt_dir") else None
            if not attempt or row.get("status") not in ("pass", "fail") or not (attempt / "terminal.json").exists():
                continue
            group = next((part for part in attempt.parts if part.startswith("group-")), None)
            if group not in starts:
                starts[group] = min(os.stat(x).st_mtime for x in
                                    (TA / "probes" / raw / group).glob(f"models/{raw}/attempts/*/attempt-0/started.json"))
            out.append([os.stat(attempt / "terminal.json").st_mtime - starts[group], row["status"] == "pass",
                        (row.get("patch_chars") or 0) == 0])
        out.sort()
        store[key] = [[bool(a), bool(b)] for _, a, b in out]
    sub = store[key][:n]
    return round(100.0 * sum(a for a, _ in sub) / len(sub), 1), round(100.0 * sum(b for _, b in sub) / len(sub), 1)


def model_report(model, cache, history):
    import rollouts  # synth/rollouts.py: ATP's slot states, as the probe harness reads them
    per_task = collections.defaultdict(list)
    states = collections.Counter()
    groups_started = 0
    infos = []
    for group in range(GROUPS):
        probe = TA / "probes" / model / f"group-{group}"
        if not (probe / "manifest.json").exists():
            continue
        groups_started += 1
        try:
            slots = rollouts.probe_states(probe)
        except Exception as error:  # noqa: BLE001 - a campaign being created: skip it this time
            states[f"unreadable:{type(error).__name__}"] += 1
            continue
        for task, _model, _repeat, state in slots:
            status = state["status"]
            states[status] += 1
            if status not in rollouts.TERMINAL_STATES:
                continue
            if status != "completed" or not state.get("attempt_dir"):
                per_task[task["task_id"]].append("infra")
                continue
            attempt = Path(state["attempt_dir"])
            info = slot_info(attempt, cache)  # cached with the outcome: terminal.json is read once per attempt
            outcome = info["outcome"]
            per_task[task["task_id"]].append(outcome)
            infos.append(info)
    graded = [i for i in infos]
    passes = sum(i["outcome"] == "pass" for i in graded)
    finished = sum(len(v) for v in per_task.values())
    full = [v for v in per_task.values() if len(v) == REPEATS and "infra" not in v]
    exported = [i for i in graded if i["export"]]
    actions = sum(i["action_turns"] for i in exported)
    counts = parser_counts(model)
    calls = counts.get("calls", 0)
    live = live_calls(model, cache)
    ready, starting = registry(model)
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None
    now = time.time()
    samples = history.setdefault(model, [])
    if not samples or samples[-1][1] != finished or now - samples[-1][0] > 1800:
        samples.append([now, finished])
    del samples[:-30]
    recent = [s for s in samples if now - s[0] <= 3600]
    rate = (recent[-1][1] - recent[0][1]) / ((recent[-1][0] - recent[0][0]) / 3600) if len(recent) >= 2 and \
        recent[-1][0] > recent[0][0] else None
    remaining = TASKS * REPEATS - finished
    return {
        "model": model, "label": LABEL.get(model, model), "groups_started": groups_started,
        "workers_ready": ready, "workers_starting": starting,
        "states": dict(states), "finished": finished, "graded": len(graded),
        "infra": sum(v.count("infra") for v in per_task.values()),
        "pass1": pct(passes, len(graded)), "passes": passes,
        "pass5": pct(sum("pass" in v for v in full), len(full)), "pass5_tasks": len(full),
        "empty": pct(sum(i["patch_chars"] == 0 for i in graded), len(graded)),
        "empty_n": sum(i["patch_chars"] == 0 for i in graded),
        "noreason": pct(actions - sum(i["action_with_reasoning"] for i in exported), actions),
        "bare": pct(sum(i["action_bare"] for i in exported), actions),
        "noreason_live": pct(live.get("noreason", 0), live.get("action", 0)),
        "bare_live": pct(live.get("bare", 0), live.get("action", 0)),
        "short_live": pct(live.get("short", 0), live.get("action", 0)),
        "live_action_calls": live.get("action", 0),
        "stage_pass1": same_stage(model, len(graded), cache)[0], "stage_empty": same_stage(model, len(graded), cache)[1],
        "stage_etp_pass1": same_stage(model, len(graded), cache, SAME_STAGE_ETP, model + "|etp")[0],
        "stage_etp_empty": same_stage(model, len(graded), cache, SAME_STAGE_ETP, model + "|etp")[1],
        "timeout": pct(sum(i.get("timeout") is True for i in graded), sum("timeout" in i for i in graded)),
        "timeout_n": sum(i.get("timeout") is True for i in graded),
        "all_reasoned": pct(sum(bool(i["all_actions_reasoned"]) for i in exported), len(exported)),
        "missing_close": round(100.0 * (counts.get("recovered", 0) + counts.get("unclosed_no_tool", 0)) / calls, 3)
        if calls else None,
        "recovered": counts.get("recovered", 0), "unclosed_no_tool": counts.get("unclosed_no_tool", 0),
        "empty_out": round(100.0 * counts.get("empty", 0) / calls, 3) if calls else None, "calls": calls,
        "rate_per_h": round(rate, 1) if rate and finished >= 10 else None,  # too noisy before ~10 finished slots
        "eta_h": round(remaining / rate, 1) if rate and finished >= 10 else None,
    }


def render(reports):
    fmt = lambda v: "-" if v is None else (v if isinstance(v, str) else f"{v}")
    cols = [(r["label"] + f" ({r['finished']}/250)", r) for r in reports] + [(name, ref) for name, ref in REF.items()]
    rows = [("pass@1 %", "pass1"), ("  同阶段 raw v2 (同 epoch, 最早完成的同样条数) pass@1 %", "stage_pass1"),
            ("  同阶段 v2 空think前缀 (同 epoch) pass@1 %", "stage_etp_pass1"),
            ("pass@5 % (5/5 已评分的 task)", "pass5"), ("空 patch %", "empty"),
            ("  同阶段 raw v2 空 patch %", "stage_empty"), ("  同阶段 v2 空think前缀 空 patch %", "stage_etp_empty"),
            ("超时 % (3 h deadline 被杀, 已评分)", "timeout"),
            ("无推理的 action 调用 % (实时, 含进行中)", "noreason_live"), ("bare action 调用 % (实时)", "bare_live"),
            ("推理 <40 字符的 action 调用 % (实时)", "short_live"),
            ("无推理的 action 轮 % (已完成 rollout)", "noreason"),
            ("每个 action 都有推理的 rollout %", "all_reasoned"),
            ("缺失 </think> % (每次调用)", "missing_close"), ("空输出 % (<think> 后直接 EOS)", "empty_out")]
    lines = ["| 指标 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for title, key in rows:
        lines.append(f"| {title} | " + " | ".join(fmt(src.get(key)) for _, src in cols) + " |")
    for r in reports:
        lines.append(f"- {r['label']}: 完成 {r['finished']}/250 (graded {r['graded']}, infra {r['infra']}), "
                     f"pass {r['passes']}, 空 patch {r['empty_n']}, 超时 {r['timeout_n']}, 组 {r['groups_started']}/8, "
                     f"worker 就绪 {r['workers_ready']} / 启动中 {r['workers_starting']}, 调用 {r['calls']} "
                     f"(recovered {r['recovered']}, unclosed {r['unclosed_no_tool']}), "
                     f"速度 {fmt(r['rate_per_h'])}/h, ETA {fmt(r['eta_h'])} h, 状态 {r['states']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["etp5", "etp10"])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(SYNTH))
    import common  # noqa: F401  (puts ATP's scripts on sys.path for rollouts)
    sys.path.insert(0, str(TR))
    cache = read_json(CACHE, {}) or {}
    history = cache.pop("_history", {})
    reports = [model_report(model, cache, history) for model in args.models]
    cache["_history"] = history
    tmp = CACHE.with_name("." + CACHE.name + ".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, CACHE)
    text = render(reports)
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(text)
    for model in args.models:
        for line in health(model):
            print("ATTENTION: " + line)
    if args.json:
        args.json.write_text(json.dumps({"at": time.time(), "reports": reports, "reference": REF}, indent=1) + "\n")


if __name__ == "__main__":
    main()
