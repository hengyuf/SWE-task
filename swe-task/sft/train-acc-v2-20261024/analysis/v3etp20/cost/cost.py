#!/usr/bin/env python3
"""train-acc-v2 (2026-10-27): rollout cost per model, and whether reasoning before every action (empty-think-prefix
SFT, etp5 / etp10) makes rollouts slower than the raw v2 SFT of the same epochs. Read-only on the inputs:
  * analysis/think/calls-<model>.jsonl (think_analysis.py; every 200 response of each slot's final attempt): completion
    and prompt tokens, latency_s = response minus request journal timestamp (server time incl. vLLM queueing);
  * analysis/empty-patch/<model>/all.jsonl (DeepSWE classifier): session wall time (wall_s), not finished, codes;
  * TA/results (status, agent_error).

    python3 -I -B cost.py -> cost.{json,md}

Per model: model calls, completion / prompt tokens per call and per rollout, latency per call, model time per rollout
(sum of call latencies), session wall time and non-model time (wall minus model time: tools, harness, OpenCode), the
3 h deadline kills (not finished; see empty-patch/table.py), and the serving speed: OLS of call latency on completion
tokens (slope = s per 1k output tokens, intercept = fixed s per call: queueing + prefill) and the median s per 1k
output tokens over calls with >= 1,000 completion tokens. Paired task-bootstrap 95% CIs (10,000 resamples of the 50
tasks, seed 20261027) of B - A for mean wall time, model calls, completion tokens and model time per rollout.
Serving load per fleet (fleets/<fleet>/logs/vllm-*.log, vLLM's 10 s engine stats; samples with >= 1 running request):
mean running / waiting requests per worker, generation throughput per worker, and per running sequence.
Counterfactual per pair: B's mean model minutes per rollout at A's serving speed (A's OLS slope and intercept applied to
B's mean completion tokens and calls per rollout), to separate more output from slower serving.

2026-11-04: copy in analysis/v3etp20/cost, + v3etp20 and its pairs; earlier models read from analysis/{think,empty-patch}.
"""
import collections
import json
from pathlib import Path
import random
import re
import statistics

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = TA / "analysis/v3etp20/cost"  # 2026-11-04 copy: outputs here (the original analysis/cost is read-only)
OWN = ("v3etp20",)  # models whose calls / classifier rows are in analysis/v3etp20/{think,empty-patch}


def think_dir(model):
    return TA / ("analysis/v3etp20/think" if model in OWN else "analysis/think")


def ep_dir(model):
    return TA / ("analysis/v3etp20/empty-patch" if model in OWN else "analysis/empty-patch")


LABEL = {"sc14": "base", "sftv2": "5 ep", "sftv2e10": "10 ep", "sftv2e20": "20 ep",
         "etp5": "etp 5 ep",
         "etp10": "etp 10 ep",
         "etp20": "etp 20 ep",
         "v3etp5": "v3 etp 5 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 5 ep (iter71)
         "v3etp10": "v3 etp 10 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 10 ep (iter142)
         "v3etp20": "v3 etp 20 ep",  # 2026-11-04: SFT-v3 empty-think-prefix 20 ep (iter284)
         }  # the one model registry (one line per model, column order)
MODELS = tuple(LABEL)
SAME_EPOCH_RAW = {"etp5": "sftv2", "etp10": "sftv2e10", "etp20": "sftv2e20"}  # used only for models in LABEL
ETP = [m for m in MODELS if m in SAME_EPOCH_RAW]
PAIRS = (tuple((SAME_EPOCH_RAW[m], m) for m in ETP) + (("sftv2e20", "etp10"),) + tuple(zip(ETP, ETP[1:]))
         + tuple(("sc14", m) for m in ETP) + (("sc14", "sftv2e10"),))
# 2026-10-31, SFT-v3 empty-think-prefix: the same pairs as analysis/accuracy/paired.py
PAIRS += (("etp5", "v3etp5"), ("etp10", "v3etp5"), ("etp10", "v3etp10"), ("etp20", "v3etp10"), ("v3etp5", "v3etp10"),
          ("sc14", "v3etp5"), ("sc14", "v3etp10"))
# 2026-11-04: v3etp20 pairs (the same as analysis/v3etp20/accuracy/paired.py)
PAIRS += (("v3etp10", "v3etp20"), ("v3etp5", "v3etp20"), ("etp20", "v3etp20"), ("sc14", "v3etp20"), ("etp10", "v3etp20"),
          ("etp5", "v3etp20"), ("sftv2e20", "v3etp20"), ("sftv2e10", "v3etp20"), ("sftv2", "v3etp20"))
DEADLINE_WALL_S = 10500
FLEET = {"sc14": "base"}  # model -> fleet dir when it differs from the model name
STATS = re.compile(rb"Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")


def serving_load(model):
    gen, running, waiting = [], [], []
    for log in (TA / "fleets" / FLEET.get(model, model) / "logs").glob("vllm-*.log"):
        with open(log, "rb") as handle:
            for line in handle:
                if b"Running:" not in line:
                    continue
                match = STATS.search(line)
                if match and int(match.group(2)) > 0:
                    gen.append(float(match.group(1)))
                    running.append(int(match.group(2)))
                    waiting.append(int(match.group(3)))
    return {"vllm_active_samples": len(running), "vllm_running_reqs_mean": round(mean(running), 1),
            "vllm_waiting_reqs_mean": round(mean(waiting), 2),
            "vllm_gen_tok_s_per_worker_mean": round(mean(gen)),
            "vllm_gen_tok_s_per_running_seq": round(sum(gen) / sum(running), 1)}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def mean(xs):
    return statistics.fmean(xs) if xs else None


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def load(model):
    results = {}
    for path in TA.glob(f"results/{model}-group-*.json"):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for r in records:
                results[(data["group"], task_id, int(r["slot"].rsplit("-r", 1)[1]))] = r
    slots = {}
    for line in open(ep_dir(model) / model / "all.jsonl"):
        c = json.loads(line)
        key = (c["ti"] // 100, c["task_id"], c["ri"])
        rec = results[key]
        slots[(c["task_id"], rec["slot"])] = {
            "task_id": c["task_id"], "status": rec["status"], "wall_s": c.get("wall_s"), "code": c["code"],
            "not_finished": c["not_finished"], "empty": c["empty_patch"], "compactions": c["f"].get("n_compactions") or 0,
            "deadline": c["not_finished"] and not c["code"].startswith("N4") and (
                rec.get("agent_error") == "TimeoutError" or (c.get("wall_s") or 0) >= DEADLINE_WALL_S),
            "calls": 0, "completion": 0, "model_s": 0.0, "max_call_s": 0.0, "cap_calls": 0}
    calls = []
    for line in open(think_dir(model) / f"calls-{model}.jsonl"):
        r = json.loads(line)
        s = slots.get((r["task_id"], r["slot"]))
        if s is None:
            continue
        s["calls"] += 1
        s["completion"] += r["completion_tokens"] or 0
        if r.get("latency_s") is not None:
            s["model_s"] += r["latency_s"]
            s["max_call_s"] = max(s["max_call_s"], r["latency_s"])
        s["cap_calls"] += r["finish"] == "length"
        calls.append((r["completion_tokens"] or 0, r.get("prompt_tokens") or 0, r.get("latency_s")))
    return slots, calls


def summarize(slots, calls):
    rows = [s for s in slots.values() if s["wall_s"] is not None]  # the sandbox-lost etp5 slot has no session
    timed = [c for c in calls if c[2] is not None]
    slope, intercept = ols([c[0] for c in timed], [c[2] for c in timed])
    long_calls = [c[2] / c[0] * 1000 for c in timed if c[0] >= 1000]
    walls = [s["wall_s"] / 60 for s in rows]
    resolved = [s["wall_s"] / 60 for s in rows if s["status"] == "pass"]
    killed = [s for s in rows if s["deadline"]]
    out = {
        "rollouts_with_session": len(rows),
        "model_calls_per_rollout_mean": round(mean([s["calls"] for s in rows]), 1),
        "model_calls_per_rollout_median": q([s["calls"] for s in rows], .5),
        "completion_tokens_per_call_mean": round(mean([c[0] for c in calls])),
        "completion_tokens_per_call_median": q([c[0] for c in calls], .5),
        "completion_tokens_per_call_p90": q([c[0] for c in calls], .9),
        "completion_ktok_per_rollout_mean": round(mean([s["completion"] for s in rows]) / 1000, 1),
        "prompt_ktok_per_call_mean": round(mean([c[1] for c in calls]) / 1000, 1),
        "latency_s_per_call_mean": round(mean([c[2] for c in timed]), 2),
        "latency_s_per_call_median": round(q([c[2] for c in timed], .5), 2),
        "latency_s_per_call_p90": round(q([c[2] for c in timed], .9), 2),
        "_ols": [slope, intercept],
        "s_per_1k_output_tokens_ols_slope": round(1000 * slope, 2),
        "s_fixed_per_call_ols_intercept": round(intercept, 2),
        "s_per_1k_output_tokens_median_calls_ge_1k": round(q(long_calls, .5), 2) if long_calls else None,
        "model_min_per_rollout_mean": round(mean([s["model_s"] for s in rows]) / 60, 1),
        "non_model_min_per_rollout_mean": round(mean([s["wall_s"] - s["model_s"] for s in rows]) / 60, 1),
        "pct_wall_in_model_calls": round(100 * sum(s["model_s"] for s in rows) / sum(s["wall_s"] for s in rows), 1),
        "wall_min_mean": round(mean(walls), 1), "wall_min_median": round(q(walls, .5), 1),
        "wall_min_p90": round(q(walls, .9), 1),
        "wall_min_median_resolved": round(q(resolved, .5), 1) if resolved else None,
        "s_per_call_wall_mean": round(sum(s["wall_s"] for s in rows) / sum(s["calls"] for s in rows), 2),
        "deadline_kills_3h": len(killed),
        "deadline_killed_calls_mean": round(mean([s["calls"] for s in killed]), 1) if killed else None,
        "deadline_killed_pct_wall_in_model_calls": round(100 * sum(s["model_s"] for s in killed) /
                                                         sum(s["wall_s"] for s in killed), 1) if killed else None,
        "deadline_killed_with_cap_call": sum(s["cap_calls"] > 0 for s in killed),
        "deadline_killed_compacted": sum(s["compactions"] > 0 for s in killed),
        "deadline_killed_max_call_min_median": round(q([s["max_call_s"] for s in killed], .5) / 60, 1) if killed else None,
        "rollouts_over_2h": sum(s["wall_s"] >= 7200 for s in rows),
        "rollouts_ge_500_calls": sum(s["calls"] >= 500 for s in rows),
        "model_calls_median_resolved": q([s["calls"] for s in rows if s["status"] == "pass"], .5),
        "completion_ktok_median_resolved": round(q([s["completion"] for s in rows if s["status"] == "pass"], .5) / 1000, 1)
        if any(s["status"] == "pass" for s in rows) else None,
    }
    return out


def paired(slots_a, slots_b, seed=20261027, n=10000):
    """Task-bootstrap CI of B - A for per-rollout means (pooled over the task's rollouts)."""
    metrics = {"wall_min": lambda s: s["wall_s"] / 60, "model_calls": lambda s: s["calls"],
               "completion_ktok": lambda s: s["completion"] / 1000, "model_min": lambda s: s["model_s"] / 60}
    per = {}
    for name, (slots) in (("a", slots_a), ("b", slots_b)):
        tasks = collections.defaultdict(list)
        for s in slots.values():
            if s["wall_s"] is not None:
                tasks[s["task_id"]].append(s)
        per[name] = tasks
    tasks = sorted(set(per["a"]) & set(per["b"]))
    out = {}
    for metric, fn in metrics.items():
        sums = {t: (sum(fn(s) for s in per["a"][t]), len(per["a"][t]), sum(fn(s) for s in per["b"][t]), len(per["b"][t]))
                for t in tasks}

        def diff(sample):
            return (sum(sums[t][2] for t in sample) / sum(sums[t][3] for t in sample)
                    - sum(sums[t][0] for t in sample) / sum(sums[t][1] for t in sample))
        rng = random.Random(seed)
        boots = sorted(diff([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(n))
        a_mean = sum(v[0] for v in sums.values()) / sum(v[1] for v in sums.values())
        out[metric] = {"a": round(a_mean, 1), "b": round(a_mean + diff(tasks), 1), "diff": round(diff(tasks), 1),
                       "ci95": [round(boots[int(0.025 * n)], 1), round(boots[int(0.975 * n) - 1], 1)],
                       "ratio": round((a_mean + diff(tasks)) / a_mean, 2)}
    return out


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    summary = {m: {**summarize(*data[m]), **serving_load(m)} for m in MODELS}
    pairs = [{"a": a, "b": b, **paired(data[a][0], data[b][0])} for a, b in PAIRS]
    for p in pairs:
        slope, intercept = summary[p["a"]]["_ols"]
        rows_b = [s for s in data[p["b"]][0].values() if s["wall_s"] is not None]
        p["b_model_min_at_a_serving"] = round(sum(slope * s["completion"] + intercept * s["calls"] for s in rows_b)
                                              / len(rows_b) / 60, 1)
    for m in MODELS:
        summary[m].pop("_ols")
    tmp = HERE / ".cost.json.tmp"
    tmp.write_text(json.dumps({"models": summary, "paired": pairs}, indent=1) + "\n")
    tmp.replace(HERE / "cost.json")
    lines = ["| per rollout / per call | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for key in summary[MODELS[0]]:
        lines.append(f"| {key} | " + " | ".join(str(summary[m][key]) for m in MODELS) + " |")
    lines += ["", "Paired by task (B - A, mean per rollout, 95% task-bootstrap CI; ratio = B / A):", "",
              "| A | B | wall min | model calls | completion ktok | model min | B model min at A's serving speed |",
              "|---|---|---|---|---|---|---|"]
    for p in pairs:
        cells = [f"{p[k]['a']} -> {p[k]['b']} ({p[k]['diff']:+.1f} [{p[k]['ci95'][0]:+.1f}, {p[k]['ci95'][1]:+.1f}], "
                 f"x{p[k]['ratio']})" for k in ("wall_min", "model_calls", "completion_ktok", "model_min")]
        lines.append(f"| {LABEL[p['a']]} | {LABEL[p['b']]} | " + " | ".join(cells) + f" | {p['b_model_min_at_a_serving']} |")
    (HERE / "cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
