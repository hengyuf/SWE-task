#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): degenerate Kimi-K3 hub replies, new arm (sentence + 262k window) vs control ctrl-old
(previous settings), read-only; writes analysis/degenerate-<out>.json and prints a summary.

    /usr/bin/python3 -I -B degenerate.py [--out compare]

Every status-200 "response" event of every attempt (all attempts, including ones later retried or killed: each reply is
one sample of the hub's behaviour) is classified:
  pad      "[PAD]" in the reply's content / reasoning / tool-call arguments
  special  a chat-template special token (<|close|>, <|im_end|>, <|tool_call...|> ...) in it, and no [PAD]
  empty    no tool call, no content, no reasoning (e.g. a 5-token empty stop)
"degenerate" = any of the three. A degenerate reply without a tool call ends the OpenCode session ("ended_by").
Arms: new = probes/kimi/sel50-g*/ (recovery campaigns included); control = probes/kimi/ctrl-old/. Comparisons:
  same_tasks   new-arm replies on the control's 4 tasks (all hours) vs control
  concurrent   new-arm replies on ANY task between the control's first and last reply vs control
  overall      every new-arm reply vs control
Statistics: per-reply rate with a two-sided Fisher exact test on (degenerate, other) x (control, new), and a 95%
bootstrap CI of the rate difference resampling ROLLOUTS within each arm (replies of one rollout are not independent);
per rollout: rollouts with >= 1 degenerate reply and sessions ended by one (final attempts of graded slots), Fisher.
Historical references (same classification): previous Kimi on the 50 tasks, Oct 17-21: 0 in 27,163 replies
(analysis/stops-previous-kimi.json); default-setting Kimi probes of the official2 builds, Oct 24 07:04-13:07: 11 [PAD]
replies in 10,134 (analysis/degenerate-replies-evidence.json).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402

HERE = Path(__file__).resolve().parent
SPECIAL = re.compile(r"<\|[a-z_]+\|>")
CTRL = "ctrl-old"


def classify(body):
    choice = (body.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    text = content + reasoning + "".join(str((c.get("function") or {}).get("arguments") or "") for c in calls)
    if "[PAD]" in text:
        kind = "pad"
    elif SPECIAL.search(text):
        kind = "special"
    elif not calls and not content.strip() and not reasoning.strip():
        kind = "empty"
    else:
        kind = None
    return kind, bool(calls)


def scan(attempt):
    """[(timestamp, kind or None, has_tool_call)] for every 200 reply of one attempt."""
    out = []
    path = Path(attempt) / "http.jsonl"
    if not path.exists():
        return attempt, out
    with open(path, "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                event = json.loads(raw)
                if event.get("status") != 200:
                    continue
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            kind, tool = classify(body)
            out.append((event.get("timestamp") or 0.0, kind, tool))
    return attempt, out


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables at most as likely as the observed one)."""
    row1, col1, n = a + b, a + c, a + b + c + d

    def logp(x):
        return (math.lgamma(col1 + 1) - math.lgamma(x + 1) - math.lgamma(col1 - x + 1)
                + math.lgamma(n - col1 + 1) - math.lgamma(row1 - x + 1) - math.lgamma(n - col1 - row1 + x + 1)
                - math.lgamma(n + 1) + math.lgamma(row1 + 1) + math.lgamma(n - row1 + 1))
    low, high = max(0, row1 + col1 - n), min(row1, col1)
    observed = logp(a)
    return min(1.0, sum(math.exp(logp(x)) for x in range(low, high + 1) if logp(x) <= observed + 1e-9))


def binom_tail_ge(x, n, p):
    """P(X >= x) for X ~ Binomial(n, p), exactly (log-space terms)."""
    if x <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + k * math.log(p) + (n - k) * math.log(1 - p)) for k in range(x, n + 1))


def clopper_pearson(x, n):
    """Exact 95% CI of a binomial proportion by bisection on the exact tails."""
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if x == 0 else solve(lambda q: binom_tail_ge(x, n, q) < 0.025)
    upper = 1.0 if x == n else solve(lambda q: 1 - binom_tail_ge(x + 1, n, q) > 0.025)
    return lower, upper


def ratio_ci(events_ctrl, replies_ctrl, events_new, replies_new):
    """Exact 95% CI of the per-reply rate ratio new/control: conditional on the total number of events, the control's
    share is binomial (Clopper-Pearson), pi = r_c e_c / (r_c e_c + r_n e_n) -> ratio = (e_c / e_n) (1 - pi) / pi."""
    n, x = events_ctrl + events_new, events_ctrl
    if n == 0:
        return None
    lo_pi, hi_pi = clopper_pearson(x, n)
    scale = replies_ctrl / replies_new
    upper = scale * (1 - lo_pi) / lo_pi if lo_pi > 0 else float("inf")
    lower = scale * (1 - hi_pi) / hi_pi if hi_pi < 1 else 0.0
    return [round(lower, 2), round(upper, 2)]


def rate_block(rollouts):
    replies = sum(len(r["replies"]) for r in rollouts)
    kinds = {k: sum(sum(1 for _, kind, _ in r["replies"] if kind == k) for r in rollouts) for k in ("pad", "special", "empty")}
    bad = sum(kinds.values())
    return {"rollouts": len(rollouts), "replies": replies, "degenerate": bad, **kinds,
            "per_1000_replies": round(1000 * bad / replies, 2) if replies else None}


def boot_diff(ctrl, new, n=4000, seed=20261026):
    """95% CI of (new rate - control rate) per 1000 replies, resampling rollouts within each arm."""
    rng = random.Random(seed)
    def counts(rs):
        return [(sum(1 for _, k, _ in r["replies"] if k), len(r["replies"])) for r in rs if r["replies"]]
    c, w = counts(ctrl), counts(new)
    if not c or not w:
        return None
    stats = []
    for _ in range(n):
        cs = [c[rng.randrange(len(c))] for _ in c]
        ws = [w[rng.randrange(len(w))] for _ in w]
        cr = sum(x for x, _ in cs) / max(1, sum(y for _, y in cs))
        wr = sum(x for x, _ in ws) / max(1, sum(y for _, y in ws))
        stats.append(1000 * (wr - cr))
    stats.sort()
    return [round(stats[int(0.025 * n)], 2), round(stats[int(0.975 * n) - 1], 2)]


def compare(ctrl, new):
    a, b = rate_block(ctrl), rate_block(new)
    out = {"control": a, "new": b,
           "fisher_p_per_reply": fisher(a["degenerate"], a["replies"] - a["degenerate"],
                                        b["degenerate"], b["replies"] - b["degenerate"]),
           "bootstrap95_new_minus_control_per_1000": boot_diff(ctrl, new)}
    if b["per_1000_replies"] is not None and a["per_1000_replies"]:
        out["rate_ratio_new_over_control"] = round(b["per_1000_replies"] / a["per_1000_replies"], 2)
        out["rate_ratio_exact95"] = ratio_ci(a["degenerate"], a["replies"], b["degenerate"], b["replies"])
    for label, key in (("with_degenerate_reply", "any"), ("ended_by_degenerate_reply", "ended")):
        ca = sum(r[key] for r in ctrl if r["final"])
        na = sum(r[key] for r in new if r["final"])
        cn = sum(1 for r in ctrl if r["final"])
        nn = sum(1 for r in new if r["final"])
        out[f"rollouts_{label}"] = {"control": f"{ca}/{cn}", "new": f"{na}/{nn}",
                                    "fisher_p": fisher(ca, cn - ca, na, nn - na) if cn and nn else None}
    return out


def final_attempts():
    """{attempt_dir: (campaign, task_id, slot, status)} of the graded/infra final slots in results files."""
    out = {}
    for path in sorted(HERE.glob("results/kimi-*.json")):
        data = json.loads(path.read_text())
        name = path.stem.removeprefix("kimi-")
        for task_id, records in data["records"].items():
            for record in records:
                if record.get("attempt_dir"):
                    out[str(Path(record["attempt_dir"]).resolve())] = (name, task_id, record["slot"], record["status"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="compare")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = HERE / "probes" / "kimi"
    attempts = sorted(str(a.resolve()) for a in root.glob("*/models/kimi/attempts/*/attempt-*")
                      if a.parent.parent.parent.parent.parent.name.startswith(("sel50-g", CTRL)))
    with ProcessPoolExecutor(args.workers) as pool:
        scanned = dict(pool.map(scan, attempts, chunksize=2))
    final = final_attempts()
    manifests = {}
    rollouts = []
    for attempt, replies in scanned.items():
        path = Path(attempt)
        campaign = path.parents[4].name
        if campaign not in manifests:
            manifest = json.loads((path.parents[4] / "manifest.json").read_text())
            manifests[campaign] = {t["_ng_task_index"]: t["task_id"] for t in manifest["tasks"]}
        index = int(path.parent.name.split("-")[0])
        last = replies[-1] if replies else None
        rollouts.append({"attempt": attempt, "campaign": campaign, "arm": "control" if campaign == CTRL else "new",
                         "task_id": manifests[campaign][index], "replies": replies, "final": attempt in final,
                         "status": final.get(attempt, (None, None, None, None))[3],
                         "any": any(k for _, k, _ in replies),
                         "ended": bool(last and last[1] and not last[2])})
    ctrl = [r for r in rollouts if r["arm"] == "control"]
    new = [r for r in rollouts if r["arm"] == "new"]
    ctrl_tasks = {r["task_id"] for r in ctrl}
    stamps = [t for r in ctrl for t, _, _ in r["replies"]]
    report = {"control_tasks": sorted(ctrl_tasks),
              "control_window": [min(stamps), max(stamps)] if stamps else None,
              "same_tasks": compare(ctrl, [r for r in new if r["task_id"] in ctrl_tasks]),
              "overall": compare(ctrl, new)}
    if stamps:
        lo, hi = min(stamps), max(stamps)
        window_new = [dict(r, replies=[x for x in r["replies"] if lo <= x[0] <= hi]) for r in new]
        window_new = [r for r in window_new if r["replies"]]
        report["concurrent"] = compare(ctrl, window_new)
        report["concurrent"]["note"] = "per-reply rows only (rollout-level fields use whole rollouts that overlap the window)"
    report["historical_reference"] = {
        "previous_kimi_50_tasks_oct17_21": {"replies": 27163, "degenerate": 0},
        "default_settings_official2_oct24": {"replies": 10134, "degenerate": 11, "per_1000_replies": 1.09}}
    out = HERE / "analysis" / f"degenerate-{args.out}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
