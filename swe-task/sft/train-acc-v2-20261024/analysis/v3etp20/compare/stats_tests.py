#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): tests for the rollout-level counts the report compares (v3etp20 vs v3etp10, v3etp5,
etp20 and base sc14). Stdlib only; read-only on the inputs.

    python3 -I -B stats_tests.py -> stats_tests.{json,txt}

Per behavior: k / n per model and
  * two-sided Fisher exact on rollouts (treats the 250 slots as independent; they are 5 per task, so it is a guide),
  * a paired task-bootstrap 95% CI of the difference in the per-slot rate (10,000 resamples of the 50 tasks, seed
    20261104, the same draw for both models) and a sign-flip p on the per-task differences (100,000 flips), which
    respect the clustering by task.
Slot flags (one row per slot, 250 per model, infra slots included as in the empty-patch table):
  empty patch, edited-uncommitted (E1), 3 h deadline kill, ends on a runaway call (E3a_runaway), stopped right after the
  compaction "continue" (empty, compacted, finished), >= 1 tool-call compaction summary, >= 1 unreasoned action turn,
  >= 1 thinking call cut at the length limit (counting-parser log; n/a for sc14, served with the plain lenient parser),
  >= 1 reasoning with a foreign-script glitch (text lens B6, any call incl. compaction summaries).
Sources: analysis/{empty-patch,think,features}/ (earlier models) and analysis/v3etp20/{empty-patch,think,features}
(v3etp20; sc14's extractor rows are teacher-rollouts-v2-20261025/features/sc14.jsonl), the endings-lens sessions (../degenerate/endings/out/sessions) and the text-lens flagged calls
(../degenerate/text/out/models_calls_flagged.jsonl).
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
V20 = TA / "analysis/v3etp20"
HERE = Path(__file__).resolve().parent
MODELS = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]
REFS = ["v3etp10", "v3etp5", "etp20", "sc14"]


def d(model, sub):
    return (V20 / sub) if model == "v3etp20" else (TA / "analysis" / sub)


def fisher(a, b, c, e):
    n1, n2, k = a + b, c + e, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def flags(model):
    rows = {}
    for line in open(d(model, "empty-patch") / model / "all.jsonl"):
        c = json.loads(line)
        f = c["f"]
        key = (c["task_id"], c["ri"])
        rows[key] = {
            "empty_patch": c["empty_patch"],
            "edited_uncommitted": c["empty_patch"] and c["code"] == "E1_UNCOMMITTED",
            "runaway_end": c["empty_patch"] and c.get("subcode") == "E3a_cap_cut" and bool(f.get("last_out_tokens")),
            "stopped_after_compaction_continue": c["empty_patch"] and bool(f.get("n_compactions")) and not c["not_finished"],
        }
    for line in gzip.open(V20 / "degenerate/endings/out/derived.jsonl.gz", "rt"):
        x = json.loads(line)
        if x["model"] == model:
            rows[(x["task_id"], x["ri"])]["deadline_kill"] = bool(x.get("deadline_kill"))
    for line in gzip.open(V20 / f"degenerate/endings/out/sessions/ta__{model}.jsonl.gz", "rt"):
        x = json.loads(line)
        s = x.get("s") or {}
        rows[(x["task_id"], x["ri"])]["toolcall_summary"] = any(
            "si" in c and (c.get("s_toolmarkup") or c.get("s_tools")) for c in (s.get("comps") or []))
    tr = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/features")
    feat = tr / f"{model}.jsonl" if model in ("sc14", "sftv2") else d(model, "features") / f"{model}.jsonl"
    for line in open(feat):
        r = json.loads(line)
        key = (r["task_id"], int(r["slot"].rsplit("-r", 1)[1]))
        rows[key]["unreasoned_action"] = bool(r.get("export")) and (r.get("action_turns") or 0) > (r.get("action_with_reasoning") or 0)
    cut = set()
    for line in open(d(model, "think") / f"calls-{model}.jsonl"):
        r = json.loads(line)
        if r.get("log_kind") == "UNCLOSED_NO_TOOL" and r.get("finish") == "length":
            cut.add((r["task_id"], int(r["slot"].rsplit("-r", 1)[1])))
    glitch = set()
    for line in open(V20 / "degenerate/text/out/models_calls_flagged.jsonl"):
        r = json.loads(line)
        # as the text lens's B6 rollout count (aggregate.py foreign_rollouts "R ANY"): any call, summaries included
        if r["dataset"] == "trainacc" and r["model"] == model and ((r.get("R") or {}).get("foreign_spont")):
            glitch.add((r["task_id"], int(str(r["rep"]).rsplit("-r", 1)[1])))
    graded = {}
    for path in TA.glob(f"results/{model}-group-*.json"):
        for task_id, recs in json.loads(path.read_text())["records"].items():
            for r in recs:
                graded[(task_id, int(r["slot"].rsplit("-r", 1)[1]))] = r["status"] in ("pass", "fail")
    for key, row in rows.items():
        row["graded"] = graded[key]
        row["length_cut"] = key in cut
        row["foreign_script_glitch"] = key in glitch
        row.setdefault("deadline_kill", False)
        row.setdefault("toolcall_summary", False)
        row.setdefault("unreasoned_action", False)
    return rows


def paired(ra, rb, flag, seed=20261104, n=10000):
    tasks = sorted({k[0] for k in ra} & {k[0] for k in rb})
    cell = {t: (sum(v[flag] for k, v in ra.items() if k[0] == t), sum(1 for k in ra if k[0] == t),
                sum(v[flag] for k, v in rb.items() if k[0] == t), sum(1 for k in rb if k[0] == t)) for t in tasks}

    def diff(sample):
        return 100 * (sum(cell[t][2] for t in sample) / sum(cell[t][3] for t in sample)
                      - sum(cell[t][0] for t in sample) / sum(cell[t][1] for t in sample))
    rng = random.Random(seed)
    boots = sorted(diff([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(n))
    diffs = [cell[t][2] / cell[t][3] - cell[t][0] / cell[t][1] for t in tasks]
    obs = sum(diffs) / len(diffs)
    flips = 100000
    ext = sum(abs(sum(x if rng.random() < 0.5 else -x for x in diffs) / len(diffs)) >= abs(obs) - 1e-12 for _ in range(flips))
    return round(diff(tasks), 1), [round(boots[int(0.025 * n)], 1), round(boots[int(0.975 * n) - 1], 1)], (ext + 1) / (flips + 1)


def main():
    F = {m: flags(m) for m in MODELS}
    names = ["empty_patch", "edited_uncommitted", "deadline_kill", "runaway_end", "stopped_after_compaction_continue",
             "toolcall_summary", "unreasoned_action", "length_cut", "foreign_script_glitch"]
    out, lines = {}, []
    lines.append("| rollouts with | " + " | ".join(MODELS) + " | v3etp20 - ref, pp [95% task-bootstrap CI], sign-flip p; Fisher p |")
    lines.append("|---|" + "---|" * (len(MODELS) + 1))
    # the same tests on the graded slots only (infra slots of the reference dropped, as pass@1 does): "<flag>|graded"
    G = {m: {k: v for k, v in F[m].items() if v["graded"]} for m in MODELS}
    for full in names + ["empty_patch|graded"]:
        name, sel = (full.split("|")[0], G) if full.endswith("|graded") else (full, F)
        cnt = {m: [sum(v[name] for v in sel[m].values()), len(sel[m])] for m in MODELS}
        tests = {}
        for ref in REFS:
            a, n = cnt["v3etp20"]
            b, m_ = cnt[ref]
            dd, ci, p = paired(sel[ref], sel["v3etp20"], name)
            tests[ref] = {"diff_pp": dd, "ci95": ci, "signflip_p": p, "fisher_p": fisher(a, n - a, b, m_ - b)}
        out[full] = {"counts": cnt, "tests": tests}
        name = full
        lines.append(f"| {name} | " + " | ".join(f"{cnt[m][0]}/{cnt[m][1]}" for m in MODELS) + " | " + "; ".join(
            f"vs {r}: {t['diff_pp']:+.1f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}] p={t['signflip_p']:.3g}; F={t['fisher_p']:.3g}"
            for r, t in tests.items()) + " |")
    (HERE / "stats_tests.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "stats_tests.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
