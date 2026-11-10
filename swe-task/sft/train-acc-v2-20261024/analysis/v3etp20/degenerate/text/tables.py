#!/usr/bin/env python3
"""Render out/summary.json (aggregate.py) as plain-text tables -> out/summary_tables.txt (stdlib only).

Cells are 'calls (per 1,000 calls) | rollouts (% of rollouts)' unless stated. Pairwise z (two-proportion, rollout
level) is given for the comparisons that matter: v3etp5 vs etp5 (same epochs), v3etp5 vs etp10 (~same steps),
v3etp10 vs etp10 (same epochs), v3etp10 vs etp20 (~same steps), each v3 run vs base.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DS = ["base", "base-rerun", "v2-5ep", "v2-10ep", "v2-20ep", "etp5", "etp10", "etp20", "v3etp5", "v3etp10"]
TA = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]  # 2026-11-04: + v3etp20
PAIRS = {"deepswe": [("v3etp5", "etp5"), ("v3etp5", "etp10"), ("v3etp10", "etp10"), ("v3etp10", "etp20"),
                     ("v3etp5", "base"), ("v3etp10", "base")],
         "trainacc": [("v3etp5", "etp5"), ("v3etp5", "etp10"), ("v3etp10", "etp10"), ("v3etp10", "etp20"),
                      ("v3etp5", "sc14"), ("v3etp10", "sc14"),
                      # 2026-11-04: v3etp20 vs v3etp10 / v3etp5 / etp20 (same epochs, v2 corpus) / base
                      ("v3etp20", "v3etp10"), ("v3etp20", "v3etp5"), ("v3etp20", "etp20"), ("v3etp20", "sc14")]}


def z2(a, n, b, m):
    if not n or not m:
        return float("nan")
    p = (a + b) / (n + m)
    if p in (0, 1):
        return 0.0
    se = math.sqrt(p * (1 - p) * (1 / n + 1 / m))
    return ((a / n) - (b / m)) / se


def main():
    d = json.loads((OUT / "summary.json").read_text())
    M = d["models"]
    lines = [f"thresholds: {d['thresholds']}", ""]

    def get(m, path):
        x = m
        for p in path:
            if x is None:
                return 0
            x = x.get(p) if isinstance(x, dict) else None
        return x or 0

    rows = [
        ("B1 reasoning repetition loop (any 2k window zlib < 0.10)", ("counts", "R_loop_calls"), ("rollouts_with", "R_loop")),
        ("B1 ... class hmm", ("loop_class_calls", "hmm"), ("loop_class_rollouts", "hmm")),
        ("B1 ... class bare dash ('— — —')", ("loop_class_calls", "bare dash"), ("loop_class_rollouts", "bare dash")),
        ("B1 ... class check marks (✓/✗)", ("loop_class_calls", "check marks"), ("loop_class_rollouts", "check marks")),
        ("B1 ... class shouting (AH/OH/WAIT)", ("loop_class_calls", "shouting (AH/OH/WAIT)"), ("loop_class_rollouts", "shouting (AH/OH/WAIT)")),
        ("B1 ... class hold on", ("loop_class_calls", "hold on"), ("loop_class_rollouts", "hold on")),
        ("B1 ... class wait", ("loop_class_calls", "wait"), ("loop_class_rollouts", "wait")),
        ("B1 ... class zero-width", ("loop_class_calls", "zero-width"), ("loop_class_rollouts", "zero-width")),
        ("B1 ... class arrows", ("loop_class_calls", "arrows"), ("loop_class_rollouts", "arrows")),
        ("B1 ... class other symbols", ("loop_class_calls", "other symbols"), ("loop_class_rollouts", "other symbols")),
        ("B1 ... class words", ("loop_class_calls", "words"), ("loop_class_rollouts", "words")),
        ("B1 ... class code/data", ("loop_class_calls", "code/data"), ("loop_class_rollouts", "code/data")),
        ("B1 ... outcome runaway (cut at length limit)", ("loop_outcome_calls", "runaway (cut at length limit)"), None),
        ("B1 ... outcome self-ended", ("loop_outcome_calls", "self-ended (loop ran to the end of the reasoning, then the call ended normally)"), None),
        ("B1 ... outcome escaped", ("loop_outcome_calls", "escaped (normal text after the loop)"), None),
        ("B1 compaction-summary calls with a loop", ("counts", "comp_R_loop_calls"), ("rollouts_with", "comp_loop")),
        ("B2 visible-text loop", ("counts", "V_loop_calls"), ("rollouts_with", "V_loop")),
        ("B3 char run longer than the corpus max for that char (R)", None, ("charrun_beyond_corpus_rollouts", "R")),
        ("B4 control tokens not in context, reasoning (any)", ("ctrl_calls_by_family", "R ANY"), ("ctrl_rollouts_by_family", "R ANY")),
        ("B4 control tokens not in context, visible (any)", ("ctrl_calls_by_family", "V ANY"), ("ctrl_rollouts_by_family", "V ANY")),
        ("B6 foreign script not in context, reasoning (any)", ("foreign_calls", "R ANY"), ("foreign_rollouts", "R ANY")),
        ("B6 ... CJK", ("foreign_calls", "R CJK"), ("foreign_rollouts", "R CJK")),
        ("B6 foreign script not in context, visible (any)", ("foreign_calls", "V ANY"), ("foreign_rollouts", "V ANY")),
        ("B8 interjection flood (>= 22 per 1k chars, window zlib >= 0.25)", ("counts", "interj_flood_calls"), ("rollouts_with", "interj_flood")),
        ("B8 ... in calls without any loop region", ("counts", "interj_flood_calls_noloop"), None),
        ("B8 ... softer: >= 15 per 1k chars (corpus p99.9 = 14)", ("counts", "interj_ge15_calls"), None),
        ("B9 em-dash flood (>= 26 per 1k chars, zlib >= 0.25)", ("counts", "emdash_flood_calls"), ("rollouts_with", "emdash_flood")),
        ("B9 '?' flood (>= 54 per 1k chars)", ("counts", "qmark_flood_calls"), ("rollouts_with", "qmark_flood")),
        ("B11 bang-separator run (>= 8 'word!' tokens)", ("counts", "bangsep_calls"), ("rollouts_with", "bangsep")),
        ("B13 no-space run >= 450 chars", ("counts", "nospace_calls"), ("rollouts_with", "nospace")),
        ("B14 reasoning >= 16,384 tokens", ("counts", "rtok_ge_16384"), None),
        ("B14 reasoning >= 32,768 tokens", ("counts", "rtok_ge_32768"), ("rollouts_with", "rtok_ge_32768")),
        ("B14 ... of which no loop region", ("counts", "rtok_ge_32768_noloop"), ("rollouts_with", "rtok_ge_32768_noloop")),
        ("B14 reasoning >= 65,536 tokens", ("counts", "rtok_ge_65536"), None),
        ("B14 ... of which no loop region", ("counts", "rtok_ge_65536_noloop"), None),
        ("B14 reasoning >= 131,072 tokens", ("counts", "rtok_ge_131072"), None),
        ("B15 sentence cycling (>= 21 recurrences, no loop region)", ("counts", "cycle_calls"), ("rollouts_with", "cycle")),
        ("B16 reasoning identical to previous call (>= 200 chars)", ("counts", "dup_calls"), ("rollouts_with", "dup")),
        ("B17 visible text with reasoning markers (hmm/hold on/oh/ah/ugh/whatever)", ("V_reasoning_marker_calls", "calls"), None),
    ]
    for ds, order in (("deepswe", DS), ("trainacc", TA)):
        names = [f"{ds}/{m}" for m in order if f"{ds}/{m}" in M]
        lines.append(f"===== {ds}: denominators")
        for nm in names:
            m = M[nm]
            lines.append(f"  {nm:22s} rollouts {m['rollouts']:4d}  calls {m['calls']:7d}  reasoned calls {m['reasoned_calls']:7d}  "
                         f"visible calls {m['visible_calls']:6d}  compaction calls {m['comp_calls']:5d}")
        lines.append("")
        for title, cpath, rpath in rows:
            lines.append(f"--- {title}")
            for nm in names:
                m = M[nm]
                c = get(m, cpath) if cpath else None
                r = get(m, rpath) if rpath else None
                if rpath == ("V_reasoning_marker_calls", "calls"):
                    pass
                cs = f"{c:6d} ({1000 * c / max(1, m['calls']):7.3f}/1k)" if c is not None else " " * 21
                rs = f"{r:4d} ({100 * r / max(1, m['rollouts']):5.1f}%)" if r is not None else ""
                lines.append(f"  {nm:22s} {cs}   {rs}")
            if rpath:
                zs = []
                for a, b in PAIRS[ds]:
                    A, B = M.get(f"{ds}/{a}"), M.get(f"{ds}/{b}")
                    if A and B:
                        zs.append(f"{a}-{b} z={z2(get(A, rpath), A['rollouts'], get(B, rpath), B['rollouts']):+.1f}")
                lines.append("    rollout z: " + "; ".join(zs))
            lines.append("")
        lines.append(f"--- loop onset: reasoning tokens before the first loop region in the call (p10/p25/p50/p75/p90); "
                     f"loop-call prompt tokens p10/p50/p90; first loop call index p25/p50/p75 and session fraction")
        for nm in names:
            m = M[nm]
            lines.append(f"  {nm:22s} {m.get('loop_onset_tokens_before_p10_p25_p50_p75_p90')}  <1k: {m.get('loop_onset_within_1k_tokens')}  "
                         f"prompt {m.get('loop_call_prompt_tokens_p10_p50_p90')}  first idx {m.get('first_loop_call_index_p25_p50_p75')} "
                         f"frac {m.get('first_loop_call_session_fraction_p25_p50_p75')}")
        lines.append("")
        lines.append("--- sequence: loop rollouts with an interjection/em-dash flood in an EARLIER call (and at or before); "
                     "no-loop rollouts with a flood; flood rollouts that later/also loop; gap flood->first loop (calls p25/p50/p75)")
        for nm in names:
            c = M[nm]["counts"]
            lr = M[nm]["rollouts_with"].get("R_loop", 0)
            lines.append(f"  {nm:22s} loop rollouts {lr}: flood before {c.get('loop_rollouts_flood_before_first_loop', 0)}, at/before "
                         f"{c.get('loop_rollouts_flood_at_or_before_first_loop', 0)} | no-loop rollouts {c.get('noloop_rollouts', 0)}: with flood "
                         f"{c.get('noloop_rollouts_with_flood', 0)} | flood rollouts {c.get('rollouts_with_flood', 0)}: with loop "
                         f"{c.get('flood_rollouts_with_loop', 0)} | gap {M[nm].get('flood_to_first_loop_gap_calls_p25_p50_p75')}")
        lines.append("")
        lines.append("--- per 1k calls by prompt tokens (context): loop calls / flood calls")
        for nm in names:
            bp = M[nm].get("by_prompt_tokens", {})
            lines.append(f"  {nm:22s} " + "  ".join(f"{k}: {1000 * v['loop_calls'] / max(1, v['calls']):.2f}/{1000 * v['flood_calls'] / max(1, v['calls']):.1f} (n={v['calls']})" for k, v in bp.items()))
        lines.append("")
        lines.append("--- normal-style densities per 1k reasoning chars outside loop regions: interjections by marker; em dash; "
                     "shouting words; '!!' runs; stutters; all-caps words; loop share of reasoning chars")
        for nm in names:
            m = M[nm]
            pk = m.get("per_1k_reasoning_chars_outside_loops", {})
            lines.append(f"  {nm:22s} hmm {pk.get('hmm', 0):.3f} wait {pk.get('wait', 0):.3f} actually {pk.get('actually', 0):.3f} "
                         f"hold-on {pk.get('hold on', 0):.3f} oh {pk.get('oh', 0):.3f} ah {pk.get('ah', 0):.3f} whatever {pk.get('whatever', 0):.3f} "
                         f"ugh {pk.get('ugh', 0):.3f} | em-dash {m.get('emdash_per_1k_outside_loops')} shout {m.get('shout_per_1k_outside_loops')} "
                         f"!! {m.get('excl_runs_per_1k')} stutter {m.get('stutters_per_1k')} caps {m.get('caps_words_per_1k')} | loop share {m.get('loop_char_share')}")
        lines.append("")
        lines.append("--- detail: loop classes, control tokens, foreign scripts, garbage, char runs, spaced runs, think tags, V markers")
        for nm in names:
            m = M[nm]
            lines.append(f"  {nm}")
            for k in ("loop_class_calls", "V_loop_class_calls", "ctrl_calls_by_family", "ctrl_tokens", "foreign_calls", "foreign_chars",
                      "cjkpunct_spont_calls", "garbage_calls", "garbage_rollouts", "charrun_beyond_corpus_calls", "spaced_run_calls",
                      "thinktag_calls", "stutter_ge_thr_calls", "V_reasoning_marker_calls", "V_reasoning_marker_rollouts",
                      "V_interj_per_1k_visible_chars", "out_tokens_p50_p90_p99_p999_max", "reasoning_chars_p50_p90_p99_max"):
                v = m.get(k)
                if v:
                    lines.append(f"      {k}: {json.dumps(v, ensure_ascii=False)[:600]}")
        lines.append("")
    lines.append("===== corpus (trained assistant turns)")
    for c, v in d["corpus"].items():
        lines.append(f"  {c}: " + json.dumps({k: vv for k, vv in v.items() if k not in ("dist",)}, ensure_ascii=False)[:3000])
    (OUT / "summary_tables.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
