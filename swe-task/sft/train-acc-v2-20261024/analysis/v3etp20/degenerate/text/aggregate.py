#!/usr/bin/env python3
"""Per-behavior prevalence tables from scan_models.py + scan_corpus.py outputs (stdlib only).

usage: python -I aggregate.py   -> out/summary.json, out/summary_tables.txt, out/examples.json

Denominators: rollouts = rollouts with a readable OpenCode export; calls = non-compaction assistant calls (compaction
summaries are reported apart); reasoned calls = calls with non-empty reasoning. Corpus: trained assistant turns of
SFT-v3 / SFT-v2 (reasoned turns = turns with reasoning before </think>).
Behaviors (B-codes) and thresholds are defined in BEHAVIORS below; thresholds that refer to the corpus are read from
out/corpus_summary.json (e.g. 'above the corpus maximum').
"""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DS_ORDER = ["base", "base-rerun", "v2-5ep", "v2-10ep", "v2-20ep", "etp5", "etp10", "etp20", "v3etp5", "v3etp10"]
TA_ORDER = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]  # 2026-11-04: + v3etp20
# 2026-11-04 copy: the SFT-corpus scan is not re-run here; its thresholds come from the published corpus_summary.json
CORPUS_SUMMARY = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis/"
                      "sftv3-etp-20261030/degenerate-behaviors-20261031/text/out/corpus_summary.json")
# tuple indices (scan_models.py)
CI, RCH, VCH, OUTT, FIN, PROMPT, RTOK, RMIN, RLOOP, RINTJ, RIMAX, REMD, TOOLS, VMIN, VLOOP, VTOK, DUP, \
    REMAX, RQMAX, RCAPS, RBANG, RSTUT, RNOSP, RIMAXR, REMAXR, RSHOUT, REXCL, RCYC, VIMAX = range(29)
CTRL_FAMILIES = [
    ("open/close/sep format", ("<|open|>", "<|close|>", "<|sep|>")),
    ("<|unk|>", ("<|unk|>",)),
    ("harmony (<|channel|>, <|start|>, <|constrain|>, <|message|>, <|end|>, <|call|>, <|return|>)",
     ("<|channel|>", "<|start|>", "<|constrain|>", "<|message|>", "<|end|>", "<|call|>", "<|return|>")),
    ("ChatML / Llama (<|im_start|>, <|im_end|>, <|endoftext|>, <|eot_id|>, <|begin_of_text|>, header ids)",
     ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|begin_of_text|>", "<|start_header_id|>",
      "<|end_header_id|>", "<|im_sep|>")),
    ("Kimi tool-call tokens (<|tool_call_begin|> ...)", ("<|tool_calls_section_begin|>", "<|tool_call_begin|>",
                                                           "<|tool_call_argument_begin|>", "<|tool_call_end|>",
                                                           "<|tool_calls_section_end|>", "<|tool_call|>")),
    ("Mistral / SentencePiece ([INST], [TOOL_CALLS], <s>, </s>, <unk>, <0xNN>)", ()),
]


def family(tok):
    for name, toks in CTRL_FAMILIES:
        if tok in toks:
            return name
    if tok.startswith("[") or tok in ("<s>", "</s>", "<unk>", "<pad>", "<bos>", "<eos>", "<mask>") or tok.startswith("<0x"):
        return CTRL_FAMILIES[-1][0]
    if tok.startswith("<｜"):
        return "DeepSeek fullwidth (<｜...｜>)"
    return "other invented <|...|> (e.g. <|reserved...|>, <|resume|>)"


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def z2(a, n, b, m):
    """two-proportion z (a/n vs b/m)."""
    if not n or not m:
        return float("nan")
    p = (a + b) / (n + m)
    se = math.sqrt(p * (1 - p) * (1 / n + 1 / m)) if 0 < p < 1 else 0
    return ((a / n) - (b / m)) / se if se else 0.0


def load():
    rolls = [json.loads(l) for l in open(OUT / "models_rollouts.jsonl")]
    flagged = [json.loads(l) for l in open(OUT / "models_calls_flagged.jsonl")]
    corpus = json.loads(CORPUS_SUMMARY.read_text())
    return rolls, flagged, corpus


def corpus_tables(corpus):
    """per corpus (v3, v2): totals over sources + per-source counters / distributions."""
    res = {}
    for c, srcs in corpus["counters"].items():
        tot = Counter()
        for s, a in srcs.items():
            for k, v in a.items():
                if k == "turn_tokens_max":
                    tot[k] = max(tot[k], v)
                else:
                    tot[k] += v
        dist = defaultdict(Counter)
        bychar = defaultdict(dict)
        for s, dd in corpus["distributions"][c].items():
            for metric, h in dd.items():
                if metric.endswith("_by_char"):
                    for ch, v in h.items():
                        bychar[metric][ch] = max(bychar[metric].get(ch, 0), v)
                else:
                    for k, v in h.items():
                        dist[metric][float(k) if "." in k else int(k)] += v
        res[c] = {"tot": dict(tot), "src": srcs, "dist": {m: dict(h) for m, h in dist.items()}, "bychar": dict(bychar)}
    return res


def dist_max(h):
    return max((k for k, v in h.items() if v), default=0)


def dist_ge(h, thr):
    return sum(v for k, v in h.items() if k >= thr)


def main():
    rolls, flagged, corpus = load()
    C = corpus_tables(corpus)
    cv3 = C["v3"]
    # thresholds just above the SFT corpora's own maxima (v3 and v2 trained reasoning, out/corpus_summary.json):
    #   interjections per 1,000 chars: corpus max 21 -> flood >= 22; em dashes per 1,000 chars: max 25 -> >= 26;
    #   '?' per 1,000 chars: max 53 -> >= 54; sentence segment recurring in one text: max 20 -> >= 21;
    #   immediate word repeats: max 4 except one code turn ('drop' x11) -> >= 5; no-space run: < 450 chars except the
    #   one garbled v4 turn (1,238) -> >= 450
    thr = {"interj_flood": 22, "emdash_flood": 26, "qmark_flood": 54, "cycle": 21, "stutter_max": 5, "nospace": 450}
    thr_check = {
        "interj_flood": dist_max(cv3["dist"].get("R_interj_maxwin", {})) + 1,
        "emdash_flood": max(dist_max(C["v3"]["dist"].get("R_emdash_maxwin", {})), dist_max(C["v2"]["dist"].get("R_emdash_maxwin", {}))) + 1,
        "qmark_flood": max(dist_max(C["v3"]["dist"].get("R_qmark_maxwin", {})), dist_max(C["v2"]["dist"].get("R_qmark_maxwin", {}))) + 1,
        "cycle": max(dist_max(C["v3"]["dist"].get("R_cycle", {})), dist_max(C["v2"]["dist"].get("R_cycle", {}))) + 1,
    }
    for k, v in thr_check.items():
        assert thr[k] == v, (k, thr[k], v)
    charmax = {}
    for c in ("v3", "v2"):
        for ch, v in C[c]["bychar"].get("R_charrun_maxlen_by_char", {}).items():
            charmax[ch] = max(charmax.get(ch, 0), v)
    # ---- model side
    M = defaultdict(lambda: defaultdict(Counter))      # (ds, model) -> counters
    roll_sets = defaultdict(lambda: defaultdict(set))   # (ds, model) -> behavior -> rollout ids
    dists = defaultdict(lambda: defaultdict(list))
    roll_index = {}
    for r in rolls:
        key = (r["dataset"], r["model"])
        if r.get("unreadable"):
            M[key]["n"]["unreadable"] += 1
            continue
        roll_index[(key, r["id"])] = r
        n = M[key]["n"]
        n["rollouts"] += 1
        n["calls"] += r["n_calls"]
        n["comp_calls"] += r["n_comp"]
        for k, v in r["sums"].items():
            M[key]["sums"][k] += v
        first_loop = None
        first_flood = None
        for t in r["calls"]:
            comp = t[CI] == -1
            if comp:
                n["comp_R_loop_calls"] += t[RLOOP] > 0
                if t[RLOOP] > 0:
                    roll_sets[key]["comp_loop"].add(r["id"])
                continue
            n["reasoned_calls"] += t[RCH] > 0
            cb = min(t[PROMPT] // 32768, 7)
            ctxb = M[key]["ctx_calls"]
            ctxb[cb] += 1
            if t[RLOOP] > 0:
                M[key]["ctx_loop"][cb] += 1
            is_flood = ((t[RIMAX] >= thr["interj_flood"] and (t[RIMAXR] == -1 or t[RIMAXR] >= 0.25))
                        or (t[REMAX] >= thr["emdash_flood"] and (t[REMAXR] == -1 or t[REMAXR] >= 0.25)))
            if is_flood:
                M[key]["ctx_flood"][cb] += 1
                if first_flood is None:
                    first_flood = t[CI]
            n["visible_calls"] += t[VCH] > 0
            dists[key]["out"].append(t[OUTT])
            dists[key]["rch"].append(t[RCH])
            if t[RLOOP] > 0:
                n["R_loop_calls"] += 1
                n["R_loop_chars"] += t[RLOOP]
                roll_sets[key]["R_loop"].add(r["id"])
                if first_loop is None:
                    first_loop = t[CI]
            if t[VLOOP] > 0:
                n["V_loop_calls"] += 1
                roll_sets[key]["V_loop"].add(r["id"])
            if t[RMIN] != -1 and t[RMIN] < 0.15:
                n["R_win_lt15_calls"] += 1
            # floods short of a loop
            if t[RIMAX] >= thr["interj_flood"] and (t[RIMAXR] == -1 or t[RIMAXR] >= 0.25):
                n["interj_flood_calls"] += 1
                roll_sets[key]["interj_flood"].add(r["id"])
                if t[RLOOP] == 0:
                    n["interj_flood_calls_noloop"] += 1
            if t[RIMAX] >= 15:
                n["interj_ge15_calls"] += 1
            if t[REMAX] >= thr["emdash_flood"] and (t[REMAXR] == -1 or t[REMAXR] >= 0.25):
                n["emdash_flood_calls"] += 1
                roll_sets[key]["emdash_flood"].add(r["id"])
            if t[RQMAX] >= thr["qmark_flood"]:
                n["qmark_flood_calls"] += 1
                roll_sets[key]["qmark_flood"].add(r["id"])
            if t[RBANG] >= 8:
                n["bangsep_calls"] += 1
                roll_sets[key]["bangsep"].add(r["id"])
            if t[RNOSP] >= thr["nospace"]:
                n["nospace_calls"] += 1
                roll_sets[key]["nospace"].add(r["id"])
            n["stutters"] += t[RSTUT]
            n["caps_words"] += t[RCAPS]
            if t[RCYC] >= thr["cycle"] and t[RLOOP] == 0:
                n["cycle_calls"] += 1
                roll_sets[key]["cycle"].add(r["id"])
            if t[DUP]:
                n["dup_calls"] += 1
                roll_sets[key]["dup"].add(r["id"])
            if t[VIMAX] > 0:
                n["V_interj_calls"] += 1
                roll_sets[key]["V_interj"].add(r["id"])
            # extreme length (reasoning tokens; -1 means < 16,384)
            rt = t[RTOK]
            for lim in (16384, 32768, 65536, 131072):
                if rt >= lim:
                    n[f"rtok_ge_{lim}"] += 1
                    n[f"rtok_ge_{lim}_noloop"] += t[RLOOP] == 0
                    if lim == 32768:
                        roll_sets[key]["rtok_ge_32768"].add(r["id"])
                        if t[RLOOP] == 0:
                            roll_sets[key]["rtok_ge_32768_noloop"].add(r["id"])
            for lim in (16384, 32768, 65536):
                n[f"out_ge_{lim}"] += t[OUTT] >= lim
        if first_loop is not None:
            dists[key]["first_loop_frac"].append(first_loop / max(1, r["n_calls"]))
            dists[key]["first_loop_ci"].append(first_loop)
            n["loop_rollouts_flood_before_first_loop"] += first_flood is not None and first_flood < first_loop
            n["loop_rollouts_flood_at_or_before_first_loop"] += first_flood is not None and first_flood <= first_loop
            if first_flood is not None and first_flood < first_loop:
                dists[key]["flood_to_loop_gap"].append(first_loop - first_flood)
        else:
            n["noloop_rollouts"] += 1
            n["noloop_rollouts_with_flood"] += first_flood is not None
        n["rollouts_with_flood"] += first_flood is not None
        if first_flood is not None:
            n["flood_rollouts_with_loop"] += first_loop is not None
    # flagged-call details
    loopcls = defaultdict(Counter)
    loopcls_roll = defaultdict(lambda: defaultdict(set))
    loopout = defaultdict(Counter)
    onset = defaultdict(list)
    onset_prompt = defaultdict(list)
    vloopcls = defaultdict(Counter)
    ctrl = defaultdict(Counter)
    ctrl_calls = defaultdict(Counter)
    ctrl_roll = defaultdict(lambda: defaultdict(set))
    thinktag = defaultdict(Counter)
    foreign = defaultdict(Counter)
    foreign_calls = defaultdict(Counter)
    foreign_roll = defaultdict(lambda: defaultdict(set))
    cjkp = defaultdict(Counter)
    garb = defaultdict(Counter)
    garb_calls = defaultdict(Counter)
    garb_roll = defaultdict(lambda: defaultdict(set))
    charrun = defaultdict(Counter)
    charrun_roll = defaultdict(lambda: defaultdict(set))
    spaced = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))
    stut3 = defaultdict(Counter)
    vint = defaultdict(Counter)
    vint_roll = defaultdict(set)

    def add_ex(key, beh, rec, text, limit=6):
        lst = examples[f"{key[0]}/{key[1]}"][beh]
        if len(lst) < limit:
            lst.append({"id": rec["id"], "call": rec["call"], "comp": rec["comp"], "finish": rec["finish"],
                        "out": rec["out"], "prompt": rec["prompt"], "excerpt": text})

    for rec in flagged:
        key = (rec["dataset"], rec["model"])
        comp = rec["comp"]
        R, V = rec.get("R") or {}, rec.get("V") or {}
        if R.get("regions") and not comp:
            regs = R["regions"]
            big = max(regs, key=lambda g: g["end"] - g["start"])
            cls = big["cls"]
            loopcls[key][cls] += 1
            loopcls_roll[key][cls].add(rec["id"])
            last = regs[-1]
            # terminal: the loop reaches (within 2,000 chars) the end of the reasoning, or the last 20,000 chars
            # still compress below 0.15 (thinkx / notool_kinds tail rule)
            if last["after"] <= 2000 or (R.get("tail") is not None and R["tail"] < 0.15):
                kind = "runaway (cut at length limit)" if rec["finish"] == "length" else "self-ended (loop ran to the end of the reasoning, then the call ended normally)"
            else:
                kind = "escaped (normal text after the loop)"
            loopout[key][kind] += 1
            if regs[0].get("tok_before") is not None:
                onset[key].append(regs[0]["tok_before"])
            onset_prompt[key].append(rec["prompt"])
            add_ex(key, f"R_loop::{cls}", rec, f"[{kind}; unit '{big['unit']}'; onset after {regs[0].get('tok_before')} reasoning tokens] ...{big['sample']}...", 3)
        if V.get("regions") and not comp:
            big = max(V["regions"], key=lambda g: g["end"] - g["start"])
            vloopcls[key][big["cls"]] += 1
            add_ex(key, f"V_loop::{big['cls']}", rec, f"unit '{big['unit']}' ...{big['sample']}...", 3)
        for where, f in (("R", R), ("V", V)):
            if not f:
                continue
            if f.get("ctrl_spont"):
                fams = set()
                for tok, c in f["ctrl_spont"].items():
                    ctrl[key][(where, tok)] += c
                    fams.add(family(tok))
                for fam in fams:
                    ctrl_calls[key][(where, fam)] += 1
                    ctrl_roll[key][(where, fam)].add(rec["id"])
                ctrl_calls[key][(where, "ANY")] += 1
                ctrl_roll[key][(where, "ANY")].add(rec["id"])
                add_ex(key, f"{where}_ctrl", rec, f.get("ctrl_ex"), 8)
            if f.get("thinktag"):
                for tag, c in f["thinktag"].items():
                    thinktag[key][(where, tag)] += 1
            if f.get("foreign_spont"):
                for sc, c in f["foreign_spont"].items():
                    foreign[key][(where, sc)] += c
                    foreign_calls[key][(where, sc)] += 1
                    foreign_roll[key][(where, sc)].add(rec["id"])
                foreign_calls[key][(where, "ANY")] += 1
                foreign_roll[key][(where, "ANY")].add(rec["id"])
                add_ex(key, f"{where}_foreign", rec, f.get("foreign_ex"), 10)
            if f.get("cjkpunct_spont"):
                cjkp[key][where] += 1
            if f.get("garbage_spont"):
                for g, c in f["garbage_spont"].items():
                    garb[key][(where, g)] += c
                    garb_calls[key][(where, g)] += 1
                    garb_roll[key][(where, g)].add(rec["id"])
                add_ex(key, f"{where}_garbage", rec, f.get("garbage_ex"), 6)
            cr = f.get("charrun")
            if cr and cr["len"] > charmax.get(cr["ch"], 19):
                charrun[key][(where, cr["ch"])] += 1
                charrun_roll[key][(where, "ANY")].add(rec["id"])
            sp = f.get("spacedrun")
            if sp and not comp:
                spaced[key][(where, sp["ch"])] += 1
            if f.get("stutter_max", 0) >= thr["stutter_max"] and not comp:
                stut3[key][where] += 1
                add_ex(key, f"{where}_stutter", rec, f.get("stutter_ex"), 4)
            if f.get("bangsep") and not comp:
                add_ex(key, f"{where}_bangsep", rec, f.get("bangsep_ex"), 4)
            if f.get("interj_out_maxwin", f.get("interj_maxwin", 0)) >= thr["interj_flood"] and not comp:
                ex = f.get("interj_out_maxwin_ex") or f.get("interj_maxwin_ex")
                ratio = f.get("interj_out_maxwin_ratio", f.get("interj_maxwin_ratio"))
                if ratio is None or ratio >= 0.25:
                    add_ex(key, f"{where}_interj_flood", rec, ex, 4)
            if where == "V" and f.get("interj") and not comp:
                marks = {k for k in f["interj"] if k in ("hmm", "hm/hmmm", "hold on", "oh", "ah", "ugh", "whatever")}
                if marks:
                    vint[key]["calls"] += 1
                    vint_roll[key].add(rec["id"])
                    for k in marks:
                        vint[key][k] += 1
                    add_ex(key, "V_reasoning_markers", rec, None, 0)
    # ---- assemble
    out = {"thresholds": thr, "corpus_charrun_max_by_char": charmax, "models": {}, "corpus": {}}
    for key in sorted(M, key=lambda k: (k[0], (DS_ORDER if k[0] == "deepswe" else TA_ORDER).index(k[1]))):
        n = M[key]["n"]
        s = M[key]["sums"]
        d = dists[key]
        rs = roll_sets[key]
        res = {"rollouts": n["rollouts"], "calls": n["calls"], "reasoned_calls": n["reasoned_calls"],
               "visible_calls": n["visible_calls"], "comp_calls": n["comp_calls"], "unreadable": n["unreadable"]}
        res["counts"] = dict(n)
        res["rollouts_with"] = {b: len(v) for b, v in rs.items()}
        res["loop_class_calls"] = dict(loopcls[key])
        res["loop_class_rollouts"] = {c: len(v) for c, v in loopcls_roll[key].items()}
        res["loop_outcome_calls"] = dict(loopout[key])
        o = sorted(onset[key])
        if o:
            q = lambda p: o[min(len(o) - 1, int(p * len(o)))]
            res["loop_onset_tokens_before_p10_p25_p50_p75_p90"] = [q(.1), q(.25), q(.5), q(.75), q(.9)]
            res["loop_onset_within_1k_tokens"] = sum(1 for x in o if x < 1000)
        op = sorted(onset_prompt[key])
        if op:
            res["loop_call_prompt_tokens_p10_p50_p90"] = [op[int(.1 * len(op))], op[len(op) // 2], op[min(len(op) - 1, int(.9 * len(op)))]]
        fl = sorted(d["first_loop_frac"])
        if fl:
            res["first_loop_call_session_fraction_p25_p50_p75"] = [round(fl[int(.25 * len(fl))], 3), round(fl[len(fl) // 2], 3), round(fl[int(.75 * len(fl))], 3)]
            ci = sorted(d["first_loop_ci"])
            res["first_loop_call_index_p25_p50_p75"] = [ci[int(.25 * len(ci))], ci[len(ci) // 2], ci[int(.75 * len(ci))]]
        res["V_loop_class_calls"] = dict(vloopcls[key])
        labels = ["0-32k", "32-64k", "64-96k", "96-128k", "128-160k", "160-192k", "192-224k", "224k+"]
        res["by_prompt_tokens"] = {labels[b]: {"calls": M[key]["ctx_calls"][b], "loop_calls": M[key]["ctx_loop"][b],
                                               "flood_calls": M[key]["ctx_flood"][b]} for b in range(8)}
        g = sorted(d["flood_to_loop_gap"])
        if g:
            res["flood_to_first_loop_gap_calls_p25_p50_p75"] = [g[int(.25 * len(g))], g[len(g) // 2], g[int(.75 * len(g))]]
        res["ctrl_tokens"] = {f"{w} {t}": c for (w, t), c in ctrl[key].most_common()}
        res["ctrl_calls_by_family"] = {f"{w} {fam}": c for (w, fam), c in ctrl_calls[key].items()}
        res["ctrl_rollouts_by_family"] = {f"{w} {fam}": len(v) for (w, fam), v in ctrl_roll[key].items()}
        res["thinktag_calls"] = {f"{w} {t}": c for (w, t), c in thinktag[key].items()}
        res["foreign_chars"] = {f"{w} {sc}": c for (w, sc), c in foreign[key].items()}
        res["foreign_calls"] = {f"{w} {sc}": c for (w, sc), c in foreign_calls[key].items()}
        res["foreign_rollouts"] = {f"{w} {sc}": len(v) for (w, sc), v in foreign_roll[key].items()}
        res["cjkpunct_spont_calls"] = dict(cjkp[key])
        res["garbage_chars"] = {f"{w} {g}": c for (w, g), c in garb[key].items()}
        res["garbage_calls"] = {f"{w} {g}": c for (w, g), c in garb_calls[key].items()}
        res["garbage_rollouts"] = {f"{w} {g}": len(v) for (w, g), v in garb_roll[key].items()}
        res["charrun_beyond_corpus_calls"] = {f"{w} {ch!r}": c for (w, ch), c in charrun[key].most_common()}
        res["charrun_beyond_corpus_rollouts"] = {f"{w}": len(v) for (w,), v in [((k[0],), v) for k, v in charrun_roll[key].items()]}
        res["spaced_run_calls"] = {f"{w} {ch!r}": c for (w, ch), c in spaced[key].most_common()}
        res["stutter_ge_thr_calls"] = dict(stut3[key])
        res["V_reasoning_marker_calls"] = dict(vint[key])
        res["V_reasoning_marker_rollouts"] = len(vint_roll[key])
        rch = s.get("R_chars_out", 0)
        res["per_1k_reasoning_chars_outside_loops"] = {k.split("::")[1]: round(1000 * v / rch, 4) for k, v in s.items() if k.startswith("R_interj_out::")} if rch else {}
        res["emdash_per_1k_outside_loops"] = round(1000 * s.get("R_emdash_out", 0) / rch, 3) if rch else None
        res["shout_per_1k_outside_loops"] = round(1000 * s.get("R_shout_out", 0) / rch, 4) if rch else None
        res["excl_runs_per_1k"] = round(1000 * s.get("R_excl_runs", 0) / max(1, s.get("R_chars", 0)), 4)
        res["stutters_per_1k"] = round(1000 * n["stutters"] / max(1, s.get("R_chars", 0)), 4)
        res["caps_words_per_1k"] = round(1000 * n["caps_words"] / max(1, s.get("R_chars", 0)), 4)
        res["loop_char_share"] = round(n["R_loop_chars"] / max(1, s.get("R_chars", 0)), 4)
        res["V_interj_per_1k_visible_chars"] = {k.split("::")[1]: round(1000 * v / max(1, s.get("V_chars", 0)), 4) for k, v in s.items() if k.startswith("V_interj::")}
        res["out_tokens_p50_p90_p99_p999_max"] = (lambda x: [x[len(x) // 2], x[int(.9 * len(x))], x[int(.99 * len(x))], x[int(.999 * len(x))], x[-1]])(sorted(d["out"])) if d["out"] else None
        res["reasoning_chars_p50_p90_p99_max"] = (lambda x: [x[len(x) // 2], x[int(.9 * len(x))], x[int(.99 * len(x))], x[-1]])(sorted(d["rch"])) if d["rch"] else None
        out["models"][f"{key[0]}/{key[1]}"] = res
    # ---- corpus
    for c in ("v3", "v2"):
        t = C[c]["tot"]
        dd = C[c]["dist"]
        rc = t.get("r_chars", 0)
        res = {"turns": t["turns"], "reasoned_turns": t["turns_reasoned"], "visible_turns": t["turns_visible"],
               "conversations": t["conversations"],
               "R_loop_region_turns": t.get("R_turns_loop_region", 0), "V_loop_region_turns": t.get("V_turns_loop_region", 0),
               "R_min_window_ratio_min": min(dd.get("R_minwin_bin", {}) or [None]),
               "interj_maxwin_max": dist_max(dd.get("R_interj_maxwin", {})),
               "interj_maxwin_ge15_turns": dist_ge(dd.get("R_interj_maxwin", {}), 15),
               "emdash_maxwin_max": dist_max(dd.get("R_emdash_maxwin", {})),
               "qmark_maxwin_max": dist_max(dd.get("R_qmark_maxwin", {})),
               "caps_maxwin_max": dist_max(dd.get("R_caps_maxwin", {})),
               "cycle_max": dist_max(dd.get("R_cycle", {})),
               "stutter_max_max": dist_max(dd.get("R_stutter_max", {})),
               "nospace_bin_max": dist_max(dd.get("R_nospace_max_bin", {})),
               "bangsep_turns": t.get("R_turns_bangsep", 0), "bangsep_hist": dd.get("R_bangsep", {}),
               "turn_tokens_ge_16k_32k_64k_max": [t.get("turn_tokens_ge_16k"), t.get("turn_tokens_ge_32k"), t.get("turn_tokens_ge_64k"), t.get("turn_tokens_max")],
               "per_1k_reasoning_chars": {k.split("_", 2)[2]: round(1000 * v / rc, 4) for k, v in t.items() if k.startswith("R_interj_")},
               "emdash_per_1k": round(1000 * t.get("R_emdash", 0) / rc, 3),
               "shout_per_1k": round(1000 * t.get("R_shout", 0) / rc, 4),
               "excl_runs_per_1k": round(1000 * t.get("R_excl_runs", 0) / rc, 4),
               "stutters_per_1k": round(1000 * t.get("R_stutter", 0) / rc, 4),
               "caps_words_per_1k": round(1000 * t.get("R_caps_words", 0) / rc, 4),
               "V_interj_per_1k": {k.split("_", 2)[2]: round(1000 * v / max(1, t.get("v_chars", 0)), 4) for k, v in t.items() if k.startswith("V_interj_")},
               "ctrl": {k: v for k, v in t.items() if "_ctrl" in k},
               "foreign": {k: v for k, v in t.items() if "foreign" in k or "cjkpunct" in k},
               "garbage": {k: v for k, v in t.items() if "garbage" in k},
               "thinktag_turns": {k: v for k, v in t.items() if "thinktag" in k},
               "by_source": {s: {"turns": a.get("turns"), "reasoned": a.get("turns_reasoned"),
                                 "foreign_spont_turns": a.get("R_turns_foreign_spont", 0),
                                 "garbage_spont_turns": a.get("R_turns_garbage_spont", 0),
                                 "ctrl_spont_turns": a.get("R_turns_ctrl_spont", 0),
                                 "bangsep_turns": a.get("R_turns_bangsep", 0),
                                 "shout_per_1k": round(1000 * a.get("R_shout", 0) / max(1, a.get("r_chars", 1)), 4),
                                 "emdash_per_1k": round(1000 * a.get("R_emdash", 0) / max(1, a.get("r_chars", 1)), 3)}
                             for s, a in C[c]["src"].items()},
               "dist": {m: h for m, h in dd.items() if m in ("R_interj_maxwin", "R_emdash_maxwin", "R_qmark_maxwin", "R_caps_maxwin", "R_cycle", "R_stutter_max", "R_bangsep", "R_nospace_max_bin")}}
        out["corpus"][c] = res
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n")
    ex = {k: dict(v) for k, v in examples.items()}
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")
    print("thresholds", thr)
    print("wrote summary.json, examples.json")


if __name__ == "__main__":
    main()
