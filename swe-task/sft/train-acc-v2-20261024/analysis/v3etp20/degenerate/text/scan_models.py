#!/usr/bin/env python3
"""Text-level degeneration detectors (textlib.py) on the models' own calls, DeepSWE + train-acc (read-only).

usage: python -I scan_models.py [--workers 16] [--only deepswe|trainacc] [--models a,b]

Rollouts:
  deepswe   113 tasks x 5 per run; rows = the empty-patch classifier's all.jsonl (the rows thinkx.py / think_watch.py
            use); export = opencode_results_fpath
  trainacc  50 training tasks x 5 per model; rows = results/<model>-group-*.json records (final attempts); export =
            <attempt_dir>/export.json
Calls are read as think_watch.py / thinkx.py do: every assistant message is one model call; reasoning R = 'reasoning'
parts + the leading <think>...</think> block of the text (all of it when unclosed); visible V = text after </think>.
Compaction summaries (info.summary / mode compaction) are kept, flagged comp=True, and reported apart.
Context for the "copied" checks = user-message texts + tool outputs/errors that precede the call.
Per-call minimal tuple (calls[]): call index (-1 = compaction summary), R chars, V chars, output tokens, finish (0 tool-calls,
1 stop, 2 length, 3 other), prompt tokens, R tokens (tokenizer; -1 if R has < 16,384 UTF-8 bytes, i.e. surely < 16,384 tokens), R min window ratio, R loop chars,
R interjections, R max interjections per 1,000 chars (outside loops), R em dashes, tool parts, V min window ratio,
V loop chars, V tokens, duplicate-of-previous-call flag, R em-dash max window, R '?' max window, R all-caps words,
R bang-separator run, R stutters, R longest no-space run, R interjection max-window zlib ratio, R em-dash max-window
zlib ratio, R shouting words, R '!!' runs, R sentence-cycling count, V max interjections per 1,000 chars.
Writes out/models_rollouts.jsonl (one record per rollout: per-call minimal tuples + sums) and
out/models_calls_flagged.jsonl (full feature dicts of calls with any finding).

2026-11-04: copied from sftv3-etp-20261030/degenerate-behaviors-20261031/text into train-acc-v2-20261024/analysis/v3etp20/
degenerate/text (OUT = ./out); + v3etp20 in TRAINACC. Run here with --only trainacc --models sc14,etp20,v3etp5,v3etp10,v3etp20.
"""
import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import textlib as T  # noqa: E402

E = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1")
A = E / "analysis"
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
TOKENIZER = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf/tokenizer.json")
OUT = HERE / "out"
DEEPSWE = {
    "base": A / "empty-patch-20261018/base-step0-rep5-out131k/all.jsonl",
    "base-rerun": A / "sftv3-etp-20261030/classifier/base-rerun/all.jsonl",
    "v2-5ep": A / "lenient-think-20261023/empty-patch/v2-lenient/all.jsonl",
    "v2-10ep": A / "lenient-think-20261023/empty-patch/v2-10ep-lenient/all.jsonl",
    "v2-20ep": A / "lenient-think-20261023/empty-patch/v2-20ep-lenient/all.jsonl",
    "etp5": A / "etp-sft-20261027/classifier/etp5/all.jsonl",
    "etp10": A / "etp-sft-20261027/classifier/etp10/all.jsonl",
    "etp20": A / "etp-sft-20261027/classifier/etp20/all.jsonl",
    "v3etp5": A / "sftv3-etp-20261030/classifier/v3etp5/all.jsonl",
    "v3etp10": A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
}
TRAINACC = ("sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20")  # 2026-11-04: + v3etp20
FIN = {"tool-calls": 0, "stop": 1, "length": 2}
_TOK = None


def ntok(text):
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER))
    return len(_TOK.encode(text, add_special_tokens=False).ids) if text else 0


class Ctx:
    def __init__(self):
        self.parts, self.joined, self.dirty = [], "", False

    def add(self, s):
        if s:
            self.parts.append(s)
            self.dirty = True

    def has(self, s):
        if self.dirty:
            self.joined = "\n".join(self.parts)
            self.dirty = False
        return s in self.joined


def flagged(f, where):
    if not f.get("chars"):
        return False
    if f.get("regions") or f.get("ctrl") or f.get("thinktag") or f.get("foreign") or f.get("cjkpunct") or f.get("garbage"):
        return True
    cr = f.get("charrun")
    if cr and (cr["len"] >= 100 or (not cr["ch"].isascii() and cr["len"] >= 20)):
        return True
    if f.get("spacedrun") or f.get("cycle", 0) >= 3:
        return True
    if f.get("interj_maxwin", 0) >= 10 or f.get("shout", 0) >= 3 or f.get("excl_runs", 0) >= 5:
        return True
    if f.get("tail") is not None and f["tail"] < 0.15:
        return True
    if f.get("bangsep") or f.get("stutter_max", 0) >= 3 or f.get("nospace_max", 0) >= 300:
        return True
    if f.get("emdash_maxwin", 0) >= 40 or f.get("qmark_maxwin", 0) >= 25 or f.get("caps_maxwin", 0) >= 25:
        return True
    if where == "V" and f.get("interj"):
        return True
    return False


def process(job):
    dataset, model, rows = job
    recs, flagged_calls = [], []
    for row in rows:
        try:
            data = json.loads(Path(row["export"]).read_text())
        except (OSError, ValueError):
            recs.append({"dataset": dataset, "model": model, "id": row["id"], "unreadable": True})
            continue
        ctx = Ctx()
        calls, sums = [], Counter()
        ci = 0
        prev_hash = None
        dup_consec = 0
        for message in data.get("messages") or []:
            info = message.get("info") or {}
            parts = message.get("parts") or []
            role = info.get("role")
            if role == "user":
                ctx.add("".join(p.get("text") or "" for p in parts if p.get("type") == "text"))
                continue
            if role != "assistant":
                continue
            comp = info.get("summary") is True or info.get("mode") == "compaction"
            text = "".join(p.get("text") or "" for p in parts
                           if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
            inner, visible = T.split_think(text)
            R = ("".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or "")).strip()
            V = (visible or "").strip()
            tok = info.get("tokens") or {}
            cache = tok.get("cache") or {}
            prompt = (tok.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
            out = tok.get("output") if isinstance(tok.get("output"), (int, float)) else -1
            fin = info.get("finish")
            tools = sum(1 for p in parts if p.get("type") == "tool")
            fr = T.text_features(R, ctx.has, ntok=ntok)
            fv = T.text_features(V, ctx.has)
            # exact token counts only where they can reach 16,384 (tokens <= UTF-8 bytes); -1 = fewer than 16,384
            rtok = ntok(R) if len(R) >= 16384 and len(R.encode()) >= 16384 else -1
            vtok = ntok(V) if len(V) >= 16384 and len(V.encode()) >= 16384 else -1
            h = T.md5(R) if len(R) >= 200 else None
            dup = h is not None and h == prev_hash
            prev_hash = h
            if not comp:
                dup_consec += dup
            # minimal tuple for every call
            calls.append([ci if not comp else -1, len(R), len(V), out, FIN.get(fin, 3), prompt, rtok,
                          fr.get("min_win", -1), fr.get("loop_chars", 0), sum((fr.get("interj") or {}).values()),
                          fr.get("interj_out_maxwin", fr.get("interj_maxwin", 0)), fr.get("emdash", 0), tools,
                          fv.get("min_win", -1), fv.get("loop_chars", 0), vtok, int(dup),
                          fr.get("emdash_maxwin", 0), fr.get("qmark_maxwin", 0), fr.get("caps_words", 0),
                          fr.get("bangsep", 0), fr.get("stutter", 0), fr.get("nospace_max", 0),
                          fr.get("interj_out_maxwin_ratio", fr.get("interj_maxwin_ratio", -1)),
                          fr.get("emdash_maxwin_ratio", -1), fr.get("shout", 0), fr.get("excl_runs", 0),
                          fr.get("cycle", 0), fv.get("interj_maxwin", 0)])
            if not comp:
                for k, v in (fr.get("interj_out") or fr.get("interj") or {}).items():
                    sums[f"R_interj_out::{k}"] += v
                sums["R_chars_out"] += fr.get("chars_out", len(R))
                sums["R_chars"] += len(R)
                sums["R_emdash_out"] += fr.get("emdash_out", fr.get("emdash", 0))
                sums["R_shout_out"] += fr.get("shout_out", fr.get("shout", 0))
                sums["R_excl_runs"] += fr.get("excl_runs", 0)
                sums["V_chars"] += len(V)
                for k, v in (fv.get("interj") or {}).items():
                    sums[f"V_interj::{k}"] += v
            fl_r, fl_v = flagged(fr, "R"), flagged(fv, "V")
            if fl_r or fl_v or rtok >= 16384 or dup:
                rec = {"dataset": dataset, "model": model, "id": row["id"], "task_id": row["task_id"], "rep": row["rep"],
                       "call": ci if not comp else None, "comp": comp, "out": out, "finish": fin, "prompt": prompt,
                       "tools": tools, "rtok": rtok, "vtok": vtok, "unclosed": inner is not None and "</think>" not in text,
                       "dup_prev": dup}
                if fl_r or rtok >= 16384:
                    rec["R"] = fr
                if fl_v:
                    rec["V"] = fv
                flagged_calls.append(rec)
            if not comp:
                ci += 1
            for p in parts:
                if p.get("type") == "tool":
                    st = p.get("state") or {}
                    o = st.get("output")
                    ctx.add(o if isinstance(o, str) else json.dumps(o) if o is not None else "")
                    er = st.get("error")
                    ctx.add(er if isinstance(er, str) else "")
        recs.append({"dataset": dataset, "model": model, "id": row["id"], "task_id": row["task_id"], "rep": row["rep"],
                     "reward": row.get("reward"), "empty_patch": row.get("empty_patch"), "code": row.get("code"),
                     "subcode": row.get("subcode"), "status": row.get("status"), "n_calls": ci,
                     "n_comp": sum(1 for c in calls if c[0] == -1), "dup_consec": dup_consec,
                     "sums": dict(sums), "calls": calls})
    return dataset, model, recs, flagged_calls


def deepswe_rows(path):
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if not (r.get("opencode_export_found") and r.get("opencode_results_fpath")):
            continue
        rows.append({"id": f"{r['task_id']}/r{r['ri']}", "task_id": r["task_id"], "rep": r["ri"],
                     "export": r["opencode_results_fpath"], "reward": r.get("reward"),
                     "empty_patch": r.get("empty_patch"), "code": r.get("code"), "subcode": r.get("subcode")})
    return rows


def trainacc_rows(model):
    rows = []
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for rec in records:
                ex = Path(rec["attempt_dir"]) / "export.json"
                if not ex.exists():
                    continue
                rows.append({"id": f"{task_id}/{rec['slot']}", "task_id": task_id, "rep": rec["slot"],
                             "export": str(ex), "reward": rec.get("reward"), "status": rec.get("status"),
                             "empty_patch": not rec.get("patch_chars") or rec.get("patch_chars") in ("0", 0, "None")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only")
    ap.add_argument("--models")
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    want = set(args.models.split(",")) if args.models else None
    jobs = []
    if args.only in (None, "deepswe"):
        for name, path in DEEPSWE.items():
            if want and name not in want:
                continue
            rows = deepswe_rows(path)
            jobs += [("deepswe", name, rows[i:i + args.chunk]) for i in range(0, len(rows), args.chunk)]
    if args.only in (None, "trainacc"):
        for model in TRAINACC:
            if want and model not in want:
                continue
            rows = trainacc_rows(model)
            jobs += [("trainacc", model, rows[i:i + args.chunk]) for i in range(0, len(rows), args.chunk)]
    # priority: the new models and the main references first
    prio = {"v3etp5": 0, "v3etp10": 0, "base": 1, "sc14": 1, "etp10": 1, "etp20": 1, "etp5": 2, "base-rerun": 2}
    jobs.sort(key=lambda j: (prio.get(j[1], 3), j[0], j[1]))
    mode = "w"
    if args.resume:
        def valid_lines(path):
            for line in open(path):
                try:
                    yield line if line.endswith("\n") else None, json.loads(line)
                except ValueError:
                    continue

        done = set()
        path = OUT / f"models_rollouts{args.suffix}.jsonl"
        if path.exists():
            keep = []
            for line, r in valid_lines(path):
                if line is None:
                    continue
                done.add((r["dataset"], r["model"], r["id"]))
                keep.append(line)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(keep))
            tmp.replace(path)
        # drop flagged-call records of rollouts that are not complete in the rollouts file
        fpath = OUT / f"models_calls_flagged{args.suffix}.jsonl"
        if fpath.exists():
            keep = [line for line, r in valid_lines(fpath)
                    if line is not None and (r["dataset"], r["model"], r["id"]) in done]
            tmp = fpath.with_suffix(".tmp")
            tmp.write_text("".join(keep))
            tmp.replace(fpath)
        jobs = [(d, m, [r for r in rows if (d, m, r["id"]) not in done]) for d, m, rows in jobs]
        jobs = [j for j in jobs if j[2]]
        mode = "a"
        print(f"resume: {len(done)} rollouts done, {sum(len(j[2]) for j in jobs)} to go", flush=True)
    n_rec = n_fl = 0
    with open(OUT / f"models_rollouts{args.suffix}.jsonl", mode) as fr, \
            open(OUT / f"models_calls_flagged{args.suffix}.jsonl", mode) as ff, Pool(args.workers) as pool:
        for k, (dataset, model, recs, fl) in enumerate(pool.imap_unordered(process, jobs)):
            for r in fl:
                ff.write(json.dumps(r, ensure_ascii=False) + "\n")
            ff.flush()
            for r in recs:
                fr.write(json.dumps(r, ensure_ascii=False) + "\n")
            fr.flush()
            n_rec += len(recs)
            n_fl += len(fl)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} jobs, {n_rec} rollouts, {n_fl} flagged calls", flush=True)
    print(f"done: {n_rec} rollouts, {n_fl} flagged calls", flush=True)


if __name__ == "__main__":
    main()
