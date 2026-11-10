#!/usr/bin/env python3
"""verify-1: foreign-script runs (non-Latin letters) in reasoning that do not occur in the session context seen so far
(task prompt + every tool output earlier in the export, compaction summaries' reasoning included), per rollout, from
each final-attempt export.json. Two variants: any non-Latin script; CJK/kana/hangul only. Stdlib only.
    python3 -I -B foreign.py MODEL... -> out/foreign.txt"""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = Path(__file__).resolve().parent
FOREIGN = re.compile("[Ͱ-ϿЀ-ԯ԰-֏֐-׿؀-ۿ܀-ࣿऀ-෿฀-๿຀-࿿က-႟Ⴀ-ჿᄀ-ᇿሀ-᎟぀-ヿ㄀-ㄯ㄰-㆏ㆠ-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿ｦ-ﾟ]+")
CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯]")
def scan(a):
    try:
        exp = json.loads((Path(a) / "export.json").read_text())
    except (OSError, ValueError):
        return None
    ctx = []
    anyf = cjk = False
    calls = 0
    for m in exp.get("messages") or []:
        info = m.get("info") or {}
        parts = m.get("parts") or []
        if info.get("role") == "user":
            ctx += [p.get("text") or "" for p in parts if p.get("type") == "text"]
            continue
        if info.get("role") != "assistant":
            continue
        text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        s = text.lstrip()
        r = s[7:].split("</think>", 1)[0] if s.startswith("<think>") else ""
        r = "".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + r
        runs = [x for x in FOREIGN.findall(r) if not (len(x) < 3 and all("\u0370" <= ch <= "\u03ff" for ch in x))]
        if runs:
            c = "\n".join(ctx)
            sp = [x for x in runs if x not in c]
            if sp:
                anyf = True
                calls += 1
                cjk |= any(CJK.search(x) for x in sp)
        for p in parts:
            if p.get("type") == "tool":
                st = p.get("state") or {}
                o = st.get("output") if isinstance(st.get("output"), str) else ((st.get("metadata") or {}).get("output") or "")
                ctx.append(o if isinstance(o, str) else "")
    return anyf, cjk, calls
def main():
    lines = []
    for model in sys.argv[1:]:
        ads = [r["attempt_dir"] for f in sorted(TA.glob(f"results/{model}-group-*.json")) for rs in json.loads(f.read_text())["records"].values() for r in rs]
        with Pool(12) as pool:
            rows = [x for x in pool.map(scan, ads, chunksize=2) if x]
        lines.append(f"[Greek runs < 3 chars skipped] {model}: rollouts {len(rows)} with a spontaneous non-Latin run in reasoning {sum(r[0] for r in rows)}; CJK-only {sum(r[1] for r in rows)}; calls {sum(r[2] for r in rows)}")
    (HERE / "out/foreign.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
