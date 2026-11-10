#!/usr/bin/env python3
"""reconcile (2026-11-04): the final visible text of every v3etp20 empty-patch slot that the DeepSWE classifier labels
E1_UNCOMMITTED (analysis/v3etp20/empty-patch/v3etp20/all.jsonl), from the slot's export.json, for a manual check of the
'asks or defers to commit' vs 'claims done' split. Read-only; stdlib only.

    /usr/bin/python3 -I -B r_uncommitted.py -> out/uncommitted_v3etp20.txt
"""
import json
import re
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
ASK = re.compile(r"(shall i commit|should i commit|want me to commit|would you like (?:me )?to commit|let me know if you(?:'d like| want)"
                 r"|ready to (?:be )?commit|before committing|not (?:yet )?committed|haven't committed|have not committed|uncommitted"
                 r"|you can commit|can be committed|to commit (?:it|these|the changes)|commit when|i can commit|ready for (?:review|commit))", re.I)


def main():
    att = {}
    for path in sorted(TA.glob("results/v3etp20-group-*.json")):
        for t, recs in json.loads(path.read_text())["records"].items():
            for r in recs:
                att[(t, int(r["slot"].rsplit("-r", 1)[1]))] = r["attempt_dir"]
    L = []
    n_ask = 0
    rows = [json.loads(x) for x in open(TA / "analysis/v3etp20/empty-patch/v3etp20/all.jsonl")]
    for c in rows:
        if c["code"] != "E1_UNCOMMITTED":
            continue
        d = json.loads((Path(att[(c["task_id"], c["ri"])]) / "export.json").read_text())
        last = [m for m in d["messages"] if (m.get("info") or {}).get("role") == "assistant"][-1]
        text = "".join(p.get("text") or "" for p in last["parts"] if p.get("type") == "text")
        vis = text.split("</think>", 1)[1] if "</think>" in text else text
        ask = bool(ASK.search(vis))
        n_ask += ask
        L.append(f"== {c['task_id']} r{c['ri']} {c.get('subcode')} asks/defers-regex {ask}\n   ...{vis.strip()[-420:]!r}")
    L.append(f"E1 rollouts {sum(1 for c in rows if c['code'] == 'E1_UNCOMMITTED')}; final text asks or defers to commit (regex) {n_ask}")
    (OUT / "uncommitted_v3etp20.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
