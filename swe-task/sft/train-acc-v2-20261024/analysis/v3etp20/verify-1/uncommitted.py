#!/usr/bin/env python3
"""verify-1: final visible text of the edited-but-uncommitted empty-patch rollouts (my ending() = 'uncommitted'),
split into asks/defers-to-commit vs claims-done by simple regexes of my own, plus the endlib regexes (definition only).
    python3 -I -B uncommitted.py MODEL... -> out/uncommitted.txt"""
import gzip, importlib.util, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stats", HERE / "stats.py"); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
spec2 = importlib.util.spec_from_file_location("scan_export", HERE / "scan_export.py"); SE = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(SE)
MY_ASK = re.compile(r"(\?\s*$|shall i|should i|would you like|want me to|do you want|let me know|not (yet )?committed|haven'?t committed|ready to commit|before committing|uncommitted)", re.I)
MY_DONE = re.compile(r"(\bdone\b|complete[d]?\b|implemented|finished|all (?:\w+ )?tests? pass|summary of (?:the )?changes|^#+\s*summary)", re.I | re.M)
END_DEFERS = re.compile(r"(want me to (go ahead and )?commit|shall i commit|should i commit|like me to commit|me to commit|ha(ve|s)n['’]?t (been )?committed|have not committed|not (yet )?committed|nothing (is |was )?committed|no commit(s)? (yet|made|was made)|uncommitted|won['’]?t commit|will not commit|wouldn['’]?t commit|without (being asked|your|an explicit|explicit)|didn['’]?t commit|did not commit|not commit(ting)? (anything|unless|until)|ready to commit|hold(ing)? (off )?on committing|say the word|go ahead and commit|commit (whenever|when you|if you|once you)|only commit(ting)? when|commit only on|no-commit|unrequested commits|before committing|explicitly ask)", re.I)
END_ASK = re.compile(r"\?\s*$|\b(would you like|do you want|let me know|should i|shall i|please confirm|which (one|option))\b", re.I)
lines = []
for m in sys.argv[1:]:
    rows = [json.loads(l) for l in gzip.open(HERE / f"out/export-{m}.jsonl.gz", "rt")]
    unc = [r for r in rows if not r.get("patch_chars") and ST.ending(r) == "uncommitted"]
    mine = {"asks_or_defers": 0, "claims_done": 0, "other": 0}
    theirs = {"asks_or_defers": 0, "claims_done/other": 0}
    for r in unc:
        exp = json.loads(Path(r["attempt_dir"], "export.json").read_text())
        last = [x for x in exp["messages"] if (x.get("info") or {}).get("role") == "assistant" and not (x["info"].get("summary") is True or x["info"].get("mode") == "compaction")][-1]
        text = "".join(p.get("text") or "" for p in last["parts"] if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        inner, vis, unclosed = SE.split_think(text)
        v = (vis or "").strip()
        k = "asks_or_defers" if MY_ASK.search(v[-400:]) or END_DEFERS.search(v) else ("claims_done" if MY_DONE.search(v) else "other")
        mine[k] += 1
        theirs["asks_or_defers" if (END_DEFERS.search(v) or END_ASK.search(v[-300:])) else "claims_done/other"] += 1
        lines.append(f"  {m} {r['task_id'][4:34]} {r['slot']} [{k}] ...{v[-160:]!r}")
    lines.insert(0, f"{m}: uncommitted {len(unc)} mine {mine} endlib-regex split {theirs}")
(HERE / "out/uncommitted.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
