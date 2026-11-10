#!/usr/bin/env python3
"""reconcile (2026-11-04): targeted scan of the HTTP journal (http.jsonl) of selected slots' FINAL attempts (attempt_dir
from results/<model>-group-*.json). Read-only; stdlib only.

    /usr/bin/python3 -I -B r_http_targeted.py -> out/http_targeted.txt

  * v3etp20 3 h deadline kills (agent_error TimeoutError or session >= 10,500 s, unfinished): model calls = HTTP 200
    responses with choices; compaction calls among them (request system prompt = OpenCode's anchored-summary agent);
    median over the kills
  * v3etp10's two slots without a session export: model calls, compaction calls, action calls (tool_calls) with empty
    reasoning (None or blank)
  * v3etp20 compaction responses: tool-call markup in content / reasoning, '## Goal' in content, parsed tool_calls
"""
import collections
import json
import re
import statistics
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent / "out"
COMP = b"You are an anchored context summarization assistant"
MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|</function>|<parameter=")
EV = re.compile(rb'"event": ?"([a-z_]+)"')
CID = re.compile(rb'"call_id": ?"([0-9a-f]+)"')


def slots(model):
    out = {}
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, recs in data["records"].items():
            for r in recs:
                out[(task_id, int(r["slot"].rsplit("-r", 1)[1]))] = r
    return out


def scan(attempt_dir):
    path = Path(attempt_dir) / "http.jsonl"
    comp_ids = set()
    calls = []
    status = collections.Counter()
    first_req = None
    with open(path, "rb") as h:
        for raw in h:
            head = raw[:2000]
            m = EV.search(head)
            ev = m.group(1).decode() if m else "?"
            if ev == "request":
                c = CID.search(head).group(1).decode()
                if COMP in raw[:12000]:
                    comp_ids.add(c)
                ts = json.loads(raw).get("timestamp") if first_req is None else None
                if first_req is None:
                    first_req = ts
                continue
            if ev != "response":
                continue
            d = json.loads(raw)
            status[str(d.get("status"))] += 1
            body = d.get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = None
            if d.get("status") != 200 or not isinstance(body, dict) or not body.get("choices"):
                continue
            msg = body["choices"][0].get("message") or {}
            r = msg.get("reasoning")
            if r is None:
                r = msg.get("reasoning_content")
            content = msg.get("content") or ""
            calls.append({"call_id": d.get("call_id"), "comp": d.get("call_id") in comp_ids, "ts": d.get("timestamp"),
                          "finish": body["choices"][0].get("finish_reason"), "tools": len(msg.get("tool_calls") or []),
                          "r_empty": r is None or not str(r).strip(), "r_none": r is None,
                          "c_markup": bool(MARKUP.search(content)), "r_markup": bool(MARKUP.search(r or "")),
                          "c_goal": "## Goal" in content, "c_head": content[:120], "r_len": len(r or "")})
    return {"calls": calls, "status": dict(status), "first_req": first_req}


def main():
    L = []
    S20 = slots("v3etp20")
    # deadline kills per the report's rule: TimeoutError flag, or the slot ran >= 10,500 s without finishing; take the
    # 7 ATP budget_timeout slots listed in compare/slots_v3etp20.txt and re-count from the journal
    dk = [("syn-gomacro-cand-c01-c696c6ca", 1), ("syn-gomacro-cand-c01-c696c6ca", 3), ("syn-gomacro-cand-c02-r1b0-e8804dd9", 1),
          ("syn-gomacro-cand-c02-r1b0-e8804dd9", 4), ("syn-wasmtime-cand-c03-353ec720", 1), ("syn-wasmtime-cand-c03-353ec720", 3),
          ("syn-zsa-cand-c02-r1b1-ae1eb794", 3)]
    L.append("== v3etp20 deadline kills: model calls (HTTP 200 with choices) in the final attempt's journal")
    counts = []
    for t, rep in dk:
        r = S20[(t, rep)]
        x = scan(r["attempt_dir"])
        n = len(x["calls"])
        nc = sum(c["comp"] for c in x["calls"])
        counts.append(n)
        vf = json.loads((Path(r["attempt_dir"]) / "generation-validity-v2-r7.json").read_text()) if (Path(r["attempt_dir"]) / "generation-validity-v2-r7.json").exists() else {}
        L.append(f"  {t} r{rep}: calls {n} (compaction {nc}, other {n - nc}); agent_error {r.get('agent_error')}; status {x['status']}; "
                 f"validity {vf.get('classification') or vf.get('status') or vf.get('verdict') or list(vf)[:6]}")
    L.append(f"  sorted {sorted(counts)}; median {statistics.median(counts)}; range {min(counts)}-{max(counts)}")
    # v3etp10 no-export slots
    S10 = slots("v3etp10")
    L.append("\n== v3etp10 slots without a session export: HTTP journal")
    for t, rep in (("syn-ibis-cand-c03-r1b4-3d3b86b5", 3), ("syn-wasmtime-cand-c03-353ec720", 1)):
        r = S10[(t, rep)]
        x = scan(r["attempt_dir"])
        acts = [c for c in x["calls"] if not c["comp"] and c["tools"]]
        un = [c for c in acts if c["r_empty"]]
        L.append(f"  {t} r{rep} status {r['status']}: calls {len(x['calls'])}, compaction {sum(c['comp'] for c in x['calls'])}, "
                 f"tool-call compaction responses {sum(1 for c in x['calls'] if c['comp'] and (c['c_markup'] or c['r_markup'] or c['tools']))}, "
                 f"action calls {len(acts)}, with empty reasoning {len(un)} {[(c['r_none'], c['finish'], c['tools']) for c in un]}")
    # v3etp20 compaction responses (all slots with >= 1 compaction in the export scan)
    import gzip
    rows = [json.loads(line) for line in gzip.open(OUT / "export-v3etp20.jsonl.gz", "rt")]
    L.append("\n== v3etp20 compaction responses in the HTTP journal (slots with >= 1 compaction)")
    tot = collections.Counter()
    goal_tc = []
    for row in rows:
        if not row.get("summaries"):
            continue
        x = scan(row["attempt_dir"])
        comps = [c for c in x["calls"] if c["comp"]]
        tot["slots"] += 1
        tot["comp_200"] += len(comps)
        tcs = [c for c in comps if c["c_markup"] or c["r_markup"] or c["tools"]]
        tot["toolcall"] += len(tcs)
        tot["parsed_tool_calls"] += sum(1 for c in comps if c["tools"])
        tot["rollouts_toolcall"] += bool(tcs)
        tot["first_toolcall"] += bool(comps) and (comps[0]["c_markup"] or comps[0]["r_markup"] or comps[0]["tools"])
        tot["length_cut"] += sum(1 for c in comps if c["finish"] == "length")
        for c in tcs:
            tot["toolcall_with_goal"] += c["c_goal"]
            if c["c_goal"]:
                goal_tc.append(f"{row['task_id']} r{row['rep']}: {c['c_head'][:80]!r}")
    L.append(f"  {dict(tot)}")
    L.extend("   goal+toolcall: " + g for g in goal_tc)
    (OUT / "http_targeted.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
