#!/usr/bin/env python3
"""Extra think-token / runaway / session-length counts per DeepSWE run (stdlib only; read-only on the runs).

usage: thinkx.py --json OUT NAME=<classifier all.jsonl> [...]

Turn parsing is think_watch.py's (assistant messages without compaction summaries; reasoning = reasoning parts + the
leading <think> block of the text, analyze.py split_think). Per run:
  reasoning markup split: turns whose reasoning holds tool-call markup (analyze.py RE_TOOLCALL; "headless" = no
      <tool_call> opener) vs a special token (RE_SPECIAL_TOK, e.g. <|im_end|>), and which special tokens
  unclosed think: turns whose text starts a <think> block that never closes (visible as such; a tool call the lenient
      parser rescued from an unclosed think is NOT visible -- it looks like a closed one)
  <=1-token turns anywhere in a session (OpenCode ends a session at the first turn without a tool call, so these are
      final turns)
  capped calls: assistant turns with >= 131,072 output tokens (= analyze.py n_calls_at_cap); a capped call is a
      "repetition loop" when zlib compresses the last 20,000 characters of its reasoning+text to < 10% (natural text
      and code: ~25-45%); location = inside the reasoning (the think block is still open at the cut or closes only in
      the last 20 characters) or visible text
  session length: median model calls (analyze.py f.n_assistant, incl. compaction summaries), median think_watch turns,
      median / mean output tokens per rollout (usage_output_tokens = Gym response usage), median wall seconds
  3 h kills: classifier N1_TIMEOUT_3H split by wall time (>= 10,700 s = the 10,800 s agent cap; shorter = a session
      that died with 'signal: killed' for another reason, e.g. the model killed its own OpenCode process)
"""
import argparse, json, re, statistics, zlib
from collections import Counter
from pathlib import Path

RE_TOOLCALL = re.compile(r"(<tool_call>|</tool_call>|<function=|</function>|<parameter=)")  # analyze.py
RE_SPECIAL_TOK = re.compile(r"<\|[A-Za-z_]+\|>")
CAP = 131072


def split_think(text):
    s = text.strip()
    if not s.startswith("<think>"):
        return None, s
    if "</think>" not in s:
        return s[len("<think>"):], None
    inner, after = s[len("<think>"):].split("</think>", 1)
    return inner, after.strip()


def turns_of(path):
    data = json.loads(Path(path).read_text())
    out = []
    for message in data.get("messages") or []:
        info = message.get("info") or {}
        parts = message.get("parts") or []
        if info.get("role") != "assistant" or info.get("summary") is True or info.get("mode") == "compaction":
            continue
        text = "".join(p.get("text") or "" for p in parts
                       if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored"))
        inner, visible = split_think(text)
        reasoning = ("".join(p.get("text") or "" for p in parts if p.get("type") == "reasoning") + (inner or "")).strip()
        tok = (info.get("tokens") or {}).get("output")
        out.append({"text": text, "reasoning": reasoning, "visible": (visible or "").strip(),
                    "unclosed": inner is not None and "</think>" not in text,
                    "tools": sum(1 for p in parts if p.get("type") == "tool"),
                    "out": tok if isinstance(tok, (int, float)) else None, "finish": info.get("finish")})
    return out


def loop_info(turn):
    body = turn["text"] if turn["text"] else turn["reasoning"]
    tail = body[-20000:].encode()
    ratio = len(zlib.compress(tail)) / max(1, len(tail))
    s = turn["text"].strip()
    in_reasoning = (s.startswith("<think>") and ("</think>" not in s or s.rfind("</think>") >= len(s) - 20)) or (not s and bool(turn["reasoning"]))
    return ratio, in_reasoning


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--json", required=True)
    args = ap.parse_args()
    out = {}
    for item in args.runs:
        name, path = item.split("=", 1)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        c = Counter(); special = Counter(); ratios = []; loop_rows = []
        calls, tw_turns, out_tok, walls = [], [], [], []
        for r in rows:
            f = r.get("f") or {}
            if isinstance(r.get("usage_output_tokens"), (int, float)):
                out_tok.append(r["usage_output_tokens"])
            if r.get("wall_s"):
                walls.append(r["wall_s"])
            if not r.get("opencode_results_fpath") or not r.get("opencode_export_found"):
                continue
            try:
                turns = turns_of(r["opencode_results_fpath"])
            except (OSError, ValueError):
                c["export_unreadable"] += 1
                continue
            c["exports"] += 1
            if f.get("n_assistant"):
                calls.append(f["n_assistant"])
            tw_turns.append(len(turns))
            has_tc = has_sp = has_unclosed = has_loop = False
            for t in turns:
                c["turns"] += 1
                tc = bool(RE_TOOLCALL.search(t["reasoning"]))
                sp = RE_SPECIAL_TOK.findall(t["reasoning"])
                if tc:
                    c["tc_markup_turns"] += 1; has_tc = True
                    c["tc_headless_turns" if "<tool_call>" not in t["reasoning"] else "tc_opener_turns"] += 1
                if sp:
                    c["special_turns"] += 1; has_sp = True
                    special.update(set(sp))
                if t["unclosed"]:
                    c["unclosed_turns"] += 1; has_unclosed = True
                if t["out"] is not None and t["out"] <= 1:
                    c["le1_token_turns"] += 1
                if t["out"] is not None and t["out"] >= CAP:
                    c["capped_calls"] += 1
                    ratio, in_r = loop_info(t)
                    ratios.append(round(ratio, 4))
                    if ratio < 0.10:
                        c["loop_calls"] += 1; has_loop = True
                        c["loop_calls_in_reasoning" if in_r else "loop_calls_visible"] += 1
            c["tc_markup_rollouts"] += has_tc; c["special_rollouts"] += has_sp; c["unclosed_rollouts"] += has_unclosed
            if has_loop:
                c["loop_rollouts"] += 1
                c["loop_rollouts_resolved"] += (r.get("reward") or 0) >= 1
                c["loop_rollouts_empty"] += bool(r["empty_patch"])
                c["loop_rollouts_final_cut"] += r.get("subcode") == "E3a_cap_cut"
                loop_rows.append({"task_id": r["task_id"], "ri": r["ri"], "shard": r["shard"], "code": r["code"],
                                  "subcode": r.get("subcode"), "reward": r.get("reward")})
        n1 = [r for r in rows if r["code"] == "N1_TIMEOUT_3H"]
        res = dict(c)
        res.update({
            "rollouts": len(rows), "special_tokens": dict(special.most_common()),
            "capped_call_zlib_ratios": sorted(ratios), "loop_rows": loop_rows,
            "median_model_calls": statistics.median(calls) if calls else None,
            "median_turns": statistics.median(tw_turns) if tw_turns else None,
            "median_output_tokens": statistics.median(out_tok) if out_tok else None,
            "mean_output_tokens": round(statistics.mean(out_tok)) if out_tok else None,
            "median_wall_s": statistics.median(walls) if walls else None,
            "n1": len(n1), "n1_at_cap": sum((r.get("wall_s") or 0) >= 10700 for r in n1),
            "n1_short": [{"task_id": r["task_id"], "ri": r["ri"], "wall_s": r.get("wall_s")} for r in n1 if (r.get("wall_s") or 0) < 10700],
            "self_kill_like": [{"task_id": r["task_id"], "ri": r["ri"], "code": r["code"], "subcode": r.get("subcode"), "wall_s": r.get("wall_s")}
                               for r in rows if r["code"] in ("N5_SELF_KILL", "N9_OTHER_NF") or (r["code"] == "N1_TIMEOUT_3H" and (r.get("wall_s") or 0) < 10700)],
        })
        out[name] = res
        print(f"{name:10s} exports {c['exports']} turns {c['turns']} | tc-markup turns {c['tc_markup_turns']}/{c['tc_markup_rollouts']} rollouts (headless {c['tc_headless_turns']}) "
              f"| special-token turns {c['special_turns']}/{c['special_rollouts']} {dict(special.most_common(4))} | unclosed think turns {c['unclosed_turns']}/{c['unclosed_rollouts']} "
              f"| <=1-tok turns {c['le1_token_turns']} | capped calls {c['capped_calls']}, loops {c['loop_calls']} (in reasoning {c['loop_calls_in_reasoning']}), loop rollouts {c['loop_rollouts']} "
              f"(resolved {c['loop_rollouts_resolved']}, empty {c['loop_rollouts_empty']}, final cut {c['loop_rollouts_final_cut']}) "
              f"| median calls {res['median_model_calls']} turns {res['median_turns']} out tok {res['median_output_tokens']} wall {res['median_wall_s']} | N1 {res['n1']} at cap {res['n1_at_cap']}")
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
