#!/usr/bin/env python3
"""kimi-rollouts-v4 final summary (2026-10-29): per-rollout behaviour features of this batch (194 v2 tasks x 5, stock
prompt, 262,144 window), of kimi-rollouts-v3's sentence arm and of the earlier 1M Kimi-K3 rollouts of the 50 sel50 tasks
(teacher-rollouts-v2's features.py, whose turn / reasoning definitions are kept unchanged so the numbers compare with
v3's phase-1 report and the earlier teacher reports). Read-only on all inputs (v3 and the synth runs included).

    <python with tokenizers> -I -B features.py [--sources kimi4-all kimi3 kimi-1m] [--workers 8]
      -> analysis/features/<source>.jsonl (one line per rollout)

v4 final changes (the 2026-10-25 code, byte-identical to v3's, is in backup-pre-final-20261029/features.py; this file
follows kimi-rollouts-v5's final features.py):
  - default sources kimi4-all (every graded v4 rollout once: 970 first attempts + the broken-reply re-runs), kimi3
    (v3's sentence arm, sel50, 250 slots) and kimi-1m (the earlier 1M Kimi-K3 rollouts of the 50 sel50 tasks, 250 slots);
    outputs go to analysis/features/;
  - every field of the source slot is kept (taskset, role, final, campaign, ended_by_broken_reply, ...);
  - the session export is read from the slot's export_dir (a regrade attempt's generation), else the attempt;
  - new fields: broken_text_parts (assistant text / reasoning / tool-call input parts of the MAIN session containing
    "[PAD]" or a chat-template special token, degenerate.py's SPECIAL), broken_tool_outputs (tool outputs containing
    them, e.g. a task subagent whose final reply was broken), prompt_max_main (max over main-session steps of
    input + cache-read tokens), context_max_main (input + cache read + output), steps_over_242144 / steps_over_200k
    (main-session steps whose prompt passed OpenCode's compaction point at limit.input 262,144 - 20,000 / 200k),
    message_errors (assistant message error names), final_finish already existed; and (as window.py)
    compaction_trigger_prompts (the prompt of the step before each compaction), turns_after_last_compaction,
    action_turns_after_last_compaction, commit_after_last_compaction, ended_on_compaction (no step after it).
Turn = one assistant message of the main session = one model call (OpenCode step), excluding OpenCode compaction
summaries (mode "compaction"), which are counted separately. As in diagnostics/r3-token-breakdown-20261023/
think_format_compare.py, the assistant text of a turn is "<think>REASONING</think>VISIBLE" (ATP's model server puts the
call's reasoning, hosted reasoning_content or local <think>, in front of the content; a call without reasoning has no
think block); "reasoning" parts (none expected) also count as reasoning. Per turn: reasoning (non-empty after strip),
its chars and tokens (base Nemotron = sc14 tokenizer, the same ruler for every source), visible text chars, tool calls.
Turn classes: action (>= 1 tool call) with / without reasoning; no-tool turns: reasoning-only (no visible text),
text (visible text, with or without reasoning), empty.
Episode traits: git commit / branch commands (bash tool input), the last commit's position among the tool calls, final
turn (no tool call) and its visible text, todowrite / task (subagent) / webfetch / skill calls, compactions, hub
token counts (step-finish tokens: input, cache read, output, reasoning; main session only), agent wall time.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(HERE))
import sources  # noqa: E402

OUT_DIR = HERE / "analysis" / "features"
TOKENIZER = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf/tokenizer.json")
THINK = re.compile(r"<think>(.*?)</think>(.*)", re.S)
COMMIT = re.compile(r"\bgit\b[^\n|;&]*\bcommit\b")
BRANCH = re.compile(r"\bgit\b[^\n|;&]*\b(checkout\s+-[bB]|switch\s+-[cC]|switch\s+--create|branch\s+(?!-)[\w./-]+)")
COMPACT_AT = 262144 - 20000
_TOK = None


def _degenerate_special():
    """degenerate.py's SPECIAL pattern (loaded by path, as brokenreply.py loads that module)."""
    spec = importlib.util.spec_from_file_location("kr4_degenerate_for_features", HERE / "degenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPECIAL


SPECIAL = _degenerate_special()


def broken_text(text) -> bool:
    text = text if isinstance(text, str) else json.dumps(text) if text is not None else ""
    return "[PAD]" in text or bool(SPECIAL.search(text))


def ntok(texts):
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER))
    texts = [t for t in texts]
    if not texts:
        return []
    return [len(e.ids) for e in _TOK.encode_batch(texts, add_special_tokens=False)]


def split_turn(message):
    """(reasoning, visible, tool parts, open_think) of one assistant message."""
    parts = message.get("parts") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text" and not p.get("synthetic")
                   and not p.get("ignored"))
    reasoning_parts = "".join(p.get("text", "") for p in parts if p.get("type") == "reasoning")
    tools = [p for p in parts if p.get("type") == "tool"]
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        match = THINK.fullmatch(stripped)
        if match:
            return reasoning_parts + match.group(1), match.group(2), tools, False
        return reasoning_parts + stripped[len("<think>"):], "", tools, True  # unclosed think block
    return reasoning_parts, text, tools, False


def command_of(tool):
    state = tool.get("state") or {}
    value = (state.get("input") or {}).get("command")
    return value if isinstance(value, str) else ""


def features(slot):
    out = {key: value for key, value in slot.items() if key not in ("all_attempts", "index_row")}
    if not out.get("campaign"):
        out["campaign"] = Path(slot["origin"]).stem if slot.get("origin") and slot["origin"].endswith(".json") else None
    attempt = Path(slot["attempt_dir"]) if slot.get("attempt_dir") else None
    generation = Path(slot.get("export_dir") or attempt) if attempt else None
    try:
        export = json.loads((generation / "export.json").read_text()) if generation else None
    except (OSError, ValueError):
        export = None
    terminal = sources.read_json(attempt / "terminal.json", {}) if attempt else {}
    out["agent_error"] = out.get("agent_error") or terminal.get("opencode_run_error_type") or terminal.get("failure_reason")
    out["evaluation_completed"] = terminal.get("evaluation_completed")
    patch_file = next((d / "model.patch" for d in (attempt, generation) if d and (d / "model.patch").exists()), None)
    out["patch_chars"] = patch_file.stat().st_size if patch_file else 0
    if not export:
        out["export"] = False
        return out
    out["export"] = True
    messages = export.get("messages") or []
    turns, compactions, user_messages = [], 0, 0
    times = []
    hub = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "reasoning": 0}
    prompts, contexts, message_errors = [], [], []
    broken_parts = broken_outputs = 0
    trigger_prompts, compaction_at = [], []  # prompt of the step before each compaction; len(turns) at it
    previous_prompt = None
    for message in messages:
        info = message.get("info") or {}
        created = (info.get("time") or {}).get("created")
        completed = (info.get("time") or {}).get("completed")
        times += [t for t in (created, completed) if isinstance(t, (int, float))]
        if info.get("role") == "user":
            user_messages += 1
            continue
        if info.get("role") != "assistant":
            continue
        tokens = info.get("tokens") or {}
        hub["input"] += tokens.get("input") or 0
        hub["output"] += tokens.get("output") or 0
        hub["reasoning"] += tokens.get("reasoning") or 0
        hub["cache_read"] += (tokens.get("cache") or {}).get("read") or 0
        hub["cache_write"] += (tokens.get("cache") or {}).get("write") or 0
        error = info.get("error")
        if error:
            message_errors.append((error.get("name") or "error") if isinstance(error, dict) else str(error)[:80])
        for part in message.get("parts") or []:
            kind = part.get("type")
            if kind in ("text", "reasoning") and broken_text(part.get("text") or ""):
                broken_parts += 1
            elif kind == "tool":
                state = part.get("state") or {}
                broken_parts += broken_text(state.get("input"))
                broken_outputs += broken_text(state.get("output") or "")
        if info.get("mode") == "compaction" or info.get("summary") is True:
            compactions += 1
            trigger_prompts.append(previous_prompt)
            compaction_at.append(len(turns))
            continue
        prompt = (tokens.get("input") or 0) + ((tokens.get("cache") or {}).get("read") or 0)
        previous_prompt = prompt
        prompts.append(prompt)
        contexts.append(prompt + (tokens.get("output") or 0))
        reasoning, visible, tools, open_think = split_turn(message)
        turns.append({"reasoning": reasoning if reasoning.strip() else "", "visible": visible.strip(),
                      "tools": [t.get("tool") for t in tools], "commands": [command_of(t) for t in tools],
                      "errors": sum((t.get("state") or {}).get("status") == "error" or bool(re.search(
                          r"(?m)^(Exit code [1-9]|.*\b(FAIL|FAILED|Traceback|panic:|error:))", str((t.get("state") or {}).get("output") or "")[-4000:]))
                                    for t in tools),
                      "open_think": open_think, "finish": info.get("finish")})
    counts = ntok([t["reasoning"] for t in turns if t["reasoning"]])
    it = iter(counts)
    for t in turns:
        t["reasoning_tokens"] = next(it) if t["reasoning"] else 0
    n = len(turns)
    action = [t for t in turns if t["tools"]]
    notool = [t for t in turns if not t["tools"]]
    calls = [name for t in turns for name in t["tools"]]
    commands = [c for t in turns for c in t["commands"]]
    commit_positions = [i for i, c in enumerate(commands) if COMMIT.search(c)]
    final = turns[-1] if turns else None
    out.update({
        "turns": n, "compactions": compactions, "user_messages": user_messages,
        "action_turns": len(action), "notool_turns": len(notool),
        "turns_with_reasoning": sum(bool(t["reasoning"]) for t in turns),
        "action_with_reasoning": sum(bool(t["reasoning"]) for t in action),
        "action_without_reasoning": sum(not t["reasoning"] for t in action),
        "action_with_visible": sum(bool(t["visible"]) for t in action),
        "action_bare": sum(not t["reasoning"] and not t["visible"] for t in action),
        "reasoning_only_turns": sum(bool(t["reasoning"]) and not t["visible"] for t in notool),
        "text_turns": sum(bool(t["visible"]) for t in notool),
        "text_turns_without_reasoning": sum(bool(t["visible"]) and not t["reasoning"] for t in notool),
        "empty_turns": sum(not t["reasoning"] and not t["visible"] for t in notool),
        "open_think_turns": sum(t["open_think"] for t in turns),
        "tool_calls": len(calls), "parallel_turns": sum(len(t["tools"]) >= 2 for t in turns),
        "max_calls_per_turn": max((len(t["tools"]) for t in turns), default=0),
        "calls_hist": {str(k): sum(len(t["tools"]) == k for t in turns)
                       for k in sorted({len(t["tools"]) for t in turns})},
        "tools": {name: calls.count(name) for name in sorted(set(calls))},
        "reasoning_chars": [len(t["reasoning"]) for t in turns if t["reasoning"]],
        "reasoning_tokens": [t["reasoning_tokens"] for t in turns if t["reasoning"]],
        "reasoning_tokens_total": sum(t["reasoning_tokens"] for t in turns),
        "visible_chars": [len(t["visible"]) for t in turns if t["visible"]],
        "visible_chars_total": sum(len(t["visible"]) for t in turns),
        "first_turn_reasoning": bool(turns and turns[0]["reasoning"]),
        "first_action_reasoning": bool(action and action[0]["reasoning"]),
        "all_actions_reasoned": bool(action) and all(t["reasoning"] for t in action),
        "longest_unreasoned_action_run": max_run([not t["reasoning"] for t in action]),
        "git_commit_calls": len(commit_positions),
        "last_commit_calls_from_end": (len(commands) - 1 - commit_positions[-1]) if commit_positions else None,
        "branch_created": any(BRANCH.search(c) for c in commands),
        "final_notool": bool(final and not final["tools"]),
        "final_visible_chars": len(final["visible"]) if final and not final["tools"] else 0,
        "final_reasoning": bool(final and final["reasoning"]),
        "final_finish": final["finish"] if final else None,
        "reasoned_by_first_tool": {name: [sum(1 for t in action if t["tools"][0] == name and t["reasoning"]),
                                          sum(1 for t in action if t["tools"][0] == name)]
                                   for name in sorted({t["tools"][0] for t in action})},
        "reasoning_openers": dict(collections.Counter(" ".join(re.findall(r"[A-Za-z']+", t["reasoning"][:60])[:2])
                                                      for t in turns if t["reasoning"]).most_common(10)),
        "unreasoned_after_reasoned": sum(1 for a, b in zip(action, action[1:]) if a["reasoning"] and not b["reasoning"]),
        "reasoned_after_prev": {"prev_reasoned": [sum(1 for a, b in zip(action, action[1:]) if a["reasoning"] and b["reasoning"]),
                                                  sum(1 for a in action[:-1] if a["reasoning"])],
                                "prev_unreasoned": [sum(1 for a, b in zip(action, action[1:]) if not a["reasoning"] and b["reasoning"]),
                                                    sum(1 for a in action[:-1] if not a["reasoning"])],
                                "prev_error": [sum(1 for a, b in zip(action, action[1:]) if a["errors"] and b["reasoning"]),
                                               sum(1 for a in action[:-1] if a["errors"])],
                                "prev_ok": [sum(1 for a, b in zip(action, action[1:]) if not a["errors"] and b["reasoning"]),
                                            sum(1 for a in action[:-1] if not a["errors"])]},
        "reasoned_by_decile": [[sum(1 for t in action[len(action) * d // 10:len(action) * (d + 1) // 10] if t["reasoning"]),
                                len(action[len(action) * d // 10:len(action) * (d + 1) // 10])] for d in range(10)],
        "todowrite_calls": calls.count("todowrite"), "task_calls": calls.count("task"),
        "webfetch_calls": calls.count("webfetch"), "skill_calls": calls.count("skill"),
        "hub_tokens": hub,
        "agent_wall_s": (max(times) - min(times)) / 1000.0 if len(times) >= 2 else None,
        "broken_text_parts": broken_parts, "broken_tool_outputs": broken_outputs,
        "prompt_max_main": max(prompts, default=0), "context_max_main": max(contexts, default=0),
        "steps_over_242144": sum(p > COMPACT_AT for p in prompts), "steps_over_200k": sum(p > 200000 for p in prompts),
        "message_errors": message_errors,
        "compaction_trigger_prompts": trigger_prompts,
    })
    if compaction_at:
        tail = turns[compaction_at[-1]:]
        out.update({"turns_after_last_compaction": len(tail),
                    "action_turns_after_last_compaction": sum(bool(t["tools"]) for t in tail),
                    "commit_after_last_compaction": any(COMMIT.search(c) for t in tail for c in t["commands"]),
                    "ended_on_compaction": not tail})
    return out


def max_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["kimi4-all", "kimi3", "kimi-1m"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        slots = sources.slots(source)
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(features, slots, chunksize=2))
        path = OUT_DIR / f"{source}.jsonl"
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        tmp.replace(path)
        print(source, len(rows), "slots;", sum(bool(r.get("export")) for r in rows), "with export", flush=True)


if __name__ == "__main__":
    main()
