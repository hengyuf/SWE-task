#!/usr/bin/env python3
"""Build the kimi-pass-v1 SFT corpus (raw_sft scheme of zhenghaox's Super SFT pipeline on HSG).

Input : the 380 Kimi-K3 passes in kimi_pass_batch_20261021 (trajectories.jsonl.gz + harness.json), read-only.
Render: sc14's chat_template.jinja through transformers.apply_chat_template (HF jinja settings), tools from harness.json.
Output: DATA_ROOT layout mirroring super-filtered-glm53-v5-20260922/data (raw_sft shards, blends, receipt, selection).

Run (CPU only, no network):
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I \
      tools/build_corpus.py [--out DIR] [--workers 8] [--invalid-policy mask|keep]

See README.md next to tools/ for the rules; every rule below is commented where it is applied.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

BATCH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021")
HF_DIR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf")
OUT_DEFAULT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v1-20261023")
HSG_ROOT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-kimi-pass-v1-20261023"
SCHEME = "raw_sft"
SHARD_NAME = "part_{:03d}.mat.jsonl.packed.512k"
N_SHARDS = 6
SEQ_LEN = 524288          # --seq-length
PAD_GRAN = 32             # context_parallel_size 16 * 2: every conversation is padded to a multiple of 32
SHUFFLE_SEED = 20261023   # row-order shuffle
GBS = 32
EPOCHS = 5
GEN_PREFIX = "<|im_start|>assistant\n<think>\n"   # sc14 add_generation_prompt with enable_thinking (default True)
THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.S)
SURROGATE_RE = re.compile("[\ud800-\udfff]")
SCHEMA = "kimi-pass-v1-hsg-raw-sft-data-v1"

_W = {}


# ----------------------------------------------------------------------------------------------- worker side
def _init(invalid_policy):
    import tokenizers
    import transformers

    transformers.logging.set_verbosity_error()
    _W["hf"] = transformers.AutoTokenizer.from_pretrained(str(HF_DIR))
    _W["tk"] = tokenizers.Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    _W["harness"] = json.loads((BATCH / "harness.json").read_text())
    _W["invalid_policy"] = invalid_policy
    added = _W["tk"].get_added_tokens_decoder()
    _W["special_ids"] = {i for i, t in added.items() if t.special}
    _W["unk_id"] = _W["tk"].token_to_id("<unk>")


def enc(text):
    return _W["tk"].encode(text, add_special_tokens=False).ids


def fix_surrogates(obj, counter):
    """Lone UTF-16 surrogates cannot be tokenized or UTF-8 encoded: replace each by U+FFFD (counted)."""
    if isinstance(obj, str):
        if SURROGATE_RE.search(obj):
            counter["surrogate_strings_fixed"] += 1
            return obj.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return obj
    if isinstance(obj, list):
        return [fix_surrogates(v, counter) for v in obj]
    if isinstance(obj, dict):
        return {k: fix_surrogates(v, counter) for k, v in obj.items()}
    return obj


def arg_str(value):
    """sc14 template: mappings / non-string sequences -> tojson, everything else -> |string (Python str)."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value
    return str(value)


def render_calls(calls):
    out = []
    for name, args in calls:
        out.append("<tool_call>\n<function=" + name + ">\n")
        for k, v in args.items():
            out.append("<parameter=" + k + ">\n" + arg_str(v) + "\n</parameter>\n")
        out.append("</function>\n</tool_call>\n")
    return "".join(out)


def convert(line_no, line):
    """One trajectory -> (conversation messages, stats). Raises on any violated invariant."""
    stats = Counter()
    rec = json.loads(line)
    meta = rec["meta"]
    tid = meta["trajectory_id"]
    traj = fix_surrogates(rec["trajectory"], stats)
    harness = _W["harness"]
    system_messages = harness["system_prompt_variants"][meta["harness_system_sha256"]]["system_messages"]
    tools = harness["tool_sets"][meta["harness_tools_sha256"]]["tools"]
    # Edge case: several system messages. Every variant has exactly one; more would be joined below, but none exist.
    assert len(system_messages) >= 1
    if len(system_messages) > 1:
        stats["system_messages_joined"] += 1
    system_text = "\n\n".join(system_messages)

    msgs = traj["messages"]
    roles = [m["info"]["role"] for m in msgs]
    assert roles[0] == "user" and roles.count("user") == 1, (tid, "expected exactly one user message, first")
    assert all(r == "assistant" for r in roles[1:]), tid
    user_parts = msgs[0]["parts"]
    # Edge case: opencode-injected text parts (synthetic / ignored) - none in this batch; they would be dropped.
    user_texts = []
    for p in user_parts:
        assert p["type"] == "text", (tid, p["type"])
        if p.get("synthetic") or p.get("ignored"):
            stats["user_injected_parts_dropped"] += 1
            continue
        user_texts.append(p["text"])
    user_text = "\n".join(user_texts)

    oai = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    turns = []  # per assistant action: (assistant_piece_content, tool_piece_or_None, weight, n_calls)
    n_asst = len(msgs) - 1
    for idx, m in enumerate(msgs[1:], start=1):
        is_final = idx == len(msgs) - 1
        texts, calls, results = [], [], []
        invalid = False
        for p in m["parts"]:
            t = p["type"]
            if t == "text":
                if p.get("synthetic") or p.get("ignored"):
                    stats["assistant_injected_parts_dropped"] += 1
                    continue
                texts.append(p["text"])
            elif t == "reasoning":
                texts.append("<think>" + p.get("text", "") + "</think>")  # none in this batch
            elif t == "tool":
                st = p["state"]
                name = p["tool"]
                if name == "invalid":
                    invalid = True
                if name == "webfetch":
                    stats["webfetch_calls"] += 1
                    if st["status"] != "completed":
                        stats["webfetch_failed"] += 1
                calls.append((name, st["input"]))
                if st["status"] == "completed":
                    # Edge case: compaction. opencode replays a compacted output as this placeholder (none here).
                    if (st.get("time") or {}).get("compacted"):
                        stats["compacted_tool_outputs"] += 1
                        results.append("[Old tool result content cleared]")
                    else:
                        results.append(st["output"])
                elif st["status"] == "error":
                    stats["tool_errors"] += 1
                    results.append(st["error"])  # opencode sends error-text = state.error
                else:
                    stats["tool_interrupted"] += 1
                    results.append("[Tool execution was interrupted]")
            elif t in ("step-start", "step-finish", "patch", "snapshot"):
                pass  # bookkeeping parts, never sent to the model
            elif t in ("compaction", "agent", "subtask", "file", "retry"):
                raise AssertionError((tid, "unexpected part type", t))
            else:
                raise AssertionError((tid, "unknown part type", t))
        assert len(texts) <= 1, (tid, idx, "more than one text part")
        text = texts[0] if texts else ""
        mt = THINK_RE.fullmatch(text) if text.startswith("<think>") else None
        if mt:
            reasoning, visible = mt.group(1), mt.group(2)
        else:
            assert "<think>" not in text and "</think>" not in text, (tid, idx, "think tag not at start")
            reasoning, visible = "", text
        if reasoning.strip():
            stats["turns_with_reasoning"] += 1
        else:
            stats["turns_empty_reasoning"] += 1
        if not visible.strip():
            stats["turns_empty_visible"] += 1
        if is_final:
            assert not calls, (tid, "final assistant has tool calls")
        else:
            assert calls, (tid, idx, "non-final assistant without tool calls")

        a = {"role": "assistant"}
        if reasoning.strip():
            a["reasoning_content"] = reasoning
            a["content"] = visible
        else:
            # Empty reasoning: keep the uniform generation prefix '<think>\n' (what the student is prompted with)
            # and an immediately closed think block, instead of the template's '<think></think>' prompt variant.
            a["content"] = "<think>\n</think>" + visible
        if calls:
            a["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": args}} for n, args in calls]
        oai.append(a)
        for r in results:
            oai.append({"role": "tool", "content": r})

        # The piece, exactly as the template renders the turn, minus the generation prefix of the previous piece.
        content_full = "<think>\n" + reasoning + "</think>" + visible
        if calls:
            rendered = "<|im_start|>assistant\n" + content_full.strip() + "\n" + render_calls(calls) + "<|im_end|>\n"
        else:
            rendered = "<|im_start|>assistant\n" + content_full.strip() + "<|im_end|>\n"
        assert rendered.startswith(GEN_PREFIX), (tid, idx)
        a_piece = rendered[len(GEN_PREFIX):]
        t_piece = None
        if results:
            t_piece = ("<|im_start|>user\n" + "".join("<tool_response>\n" + r + "\n</tool_response>\n" for r in results)
                       + "<|im_end|>\n" + GEN_PREFIX)
        weight = 1.0
        if invalid:
            stats["invalid_tool_turns"] += 1
            if _W["invalid_policy"] == "mask":
                weight = 0.0
                stats["invalid_tool_turns_masked"] += 1
        turns.append((a_piece, t_piece, weight, len(calls)))

    # Pieces (role, content) of the conversation.
    hf = _W["hf"]
    sys_piece = hf.apply_chat_template([oai[0]], tools=tools, tokenize=False, add_generation_prompt=False)
    user_piece = "<|im_start|>user\n" + user_text + "<|im_end|>\n" + GEN_PREFIX
    pieces = [("system", sys_piece), ("user", user_piece)]
    for a_piece, t_piece, _, _ in turns:
        pieces.append(("assistant", a_piece))
        if t_piece is not None:
            pieces.append(("tool", t_piece))
    assert pieces[-1][0] == "assistant"

    # Source of truth: the full template render of the chat request must equal the concatenated pieces.
    full = hf.apply_chat_template(oai, tools=tools, tokenize=False, add_generation_prompt=False)
    joined = "".join(c for _, c in pieces)
    if full != joined:
        k = next((i for i, (x, y) in enumerate(zip(full, joined)) if x != y), min(len(full), len(joined)))
        raise AssertionError((tid, "template render != pieces", k, full[k - 80:k + 80], joined[k - 80:k + 80]))
    # The user piece's tail must be what the template emits as generation prompt.
    gp = hf.apply_chat_template(oai[:2], tools=tools, tokenize=False, add_generation_prompt=True)
    assert gp == sys_piece + user_piece, (tid, "generation prefix mismatch")

    # Invariant 1: every message boundary is a token boundary (minimal fixes would go here; count them).
    full_ids = enc(joined)
    piece_ids = [enc(c) for _, c in pieces]
    cat = [i for ids in piece_ids for i in ids]
    if cat != full_ids:
        off = 0
        for j, ids in enumerate(piece_ids):
            if full_ids[off:off + len(ids)] != ids:
                raise AssertionError((tid, "token boundary broken at piece", j, pieces[j][0], pieces[j][1][:120]))
            off += len(ids)
        raise AssertionError((tid, "token boundary broken (tail)"))

    # Invariant 2: first message system, assistant non-empty and after user/tool.
    for j, (role, c) in enumerate(pieces):
        if role == "assistant":
            assert c and pieces[j - 1][0] in ("user", "tool"), (tid, j)
    assert pieces[0][0] == "system" and sum(1 for r, _ in pieces if r == "system") == 1

    # Invariant 3: assistant_token_counts == per-message token counts, sum == non-masked targets.
    a_counts = [len(ids) for (role, _), ids in zip(pieces, piece_ids) if role == "assistant"]
    weights = [w for _, _, w, _ in turns]
    assert len(a_counts) == len(weights) == n_asst and all(c > 0 for c in a_counts)
    # Simulate SFTTokenizer.tokenize_conversation (identity, has_bos False, assistant_prefix_len 0).
    n_target = 0
    idx_tok = 0
    for (role, _), ids in zip(pieces, piece_ids):
        assert full_ids[idx_tok:idx_tok + len(ids)] == ids
        if role == "assistant":
            n_target += len(ids)
        idx_tok += len(ids)
    assert idx_tok == len(full_ids) and n_target == sum(a_counts)

    # Special tokens inside assistant targets (informational; <unk> is the pad id and would be loss-masked).
    special = _W["special_ids"]
    unk = _W["unk_id"]
    for (role, _), ids in zip(pieces, piece_ids):
        if role == "assistant":
            body = ids[:-2] if len(ids) >= 2 else ids  # the trailing <|im_end|> + '\n'
            stats["assistant_unk_tokens"] += sum(1 for i in ids if i == unk)
            stats["assistant_special_tokens_in_body"] += sum(1 for i in body if i in special)

    messages = []
    for j, (role, c) in enumerate(pieces):
        msg = {"role": role, "content": c}
        if j == 0:
            msg.update({"weight": 1.0, "step_weight_agent_run_id": tid,
                        "assistant_weights": [float(w) for w in weights],
                        "assistant_token_counts": a_counts})
        messages.append(msg)

    conv_json = json.dumps({"messages": messages}, ensure_ascii=False)
    info = {
        "line_no": line_no,
        "meta": meta,
        "trajectory_id": tid,
        "token_length": len(full_ids),
        "assistant_tokens": n_target,
        "n_assistant": n_asst,
        "n_tool_messages": sum(1 for r, _ in pieces if r == "tool"),
        "n_tool_calls": sum(t[3] for t in turns),
        "system_tokens": len(piece_ids[0]),
        "user_tokens": len(piece_ids[1]),
        "masked_assistant_turns": sum(1 for w in weights if w == 0.0),
        "stats": dict(stats),
        "training_row_sha256": hashlib.sha256(conv_json.encode("utf-8")).hexdigest(),
    }
    return info, messages


def work(arg):
    return convert(*arg)


# ------------------------------------------------------------------------------------------------- main side
def ceil32(n):
    return (n + PAD_GRAN - 1) // PAD_GRAN * PAD_GRAN


def ffd(items):
    """Deterministic first-fit-decreasing on padded length; items = [(key, padded_len)]. Returns rows of keys."""
    order = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    rows, free = [], []
    for key, size in order:
        assert size <= SEQ_LEN
        for r, f in enumerate(free):
            if size <= f:
                rows[r].append(key)
                free[r] -= size
                break
        else:
            rows.append([key])
            free.append(SEQ_LEN - size)
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--invalid-policy", choices=("mask", "keep"), default="keep")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N trajectories only, writes nothing")
    args = ap.parse_args()
    out = args.out
    os.umask(0o027)

    with gzip.open(BATCH / "trajectories.jsonl.gz", "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 380, len(lines)
    tasks = list(enumerate(lines))
    if args.limit:
        tasks = tasks[:args.limit]
    del lines

    infos, convs = [], {}
    with Pool(args.workers, initializer=_init, initargs=(args.invalid_policy,)) as pool:
        for info, messages in pool.imap(work, tasks, chunksize=1):
            infos.append(info)
            convs[info["trajectory_id"]] = messages
            print(f"[{len(infos)}/{len(tasks)}] {info['trajectory_id']} tokens={info['token_length']}", flush=True)
    del tasks
    if args.limit:
        print(json.dumps([{k: v for k, v in i.items() if k != "meta"} for i in infos], indent=1))
        return

    # Invariant 4: no conversation may exceed the sequence length (drop and list; none expected).
    dropped = [i for i in infos if ceil32(i["token_length"]) > SEQ_LEN]
    kept = [i for i in infos if ceil32(i["token_length"]) <= SEQ_LEN]
    assert len({i["trajectory_id"] for i in kept}) == len(kept)

    # Packing: FFD on ceil32 lengths, then a seeded shuffle of row order, then 6 contiguous near-equal shards.
    rows = ffd([(i["trajectory_id"], ceil32(i["token_length"])) for i in kept])
    random.Random(SHUFFLE_SEED).shuffle(rows)
    n_rows = len(rows)
    base, extra = divmod(n_rows, N_SHARDS)
    sizes = [base + (1 if s < extra else 0) for s in range(N_SHARDS)]
    assert min(sizes) >= 1, "fewer rows than shards"
    by_id = {i["trajectory_id"]: i for i in kept}

    (out / SCHEME).mkdir(parents=True, exist_ok=True)
    (out / "blends").mkdir(exist_ok=True)
    (out / "selection").mkdir(exist_ok=True)
    shards = []
    cursor = 0
    for s in range(N_SHARDS):
        path = out / SCHEME / SHARD_NAME.format(s)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in range(sizes[s]):
                row = rows[cursor + r]
                packed = []
                used = 0
                for pos, key in enumerate(row):
                    packed.extend(convs[key])
                    i = by_id[key]
                    i.update(shard=SHARD_NAME.format(s).split(".")[0], shard_file=SHARD_NAME.format(s),
                             shard_row=r, row_position=pos)
                    used += ceil32(i["token_length"])
                assert used <= SEQ_LEN
                f.write(json.dumps({"messages": packed}, ensure_ascii=False) + "\n")
                for key in row:
                    by_id[key]["row_padded_tokens"] = used
        os.replace(tmp, path)
        cursor += sizes[s]
        shards.append({"bytes": path.stat().st_size, "packed_rows": sizes[s],
                       "path": f"{HSG_ROOT}/{SCHEME}/{SHARD_NAME.format(s)}"})
    assert cursor == n_rows

    # Re-read every shard and re-verify the packing/weight invariants exactly as the dataset reads them.
    for s in range(N_SHARDS):
        with open(out / SCHEME / SHARD_NAME.format(s), encoding="utf-8") as f:
            for line in f:
                ms = json.loads(line)["messages"]
                assert ms[0]["role"] == "system"
                cur = None
                for m in ms:
                    if m["role"] == "system":
                        cur = m
                        assert len(cur["assistant_weights"]) == len(cur["assistant_token_counts"]) > 0
                        assert all(isinstance(w, float) and 0.0 <= w <= 1.0 for w in cur["assistant_weights"])
                    else:
                        assert set(m) == {"role", "content"}

    blend = {"train": [x for s in shards for x in ("0.16666667", s["path"])], "valid": None, "test": None}
    (out / "blends" / f"{SCHEME}.json").write_text(json.dumps(blend, indent=2) + "\n")

    # Selection (HSG schema fields are strings, as in all_three_zero_3205.jsonl) + our fields.
    sel_path = out / "selection" / "kimi_pass_v1.jsonl"
    with open(sel_path, "w", encoding="utf-8") as f:
        for i in sorted(kept, key=lambda x: x["line_no"]):
            m = i["meta"]
            f.write(json.dumps({
                "agent_run_id": i["trajectory_id"],
                "record_id": "r_" + hashlib.sha256(i["trajectory_id"].encode()).hexdigest()[:24],
                "source_line": str(i["line_no"]),
                "source_part": "kimi_pass_batch_20261021/trajectories.jsonl.gz",
                "token_length": str(i["token_length"]),
                "training_row_sha256": i["training_row_sha256"],
                "run": m["run"], "task_id": m["task_id"], "slot": m["slot"],
                "included_because": m["included_because"], "grader": m["grader"],
                "sanitizer_noop": m["sanitizer_noop"], "grading_note": m["grading_note"],
                "n_assistant": i["n_assistant"], "assistant_tokens": i["assistant_tokens"],
                "masked_assistant_turns": i["masked_assistant_turns"],
                "webfetch_calls": i["stats"].get("webfetch_calls", 0),
                "shard": i["shard"], "shard_row": i["shard_row"], "row_position": i["row_position"],
            }, ensure_ascii=False) + "\n")
    selection_sha = sha256_file(sel_path)

    # metadata.jsonl: the batch meta + corpus fields, batch order.
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(infos, key=lambda x: x["line_no"]):
            row = dict(i["meta"])
            row["corpus"] = {k: i.get(k) for k in (
                "line_no", "token_length", "assistant_tokens", "n_assistant", "n_tool_messages", "n_tool_calls",
                "system_tokens", "user_tokens", "masked_assistant_turns", "training_row_sha256", "shard",
                "shard_row", "row_position", "row_padded_tokens")}
            row["corpus"]["stats"] = i["stats"]
            row["corpus"]["dropped"] = i in dropped
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tot = Counter()
    for i in kept:
        tot.update(i["stats"])
    tl = [i["token_length"] for i in kept]
    stats = {
        "conversations": len(kept), "dropped": [i["trajectory_id"] for i in dropped], "packed_rows": n_rows,
        "rows_per_shard": sizes, "conversations_per_row_max": max(len(r) for r in rows),
        "tokens_total": sum(tl), "tokens_padded_total": sum(ceil32(x) for x in tl),
        "assistant_tokens_total": sum(i["assistant_tokens"] for i in kept),
        "assistant_tokens_weighted": None,
        "tokens_max": max(tl), "tokens_min": min(tl), "tokens_mean": sum(tl) / len(tl),
        "max_conversation": max(kept, key=lambda x: x["token_length"])["trajectory_id"],
        "assistant_messages": sum(i["n_assistant"] for i in kept),
        "tool_messages": sum(i["n_tool_messages"] for i in kept),
        "tool_calls": sum(i["n_tool_calls"] for i in kept),
        "row_fill": sum(ceil32(x) for x in tl) / (n_rows * SEQ_LEN),
        "edge_case_counts": dict(sorted(tot.items())),
        "grader_deepswe_recorded": sum(1 for i in kept if i["meta"]["grader"] == "deepswe-recorded"),
        "train_samples_5epoch": math.ceil(EPOCHS * n_rows / GBS) * GBS,
        "target_iteration_5epoch": math.ceil(EPOCHS * n_rows / GBS),
        "invalid_policy": args.invalid_policy,
        "shuffle_seed": SHUFFLE_SEED,
    }
    # weighted assistant tokens (masked invalid turns excluded)
    wsum = 0
    for key, ms in convs.items():
        if key in by_id:
            s0 = ms[0]
            wsum += sum(c * w for c, w in zip(s0["assistant_token_counts"], s0["assistant_weights"]))
    stats["assistant_tokens_weighted"] = int(wsum)

    receipt = {
        "blend": f"{HSG_ROOT}/blends/{SCHEME}.json",
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filter_policy": "all Kimi-K3 passes of kimi_pass_batch_20261021 (accepted or audit-strong tasks, clean glm "
                         "teacher audit); no further filtering; grading flags carried in selection/metadata",
        "packed_rows": n_rows,
        "physical_bytes": sum(s["bytes"] for s in shards),
        "raw_weight_policy": "all assistant action weights = 1.0",
        "schema": SCHEMA,
        "selected_trajectories": len(kept),
        "selection": f"{HSG_ROOT}/selection/kimi_pass_v1.jsonl",
        "selection_sha256": selection_sha,
        "shards": shards,
        "source_cluster": "cw-dfw",
        "state": "complete",
    }
    if args.invalid_policy == "mask":
        receipt["raw_weight_exceptions"] = (
            f"{stats['edge_case_counts'].get('invalid_tool_turns_masked', 0)} assistant actions whose tool call "
            "opencode rewrote to the 'invalid' tool (truncated write JSON) carry weight 0.0 (context only)")
    receipt["build_stats"] = stats
    (out / "DATA_PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    sys.exit(main())
