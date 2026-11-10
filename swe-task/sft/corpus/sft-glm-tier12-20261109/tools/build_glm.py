#!/usr/bin/env python3
"""Build the GLM-5.3 tier-1/2 SFT corpus (raw_sft scheme of zhenghaox's Super SFT pipeline; the on-disk format of
SFT-v3 / kimi-pass-v2) from tools/sources_glm.py's <sources>/glm/ (2026-11-09).

User request (hengyuf, GitHub issue hengyuf/SWE-task#1, comment 6758873905, verbatim): "现在收集tier1/2 的 successful
traces （包括有compaction的）， 将其加工为类似SFT v3 一样的dataset， 这里对于一个rollout 的多个segment（有compaction的），
将其视为多条独立的traces，task ID 可以用一个用于标记，但训练时保留各自边界。标准 SFT 可以把 segment 当独立样本，做
packing时，用现有 packer 的边界/attention 语义，别让 loss 或上下文跨样本串起来。pack 好data 把data 用datamover 移动到 AGA
（oci-aga-slurm-1-login-01.nvidia.com）上。（AGA 访问密码与HSG 相同）"
Design: /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/DESIGN.md (rev 2).

Samples (one raw_sft conversation each = the unit of packing and of the loss boundaries; id
glm53-<batch>/<campaign>/<task_id>/<slot>#seg<k>of<K>, K = 1 for a rollout without compaction):
  - route "v2_convert" (a non-compacted pass that ends with a final answer, no dropped turn): v2's convert(), unchanged:
    its source text (tools/build_corpus.py of kimi-pass-v2-20261023, `def enc` .. `def work`) and its constants
    GEN_PREFIX / THINK_RE / SURROGATE_RE are exec'd here, whole-file and block sha256 asserted, exactly as
    build_sft_v3.py lines 87-95.
  - route "segment" (every window of a compacted pass; a non-compacted pass whose export does not end in a valid final
    answer, e.g. the errored empty final message of tasksall t1-h00 11-2): convert_window() below, v2's per-turn rules:
      window 1:   v2's system piece and user piece (the task), then v2's assistant / tool pieces for the window's target
                  calls (sources_glm.py window info); the last target's tool results are not included (the sample ends on
                  an assistant piece; the results reappear in the next window's context only when OpenCode's tail keeps
                  that turn, else they are in no sample: window info end_tool_results);
      window k>1: the system piece of the window's first request (render of [system] with its tools), then ONE non-target
                  "user" piece = render([system] + context, tools, add_generation_prompt=True) minus the system piece,
                  context = that request's messages[1:] ("What did we do so far?", the summary with its reasoning, the
                  tail, the synthetic "Continue ..." message; redacted, from sources_glm.py), then v2's assistant / tool
                  pieces for the window's own targets;
      system prompt / tools = the window's FIRST request; render = sc14's chat template via transformers
      apply_chat_template with truncate_history_thinking=False (GLM saw all earlier reasoning; the student's RL / eval
      serving sets it false too). Dropped turns (empty / errored export messages, finish "length" replies) are never
      targets (sources_glm.py).
  - self-test (asserted): on every "v2_convert" pass the segment converter run as one window reproduces convert() byte
    for byte (messages = pieces + system fields, training_row_sha256, token_length, assistant_tokens, and the other
    counts and stats).
  - v2 invariants per sample (asserted): join(pieces) == the sc14 render of the sample's conversation; every prompt piece
    ends with GEN_PREFIX; concat(enc(piece)) == enc(join); exactly one system message, first; every assistant piece
    non-empty and right after a user / tool piece; the sample ends with an assistant piece.
Sample filters (v3): literal generation stop text ('</s>', generation_config eos id 2) in an assistant target -> sample
dropped (v3's rule; _stop_strings exec'd from build_sft_v3.py's source text); literal '<s>' only counted;
ceil32(tokens) > 524,288 -> dropped; no 262,144 cap (the count over 262,144 is reported).
Packing (v2 / v3 unchanged; ceil32 / ffd / epochs_table exec'd from build_sft_v3.py's source text): FFD on ceil32 lengths
into 524,288-token rows over all samples, rows shuffled with random.Random(20261109), 6 contiguous near-equal shards,
weight 0.16666667 each. Windows of one rollout may share a row (the trainer splits a row at system messages into THD
segments; attention, Mamba state and the MTP roll reset per segment).
Output (<out>): raw_sft/part_00{0..5}.mat.jsonl.packed.512k, blends/raw_sft.json (paths under --data-root),
selection/sft_glm_tier12.jsonl, metadata.jsonl, DATA_PREP_COMPLETE.json, README.md. Deterministic: the same inputs give
byte-identical outputs except the receipt's / README's completed_at.

  PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
  $PY -I -B tools/build_glm.py [--out DIR] [--sources DIR] [--data-root PATH] [--workers 16] [--invalid-policy keep]
      [--allow-other-data-root]
The real corpus dir refuses a dry run (partial sources), a rebuild over an existing raw_sft/, and a --data-root other
than DESIGN's AGA DATA_ROOT (DATA_ROOT_DEFAULT) unless --allow-other-data-root.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys  # noqa: E402

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
WORK = S / "sft/corpus/glm-tier12-20261109"
CORPUS = S / "sft/corpus/sft-glm-tier12-20261109"
TOOLS = Path(__file__).resolve().parent
V2 = S / "sft/corpus/kimi-pass-v2-20261023"
V2_BUILDER = V2 / "tools/build_corpus.py"
V2_BUILDER_SHA256 = "2c6977864c82ab8bfc68c6a62bb37ccd8ca68f9d9e96b289e447c51748f3478d"  # v2 SHA256SUMS
CONVERTER_SHA256 = "807295d4abe8cfdef362ce7dee3368b102e42e9596c0dd77e4b7a0909791e6f9"  # `def enc` .. `def work`
V3_BUILD = S / "sft/corpus/sft-v3-20261029/tools/build_sft_v3.py"
V3_BUILD_SHA256 = "64b2316e8e39ea4f6a6fa9e73a768ce0fcd2f15eaa06e3eebb0ea32ef3190069"  # sft-v3 SHA256SUMS
V3_BLOCKS_SHA256 = {  # exact source blocks of build_sft_v3.py exec'd below
    "_stop_strings": "f7db66c8c08f7bd6117c16749f9ce72bbd0c614beddc3fb732a76eb59b28fce8",
    "ceil32_ffd": "bb5bc562bd865574a34ff9ac852691632d8e6c1b37647ec7bc14a5c22d9e1595",
    "epochs_table_flag_summary": "cadfd84a26f5d9d189174108b48d549e2f4273362e4fd3a3e025496724ee3f0a",
}
HF_DIR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf")
OUT_DEFAULT = CORPUS
DATA_ROOT_DEFAULT = "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/sft-glm-tier12-20261109"
SCHEME = "raw_sft"
SHARD_NAME = "part_{:03d}.mat.jsonl.packed.512k"
N_SHARDS = 6
SEQ_LEN = 524288          # --seq-length
PAD_GRAN = 32             # context_parallel_size 16 * 2: every conversation is padded to a multiple of 32
SHUFFLE_SEED = 20261109   # row-order shuffle (the build date, as v2 / v3)
GBS = 32
EPOCHS = (5, 10, 20)
SCHEMA = "sft-glm-tier12-hsg-raw-sft-data-v1"
SELECTION_NAME = "sft_glm_tier12.jsonl"
SOURCE = "glm"
SOURCE_PART = "sources/glm/trajectories.jsonl.gz"
V2_MAX_TOKENS = 262144    # reported only (v3's v2-part cap; not applied here)
WDWDSF = "What did we do so far?"
CONTINUE = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
BATCHES = ("tasksall", "sftv3")

_W = {}

# ---------------------------------------------------------------- v2's converter, exec'd from v2's source text (as v3)
_V2_BYTES = V2_BUILDER.read_bytes()
assert hashlib.sha256(_V2_BYTES).hexdigest() == V2_BUILDER_SHA256, "v2 build_corpus.py changed"
_V2_TEXT = _V2_BYTES.decode("utf-8")
for _name in ("GEN_PREFIX", "THINK_RE", "SURROGATE_RE"):
    exec(re.search(r"^" + _name + r" = .*$", _V2_TEXT, re.M).group(0), globals())
CONVERTER_SOURCE = _V2_TEXT[_V2_TEXT.index("def enc(text):"):_V2_TEXT.index("def work(arg):")]
assert hashlib.sha256(CONVERTER_SOURCE.encode("utf-8")).hexdigest() == CONVERTER_SHA256, "v2 converter changed"
exec(compile(CONVERTER_SOURCE, str(V2_BUILDER), "exec"), globals())  # enc, fix_surrogates, arg_str, render_calls, convert

# ---------------------------------------------------------------- v3's helpers, exec'd from build_sft_v3.py's source text
_V3_BYTES = V3_BUILD.read_bytes()
assert hashlib.sha256(_V3_BYTES).hexdigest() == V3_BUILD_SHA256, "sft-v3 build_sft_v3.py changed"
_V3_TEXT = _V3_BYTES.decode("utf-8")
for _block, (_start, _end) in {
        "_stop_strings": ("def _stop_strings():", "# -----------------------------------------------------------------"
                                                  "------------------------------ worker side"),
        "ceil32_ffd": ("def ceil32(n):", "def sha256_file(path):"),
        "epochs_table_flag_summary": ("def epochs_table(n_rows):", "def main():")}.items():
    _i = _V3_TEXT.index(_start)
    _src = _V3_TEXT[_i:_V3_TEXT.index(_end, _i)]
    assert hashlib.sha256(_src.encode("utf-8")).hexdigest() == V3_BLOCKS_SHA256[_block], f"v3 {_block} changed"
    exec(compile(_src, str(V3_BUILD), "exec"), globals())  # _stop_strings, ceil32, ffd, epochs_table, flag_summary


# ----------------------------------------------------------------------------------------------- worker side
def _init(invalid_policy, harness_path):
    import tokenizers
    import transformers

    transformers.logging.set_verbosity_error()
    _W["hf"] = transformers.AutoTokenizer.from_pretrained(str(HF_DIR))
    _W["tk"] = tokenizers.Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    h = json.loads(Path(harness_path).read_text())
    _W["harness"] = {"system_prompt_variants": h["system_prompt_variants"], "tool_sets": h["tool_sets"]}
    _W["invalid_policy"] = invalid_policy
    added = _W["tk"].get_added_tokens_decoder()
    _W["special_ids"] = {i for i, t in added.items() if t.special}
    _W["unk_id"] = _W["tk"].token_to_id("<unk>")


def convert_window(source, line_no, rec, w, meta):
    """One window of a rollout -> (info, messages): v2's convert() generalized to a window (module docstring). The
    per-turn code is v2's (kimi-pass-v2 build_corpus.py lines 165-258) with is_final replaced by "last target of the
    window"; the invariant checks are v2's (lines 260-347)."""
    stats = Counter()
    tid = meta["trajectory_id"]
    traj = fix_surrogates(rec["trajectory"], stats)
    harness = _W["harness"]
    system_messages = harness["system_prompt_variants"][w["system_sha256"]]["system_messages"]
    tools = harness["tool_sets"][w["tools_sha256"]]["tools"]
    assert len(system_messages) >= 1
    if len(system_messages) > 1:
        stats["system_messages_joined"] += 1
    system_text = "\n\n".join(system_messages)
    msgs = traj["messages"]
    oai = [{"role": "system", "content": system_text}]
    user_text = None
    if w["segment"] == 1:
        assert msgs[0]["info"]["role"] == "user", (tid, "export does not start with the user task")
        user_texts = []
        for p in msgs[0]["parts"]:
            assert p["type"] == "text", (tid, p["type"])
            if p.get("synthetic") or p.get("ignored"):
                stats["user_injected_parts_dropped"] += 1
                continue
            user_texts.append(p["text"])
        user_text = "\n".join(user_texts)
        oai.append({"role": "user", "content": user_text})
    else:
        context = fix_surrogates(w["context"], stats)
        assert context[0] == {"role": "user", "content": WDWDSF}, (tid, "context does not start with the marker")
        assert context[-1] == {"role": "user", "content": CONTINUE}, (tid, "context does not end with Continue")
        oai += context
    n_prompt = len(oai)
    targets = w["target_export_indices"]
    assert targets, (tid, "window without a target")
    turns = []  # per target: (assistant_piece_content, tool_piece_or_None, weight, n_calls)
    for q, idx in enumerate(targets):
        m = msgs[idx]
        assert m["info"]["role"] == "assistant", (tid, idx)
        is_last = q == len(targets) - 1
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
                texts.append("<think>" + p.get("text", "") + "</think>")
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
                    if (st.get("time") or {}).get("compacted"):
                        stats["compacted_tool_outputs"] += 1
                        results.append("[Old tool result content cleared]")
                    else:
                        results.append(st["output"])
                elif st["status"] == "error":
                    stats["tool_errors"] += 1
                    results.append(st["error"])
                else:
                    stats["tool_interrupted"] += 1
                    results.append("[Tool execution was interrupted]")
            elif t in ("step-start", "step-finish", "patch", "snapshot"):
                pass
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
        if not is_last:
            assert calls, (tid, idx, "non-last target without tool calls")
        a = {"role": "assistant"}
        if reasoning.strip():
            a["reasoning_content"] = reasoning
            a["content"] = visible
        else:
            a["content"] = "<think>\n</think>" + visible
        if calls:
            a["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": args}} for n, args in calls]
        oai.append(a)
        if not is_last:
            for r in results:
                oai.append({"role": "tool", "content": r})
        elif results:
            stats["window_end_tool_results_cut"] += 1  # the window ends on this call; its results are next window's
        content_full = "<think>\n" + reasoning + "</think>" + visible
        if calls:
            rendered = "<|im_start|>assistant\n" + content_full.strip() + "\n" + render_calls(calls) + "<|im_end|>\n"
        else:
            rendered = "<|im_start|>assistant\n" + content_full.strip() + "<|im_end|>\n"
        assert rendered.startswith(GEN_PREFIX), (tid, idx)
        a_piece = rendered[len(GEN_PREFIX):]
        t_piece = None
        if results and not is_last:
            t_piece = ("<|im_start|>user\n" + "".join("<tool_response>\n" + r + "\n</tool_response>\n" for r in results)
                       + "<|im_end|>\n" + GEN_PREFIX)
        weight = 1.0
        if invalid:
            stats["invalid_tool_turns"] += 1
            if _W["invalid_policy"] == "mask":
                weight = 0.0
                stats["invalid_tool_turns_masked"] += 1
        turns.append((a_piece, t_piece, weight, len(calls)))

    # Pieces: system piece and the prompt (user / context) piece come from the template itself.
    hf = _W["hf"]
    kw = {"tokenize": False, "truncate_history_thinking": False}
    sys_piece = hf.apply_chat_template([oai[0]], tools=tools, add_generation_prompt=False, **kw)
    gp = hf.apply_chat_template(oai[:n_prompt], tools=tools, add_generation_prompt=True, **kw)
    assert gp.startswith(sys_piece) and gp.endswith(GEN_PREFIX), (tid, "generation prompt")
    prompt_piece = gp[len(sys_piece):]
    if user_text is not None:  # window 1: exactly v2's user piece
        assert prompt_piece == "<|im_start|>user\n" + user_text + "<|im_end|>\n" + GEN_PREFIX, (tid, "user piece")
    pieces = [("system", sys_piece), ("user", prompt_piece)]
    for a_piece, t_piece, _, _ in turns:
        pieces.append(("assistant", a_piece))
        if t_piece is not None:
            pieces.append(("tool", t_piece))
    assert pieces[-1][0] == "assistant"
    full = hf.apply_chat_template(oai, tools=tools, add_generation_prompt=False, **kw)
    joined = "".join(c for _, c in pieces)
    if full != joined:
        k = next((i for i, (x, y) in enumerate(zip(full, joined)) if x != y), min(len(full), len(joined)))
        raise AssertionError((tid, "template render != pieces", k, full[k - 80:k + 80], joined[k - 80:k + 80]))
    for role, c in pieces:
        if role in ("user", "tool"):
            assert c.endswith(GEN_PREFIX), (tid, "prompt piece without the generation prefix")

    # Invariant 1 (v2): every message boundary is a token boundary.
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
    # Invariant 2 (v2): first message system, assistant non-empty and after user/tool.
    for j, (role, c) in enumerate(pieces):
        if role == "assistant":
            assert c and pieces[j - 1][0] in ("user", "tool"), (tid, j)
    assert pieces[0][0] == "system" and sum(1 for r, _ in pieces if r == "system") == 1
    # Invariant 3 (v2): assistant_token_counts == per-message token counts, sum == non-masked targets.
    a_counts = [len(ids) for (role, _), ids in zip(pieces, piece_ids) if role == "assistant"]
    weights = [w_ for _, _, w_, _ in turns]
    n_asst = len(targets)
    assert len(a_counts) == len(weights) == n_asst and all(c > 0 for c in a_counts)
    n_target = 0
    idx_tok = 0
    for (role, _), ids in zip(pieces, piece_ids):
        assert full_ids[idx_tok:idx_tok + len(ids)] == ids
        if role == "assistant":
            n_target += len(ids)
        idx_tok += len(ids)
    assert idx_tok == len(full_ids) and n_target == sum(a_counts)
    special = _W["special_ids"]
    unk = _W["unk_id"]
    for (role, _), ids in zip(pieces, piece_ids):
        if role == "assistant":
            body = ids[:-2] if len(ids) >= 2 else ids
            stats["assistant_unk_tokens"] += sum(1 for i in ids if i == unk)
            stats["assistant_special_tokens_in_body"] += sum(1 for i in body if i in special)

    messages = []
    for j, (role, c) in enumerate(pieces):
        msg = {"role": role, "content": c}
        if j == 0:
            msg.update({"weight": 1.0, "step_weight_agent_run_id": tid,
                        "assistant_weights": [float(x) for x in weights],
                        "assistant_token_counts": a_counts})
        messages.append(msg)
    conv_json = json.dumps({"messages": messages}, ensure_ascii=False)
    info = {
        "source": source, "line_no": line_no, "meta": meta, "trajectory_id": tid,
        "token_length": len(full_ids), "assistant_tokens": n_target, "n_assistant": n_asst,
        "n_tool_messages": sum(1 for r, _ in pieces if r == "tool"), "n_tool_calls": sum(t[3] for t in turns),
        "system_tokens": len(piece_ids[0]), "user_tokens": len(piece_ids[1]),
        "masked_assistant_turns": sum(1 for x in weights if x == 0.0), "stats": dict(stats),
        "training_row_sha256": hashlib.sha256(conv_json.encode("utf-8")).hexdigest(),
    }
    return info, messages


SELFTEST_FIELDS = ("training_row_sha256", "token_length", "assistant_tokens", "n_assistant", "n_tool_messages",
                   "n_tool_calls", "system_tokens", "user_tokens", "masked_assistant_turns", "stats")


def sample_meta(meta, w, sid):
    m = {k: v for k, v in meta.items() if k != "glm"}
    m.update({"trajectory_id": sid, "parent_trajectory_id": meta["trajectory_id"], "segment": w["segment"],
              "n_segments": meta["glm"]["n_windows"],
              "rollout_harness_system_sha256": meta.get("harness_system_sha256"),
              "rollout_harness_tools_sha256": meta.get("harness_tools_sha256"),
              "rollout_harness_params_sha256": meta.get("harness_params_sha256"),
              "harness_system_sha256": w["system_sha256"], "harness_tools_sha256": w["tools_sha256"],
              "harness_params_sha256": w["params_sha256"],
              "glm": {k: v for k, v in meta["glm"].items() if k != "windows"},
              "window": {k: v for k, v in w.items() if k != "context"}})
    return m


def work(arg):
    """One rollout line -> [(info, messages)] for its samples, and its self-test record (v2_convert route)."""
    line_no, line = arg
    rec = json.loads(line)
    meta = rec["meta"]
    g = meta["glm"]
    rid = meta["trajectory_id"]
    K = g["n_windows"]
    assert K == len(g["windows"]) >= 1, rid
    out, selftest = [], None
    for w in g["windows"]:
        sid = f"{rid}#seg{w['segment']}of{K}"
        if g["route"] == "v2_convert":
            assert K == 1 and not w["dropped"] and w["ends_with"] == "final_answer", rid
            line1 = json.dumps({"meta": sample_meta(meta, w, sid), "trajectory": rec["trajectory"],
                                "final_patch": rec["final_patch"]}, ensure_ascii=False)
            info, messages = convert(SOURCE, line_no, line1)
            info_s, messages_s = convert_window(SOURCE, line_no, rec, w, sample_meta(meta, w, sid))
            diff = [f for f in SELFTEST_FIELDS if info[f] != info_s[f]]
            if messages != messages_s:
                diff.append("messages")
            assert not diff, (sid, "self-test: the segment converter differs from v2's convert()", diff)
            selftest = {"trajectory_id": sid, "equal": True, "training_row_sha256": info["training_row_sha256"]}
            info["converter"] = "v2_convert"
        else:
            info, messages = convert_window(SOURCE, line_no, rec, w, sample_meta(meta, w, sid))
            info["converter"] = "segment"
        info.update(segment=w["segment"], n_segments=K, parent_trajectory_id=rid, context_tokens=info["user_tokens"])
        out.append((info, messages))
    return out, selftest


# ------------------------------------------------------------------------------------------------- main side
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def group_stats(samples, rows, by_id):
    """Stats of a group of kept samples."""
    if not samples:
        return {"samples": 0}
    tl = sorted(i["token_length"] for i in samples)
    ids = {i["trajectory_id"] for i in samples}
    return {"samples": len(samples), "rollouts": len({i["parent_trajectory_id"] for i in samples}),
            "tasks": len({i["meta"]["task_key"] for i in samples}),
            "windows_k_ge_2": sum(1 for i in samples if i["segment"] >= 2),
            "rows_containing": sum(1 for r in rows if any(k in ids for k in r)),
            "tokens_total": sum(tl), "tokens_padded_total": sum(ceil32(x) for x in tl),
            "assistant_tokens_total": sum(i["assistant_tokens"] for i in samples),
            "tokens_min": tl[0], "tokens_median": tl[len(tl) // 2], "tokens_p90": pct(tl, 0.9), "tokens_max": tl[-1],
            "over_262144": sum(1 for x in tl if x > V2_MAX_TOKENS),
            "assistant_messages": sum(i["n_assistant"] for i in samples),
            "tool_calls": sum(i["n_tool_calls"] for i in samples),
            "max_sample": max(samples, key=lambda x: x["token_length"])["trajectory_id"],
            "flags": flag_summary([by_id[k].get("flags") for k in ids])}


def data_root_refusal(out, data_root, allow_other):
    """The real corpus dir bakes in DESIGN's AGA DATA_ROOT unless explicitly allowed (a dry run may use any root)."""
    if out != OUT_DEFAULT.resolve() or allow_other or data_root == DATA_ROOT_DEFAULT:
        return None
    return (f"--data-root {data_root} is not DESIGN's AGA DATA_ROOT {DATA_ROOT_DEFAULT}: refusing to bake it into the "
            "real corpus (pass --allow-other-data-root if intended)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--sources", type=Path, help="dir with glm/ and task_tests.json (default <out>/sources)")
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT, help="the corpus root on the training cluster (AGA)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--invalid-policy", choices=("mask", "keep"), default="keep")
    ap.add_argument("--allow-other-data-root", action="store_true",
                    help="the real corpus dir with a --data-root other than DESIGN's AGA DATA_ROOT")
    args = ap.parse_args()
    out = args.out.resolve()
    src_dir = (args.sources or out / "sources").resolve()
    data_root = args.data_root.rstrip("/")
    assert data_root.startswith("/"), "--data-root must be absolute"
    os.umask(0o027)
    started = time.time()

    report = json.loads((src_dir / "glm" / "selection_report.json").read_text())
    dry_run = bool(report["partial"])
    if out == OUT_DEFAULT.resolve() and dry_run:
        raise SystemExit("refusing a dry-run build (partial sources) into the real corpus dir")
    if out == OUT_DEFAULT.resolve() and (out / SCHEME).exists():
        raise SystemExit(f"{out / SCHEME} exists: refusing to overwrite a built corpus")
    refusal = data_root_refusal(out, data_root, args.allow_other_data_root)
    if refusal:
        raise SystemExit(refusal)
    traj_path = src_dir / "glm" / "trajectories.jsonl.gz"
    harness_path = src_dir / "glm" / "harness.json"
    input_sha = {SOURCE_PART: sha256_file(traj_path), "sources/glm/harness.json": sha256_file(harness_path),
                 "sources/glm/selection_report.json": sha256_file(src_dir / "glm" / "selection_report.json"),
                 "sources/task_tests.json": sha256_file(src_dir / "task_tests.json")}
    for name in ("classes.json", "results.json"):
        p = src_dir / "regrade" / name
        input_sha[f"sources/regrade/{name}"] = sha256_file(p) if p.exists() else None
    with gzip.open(traj_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == report["kept"], (len(lines), report["kept"])
    tasks = list(enumerate(lines))
    del lines
    print(f"converting {len(tasks)} rollouts", flush=True)

    infos, convs, selftests = [], {}, []
    with get_context("fork").Pool(args.workers, initializer=_init,
                                  initargs=(args.invalid_policy, str(harness_path))) as pool:
        for n, (samples, selftest) in enumerate(pool.imap(work, tasks, chunksize=1), 1):
            for info, messages in samples:
                infos.append(info)
                assert info["trajectory_id"] not in convs, info["trajectory_id"]
                convs[info["trajectory_id"]] = messages
            if selftest:
                selftests.append(selftest)
            print(f"[{n}/{len(tasks)}] {samples[0][0]['parent_trajectory_id']} samples={len(samples)} "
                  f"tokens={[i['token_length'] for i, _ in samples]} converter={samples[0][0]['converter']}",
                  flush=True)
    del tasks
    infos.sort(key=lambda i: (i["line_no"], i["segment"]))
    assert len({i["trajectory_id"] for i in infos}) == len(infos), "sample id collision"
    by_parent = defaultdict(list)
    for i in infos:
        by_parent[i["parent_trajectory_id"]].append(i)

    # v2 sample invariants re-checked on the emitted messages (both converters)
    lead_ws = 0
    for i in infos:
        ms = convs[i["trajectory_id"]]
        assert ms[0]["role"] == "system" and sum(1 for m in ms if m["role"] == "system") == 1
        assert ms[-1]["role"] == "assistant"
        for j, m in enumerate(ms):
            if m["role"] == "assistant":
                assert m["content"] and ms[j - 1]["role"] in ("user", "tool"), (i["trajectory_id"], j)
                lead_ws += m["content"][:1].isspace()
            elif m["role"] in ("user", "tool"):
                assert m["content"].endswith(GEN_PREFIX), (i["trajectory_id"], j)
        assert set(ms[0]) == {"role", "content", "weight", "step_weight_agent_run_id", "assistant_weights",
                              "assistant_token_counts"}
        assert all(set(m) == {"role", "content"} for m in ms[1:])

    # v3 literal stop-token rule (2026-10-29): a literal generation stop text in an assistant target -> drop the sample
    stops, bos_text = _stop_strings()
    stop_texts = [t for _, t in stops]
    for i in infos:
        asst = [m["content"] for m in convs[i["trajectory_id"]] if m["role"] == "assistant"]
        i["assistant_literal_eos"] = sum(c.count(t) for c in asst for t in stop_texts)
        i["assistant_literal_bos"] = sum(c.count(bos_text) for c in asst)
    eos_dropped = [i for i in infos if i["assistant_literal_eos"]]
    dropped = [i for i in infos if ceil32(i["token_length"]) > SEQ_LEN]
    kept = [i for i in infos if ceil32(i["token_length"]) <= SEQ_LEN and not i["assistant_literal_eos"]]
    assert kept, "no sample left"

    # ---- packing (v2 / v3): FFD on ceil32 lengths, seeded shuffle of the row order, 6 contiguous near-equal shards
    rows = ffd([(i["trajectory_id"], ceil32(i["token_length"])) for i in kept])
    random.Random(SHUFFLE_SEED).shuffle(rows)
    n_rows = len(rows)
    base, extra = divmod(n_rows, N_SHARDS)
    sizes = [base + (1 if s < extra else 0) for s in range(N_SHARDS)]
    assert min(sizes) >= 1, "fewer rows than shards"
    by_id = {i["trajectory_id"]: i for i in kept}
    for i in kept:
        i["flags"] = i["meta"].get("flags")

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
                    by_id[key]["row_samples"] = len(row)
        os.replace(tmp, path)
        cursor += sizes[s]
        shards.append({"bytes": path.stat().st_size, "packed_rows": sizes[s],
                       "path": f"{data_root}/{SCHEME}/{SHARD_NAME.format(s)}"})
    assert cursor == n_rows

    # Re-read every shard and re-verify the packing / weight invariants exactly as the dataset reads them (v2 / v3),
    # plus: the 6 system keys on every sample, one system message per sample, ids unique
    seen_ids = []
    for s in range(N_SHARDS):
        with open(out / SCHEME / SHARD_NAME.format(s), encoding="utf-8") as f:
            for line in f:
                ms = json.loads(line)["messages"]
                assert ms[0]["role"] == "system"
                for m in ms:
                    if m["role"] == "system":
                        assert set(m) == {"role", "content", "weight", "step_weight_agent_run_id", "assistant_weights",
                                          "assistant_token_counts"}
                        assert len(m["assistant_weights"]) == len(m["assistant_token_counts"]) > 0
                        assert all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in m["assistant_weights"])
                        seen_ids.append(m["step_weight_agent_run_id"])
                    else:
                        assert set(m) == {"role", "content"}
    assert sorted(seen_ids) == sorted(by_id) and len(set(seen_ids)) == len(seen_ids)

    blend = {"train": [x for s in shards for x in ("0.16666667", s["path"])], "valid": None, "test": None}
    (out / "blends" / f"{SCHEME}.json").write_text(json.dumps(blend, indent=2) + "\n")

    # ---- selection (HSG schema fields are strings, as v2 / v3) + v3's fields + the GLM / window fields
    sel_path = out / "selection" / SELECTION_NAME
    with open(sel_path, "w", encoding="utf-8") as f:
        for i in kept:  # rollout line order, then segment
            m = i["meta"]
            w = m["window"]
            fl = i.get("flags") or {}
            f.write(json.dumps({
                "agent_run_id": i["trajectory_id"],
                "record_id": "r_" + hashlib.sha256(i["trajectory_id"].encode()).hexdigest()[:24],
                "source_line": str(i["line_no"]),
                "source_part": SOURCE_PART,
                "token_length": str(i["token_length"]),
                "training_row_sha256": i["training_row_sha256"],
                "source": i["source"],
                "run": m["run"], "task_id": m["task_id"], "slot": m["slot"],
                "included_because": m["included_because"], "grader": m["grader"],
                "grader_detail": m.get("grader_detail"), "status_source": m.get("status_source"),
                "f2p": m.get("f2p"), "p2p": m.get("p2p"), "attempt_dir": m.get("attempt_dir"),
                "teacher_audit": m.get("teacher_audit"),
                "harness_system_sha256": m.get("harness_system_sha256"),
                "harness_tools_sha256": m.get("harness_tools_sha256"),
                "harness_params_sha256": m.get("harness_params_sha256"),
                "sanitizer_noop": m["sanitizer_noop"], "grading_note": m["grading_note"],
                "n_assistant": i["n_assistant"], "assistant_tokens": i["assistant_tokens"],
                "masked_assistant_turns": i["masked_assistant_turns"],
                "webfetch_calls": i["stats"].get("webfetch_calls", 0),
                "shard": i["shard"], "shard_row": i["shard_row"], "row_position": i["row_position"],
                "rollout_batch": m["rollout_batch"], "campaign": m.get("campaign"), "rerun": m.get("rerun"),
                "flags_any": bool(fl.get("any")), "flags_strong": bool(fl.get("strong")),
                "parent_trajectory_id": i["parent_trajectory_id"], "segment": i["segment"],
                "n_segments": i["n_segments"], "context_tokens": i["context_tokens"],
                "first_call_index": w["first_call_index"], "last_call_index": w["last_call_index"],
                "system_changed_in_window": w["system_changed_in_window"],
                "system_change_kinds": w["system_change_kinds"],
                "system_changed_targets": w["system_changed_targets"],
                "system_content_changed_targets": w["system_content_changed_targets"],
                "post_final_continue": w["post_final_continue"], "converter": i["converter"],
                "tier": m["tier"], "task_status": m["task_status"], "audit": m["audit"],
                "glm_verdict": m["glm_verdict"], "grader_kind": m["grader_kind"],
                "test_touch_class": m["test_touch_class"], "regrade": m["regrade"],
            }, ensure_ascii=False) + "\n")
    selection_sha = sha256_file(sel_path)

    # ---- metadata.jsonl: every sample (kept or dropped at build) + every excluded final record, rollout id order
    meta_rows = []
    for i in infos:
        row = dict(i["meta"])
        row["corpus"] = {k: i.get(k) for k in (
            "source", "line_no", "token_length", "assistant_tokens", "n_assistant", "n_tool_messages", "n_tool_calls",
            "system_tokens", "user_tokens", "context_tokens", "masked_assistant_turns", "training_row_sha256",
            "shard", "shard_row", "row_position", "row_padded_tokens", "row_samples", "segment", "n_segments",
            "converter")}
        row["corpus"]["rollout_batch"] = i["meta"]["rollout_batch"]
        row["corpus"]["stats"] = i["stats"]
        row["corpus"]["flags"] = i["meta"].get("flags")
        row["corpus"]["dropped"] = i in dropped or bool(i["assistant_literal_eos"])
        row["corpus"]["dropped_reason"] = ("over_524288" if i in dropped else
                                           "assistant_literal_eos" if i["assistant_literal_eos"] else None)
        row["corpus"]["assistant_literal_eos"] = i["assistant_literal_eos"]
        row["corpus"]["assistant_literal_bos"] = i["assistant_literal_bos"]
        meta_rows.append(((i["parent_trajectory_id"], i["segment"]), row))
    for e in report["exclusions"]:
        row = dict(e)
        row["corpus"] = {"source": SOURCE, "rollout_batch": e["rollout_batch"], "dropped": True,
                         "dropped_reason": "selection:" + e["reason"]}
        meta_rows.append(((e["trajectory_id"], 0), row))
    meta_rows.sort(key=lambda kv: kv[0])
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for _, row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- stats and receipt
    tot = Counter()
    for i in kept:
        tot.update(i["stats"])
    tl = [i["token_length"] for i in kept]
    seg_kind = Counter(f"{i['segment']}of{i['n_segments']}" for i in kept)
    ctx_tokens = sorted(i["context_tokens"] for i in kept if i["segment"] >= 2)
    parents_rows = Counter()
    for r in rows:
        c = Counter(by_id[k]["parent_trajectory_id"] for k in r)
        parents_rows["rows_with_two_or_more_windows_of_one_rollout"] += any(v >= 2 for v in c.values())
    windows_stats = {
        "samples_by_segment": dict(sorted(seg_kind.items())),
        "rollouts_by_n_segments": dict(sorted(Counter(str(i["n_segments"]) for i in kept if i["segment"] == 1).items())),
        "samples_k_ge_2": sum(1 for i in kept if i["segment"] >= 2),
        "post_final_continue_samples": [i["trajectory_id"] for i in kept if i["meta"]["window"]["post_final_continue"]],
        "system_changed_in_window_samples": sum(1 for i in kept if i["meta"]["window"]["system_changed_in_window"]),
        "system_change_kinds": dict(sorted(Counter("+".join(i["meta"]["window"]["system_change_kinds"]) for i in kept
                                                   if i["meta"]["window"]["system_change_kinds"]).items())),
        "system_content_changed_samples": [
            {"trajectory_id": i["trajectory_id"], "kinds": i["meta"]["window"]["system_change_kinds"],
             "first_change_call": i["meta"]["window"]["system_first_change_call"],
             "content_changed_targets": i["meta"]["window"]["system_content_changed_targets"],
             "changed_targets": i["meta"]["window"]["system_changed_targets"],
             "targets": i["meta"]["window"]["n_targets"]}
            for i in kept if i["meta"]["window"]["system_content_changed_targets"]],
        "system_changed_targets_total": sum(i["meta"]["window"]["system_changed_targets"] for i in kept),
        "system_content_changed_targets_total": sum(i["meta"]["window"]["system_content_changed_targets"]
                                                    for i in kept),
        "target_calls_total": sum(i["meta"]["window"]["n_targets"] for i in kept),
        "window_end_tool_results": dict(sorted(Counter(i["meta"]["window"]["end_tool_results"] for i in kept
                                                       if i["meta"]["window"]["end_tool_results"]).items())),
        "window_end_results_in_no_sample": [i["trajectory_id"] for i in kept
                                            if i["meta"]["window"]["end_tool_results"] == "in_no_sample"],
        "context_tokens_k_ge_2": ({"min": ctx_tokens[0], "median": ctx_tokens[len(ctx_tokens) // 2],
                                   "max": ctx_tokens[-1], "total": sum(ctx_tokens)} if ctx_tokens else None),
        "dropped_turns": dict(Counter(d["reason"] for i in kept for d in i["meta"]["window"]["dropped"])),
        "dropped_turn_list": [{"trajectory_id": i["trajectory_id"], **d} for i in kept
                              for d in i["meta"]["window"]["dropped"]],
        "samples_ending_with_tool_calls": sum(1 for i in kept if i["meta"]["window"]["ends_with"] == "tool_calls"),
        **parents_rows,
    }
    route_stats = {"samples_by_converter": dict(Counter(i["converter"] for i in kept)),
                   "self_test": {"rule": "segment converter == v2 convert() on every v2_convert pass (messages, "
                                         + ", ".join(SELFTEST_FIELDS) + ")",
                                 "checked": len(selftests), "equal": sum(1 for x in selftests if x["equal"]),
                                 "non_compacted_segment_route": sorted(
                                     {i["parent_trajectory_id"] for i in infos
                                      if i["n_segments"] == 1 and i["converter"] == "segment"})}}
    selection_keys = ("partial", "only", "limit_per_batch", "considered", "considered_all_tier12_records",
                      "considered_by_batch", "considered_by_tier", "final_status", "exclusion_counts",
                      "exclusion_counts_by_batch", "independent_counts_among_final_passes",
                      "old_grader_among_final_passes", "old_grader_inputs", "kept", "kept_by_batch", "kept_by_tier",
                      "kept_by_class", "kept_by_campaign", "kept_reruns", "kept_reruns_by_kind",
                      "kept_reruns_by_campaign", "reruns_unresolved", "rerun_mapping_entries",
                      "rerun_mapping_files_unreadable", "kept_tasks",
                      "tasks_considered", "kept_windows", "kept_rollouts_by_windows", "kept_routes",
                      "kept_post_final_continue_windows", "kept_system_changed_windows", "kept_system_change_kinds",
                      "kept_system_content_changed_windows", "kept_system_changed_targets",
                      "kept_system_content_changed_targets", "kept_window_end_tool_results", "kept_dropped_turns",
                      "language_rule", "languages_considered_tasks", "language_mismatch_with_classes",
                      "harness_comparison_with_v1", "harness_missing", "first_user_mismatch", "patch_mismatch_record",
                      "redactions_trajectories", "redactions_harness", "v3_sources", "skipped_sftv3_not_tier12",
                      "inputs_sha256")
    stats = {
        "per_batch": {b: group_stats([i for i in kept if i["meta"]["rollout_batch"] == b], rows, by_id)
                      for b in BATCHES},
        "per_tier": {str(t): group_stats([i for i in kept if i["meta"]["tier"] == t], rows, by_id) for t in (1, 2)},
        "per_old_grader_class": {c: group_stats([i for i in kept if i["meta"]["test_touch_class"] == c], rows, by_id)
                                 for c in ("none", "A", "B")},
        "samples": len(kept), "rollouts": len({i["parent_trajectory_id"] for i in kept}),
        "tasks": len({i["meta"]["task_key"] for i in kept}),
        "dropped": [i["trajectory_id"] for i in dropped], "packed_rows": n_rows, "rows_per_shard": sizes,
        "samples_per_row_max": max(len(r) for r in rows),
        "tokens_total": sum(tl), "tokens_padded_total": sum(ceil32(x) for x in tl),
        "assistant_tokens_total": sum(i["assistant_tokens"] for i in kept), "assistant_tokens_weighted": None,
        "tokens_max": max(tl), "tokens_min": min(tl), "tokens_mean": sum(tl) / len(tl),
        "tokens_median": sorted(tl)[len(tl) // 2], "tokens_p90": pct(tl, 0.9),
        "over_262144": sum(1 for x in tl if x > V2_MAX_TOKENS),
        "over_262144_by_segment": dict(sorted(Counter(f"{i['segment']}of{i['n_segments']}" for i in kept
                                                      if i["token_length"] > V2_MAX_TOKENS).items())),
        "max_sample": max(kept, key=lambda x: x["token_length"])["trajectory_id"],
        "assistant_messages": sum(i["n_assistant"] for i in kept),
        "tool_messages": sum(i["n_tool_messages"] for i in kept), "tool_calls": sum(i["n_tool_calls"] for i in kept),
        "row_fill": sum(ceil32(x) for x in tl) / (n_rows * SEQ_LEN),
        "rows_by_batch_mix": dict(Counter("+".join(sorted({by_id[k]["meta"]["rollout_batch"] for k in r}))
                                          for r in rows)),
        "windows": windows_stats, "routes": route_stats,
        "edge_case_counts": dict(sorted(tot.items())),
        "assistant_pieces_leading_whitespace": lead_ws,
        "train_samples_5epoch": epochs_table(n_rows)["5"]["train_samples"],
        "target_iteration_5epoch": epochs_table(n_rows)["5"]["target_iteration"],
        "epochs_gbs32": epochs_table(n_rows),
        "invalid_policy": args.invalid_policy, "shuffle_seed": SHUFFLE_SEED, "data_root": data_root,
        "data_root_design": DATA_ROOT_DEFAULT, "data_root_is_design": data_root == DATA_ROOT_DEFAULT,
        "assistant_literal_stop_rule": {
            "rule": "drop a sample whose assistant targets contain the literal text of a generation stop token "
                    "(generation_config.json eos_token_id other than the template turn end '<|im_end|>'): sc14's "
                    "tokenizer turns it into that special id, i.e. EOS mid-turn (v3 rule)",
            "stop_tokens": [{"id": i, "text": t} for i, t in stops],
            "dropped": [{"trajectory_id": i["trajectory_id"], "token_length": i["token_length"],
                         "literal_eos": i["assistant_literal_eos"], "literal_bos": i["assistant_literal_bos"]}
                        for i in eos_dropped],
            "bos_text": bos_text,
            "literal_bos_kept": [{"trajectory_id": i["trajectory_id"], "occurrences": i["assistant_literal_bos"]}
                                 for i in kept if i["assistant_literal_bos"]]},
        "selection": {k: report.get(k) for k in selection_keys},
        "dry_run": dry_run,
        "converter": {"file": str(V2_BUILDER), "file_sha256": V2_BUILDER_SHA256,
                      "block": "def enc .. def work (enc, fix_surrogates, arg_str, render_calls, convert)",
                      "block_sha256": CONVERTER_SHA256},
        "segment_converter": {"file": "tools/build_glm.py", "function": "convert_window",
                              "file_sha256": sha256_file(Path(__file__).resolve()),
                              "render": "transformers apply_chat_template, truncate_history_thinking=False"},
        "v3_helpers": {"file": str(V3_BUILD), "file_sha256": V3_BUILD_SHA256, "blocks_sha256": V3_BLOCKS_SHA256},
        "inputs_sha256": input_sha,
    }
    wsum = 0
    for key, ms in convs.items():
        if key in by_id:
            s0 = ms[0]
            wsum += sum(c * w_ for c, w_ in zip(s0["assistant_token_counts"], s0["assistant_weights"]))
    stats["assistant_tokens_weighted"] = int(wsum)

    receipt = {
        "blend": f"{data_root}/blends/{SCHEME}.json",
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filter_policy": "GLM-5.3 (self-hosted, OpenCode 1.17.11, 262,144 window, compaction at 242,144) final records "
                         "of the tier-1/2 tasks (glm-rollouts-tasksall-20261108 tier 1 / 2, glm-rollouts-sftv3-20261107 "
                         "all): final pass -> clean completion -> old-grader rule (class B passes need a "
                         "current-verifier re-grade with reward 1) -> no degenerate reply anywhere (degenerate.py "
                         "classify over every hub reply incl. subagents and summary-writing calls) -> v1 harness base "
                         "and tool set -> two-source gate (export.json rebuild == http.jsonl prefix chain for every "
                         "window), see sources/glm/selection_report.json; compacted rollouts are cut into their OpenCode "
                         "context windows, one sample each (the post-compaction context as one non-target prompt "
                         "piece); turns never trained: empty / errored export messages, finish 'length' replies, "
                         "summary-writing calls, task subagents; samples over 524288 tokens dropped; samples with the "
                         "literal text of a generation stop token ('</s>', generation_config eos id 2) in an assistant "
                         "target dropped (build_stats.assistant_literal_stop_rule); no LLM audit (informational flags)",
        "packed_rows": n_rows,
        "physical_bytes": sum(s["bytes"] for s in shards),
        "raw_weight_policy": "all assistant action weights = 1.0" + (
            "" if args.invalid_policy == "keep" else " except masked 'invalid' tool turns"),
        "schema": SCHEMA,
        "selected_trajectories": len(kept),
        "selection": f"{data_root}/selection/{SELECTION_NAME}",
        "selection_sha256": selection_sha,
        "shards": shards,
        "source_cluster": "cw-dfw",
        "state": "dry-run" if dry_run else "complete",
        "data_root": data_root,
        "sample_unit": "one sample (raw_sft conversation, one system message) per OpenCode context window; "
                       "selected_trajectories counts samples",
    }
    if args.invalid_policy == "mask":
        receipt["raw_weight_exceptions"] = (
            f"{stats['edge_case_counts'].get('invalid_tool_turns_masked', 0)} assistant actions whose tool call "
            "opencode rewrote to the 'invalid' tool carry weight 0.0 (context only)")
    receipt["build_stats"] = stats
    (out / "DATA_PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme(receipt, report))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_batch", "per_tier", "per_old_grader_class",
                                                                  "selection", "inputs_sha256", "windows")},
                     indent=1, ensure_ascii=False))
    print(json.dumps({"windows": {k: v for k, v in windows_stats.items() if k != "dropped_turn_list"},
                      "per_batch": {b: {k: v for k, v in p.items() if k != "flags"}
                                    for b, p in stats["per_batch"].items()}}, indent=1))
    print(f"build {time.time() - started:.1f} s", flush=True)
    return 0


# ---------------------------------------------------------------- README (generated from the receipt)
def _n(x):
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def readme(receipt, report):
    st = receipt["build_stats"]
    sel = st["selection"]
    ws = st["windows"]
    ep = st["epochs_gbs32"]
    lsr = st["assistant_literal_stop_rule"]
    og = sel.get("old_grader_inputs") or {}
    ind = sel.get("independent_counts_among_final_passes") or {}
    data_root = st["data_root"]
    groups = [("tasksall", st["per_batch"]["tasksall"]), ("sftv3", st["per_batch"]["sftv3"]),
              ("tier 1", st["per_tier"]["1"]), ("tier 2", st["per_tier"]["2"]),
              ("old grader: none", st["per_old_grader_class"]["none"]), ("class A", st["per_old_grader_class"]["A"]),
              ("class B", st["per_old_grader_class"]["B"])]

    def row(label, fn):
        cells = []
        for _, g in groups:
            try:
                cells.append(fn(g) if g.get("samples") else "-")
            except (KeyError, TypeError):
                cells.append("-")
        return "| " + label + " | " + " | ".join(cells) + " |"

    excl = sel.get("exclusion_counts_by_batch") or {}
    reasons = sorted({r for b in excl.values() for r in b})
    lines = [
        "# sft-glm-tier12-20261109: GLM-5.3 tier-1/2 SFT corpus (successful OpenCode rollouts, compaction windows as "
        "independent samples, raw_sft scheme)",
        "",
        ("**DRY RUN** (partial sources: `--only` / `--limit` / a non-default `--regrade-results`): not for training. "
         if st["dry_run"] else "")
        + "User request (hengyuf, GitHub issue hengyuf/SWE-task#1, comment 6758873905, verbatim): \"现在收集tier1/2 的 "
        "successful traces （包括有compaction的）， 将其加工为类似SFT v3 一样的dataset， 这里对于一个rollout 的多个segment"
        "（有compaction的）， 将其视为多条独立的traces，task ID 可以用一个用于标记，但训练时保留各自边界。标准 SFT 可以把 "
        "segment 当独立样本，做packing时，用现有 packer 的边界/attention 语义，别让 loss 或上下文跨样本串起来。pack 好data "
        "把data 用datamover 移动到 AGA（oci-aga-slurm-1-login-01.nvidia.com）上。（AGA 访问密码与HSG 相同）\"",
        "",
        "Design: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/glm-tier12-20261109/DESIGN.md` (rev 2). "
        "The on-disk format is SFT-v3's (`sft-v3-20261029`, itself kimi-pass-v2's), i.e. zhenghaox's HSG Super SFT `raw_sft` "
        "DATA_ROOT: v2's converter (exec'd from its source text, unchanged) for every non-compacted pass that ends in a "
        "final answer, a segment converter with v2's per-turn rules for the context windows, the sc14 tokenizer and chat "
        "template, identity prompt format, the same `weight` / `step_weight_agent_run_id` / `assistant_weights` / "
        "`assistant_token_counts` fields in each sample's system message, and the same packing into 524,288-token rows. "
        "Every OpenCode context window is its OWN sample (one system message, its own loss weights); the task / rollout is "
        "only a marker (`run`, `task_id`, `slot`, `parent_trajectory_id` in the selection and metadata). Data only.",
        "",
        f"- AGA DATA_ROOT: `{data_root}` (baked into `blends/raw_sft.json`, the receipt's shard paths and verify's path "
        "check; " + ("DESIGN's AGA DATA_ROOT" if st["data_root_is_design"] else
                     f"**not DESIGN's AGA DATA_ROOT `{st['data_root_design']}`**") + ").",
        f"- Built on cw-dfw with `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B`, CPU only, no network "
        f"({receipt['completed_at']}).",
        "",
        "## Files",
        "| path | what |",
        "|---|---|",
        f"| `raw_sft/part_00{{0..5}}.mat.jsonl.packed.512k` | 6 shards, {_n(st['packed_rows'])} packed rows "
        f"({'/'.join(map(str, st['rows_per_shard']))}). Each line is `{{\"messages\": [...]}}` |",
        "| `blends/raw_sft.json` | `{\"train\": [\"0.16666667\", <DATA_ROOT part_000>, ...], \"valid\": null, \"test\": null}` |",
        "| `DATA_PREP_COMPLETE.json` | Receipt with the HSG fields plus `build_stats` (`per_batch`, `per_tier`, "
        "`per_old_grader_class`, `windows`, `routes` (self-test), `selection`, `epochs_gbs32`) |",
        f"| `selection/{SELECTION_NAME}` | {_n(st['samples'])} rows (one per sample): v3's selection fields + "
        "`parent_trajectory_id`, `segment`, `n_segments`, `context_tokens`, `first_call_index`, `last_call_index`, "
        "`system_changed_in_window`, `system_change_kinds`, `system_changed_targets`, `system_content_changed_targets`, "
        "`post_final_continue`, `tier`, `task_status`, `audit`, `glm_verdict`, `grader_kind`, `test_touch_class`, "
        "`regrade`, `converter` |",
        "| `metadata.jsonl` | One row per sample (its rollout meta + window info + a `corpus` block) and one row per "
        "excluded final record (`corpus.dropped_reason` = `selection:<rule>`), rollout id order |",
        "| `sources/glm/` | `trajectories.jsonl.gz` (v1 batch line layout `{meta, trajectory, final_patch}`, redacted; "
        "`meta.glm.windows` = the window info incl. the post-compaction contexts), `harness.json`, `selection_report.json` "
        "(every considered final record kept or excluded with its first failing rule, independent counts) |",
        "| `sources/regrade/` | Byte copies of the old-grader inputs read: `classes.json`"
        + (", `results.json`" if og.get("results", {}).get("exists") else " (no `results.json` existed)") + " |",
        "| `sources/task_tests.json` | Per task (inventory key): test.patch / grader.py sha256, the test files test.patch "
        "creates (flags) |",
        "| `tools/sources_glm.py` | Selection + extraction (v3's `sources_v4v5.py` imported for examine / flags / "
        "journal / meta / write_batch; the two-source gate) |",
        "| `tools/build_glm.py` | Converter (v2's `convert()` + the window converter), filters, packing, receipt, "
        "this README. Deterministic |",
        "| `tools/verify_glm.py` | Independent re-check (below) |",
        "| `tools/sim_sft_getitem.py` | Copy of scratch/packing-semantics/tools/sim_sft_getitem.py: the real pinned "
        "`SFTDataset.__getitem__` on one packed row (verify runs it) |",
        "| `tools/glm_sft.sbatch` | The CPU job (schedule helper babysit pair) running the three tools + SHA256SUMS "
        "(modes dryrun / build / rebuild / verify) |",
        "| `SHA256SUMS` | sha256 of every file above |",
        "",
        "Training reads only `blends/` + `raw_sft/`.",
        "",
        "## Sources (read-only)",
        "- **tasksall**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/"
        "analysis/rollouts.jsonl` (final records, every re-run applied), tier 1 and 2.",
        "- **sftv3**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/analysis/"
        "final-rollouts.jsonl` (the 220 SFT-v3 tasks, all tier 1).",
        "- Task facts: `sft/tasks-all-20261106/all_tasks.jsonl` (tier 1 = accepted_or_strong, tier 2 = audit partial) and "
        "`glm-rollouts-tasksall-20261108/analysis/tier12-accept-compare.json` (GLM-teacher verdict, informational).",
        "- Old-grader classes and re-grades: `glm-tier12-20261109/regrade/classes.json`, `regrade/results.json`.",
        "- Student template and tokenizer: sc14 `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (as v1 - v3).",
        "- Not included: the hosted GLM-5.3 rollouts of the 2 official1-scriggo gomacro tasks (teacher-rollouts-v2: hub "
        "serving, 1M context, 131k output cap).",
        "",
        "## Selection (`tools/sources_glm.py`)",
        "Per final record of a tier-1/2 task, the first failing rule is the exclusion reason: " + str(report.get("rules"))
        + ".",
        "",
        "| | tasksall | sftv3 |",
        "|---|---|---|",
        f"| final records considered | {(sel.get('considered_by_batch') or {}).get('tasksall', 0)} | "
        f"{(sel.get('considered_by_batch') or {}).get('sftv3', 0)} |",
        f"| kept rollouts | {(sel.get('kept_by_batch') or {}).get('tasksall', 0)} | "
        f"{(sel.get('kept_by_batch') or {}).get('sftv3', 0)} |",
    ]
    lines += [f"| excluded: {r} | {excl.get('tasksall', {}).get(r, 0)} | {excl.get('sftv3', {}).get(r, 0)} |"
              for r in reasons]
    lines += [
        "",
        "Independent counts among final passes (not first-failing): " + ", ".join(
            f"{k} {v}" for k, v in sorted(ind.items())) + ".",
        "",
        f"Re-runs among the kept rollouts (final records whose attempt lies in another probe campaign): "
        f"{sel.get('kept_reruns', 0)} (" + (", ".join(f"{k} {v}" for k, v in sorted(
            (sel.get("kept_reruns_by_kind") or {}).items())) or "none")
        + "); `rerun_campaign` / `rerun_slot` / `rerun_reason` / `original_attempt_dir` / `original_status` in the metadata "
        "come from the campaigns' re-run mapping files (tasksall `results/glmlocal-dip*`, sftv3 "
        "`results/glmlocal-infra-rerun1.json`, the drivers' `sandbox_failure_reruns`); unresolved: "
        + str(len(sel.get("reruns_unresolved") or [])) + ". Task `language` = " + str(sel.get("language_rule"))
        + f"; mismatches with classes.json: {len(sel.get('language_mismatch_with_classes') or [])}.",
        "",
        "**Two-source gate** (hard): every window rebuilt from `export.json` (OpenCode 1.17.11 + Gym semantics; port of "
        "`scratch/glm-compaction/segments.py`) equals the `http.jsonl` prefix chain (the window's last request + its "
        "response) message by message (role, content, reasoning, tool calls with parsed arguments and ids, tool results); "
        "every transition of the chain adds exactly [the previous response, its tool results]; #windows == #main "
        "compaction calls + 1 == #compaction parts + 1; no branch point, no main-session call outside the windows.",
        "",
        "## Old-grader rule",
        "Tasks graded by DeepSWE's old `grader.py` (no test-path sanitizer) are classified by `regrade/classes.json` "
        f"(counts {json.dumps((og.get('classes') or {}).get('counts'))}): class A (no effective test-path change, or "
        "only harmless new test files) is kept; class B (an existing test file modified / deleted, or a risky new file) "
        "enters only with a current-verifier re-grade of reward 1 (`regrade/results.json`: "
        + ((f"{og['results']['n_results']} results, sha256 `{og['results']['sha256']}`, updated_at "
            f"{og['results'].get('updated_at')}; class B passes without a result: "
            f"{og['results'].get('class_b_without_result')}"
             + ("" if og["results"].get("default_path", True) else
                f"; **NOT the re-grade: a non-default results file `{og['results']['path']}` (dry-run input)**"))
           if (og.get("results") or {}).get("exists")
           else "**did not exist at selection time: every class B pass is excluded as old_grader_not_regraded**")
        + "); a class A / current-grader pass whose check re-grade returned reward 0 is excluded too. Among final passes: "
        + ", ".join(f"{k} {v}" for k, v in sorted((sel.get("old_grader_among_final_passes") or {}).items())) + ".",
        "",
        "## Samples: one per OpenCode context window",
        "- **Non-compacted pass ending in a final answer** -> 1 sample = v2's `convert()` unchanged. The window converter "
        "reproduces it byte for byte on every such pass (build-time self-test: "
        f"{st['routes']['self_test']['equal']} of {st['routes']['self_test']['checked']} equal).",
        "- **Window samples** (every window of a compacted pass, and a non-compacted pass that does not end in a valid final "
        "answer: " + (", ".join(f"`{x}`" for x in st["routes"]["self_test"]["non_compacted_segment_route"]) or "none")
        + "): window 1 = v2's system and user (task) pieces + the window's target turns; window k >= 2 = the system piece "
        "of the window's first request + ONE non-target prompt piece holding OpenCode's post-compaction context (\"What "
        "did we do so far?\", the summary with its reasoning, the tail, the synthetic \"Continue ...\" message; rendered "
        "with `truncate_history_thinking=False`, what GLM saw and what the student's serving uses) + the window's own "
        "target turns. A window's last target keeps its tool calls; their results are not in that sample. They reappear in "
        "the next window's context piece only when OpenCode's tail keeps that turn ("
        + f"{ws['window_end_tool_results'].get('next_window_context', 0)} samples); otherwise they are in no sample "
        f"({ws['window_end_tool_results'].get('in_no_sample', 0)} samples: the next window's tail does not hold that turn, "
        "or no window follows; GLM saw those results only in calls that are not targets: a dropped or unanswered reply, "
        "the summary-writing call"
        + (": " + ", ".join(f"`{x}`" for x in ws["window_end_results_in_no_sample"])
           if ws["window_end_results_in_no_sample"] else "") + ").",
        f"- Samples by window (`<k>of<K>`): " + ", ".join(f"{k} {v}" for k, v in ws["samples_by_segment"].items())
        + f"; rollouts by number of windows: " + ", ".join(f"{k} {v}" for k, v in ws["rollouts_by_n_segments"].items())
        + f". Context piece of windows k >= 2: {json.dumps(ws['context_tokens_k_ge_2'])} sc14 tokens.",
        "- Never targets: " + (", ".join(f"{k} {v}" for k, v in ws["dropped_turns"].items()) or "none") + " ("
        + "; ".join(f"`{d['trajectory_id']}` {d['reason']}" for d in ws["dropped_turn_list"]) + "); OpenCode's "
        "summary-writing (compaction) calls and task-subagent sessions are not samples (a subagent's report is its "
        "parent's tool result, as v1 - v3).",
        f"- Tagged, kept: `post_final_continue` = a window after \"final answer, then compaction\" "
        f"({len(ws['post_final_continue_samples'])} samples: "
        + (", ".join(f"`{x}`" for x in ws["post_final_continue_samples"]) or "none")
        + f"); `system_changed_in_window` ({ws['system_changed_in_window_samples']} samples: the system prompt of a "
        "later request of the window differs; the sample uses the window's first request). Parts that change "
        "(`system_change_kinds`: `date` = the env block's date line, `env` = another env line, `instructions` = "
        "AGENTS.md / CLAUDE.md content or the skills list, `base` = the base instructions): "
        + (", ".join(f"{k} {v}" for k, v in ws["system_change_kinds"].items()) or "none")
        + f" samples; {_n(ws['system_changed_targets_total'])} of {_n(ws['target_calls_total'])} target calls were "
        f"made with another system prompt than their sample's, {_n(ws['system_content_changed_targets_total'])} of them "
        f"with other content than the date line (`system_content_changed_targets`), in "
        f"{len(ws['system_content_changed_samples'])} samples"
        + (": " + ", ".join(f"`{x['trajectory_id']}` ({x['content_changed_targets']} of {x['targets']} targets, "
                            f"{'+'.join(x['kinds'])})" for x in ws["system_content_changed_samples"])
           if ws["system_content_changed_samples"] else "") + ".",
        f"- Rows holding two or more windows of one rollout: {ws.get('rows_with_two_or_more_windows_of_one_rollout', 0)} "
        "(safe: the trainer splits a row at system messages into THD segments).",
        "",
        "## Sample filters",
        f"- Literal stop-token rule (v3): a literal `</s>` (generation_config eos id 2) in an assistant target would be "
        f"EOS mid-turn: **{len(lsr['dropped'])} sample(s) dropped**: "
        + ("; ".join(f"`{d['trajectory_id']}` ({d['literal_eos']} x `</s>`, {d['literal_bos']} x `<s>`)"
                     for d in lsr["dropped"]) or "none")
        + f". Literal BOS `{lsr['bos_text']}` kept and counted: {sum(b['occurrences'] for b in lsr['literal_bos_kept'])} "
        f"occurrences in {len(lsr['literal_bos_kept'])} samples.",
        f"- ceil32(tokens) > 524,288: {len(st['dropped'])} dropped. No 262,144 cap (v3 kept v4 / v5 conversations up to "
        f"524,288): {_n(st['over_262144'])} samples are over 262,144 tokens ("
        + ", ".join(f"{k} {v}" for k, v in st["over_262144_by_segment"].items()) + ").",
        "",
        "## Packing",
        f"FFD of ceil32 lengths into 524,288-token rows (key: padded length desc, then sample id) over all samples; row "
        f"order shuffled with `random.Random({st['shuffle_seed']})`; 6 contiguous shards "
        f"{'/'.join(map(str, st['rows_per_shard']))}, weight 0.16666667 each. {_n(st['packed_rows'])} rows, <= "
        f"{st['samples_per_row_max']} samples per row, {100 * st['row_fill']:.2f}% fill. Rows by batch mix: "
        + ", ".join(f"{k} {v}" for k, v in sorted(st["rows_by_batch_mix"].items())) + ".",
        "",
        "## Stats",
        "| | " + " | ".join(g for g, _ in groups) + " |",
        "|---|" + "---|" * len(groups),
        row("samples", lambda p: _n(p["samples"])),
        row("rollouts", lambda p: _n(p["rollouts"])),
        row("tasks", lambda p: _n(p["tasks"])),
        row("windows k >= 2", lambda p: _n(p["windows_k_ge_2"])),
        row("tokens", lambda p: _n(p["tokens_total"])),
        row("assistant (trained) tokens, all weight 1.0", lambda p: _n(p["assistant_tokens_total"])),
        row("tokens/sample min / median / p90 / max", lambda p: f"{_n(p['tokens_min'])} / {_n(p['tokens_median'])} / "
                                                                f"{_n(p['tokens_p90'])} / {_n(p['tokens_max'])}"),
        row("samples over 262,144 tokens", lambda p: _n(p["over_262144"])),
        row("assistant actions / tool calls", lambda p: f"{_n(p['assistant_messages'])} / {_n(p['tool_calls'])}"),
        row("packed rows containing", lambda p: _n(p["rows_containing"])),
        "",
        f"Total: {_n(st['samples'])} samples from {_n(st['rollouts'])} rollouts on {_n(st['tasks'])} tasks, "
        f"{_n(st['tokens_total'])} tokens (padded {_n(st['tokens_padded_total'])}), {_n(st['assistant_tokens_total'])} "
        f"assistant tokens; assistant pieces starting with whitespace: {st['assistant_pieces_leading_whitespace']}.",
        "",
        "## Epochs at GBS 32 (MBS 1, seq 524,288)",
        "| epochs | target_iteration = ceil(E * rows / 32) | train_samples = iterations * 32 | nominal epochs |",
        "|---|---|---|---|",
    ]
    lines += [f"| {e} | {v['target_iteration']} | {v['train_samples']} | {v['nominal_epochs']} |" for e, v in ep.items()]
    lines += [
        "",
        "## Training notes",
        "- Identity prompt format (`--sft-tokenizer-prompt-format identity`, SFTTokenizer, sc14 tokenizer), "
        "`--seq-length 524288`, MBS 1, GBS 32 (table above). The receipt's `selected_trajectories` (= samples) and "
        "`selection_sha256` must be adapted in a sbatch that hard-codes them (as for v1 - v3).",
        "- Packed samples stay independent only with the THD packing path: PP = 1, or PP = 2 with MTP on the last stage "
        "(other pipeline layouts give some ranks `cu_seqlens = None`); CP <= 16 (per-sample padding to 2 x CP = 32 is what "
        "the ceil32 row budget assumes). The AGA SFT-v3 GB300 setup (TP4 CP16 EP32 PP1, MTP off) satisfies both.",
        "- This corpus is the RAW format (like SFT-v3 itself). The AGA v3 run trains on an empty-think-prefix transform of "
        "v3; an etp variant of this corpus is a separate follow-up and must keep `assistant_token_counts` consistent.",
        "- Options not built (DESIGN): OpenCode's summary-writing calls as a separate sample type (loss on the summary "
        "only), task-subagent sessions, the 2 hosted gomacro tasks.",
        "",
        "## Verification (`tools/verify_glm.py`)",
        "Independent of the builder: v3's re-implementation of `SFTTokenizer.tokenize_conversation` (identity) and "
        "`SFTDataset.__getitem__` packing on every row (32-token padding, 524,289 truncation guard, step-weight spans, "
        "loss_mask = weights[1:]; loss-mask sum == sum(counts x weights); no generation-stop target); every sample's "
        "`weight` == 1.0 and `assistant_weights` == the weight policy (all 1.0 with `--invalid-policy keep`); selection == "
        "shards (ids, token_length, training_row_sha256, assistant_tokens / n_assistant / masked turns == the system "
        "fields); receipt / blend paths == DATA_ROOT, and DATA_ROOT == DESIGN's AGA DATA_ROOT for a real build; per kept "
        "rollout, re-derived from the raw attempt files with verify's own chain + export code: the final record, "
        "terminal.json, the window count, the two-source equality, every sample's pieces re-rendered from http.jsonl "
        "(system and context pieces from the independently located first request of the window, every target turn and "
        "tool piece from the chain's responses and tool results) and its `assistant_token_counts` == the token counts of "
        "the re-rendered assistant pieces, coverage (every main-chain call except summary-writing calls and dropped length "
        "/ errored replies is a target exactly once), the system-change fields and where the last target's tool results "
        "are, a degenerate re-scan with degenerate.py's classify(), the old-grader class and re-grade consistency, and "
        "completeness (every considered final record kept or excluded once, every exclusion reason re-derived: status, "
        "clean rule, old grader, degenerate re-scan, v1 harness base / tool set, two-source mismatch with verify's own "
        "code); the real pinned `SFTDataset.__getitem__` (`tools/sim_sft_getitem.py`) on sample rows incl. rows holding "
        "two windows of one rollout (cu_seqlens, labels never from another sample, per-sample loss mass, Mamba seq_idx, "
        "MTP roll).",
        "",
        "VERIFY_RESULT_PLACEHOLDER",
        "",
        "## Rebuild",
        "```",
        "PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python",
        "$PY -I -B tools/sources_glm.py --workers 16      # sources/glm, sources/regrade, sources/task_tests.json",
        f"$PY -I -B tools/build_glm.py --workers 16 --data-root {data_root}",
        f"$PY -I -B tools/verify_glm.py --workers 16 --data-root {data_root} --readme",
        "```",
        "(normally all three run in one CPU job, `tools/glm_sft.sbatch build`, which also writes SHA256SUMS).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
