#!/usr/bin/env python3
"""Build SFT-v3 (raw_sft scheme of zhenghaox's Super SFT pipeline on HSG; the on-disk format of kimi-pass-v2-20261023).

User request (hengyuf, 2026-10-29, verbatim): "v4 rollout 结束之后, 我想造一个新的SFT dataset (命名为SFT-v3), 他包含v2
所有长度在256k以内的rollout, 以及v4, v5 里成功且没有compaction 的rollout,帮我build 一个新的SFT dataset 并用data mover
transfer 到HSG 上."

SFT-v3 = (a) every v2 conversation whose length is at most 262,144 tokens, measured exactly as the v2 builder recorded it
(selection/kimi_pass_v2.jsonl token_length: sc14 tokens of the rendered conversation). These are re-rendered from v2's
own sources (the v1 batch kimi_pass_batch_20261021 and v2's sources/baihe) and must be byte-identical to v2
(training_row_sha256, token_length, assistant_tokens; asserted); plus (b) the Kimi-K3 v4 / v5 passes that
tools/sources_v4v5.py extracted into <sources>/{v4,v5}/ (final graded pass incl. the approved broken-reply re-run, no
OpenCode compaction, no degenerate reply anywhere, no infra / needs_review; no length filter besides v2's 524,288 drop).
Snapshot builds (sources_v4v5.py --snapshot-incomplete; user 2026-10-29 ~16:25: do not wait for the v4 long tail): the
campaigns still running contribute their terminal slots; their pending slots are recorded in the receipt
("snapshot"), build_stats.v4v5_selection.<batch>.snapshot, metadata.jsonl (dropped_reason pending_at_snapshot) and the
README.

The converter is v2's, unchanged: its source text (tools/build_corpus.py of kimi-pass-v2-20261023, from `def enc` to
`def work`: enc, fix_surrogates, arg_str, render_calls, convert) and its constants GEN_PREFIX / THINK_RE / SURROGATE_RE
are exec'd here from that file (whole-file and block sha256 asserted, recorded in the receipt). Render: sc14's
chat_template.jinja via transformers.apply_chat_template, tools from the harness.json of each source. Packing as v2:
FFD of ceil32 lengths into 524,288-token rows over all sources together, rows shuffled with random.Random(20261029), 6
contiguous shards, weight 0.16666667 each. Output: the v2 DATA_ROOT layout (raw_sft shards, blends, receipt,
selection, metadata) plus a generated README.md. Deterministic: the same inputs give byte-identical shards, selection
and metadata (the receipt's completed_at aside).

Run (CPU only, no network; normally inside tools/sft_v3.sbatch):
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B \
      tools/build_sft_v3.py [--out DIR] [--sources DIR] [--workers 16] [--v2-limit N]
--v2-limit N (dry runs only) converts the first N kept conversations of each v2 source. The real corpus dir refuses
partial v4 / v5 sources and --v2-limit.
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
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
V2 = S / "sft/corpus/kimi-pass-v2-20261023"
V2_BUILDER = V2 / "tools/build_corpus.py"
V2_BUILDER_SHA256 = "2c6977864c82ab8bfc68c6a62bb37ccd8ca68f9d9e96b289e447c51748f3478d"  # v2 SHA256SUMS
CONVERTER_SHA256 = "807295d4abe8cfdef362ce7dee3368b102e42e9596c0dd77e4b7a0909791e6f9"  # `def enc` .. `def work`
V2_SELECTION = V2 / "selection/kimi_pass_v2.jsonl"
V2_METADATA = V2 / "metadata.jsonl"
V2_RECEIPT = V2 / "DATA_PREP_COMPLETE.json"
BATCH = S / "export/kimi_pass_batch_20261021"
HF_DIR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf")
TOOLS = Path(__file__).resolve().parent
OUT_DEFAULT = S / "sft/corpus/sft-v3-20261029"
HSG_ROOT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-v3-20261029"
V2_MAX_TOKENS = 262144    # the v2 filter: selection token_length <= 262,144 ("256k")
SCHEME = "raw_sft"
SHARD_NAME = "part_{:03d}.mat.jsonl.packed.512k"
N_SHARDS = 6
SEQ_LEN = 524288          # --seq-length
PAD_GRAN = 32             # context_parallel_size 16 * 2: every conversation is padded to a multiple of 32
SHUFFLE_SEED = 20261029   # row-order shuffle (v2: 20261023, its build date)
GBS = 32
EPOCHS = (5, 10, 20)
SCHEMA = "sft-v3-hsg-raw-sft-data-v1"
SELECTION_NAME = "sft_v3.jsonl"
# v2's source_part labels (its selection) and v3's labels for the same files
V2_PART = {"hengyuf": "kimi_pass_batch_20261021/trajectories.jsonl.gz", "baihe": "sources/baihe/trajectories.jsonl.gz"}

_W = {}

# ---------------------------------------------------------------- v2's converter, exec'd from v2's source text
_V2_BYTES = V2_BUILDER.read_bytes()
assert hashlib.sha256(_V2_BYTES).hexdigest() == V2_BUILDER_SHA256, "v2 build_corpus.py changed"
_V2_TEXT = _V2_BYTES.decode("utf-8")
for _name in ("GEN_PREFIX", "THINK_RE", "SURROGATE_RE"):
    exec(re.search(r"^" + _name + r" = .*$", _V2_TEXT, re.M).group(0), globals())
CONVERTER_SOURCE = _V2_TEXT[_V2_TEXT.index("def enc(text):"):_V2_TEXT.index("def work(arg):")]
assert hashlib.sha256(CONVERTER_SOURCE.encode("utf-8")).hexdigest() == CONVERTER_SHA256, "v2 converter changed"
exec(compile(CONVERTER_SOURCE, str(V2_BUILDER), "exec"), globals())  # enc, fix_surrogates, arg_str, render_calls, convert


def _load_sources_tool():
    spec = importlib.util.spec_from_file_location("sft_v3_sources_v4v5", TOOLS / "sources_v4v5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stop_strings():
    """Token text of generation_config.json's eos_token_id(s) other than the template's own turn end (tokenizer_config
    eos_token '<|im_end|>', which closes every assistant turn and is trained on purpose), and the BOS token text."""
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    gen = json.loads((HF_DIR / "generation_config.json").read_text())
    tcfg = json.loads((HF_DIR / "tokenizer_config.json").read_text())
    text = lambda t: t["content"] if isinstance(t, dict) else t  # noqa: E731
    eos = gen.get("eos_token_id")
    ids = eos if isinstance(eos, list) else [eos]
    stops = [(i, tk.id_to_token(i)) for i in ids if tk.id_to_token(i) != text(tcfg.get("eos_token"))]
    assert stops, "no generation stop token besides the turn end"
    return stops, text(tcfg.get("bos_token"))


# ----------------------------------------------------------------------------------------------- worker side
def _init(invalid_policy, harness_paths, task_tests_path):
    import tokenizers
    import transformers

    transformers.logging.set_verbosity_error()
    _W["hf"] = transformers.AutoTokenizer.from_pretrained(str(HF_DIR))
    _W["tk"] = tokenizers.Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    # Merged harness lookup of all sources (v2's _init over more files): system variants and tool sets keyed by the
    # same sha256 (build_batch.py's sha256_json of the redacted value), so a key present in two files is the same
    # content (asserted).
    merged = {"system_prompt_variants": {}, "tool_sets": {}}
    for hpath in harness_paths:
        h = json.loads(Path(hpath).read_text())
        for field in merged:
            for key, value in h[field].items():
                if key in merged[field]:
                    a = merged[field][key].get("system_messages", merged[field][key].get("tools"))
                    b = value.get("system_messages", value.get("tools"))
                    assert a == b, (field, key)
                else:
                    merged[field][key] = value
    _W["harness"] = merged
    _W["invalid_policy"] = invalid_policy
    added = _W["tk"].get_added_tokens_decoder()
    _W["special_ids"] = {i for i, t in added.items() if t.special}
    _W["unk_id"] = _W["tk"].token_to_id("<unk>")
    _W["sv"] = _load_sources_tool()  # flags_of / finish_flags for the v2 baseline flags
    _W["task_tests"] = json.loads(Path(task_tests_path).read_text()) if Path(task_tests_path).exists() else {}


def work(arg):
    source, line_no, line, baseline_flags = arg
    info, messages = convert(source, line_no, line)
    if baseline_flags:  # v2 lines: the same informational flags as v4 / v5 (export-based only; no journal here)
        rec = json.loads(line)
        tests = _W["task_tests"].get(rec["meta"]["task_id"]) or {}
        info["flags"] = _W["sv"].finish_flags(_W["sv"].flags_of(rec["trajectory"], tests), {})
        info["flags"]["task_tests_known"] = bool(tests)
    else:
        info["flags"] = info["meta"].get("flags")
    return info, messages


# ------------------------------------------------------------------------------------------------- main side
def ceil32(n):
    return (n + PAD_GRAN - 1) // PAD_GRAN * PAD_GRAN


def ffd(items):
    """Deterministic first-fit-decreasing on padded length; items = [(key, padded_len)]. Returns rows of keys (v2)."""
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


def epochs_table(n_rows):
    out = {}
    for e in EPOCHS:
        it = math.ceil(e * n_rows / GBS)
        out[str(e)] = {"target_iteration": it, "train_samples": it * GBS, "nominal_epochs": round(it * GBS / n_rows, 3)}
    return out


def flag_summary(rows):
    keys = ("any", "strong", "webfetch_calls", "webfetch_completed", "net_fetch_cmds", "net_fetch_failed",
            "net_fetch_ok", "git_history_cmds", "hidden_artifact_refs", "hidden_test_seen_before_touch",
            "hidden_test_first_touch_reads_ok", "hidden_test_read_attempts", "hidden_test_bash_first_mentions",
            "subagent_calls", "subagent_beyond_export", "journal_net_fetch_calls", "journal_webfetch_calls",
            "journal_hidden_refs", "journal_git_history_calls")
    flags = [r or {} for r in rows]
    out = {"conversations": len(flags)}
    for k in keys:
        out[k] = sum(1 for f in flags if f.get(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--sources", type=Path, help="dir with v4/, v5/, task_tests.json (default <out>/sources)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--invalid-policy", choices=("mask", "keep"), default="keep")
    ap.add_argument("--v2-limit", type=int, default=0, help="dry run: first N kept conversations per v2 source")
    args = ap.parse_args()
    out = args.out.resolve()
    src_dir = (args.sources or out / "sources").resolve()
    os.umask(0o027)
    started = time.time()

    reports = {}
    for b in ("v4", "v5"):
        path = src_dir / b / "selection_report.json"
        reports[b] = json.loads(path.read_text()) if path.exists() else {"batch": b, "partial": True, "kept": 0,
                                                                          "missing_report": True}
    partial = any(r["partial"] or r.get("limit_per_campaign") for r in reports.values())  # pilots are a choice, not partial
    # a snapshot build is not partial: every campaign is covered (results file, or the terminal slots of its snapshot)
    snapshots = {b: r["snapshot"] for b, r in reports.items() if r.get("snapshot")}
    pending_rows = [dict(p, batch=b) for b, s in snapshots.items() for p in s["pending"]]
    dry_run = bool(partial or args.v2_limit)
    if out == OUT_DEFAULT.resolve() and dry_run:
        raise SystemExit("refusing a dry-run build (partial v4/v5 sources or --v2-limit) into the real corpus dir")
    if out == OUT_DEFAULT.resolve() and (out / SCHEME).exists():
        raise SystemExit(f"{out / SCHEME} exists: refusing to overwrite a built corpus")
    SOURCES = [("hengyuf", BATCH / "trajectories.jsonl.gz", BATCH / "harness.json",
                "kimi_pass_batch_20261021/trajectories.jsonl.gz", 380, "v2"),
               ("baihe", V2 / "sources/baihe/trajectories.jsonl.gz", V2 / "sources/baihe/harness.json",
                "kimi-pass-v2-20261023/sources/baihe/trajectories.jsonl.gz", 418, "v2")]
    for b in ("v4", "v5"):
        if reports[b]["kept"] == 0 and dry_run:  # dry run without this batch (nothing extracted)
            continue
        SOURCES.append((b, src_dir / b / "trajectories.jsonl.gz", src_dir / b / "harness.json",
                        f"sources/{b}/trajectories.jsonl.gz", reports[b]["kept"], b))
    source_order = {s[0]: k for k, s in enumerate(SOURCES)}
    source_part = {s[0]: s[3] for s in SOURCES}
    batch_of = {s[0]: s[5] for s in SOURCES}

    # ---- v2 filter, on v2's own recorded token counts (selection token_length)
    v2sel = [json.loads(line) for line in V2_SELECTION.read_text().splitlines() if line.strip()]
    assert len(v2sel) == json.loads(V2_RECEIPT.read_text())["selected_trajectories"] == 797
    v2_by_line = {(r["source_part"], int(r["source_line"])): r for r in v2sel}
    v2_by_id = {r["agent_run_id"]: r for r in v2sel}
    v2meta = {}
    for line in V2_METADATA.read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            v2meta[(m["corpus"]["source"], m["corpus"]["line_no"])] = m
    tasks, v2_dropped, v2_skipped = [], [], 0
    input_sha = {}
    for name, path, hpath, part, expected, batch in SOURCES:
        input_sha[part] = sha256_file(path)
        input_sha[part.replace("trajectories.jsonl.gz", "harness.json")] = sha256_file(hpath)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == expected, (name, len(lines), expected)
        if batch == "v2":
            taken = 0
            for n, line in enumerate(lines):
                row = v2_by_line.get((V2_PART[name], n))
                if row is None:  # v2 dropped this one itself (> 524,288 tokens); its meta is in v2's metadata.jsonl
                    m = v2meta[(name, n)]
                    assert m["corpus"]["dropped"] is True
                    v2_dropped.append({"source": name, "line_no": n, "trajectory_id": m["trajectory_id"],
                                       "reason": "v2_dropped_over_524288", "v2_token_length": m["corpus"]["token_length"]})
                    continue
                if int(row["token_length"]) > V2_MAX_TOKENS:
                    v2_dropped.append({"source": name, "line_no": n, "trajectory_id": row["agent_run_id"],
                                       "reason": "v2_over_262144", "v2_token_length": int(row["token_length"])})
                    continue
                if args.v2_limit and taken >= args.v2_limit:
                    v2_skipped += 1
                    continue
                taken += 1
                tasks.append((name, n, line, True))
        else:
            tasks += [(name, n, line, False) for n, line in enumerate(lines)]
        del lines
    print(f"converting {len(tasks)} conversations ({sum(t[3] for t in tasks)} v2); v2 dropped {len(v2_dropped)}, "
          f"dry-run skipped {v2_skipped}", flush=True)

    infos, convs = [], {}
    harness_paths = [str(s[2]) for s in SOURCES]
    with Pool(args.workers, initializer=_init,
              initargs=(args.invalid_policy, harness_paths, str(src_dir / "task_tests.json"))) as pool:
        for info, messages in pool.imap(work, tasks, chunksize=1):
            infos.append(info)
            convs[info["trajectory_id"]] = messages
            print(f"[{len(infos)}/{len(tasks)}] {info['trajectory_id']} tokens={info['token_length']}", flush=True)
    del tasks

    infos.sort(key=lambda i: (source_order[i["source"]], i["line_no"]))
    assert len({i["trajectory_id"] for i in infos}) == len(infos), "trajectory_id collision across sources"
    # v2 identity: every re-rendered v2 conversation is byte-identical to v2's (as v2 checked v1's)
    mismatch = [i["trajectory_id"] for i in infos if batch_of[i["source"]] == "v2" and (
        i["training_row_sha256"], i["token_length"], i["assistant_tokens"]) != (
        v2_by_id[i["trajectory_id"]]["training_row_sha256"], int(v2_by_id[i["trajectory_id"]]["token_length"]),
        int(v2_by_id[i["trajectory_id"]]["assistant_tokens"]))]
    assert not mismatch, f"v2 conversations rendered differently from v2: {mismatch[:5]}"
    # v3 rule (main agent, 2026-10-29, from the first build's verification): sc14's tokenizer parses special-token TEXT
    # inside the content, so a literal generation stop token (generation_config.json eos_token_id 2 = '</s>', e.g. HTML
    # strikethrough) in an ASSISTANT target trains the student to emit EOS mid-turn, where vLLM stops. Such conversations
    # are dropped and listed (metadata.jsonl corpus.dropped_reason = assistant_literal_eos); v2 has none (asserted, so
    # v2 stays complete and byte-identical). The literal BOS '<s>' (id 1, not a stop token) is only counted: v2 already
    # trains on it.
    stops, bos_text = _stop_strings()
    stop_texts = [t for _, t in stops]
    for i in infos:
        asst = [m["content"] for m in convs[i["trajectory_id"]] if m["role"] == "assistant"]
        i["assistant_literal_eos"] = sum(c.count(t) for c in asst for t in stop_texts)
        i["assistant_literal_bos"] = sum(c.count(bos_text) for c in asst)
    eos_v2 = [i["trajectory_id"] for i in infos if batch_of[i["source"]] == "v2" and i["assistant_literal_eos"]]
    assert not eos_v2, f"v2 conversations with a literal stop token in an assistant target: {eos_v2[:5]}"
    eos_dropped = [i for i in infos if i["assistant_literal_eos"]]
    # Invariant 4 (v2): no conversation may exceed the sequence length (drop and list)
    dropped = [i for i in infos if ceil32(i["token_length"]) > SEQ_LEN]
    kept = [i for i in infos if ceil32(i["token_length"]) <= SEQ_LEN and not i["assistant_literal_eos"]]
    assert len({i["trajectory_id"] for i in kept}) == len(kept)

    # ---- packing (v2): FFD on ceil32 lengths, seeded shuffle of the row order, 6 contiguous near-equal shards
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

    # Re-read every shard and re-verify the packing / weight invariants exactly as the dataset reads them (v2)
    for s in range(N_SHARDS):
        with open(out / SCHEME / SHARD_NAME.format(s), encoding="utf-8") as f:
            for line in f:
                ms = json.loads(line)["messages"]
                assert ms[0]["role"] == "system"
                for m in ms:
                    if m["role"] == "system":
                        assert len(m["assistant_weights"]) == len(m["assistant_token_counts"]) > 0
                        assert all(isinstance(w, float) and 0.0 <= w <= 1.0 for w in m["assistant_weights"])
                    else:
                        assert set(m) == {"role", "content"}

    blend = {"train": [x for s in shards for x in ("0.16666667", s["path"])], "valid": None, "test": None}
    (out / "blends" / f"{SCHEME}.json").write_text(json.dumps(blend, indent=2) + "\n")

    # ---- selection (HSG schema fields are strings, as v2) + our fields
    sel_path = out / "selection" / SELECTION_NAME
    with open(sel_path, "w", encoding="utf-8") as f:
        for i in kept:  # source order (hengyuf, baihe, v4, v5), then source line
            m = i["meta"]
            fl = i.get("flags") or {}
            f.write(json.dumps({
                "agent_run_id": i["trajectory_id"],
                "record_id": "r_" + hashlib.sha256(i["trajectory_id"].encode()).hexdigest()[:24],
                "source_line": str(i["line_no"]),
                "source_part": source_part[i["source"]],
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
                "rollout_batch": batch_of[i["source"]], "campaign": m.get("campaign"), "rerun": m.get("rerun"),
                "flags_any": bool(fl.get("any")), "flags_strong": bool(fl.get("strong")),
            }, ensure_ascii=False) + "\n")
    selection_sha = sha256_file(sel_path)

    # ---- metadata.jsonl: every considered conversation (v2's dropped ones from v2's metadata), source order
    meta_rows = []
    for i in infos:
        row = dict(i["meta"])
        row["corpus"] = {k: i.get(k) for k in (
            "source", "line_no", "token_length", "assistant_tokens", "n_assistant", "n_tool_messages", "n_tool_calls",
            "system_tokens", "user_tokens", "masked_assistant_turns", "training_row_sha256", "shard",
            "shard_row", "row_position", "row_padded_tokens")}
        row["corpus"]["rollout_batch"] = batch_of[i["source"]]
        row["corpus"]["stats"] = i["stats"]
        row["corpus"]["flags"] = i.get("flags")
        row["corpus"]["dropped"] = i in dropped or bool(i["assistant_literal_eos"])
        row["corpus"]["dropped_reason"] = ("over_524288" if i in dropped else
                                           "assistant_literal_eos" if i["assistant_literal_eos"] else None)
        row["corpus"]["assistant_literal_eos"] = i["assistant_literal_eos"]
        row["corpus"]["assistant_literal_bos"] = i["assistant_literal_bos"]
        meta_rows.append(((source_order[i["source"]], i["line_no"]), row))
    for d in v2_dropped:
        m = dict(v2meta[(d["source"], d["line_no"])])
        m.pop("corpus", None)
        m["corpus"] = {"source": d["source"], "line_no": d["line_no"], "rollout_batch": "v2", "dropped": True,
                       "dropped_reason": d["reason"], "v2_token_length": d["v2_token_length"]}
        meta_rows.append(((source_order[d["source"]], d["line_no"]), m))
    # v4 / v5 slots pending at the snapshot (campaigns still running): listed, never converted (no conversation)
    for k, p in enumerate(pending_rows):
        snap = snapshots[p["batch"]]
        meta_rows.append(((source_order.get(p["batch"], len(SOURCES)), 10 ** 9 + k), {
            "source": p["batch"], "rollout_batch": p["batch"], "trajectory_id": p["trajectory_id"],
            "task_id": p["task_id"], "slot": p["slot"], "campaign": p["campaign"], "record_source": "snapshot",
            "snapshot_at": snap["campaigns"][p["campaign"]]["scanned_at"], "snapshot_file": snap["snapshot_file"],
            "pending_state": p["state"], "attempts": p.get("attempts"), "attempt_dir": p.get("attempt_dir"),
            "pending_detail": p.get("detail"),
            "corpus": {"source": p["batch"], "rollout_batch": p["batch"], "dropped": True,
                       "dropped_reason": "pending_at_snapshot", "pending_state": p["state"]}}))
    meta_rows.sort(key=lambda kv: kv[0])
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for _, row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- stats and receipt
    tot = Counter()
    for i in kept:
        tot.update(i["stats"])
    tl = [i["token_length"] for i in kept]

    def per_source(src):
        ks = [i for i in kept if i["source"] == src]
        eos_src = [i["trajectory_id"] for i in eos_dropped if i["source"] == src]
        if not ks:
            return {"conversations": 0, "dropped": [i["trajectory_id"] for i in dropped if i["source"] == src],
                    "dropped_assistant_literal_eos": eos_src}
        rows_with = sum(1 for r in rows if any(by_id[k]["source"] == src for k in r))
        tl_s = sorted(i["token_length"] for i in ks)
        return {"conversations": len(ks), "dropped": [i["trajectory_id"] for i in dropped if i["source"] == src],
                "dropped_assistant_literal_eos": eos_src,
                "assistant_literal_bos_kept": {"conversations": sum(1 for i in ks if i["assistant_literal_bos"]),
                                               "occurrences": sum(i["assistant_literal_bos"] for i in ks)},
                "tasks": len({(i["meta"]["run"], i["meta"]["task_id"]) for i in ks}),
                "rows_containing": rows_with, "tokens_total": sum(tl_s),
                "tokens_padded_total": sum(ceil32(x) for x in tl_s),
                "assistant_tokens_total": sum(i["assistant_tokens"] for i in ks),
                "tokens_max": tl_s[-1], "tokens_min": tl_s[0], "tokens_median": tl_s[len(tl_s) // 2],
                "tokens_p90": tl_s[min(len(tl_s) - 1, int(0.9 * len(tl_s)))],
                "over_262144": sum(1 for x in tl_s if x > V2_MAX_TOKENS),
                "max_conversation": max(ks, key=lambda x: x["token_length"])["trajectory_id"],
                "assistant_messages": sum(i["n_assistant"] for i in ks),
                "tool_calls": sum(i["n_tool_calls"] for i in ks),
                "edge_case_counts": dict(sorted(sum((Counter(i["stats"]) for i in ks), Counter()).items())),
                "grader": dict(Counter(i["meta"]["grader"] for i in ks)),
                "reruns": sum(1 for i in ks if i["meta"].get("rerun")),
                "flags": flag_summary([i.get("flags") for i in ks])}

    v2_kept_ids = {i["trajectory_id"] for i in kept if batch_of[i["source"]] == "v2"}
    over = [d for d in v2_dropped if d["reason"] == "v2_over_262144"]
    v2_tasks_all = {(r["run"], r["task_id"]) for r in v2sel}
    v2_tasks_left = {(r["run"], r["task_id"]) for r in v2sel if int(r["token_length"]) <= V2_MAX_TOKENS}
    target_of = {m["trajectory_id"]: m.get("deepswe_target") for m in v2meta.values()}
    v2_filter = {
        "rule": f"v2 selection token_length <= {V2_MAX_TOKENS} (sc14 tokens of the rendered conversation, as the v2 "
                "builder recorded them)",
        "v2_conversations": len(v2sel),
        "kept": sum(1 for r in v2sel if int(r["token_length"]) <= V2_MAX_TOKENS),
        "converted": len(v2_kept_ids), "dry_run_not_converted": v2_skipped,
        "dropped_over_262144": len(over),
        "dropped_by_source": dict(Counter(d["source"] for d in over)),
        "dropped_by_target": dict(Counter(target_of.get(d["trajectory_id"]) for d in over).most_common()),
        "dropped_by_run": dict(Counter(d["trajectory_id"].split("/")[0] for d in over).most_common()),
        "dropped": [{k: d[k] for k in ("trajectory_id", "source", "v2_token_length")} for d in over],
        "tasks_without_conversation": sorted("/".join(t) for t in v2_tasks_all - v2_tasks_left),
        "v2_dropped_earlier": [d for d in v2_dropped if d["reason"] == "v2_dropped_over_524288"],
    }
    selection_summary = {}
    for b, r in reports.items():
        selection_summary[b] = {k: r.get(k) for k in (
            "partial", "campaigns_missing", "pilots_included", "limit_per_campaign", "snapshot_mode",
            "snapshot_campaigns", "pending_at_snapshot", "slots_expected", "considered", "considered_main",
            "final_status", "exclusion_counts", "independent_counts_among_final_passes", "kept", "kept_reruns",
            "kept_by_record_source", "kept_by_campaign", "kept_tasks", "tasks_considered", "tasks_without_kept",
            "harness_comparison_with_v1", "harness_missing", "first_user_mismatch", "patch_mismatch_record",
            "v5_sft_usable_crosscheck", "record_sources")}
        selection_summary[b]["results_files"] = {c: {k: v.get(k) for k in ("sha256", "finished_at", "stats_final")}
                                                 for c, v in (r.get("results_files") or {}).items()}
        if r.get("snapshot"):  # snapshot time, per-campaign scan, logic check and the full pending list
            selection_summary[b]["snapshot"] = r["snapshot"]

    stats = {
        "per_source": {s[0]: per_source(s[0]) for s in SOURCES},
        "conversations": len(kept), "dropped": [i["trajectory_id"] for i in dropped], "packed_rows": n_rows,
        "rows_per_shard": sizes, "conversations_per_row_max": max(len(r) for r in rows),
        "tokens_total": sum(tl), "tokens_padded_total": sum(ceil32(x) for x in tl),
        "assistant_tokens_total": sum(i["assistant_tokens"] for i in kept),
        "assistant_tokens_weighted": None,
        "tokens_max": max(tl), "tokens_min": min(tl), "tokens_mean": sum(tl) / len(tl),
        "tokens_median": sorted(tl)[len(tl) // 2], "tokens_p90": sorted(tl)[min(len(tl) - 1, int(0.9 * len(tl)))],
        "max_conversation": max(kept, key=lambda x: x["token_length"])["trajectory_id"],
        "assistant_messages": sum(i["n_assistant"] for i in kept),
        "tool_messages": sum(i["n_tool_messages"] for i in kept),
        "tool_calls": sum(i["n_tool_calls"] for i in kept),
        "row_fill": sum(ceil32(x) for x in tl) / (n_rows * SEQ_LEN),
        "rows_by_batch_mix": dict(Counter("+".join(sorted({batch_of[by_id[k]["source"]] for k in r})) for r in rows)),
        "edge_case_counts": dict(sorted(tot.items())),
        "grader_deepswe_recorded": sum(1 for i in kept if i["meta"]["grader"] == "deepswe-recorded"),
        "train_samples_5epoch": epochs_table(n_rows)["5"]["train_samples"],
        "target_iteration_5epoch": epochs_table(n_rows)["5"]["target_iteration"],
        "epochs_gbs32": epochs_table(n_rows),
        "invalid_policy": args.invalid_policy,
        "shuffle_seed": SHUFFLE_SEED,
        "assistant_literal_stop_rule": {
            "rule": "drop a conversation whose assistant targets contain the literal text of a generation stop token "
                    "(generation_config.json eos_token_id other than the template turn end '<|im_end|>'): sc14's "
                    "tokenizer turns it into that special id, i.e. EOS mid-turn",
            "stop_tokens": [{"id": i, "text": t} for i, t in stops],
            "dropped": [{"trajectory_id": i["trajectory_id"], "source": i["source"], "token_length": i["token_length"],
                         "literal_eos": i["assistant_literal_eos"], "literal_bos": i["assistant_literal_bos"]}
                        for i in eos_dropped],
            "bos_text": bos_text,
            "literal_bos_kept": [{"trajectory_id": i["trajectory_id"], "source": i["source"],
                                  "occurrences": i["assistant_literal_bos"]} for i in kept if i["assistant_literal_bos"]],
        },
        "v2_filter": v2_filter,
        "v4v5_selection": selection_summary,
        "dry_run": dry_run, "v2_limit": args.v2_limit or None,
        "converter": {"file": str(V2_BUILDER), "file_sha256": V2_BUILDER_SHA256,
                      "block": "def enc .. def work (enc, fix_surrogates, arg_str, render_calls, convert)",
                      "block_sha256": CONVERTER_SHA256},
        "inputs_sha256": input_sha,
        "build_seconds": round(time.time() - started, 1),
    }
    wsum = 0
    for key, ms in convs.items():
        if key in by_id:
            s0 = ms[0]
            wsum += sum(c * w for c, w in zip(s0["assistant_token_counts"], s0["assistant_weights"]))
    stats["assistant_tokens_weighted"] = int(wsum)

    receipt = {
        "blend": f"{HSG_ROOT}/blends/{SCHEME}.json",
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filter_policy": "v2 (kimi-pass-v2-20261023: hengyuf + baihe Kimi-K3 passes with a clean glm teacher audit): "
                         f"every conversation whose v2-recorded token_length is <= {V2_MAX_TOKENS}, re-rendered and "
                         "byte-identical to v2; v4 / v5 (kimi-rollouts-v4-stock256k-20261028, "
                         "kimi-rollouts-v5-official2-20261028; Kimi-K3, OpenCode stock prompt, 262144 window): every "
                         "final graded pass (a broken-ended slot's approved re-run in its place) with evaluation "
                         "completed, no OpenCode compaction, exactly one user message, no degenerate reply anywhere "
                         "(degenerate.py classify over every hub reply) and v1's harness base, see "
                         "sources/<batch>/selection_report.json; no LLM audit for v4 / v5 (informational flags in "
                         "selection/metadata); conversations over 524288 tokens dropped; conversations with the literal "
                         "text of a generation stop token ('</s>', generation_config eos id 2) in an assistant target "
                         "dropped (sc14's tokenizer would make it EOS mid-turn; build_stats.assistant_literal_stop_rule)",
        "packed_rows": n_rows,
        "physical_bytes": sum(s["bytes"] for s in shards),
        "raw_weight_policy": "all assistant action weights = 1.0",
        "schema": SCHEMA,
        "selected_trajectories": len(kept),
        "selection": f"{HSG_ROOT}/selection/{SELECTION_NAME}",
        "selection_sha256": selection_sha,
        "shards": shards,
        "source_cluster": "cw-dfw",
        "state": "dry-run" if dry_run else "complete",
    }
    if snapshots:  # v4 / v5 campaigns still running when the corpus was built (user: do not wait for the long tail)
        receipt["filter_policy"] += (
            "; v4 / v5 campaigns without a results file at build time were read at a snapshot (their terminal slots, "
            "the driver's record logic, read-only; slots still pending at the snapshot are excluded and listed: "
            "build_stats.v4v5_selection.<batch>.snapshot.pending, metadata.jsonl corpus.dropped_reason = "
            "pending_at_snapshot)")
        receipt["snapshot"] = {
            b: {"request": s["request"], "snapshot_started_at": s["snapshot_started_at"],
                "campaign_sources": reports[b]["record_sources"],
                "snapshot_campaigns": {c: {k: v[k] for k in ("scanned_at", "slots", "terminal_slots", "pending_slots",
                                                             "plan_slots", "rerun_outcomes")}
                                       for c, v in s["campaigns"].items()},
                "pending_at_snapshot": s["pending_at_snapshot"], "pending_by_state": s["pending_by_state"],
                "pending_by_campaign": s["pending_by_campaign"],
                "pending": [{k: p.get(k) for k in ("trajectory_id", "state", "attempt_dir", "detail")}
                            for p in s["pending"]],
                "kept_from_snapshot": (reports[b].get("kept_by_record_source") or {}).get("snapshot", 0),
                "logic_check": s["logic_check"], "snapshot_file": f"sources/{b}/snapshot_records.json",
                "snapshot_file_sha256": s["snapshot_file_sha256"]}
            for b, s in snapshots.items()}
    if args.invalid_policy == "mask":
        receipt["raw_weight_exceptions"] = (
            f"{stats['edge_case_counts'].get('invalid_tool_turns_masked', 0)} assistant actions whose tool call "
            "opencode rewrote to the 'invalid' tool (truncated write JSON) carry weight 0.0 (context only)")
    receipt["build_stats"] = stats
    (out / "DATA_PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "README.md").write_text(readme(receipt, reports))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_source", "v4v5_selection", "inputs_sha256")},
                     indent=1))
    print(json.dumps({s: {k: v for k, v in p.items() if k not in ("edge_case_counts", "dropped")}
                      for s, p in stats["per_source"].items()}, indent=1))


# ---------------------------------------------------------------- README (generated from the receipt)
def _n(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def readme(receipt, reports):
    st = receipt["build_stats"]
    ps = st["per_source"]
    srcs = [s for s in ("hengyuf", "baihe", "v4", "v5") if ps.get(s, {}).get("conversations")]
    v2f = st["v2_filter"]
    ep = st["epochs_gbs32"]
    sel = st["v4v5_selection"]

    def row(label, fn):
        return "| " + label + " | " + " | ".join(fn(ps[s]) for s in srcs) + " | " + fn_total(fn) + " |"

    def fn_total(fn):
        try:
            return fn({"conversations": st["conversations"], "tokens_total": st["tokens_total"],
                       "assistant_tokens_total": st["assistant_tokens_total"], "tasks": "",
                       "rows_containing": st["packed_rows"], "reruns": sum(ps[s].get("reruns", 0) for s in srcs),
                       "over_262144": sum(ps[s].get("over_262144", 0) for s in srcs),
                       "tokens_min": st["tokens_min"], "tokens_max": st["tokens_max"],
                       "tokens_median": st["tokens_median"], "tokens_p90": st["tokens_p90"],
                       "assistant_messages": st["assistant_messages"], "tool_calls": st["tool_calls"]})
        except (KeyError, TypeError):
            return ""

    lines = [
        f"# sft-v3-20261029: SFT corpus v3 (v2 conversations <= 262,144 tokens + Kimi-K3 v4 / v5 passes, raw_sft scheme)",
        "",
        ("**DRY RUN** (partial v4 / v5 sources and/or `--v2-limit`): not for training. " if st["dry_run"] else "")
        + "User request (hengyuf, 2026-10-29, verbatim): \"v4 rollout 结束之后, 我想造一个新的SFT dataset (命名为SFT-v3), "
        "他包含v2 所有长度在256k以内的rollout, 以及v4, v5 里成功且没有compaction 的rollout,帮我build 一个新的SFT dataset "
        "并用data mover transfer 到HSG 上.\"",
        "",
    ]
    snaps = {b: s for b, s in sel.items() if s.get("snapshot")}
    if snaps:
        total_pending = sum(s["snapshot"]["pending_at_snapshot"] for s in snaps.values())
        lines += [
            f"**Snapshot build.** {next(iter(snaps.values()))['snapshot']['request']}. So the "
            + "; ".join(f"{b} campaigns {', '.join(s['snapshot_campaigns'])}" for b, s in snaps.items())
            + f" (still running) were read at a snapshot ({', '.join(sorted({c['scanned_at'] for s in snaps.values() for c in s['snapshot']['campaigns'].values()}))}): "
            f"their terminal slots are included like the finished campaigns; {total_pending} slots pending at the "
            "snapshot are excluded and listed (section \"Snapshot\" below, `metadata.jsonl`, `DATA_PREP_COMPLETE.json` "
            "`snapshot`).",
            "",
        ]
    lsr = st.get("assistant_literal_stop_rule") or {}
    if lsr:
        bos_k = lsr["literal_bos_kept"]
        lines += [
            "**Literal stop-token rule** (added after the first build's verification, 2026-10-29). sc14's tokenizer parses "
            "special-token text inside the content, so the literal text "
            + ", ".join(f"`{s['text']}`" for s in lsr["stop_tokens"])
            + " (generation_config.json eos id " + ", ".join(str(s["id"]) for s in lsr["stop_tokens"])
            + ") in an assistant target would train the model to emit EOS mid-turn (vLLM stops there). "
            f"**{len(lsr['dropped'])} conversation(s) dropped**: "
            + ("; ".join(f"`{d['trajectory_id']}` ({d['literal_eos']} x `</s>`, {d['literal_bos']} x `<s>`)"
                         for d in lsr["dropped"]) or "none")
            + f". The literal BOS `{lsr['bos_text']}` (id 1, not a stop token) is kept as v2 did: "
            f"{sum(b['occurrences'] for b in bos_k)} occurrences in {len(bos_k)} kept conversations ("
            + ", ".join(f"{b['source']} {sum(1 for x in bos_k if x['source'] == b['source'])}"
                        for b in {x['source']: x for x in bos_k}.values()) + ").",
            "",
        ]
    lines += [
        "The on-disk format is v2's (`kimi-pass-v2-20261023`), i.e. zhenghaox's HSG Super SFT `raw_sft` DATA_ROOT: the same "
        "converter (v2's code, exec'd from its source text), the sc14 tokenizer and chat template, identity prompt format, the "
        "same `weight` / `step_weight_agent_run_id` / `assistant_weights` / `assistant_token_counts` fields in each system "
        "message, and the same packing into 524,288-token rows, so the HSG training recipe works unchanged. Data only.",
        "",
        f"- HSG DATA_ROOT: `{HSG_ROOT}`",
        f"- Built on cw-dfw with `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B`, CPU only, no network "
        f"({receipt['completed_at']}).",
        "",
        "## Files",
        "| path | what |",
        "|---|---|",
        f"| `raw_sft/part_00{{0..5}}.mat.jsonl.packed.512k` | 6 shards, {_n(st['packed_rows'])} packed rows "
        f"({'/'.join(map(str, st['rows_per_shard']))}). Each line is `{{\"messages\": [...]}}` |",
        "| `blends/raw_sft.json` | `{\"train\": [\"0.16666667\", <HSG part_000>, ...], \"valid\": null, \"test\": null}` |",
        "| `DATA_PREP_COMPLETE.json` | Receipt with the HSG fields plus `build_stats` (`per_source`, `v2_filter`, `v4v5_selection`, "
        "`epochs_gbs32`) |",
        f"| `selection/{SELECTION_NAME}` | {_n(st['conversations'])} rows: v2's HSG selection fields + `source` (hengyuf, baihe, v4, "
        "v5), grading provenance, `rollout_batch`, `campaign`, `rerun`, `flags_any`, `flags_strong` |",
        "| `metadata.jsonl` | Every considered conversation's meta (v2's dropped ones included) with a `corpus` block (`dropped`, "
        "`dropped_reason`, shard/row, tokens, `flags`); v4 / v5 rows carry `campaign`, `record_source` (`results_file` / "
        "`snapshot`) and `snapshot_at`"
        + ("; the v4 / v5 slots pending at the snapshot are listed too (`corpus.dropped_reason` = `pending_at_snapshot`, "
           "`pending_state`)" if snaps else "") + " |",
        "| `sources/v4/`, `sources/v5/` | `trajectories.jsonl.gz` (v1 batch line layout `{meta, trajectory, final_patch}`, "
        "redacted), `harness.json`, `selection_report.json` (every slot's exclusion with reason, independent counts, flags"
        + (", the per-campaign record source and the snapshot with its pending list); `snapshot_records.json` (the "
           "snapshot campaigns' records, first-attempt records and pending slots, and the logic check)" if snaps else ")")
        + " |",
        "| `sources/task_tests.json` | Per task: test.patch / grader.py sha256 and the test files test.patch creates (flags) |",
        "| `tools/sources_v4v5.py` | Selects and extracts the v4 / v5 passes |",
        "| `tools/build_sft_v3.py` | v2's builder extended with the v2 filter and the v4 / v5 sources (converter unchanged). "
        "Deterministic |",
        "| `tools/verify_sft_v3.py` | Independent re-check (v2's verify plus the v3 checks below) |",
        "| `tools/sft_v3.sbatch` | The CPU job (schedule helper babysit pair) that runs the three tools and writes SHA256SUMS |",
        "| `SHA256SUMS` | sha256 of every file above |",
        "",
        "Training reads only `blends/` + `raw_sft/`; the v2 parts are read from v2's own sources (not copied here).",
        "",
        "## Sources (read-only)",
        "- **hengyuf** (v2): `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021`.",
        "- **baihe** (v2): `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023/sources/baihe`.",
        "- **v4**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028` (Kimi-K3 x5 "
        "on all 194 v2 tasks, OpenCode stock prompt, 262,144 window).",
        "- **v5**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028` (Kimi-K3 x5 "
        "on the 33 official2 accepted + strong tasks, same settings).",
        "- Student template and tokenizer: sc14 `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (as v1 / v2).",
        "",
        "## v2 part: conversations <= 262,144 tokens",
        f"Rule: {v2f['rule']}. Of v2's {v2f['v2_conversations']} conversations {v2f['kept']} are kept and "
        f"**{v2f['dropped_over_262144']} dropped** ({', '.join(f'{k} {v}' for k, v in v2f['dropped_by_source'].items())}); "
        "v2 itself had already dropped one over 524,288 (`v2_dropped_earlier`). Every kept conversation is re-rendered and "
        "checked byte-identical to v2 (training_row_sha256, token_length, assistant_tokens).",
        "",
        "| dropped, by DeepSWE target | conversations |",
        "|---|---|",
    ]
    lines += [f"| {t} | {c} |" for t, c in v2f["dropped_by_target"].items()]
    lines += [
        "",
        f"Tasks left without any v2 conversation: {len(v2f['tasks_without_conversation'])} "
        f"({', '.join('`' + t + '`' for t in v2f['tasks_without_conversation'])}).",
        "",
        "## v4 / v5 part: successful, no compaction, no degenerate reply (`tools/sources_v4v5.py`)",
        "Per slot of each campaign's results file (`records` = the final record; a slot ended by a broken reply carries its one "
        "approved re-run, which replaced it)"
        + (" or, for a campaign still running at the build, per TERMINAL slot of its snapshot (section Snapshot)" if snaps
           else "")
        + ", the first failing rule is the exclusion reason: final status not pass (fail, or "
        "infra = ATP exhausted / fatal / needs_review) -> not clean (evaluation not completed, reward != 1, cached "
        "generation-validity failure, files missing) -> OpenCode compaction (assistant `mode: compaction` / `summary: true`, "
        "a compaction part, or a compacted tool output) -> not exactly one user message -> a degenerate reply anywhere "
        "(any status-200 reply of http.jsonl, task subagents included, that degenerate.py's `classify()` labels pad / "
        "special / empty) -> a harness base other than v1's. Trajectory = the attempt's `export.json`; harness = the first "
        "request BODY of its `http.jsonl`; redaction = build_batch.py's patterns (as v2).",
        "",
        "| | v4 | v5 |",
        "|---|---|---|",
    ]
    for label, key in (("final slots considered", "considered"), ("kept", "kept"), ("kept that are re-runs", "kept_reruns"),
                       ("tasks with >= 1 kept", "kept_tasks")):
        lines.append(f"| {label} | {sel.get('v4', {}).get(key)} | {sel.get('v5', {}).get(key)} |")
    reasons = sorted({r for b in sel.values() for r in (b.get("exclusion_counts") or {})})
    for r in reasons:
        lines.append(f"| excluded: {r} | {(sel.get('v4', {}).get('exclusion_counts') or {}).get(r, 0)} | "
                     f"{(sel.get('v5', {}).get('exclusion_counts') or {}).get(r, 0)} |")
    for key in ("compaction", "degenerate_reply", "compaction_and_degenerate"):
        lines.append(f"| final passes with {key} (independent) | "
                     f"{(sel.get('v4', {}).get('independent_counts_among_final_passes') or {}).get(key, 0)} | "
                     f"{(sel.get('v5', {}).get('independent_counts_among_final_passes') or {}).get(key, 0)} |")
    if snaps:
        lines.append(f"| slots pending at the snapshot (not considered, listed below) | "
                     f"{sel.get('v4', {}).get('pending_at_snapshot') or 0} | {sel.get('v5', {}).get('pending_at_snapshot') or 0} |")
        lines.append(f"| kept from snapshot campaigns | "
                     f"{(sel.get('v4', {}).get('kept_by_record_source') or {}).get('snapshot', 0)} | "
                     f"{(sel.get('v5', {}).get('kept_by_record_source') or {}).get('snapshot', 0)} |")
    lines += [
        "",
        "Pilot rollouts (`pilot`, `baihe-pilot` of v4, `pilot` of v5; outside the 970 / 165 slots) are "
        + ("included when they pass the same rules." if any(s.get("pilots_included") for s in sel.values())
           else "left out.")]
    if snaps:
        lines += ["", "### Snapshot", "",
                  "Campaigns whose driver had not written `results/kimi-<campaign>.json` yet were read from their probe dirs, "
                  "read-only, with the driver's record logic: ATP slot_state's `completed` test from ATP's own cached "
                  "generation-validity assessment, `rollouts.probe_records` -> `driver.compact` (exec'd from their source "
                  "texts, sha256 recorded), and the broken-reply re-run plan `reruns/<campaign>.json`. Only TERMINAL slots "
                  "are considered; a graded slot that a broken reply ended counts only through its graded approved re-run. "
                  "Before it is used, the same code replays every finished campaign and must reproduce its results file "
                  "exactly.", ""]
        for b, s in snaps.items():
            snap = s["snapshot"]
            lc = snap["logic_check"]
            lines += [f"**{b}**: snapshot started {snap['snapshot_started_at']}; logic check: {lc['equal']} of {lc['slots']} "
                      f"records of {len(lc['campaigns'])} finished campaigns reproduced exactly ({lc['n_mismatches']} "
                      f"differences). Per-campaign record source:", "",
                      "| campaign | record source | finished / scanned at | terminal slots considered | pending at snapshot | "
                      "kept |", "|---|---|---|---|---|---|"]
            kept_by = s.get("kept_by_campaign") or {}
            for c, rs in (s.get("record_sources") or {}).items():
                if rs["source"] == "results_file":
                    total = (s.get("results_files") or {}).get(c, {}).get("stats_final") or {}
                    n = sum(total.get(k, 0) for k in ("pass", "fail", "infra"))
                    lines.append(f"| {c} | results file | {rs.get('finished_at')} | {n} | 0 | {kept_by.get(c, 0)} |")
                else:
                    lines.append(f"| {c} | **snapshot** | {rs.get('scanned_at')} | {rs.get('terminal_slots')} | "
                                 f"{rs.get('pending_slots')} | {kept_by.get(c, 0)} |")
            lines += ["", f"{snap['pending_at_snapshot']} slots pending at the snapshot (excluded; by state: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(snap["pending_by_state"].items())) + "):", "",
                      "| slot | state | detail |", "|---|---|---|"]
            for p in snap["pending"]:
                d = p.get("detail") or {}
                detail = "; ".join(f"{k} {v}" for k, v in d.items() if k in (
                    "original_status", "last_reply_kind", "rerun_campaign", "rerun_slot", "rerun_state", "note"))
                lines.append(f"| `{p['trajectory_id']}` | {p['state']} | {detail or '-'} |")
            lines.append("")
        lines += ["States: in_flight = an attempt is running; not_started = no attempt yet; ended_ungraded = the latest "
                  "attempt ended without a grade (awaiting ATP's retry or review); unassessed = graded but ATP has not "
                  "cached its generation-validity assessment yet; broken_ended_rerun_pending = graded, ended by a broken "
                  "reply, its one approved re-run has no grade yet; broken_ended_rerun_not_planned = the same, the "
                  "campaign's main slots still running (the driver plans the re-runs once they are all terminal)."]
    lines += [
        "",
        "No LLM teacher audit exists for v4 / v5 (v2 required a clean glm verdict); none was run. Informational flags per "
        "conversation (`metadata.jsonl` corpus.flags, `selection` flags_any / flags_strong; definitions in "
        "`tools/sources_v4v5.py`), never used to exclude. The same export-based flags are computed for the v2 part as a "
        "baseline (v2 passed the glm audit):",
        "",
        "| conversations with | " + " | ".join(srcs) + " |",
        "|---|" + "---|" * len(srcs),
    ]
    for key in ("any", "strong", "webfetch_calls", "webfetch_completed", "net_fetch_cmds", "net_fetch_failed",
                "net_fetch_ok", "git_history_cmds", "hidden_artifact_refs", "hidden_test_seen_before_touch",
                "hidden_test_first_touch_reads_ok", "hidden_test_read_attempts", "hidden_test_bash_first_mentions",
                "subagent_calls", "subagent_beyond_export"):
        lines.append(f"| {key} | " + " | ".join(str(ps[s].get("flags", {}).get(key, "")) for s in srcs) + " |")
    lines += [
        "",
        "## Packing",
        f"FFD of ceil32 lengths into 524,288-token rows (key: padded length desc, then trajectory_id) over all sources together; "
        f"row order shuffled with `random.Random({st['shuffle_seed']})`; 6 contiguous shards "
        f"{'/'.join(map(str, st['rows_per_shard']))}, weight 0.16666667 each. {_n(st['packed_rows'])} rows, <= "
        f"{st['conversations_per_row_max']} conversations per row, {100 * st['row_fill']:.2f}% fill. Rows by batch mix: "
        + ", ".join(f"{k} {v}" for k, v in sorted(st["rows_by_batch_mix"].items())) + ".",
        "",
        "## Stats",
        "| | " + " | ".join(srcs) + " | total |",
        "|---|" + "---|" * (len(srcs) + 1),
        row("conversations", lambda p: _n(p.get("conversations"))),
        row("tasks", lambda p: _n(p.get("tasks"))),
        row("tokens", lambda p: _n(p.get("tokens_total"))),
        row("assistant (trained) tokens, all weight 1.0", lambda p: _n(p.get("assistant_tokens_total"))),
        row("tokens/conversation min / median / p90 / max",
            lambda p: f"{_n(p.get('tokens_min'))} / {_n(p.get('tokens_median'))} / {_n(p.get('tokens_p90'))} / "
                      f"{_n(p.get('tokens_max'))}"),
        row("conversations over 262,144 tokens", lambda p: _n(p.get("over_262144"))),
        row("assistant actions / tool calls", lambda p: f"{_n(p.get('assistant_messages'))} / {_n(p.get('tool_calls'))}"),
        row("packed rows containing", lambda p: _n(p.get("rows_containing"))),
        "",
        f"Total: {_n(st['conversations'])} conversations, {_n(st['tokens_total'])} tokens (padded "
        f"{_n(st['tokens_padded_total'])}), {_n(st['assistant_tokens_total'])} assistant tokens; dropped over 524,288: "
        f"{len(st['dropped'])}.",
        "",
        "## Epochs at GBS 32 (HSG recipe: MBS 1, seq 524,288)",
        "| epochs | target_iteration = ceil(E * rows / 32) | train_samples = iterations * 32 | nominal epochs |",
        "|---|---|---|---|",
    ]
    lines += [f"| {e} | {v['target_iteration']} | {v['train_samples']} | {v['nominal_epochs']} |" for e, v in ep.items()]
    lines += [
        "",
        f"As for v1 / v2, the HSG sbatch hard-codes the corpus (train_samples, target_iteration, save_interval, the "
        f"`iter_*` checks, `selection_sha256`, `selected_trajectories`): adapt them (here selected_trajectories "
        f"{receipt['selected_trajectories']}, selection_sha256 from DATA_PREP_COMPLETE.json).",
        "",
        "## Verification (`tools/verify_sft_v3.py`)",
        "Reloads the shards with `datasets.load_dataset(\"json\")`, re-implements `SFTTokenizer.tokenize_conversation` "
        "(identity) and `SFTDataset.__getitem__` packing (32-token padding, 524,289 truncation guard, step-weight spans, "
        "loss_mask = weights[1:] with pad/IGNORE masking), then the v3 checks: shard conversations == selection rows (by "
        "source), training_row_sha256 of every conversation, every v2 conversation byte-identical to v2 and <= 262,144 "
        "tokens (and, for the real build, all of them present), receipt / blend consistency, and for every v4 / v5 "
        "conversation an independent re-read of its results record, export (compaction) and http.jsonl (degenerate.py's "
        "own scan())"
        + ("; a snapshot conversation's record is re-checked against its probe slot dir (the campaign manifest's slot, "
           "the re-run plan for a re-run, the first attempt with a full-reward completed grade and ATP's cached validity "
           "verdict None) and, if the campaign's results file has appeared since, against that file; every slot of a "
           "snapshot campaign is kept, excluded with a reason or pending, exactly once" if snaps else "") + ".",
        "",
        "VERIFY_RESULT_PLACEHOLDER",
        "",
        "## Rebuild",
        "```",
        "PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python",
        "$PY -I -B tools/sources_v4v5.py --workers 16    # sources/v4, sources/v5, sources/task_tests.json",
        "$PY -I -B tools/build_sft_v3.py --workers 16",
        "$PY -I -B tools/verify_sft_v3.py --workers 16 --readme",
        "```",
        "(normally all three run in one CPU job, `tools/sft_v3.sbatch build`, which also writes SHA256SUMS).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
