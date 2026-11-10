#!/usr/bin/env python3
"""Independent re-check of an SFT-v3 build: v2's tools/verify_corpus.py (datasets json loader + a re-implementation of
SFTTokenizer.tokenize_conversation (identity) and SFTDataset.__getitem__: CP16 padding, 524288 truncation, step
weights, with transformers' tokenizer) generalized to any output dir, plus the v3 checks:
  - shard conversations == selection rows (once each), counts by source, every training_row_sha256 and token_length;
  - receipt: selection_sha256, packed_rows, selected_trajectories, shard rows / bytes / HSG paths, blend weights;
  - v2 part: every v2 conversation byte-identical to v2 (its selection's training_row_sha256) and <= 262,144 tokens by
    v2's own count; for a real (non dry-run) build, exactly v2's conversations <= 262,144 are present;
  - v4 / v5 part (independent of sources_v4v5.py's code): each conversation's results record is a final "pass" for that
    attempt; its export.json has no compaction (assistant mode compaction / summary true, compaction part, compacted
    tool output) and exactly one user message, as many assistant messages as the selection's n_assistant; its
    http.jsonl has no degenerate reply by degenerate.py's OWN scan() + classify() (exec'd from its source text;
    --no-rescan skips this ~100 GB read); every final pass of the processed results files is either in the selection
    or excluded with a reason in selection_report.json; v5 vs analysis/sft-usable-passes.jsonl is reported.
  - assistant targets (2026-10-29 rule): no loss-target token is a generation stop id (generation_config.json
    eos_token_id other than the tokenizer's turn-end eos '<|im_end|>'), counted on the real tokenization; literal BOS
    targets are counted; every build-stage drop (metadata corpus.dropped_reason assistant_literal_eos) is confirmed
    in its own export.json and counts as an exclusion in the completeness check.
--readme writes the result into README.md (the VERIFY_RESULT_PLACEHOLDER line). Exit status 0 only if all hold.

    <vllm python> -I -B tools/verify_sft_v3.py [--out DIR] [--workers 16] [--no-rescan] [--readme]
"""
import os
import sys

sys.dont_write_bytecode = True
_TMP = os.environ.get("TMPDIR") or "/tmp"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = _TMP + "/hf"
os.environ["HF_DATASETS_CACHE"] = _TMP + "/hf/ds"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import datasets  # noqa: E402
import numpy as np  # noqa: E402
import transformers  # noqa: E402

H = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf"
OUT_DEFAULT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/sft-v3-20261029"
HSG_ROOT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-v3-20261029"
V2 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023")
V4 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028")
V5 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028")
IDT = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
SEQ = 524288
IGN = -100
V2_MAX = 262144
ALIAS = "/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/"
CANON = "/lustre/fsw/portfolios/nemotron/users/hengyuf/"
tok = None

# degenerate.py's own scan() and classify() (the definition), exec'd from its source text
_DEG = (V4 / "degenerate.py").read_text()
DG = {"json": json, "re": re, "Path": Path,
      "SPECIAL": re.compile(re.search(r'^SPECIAL = re\.compile\(r"(.*)"\)$', _DEG, re.M).group(1))}
exec(compile(_DEG[_DEG.index("def classify(body):"):_DEG.index("def fisher(")], str(V4 / "degenerate.py"), "exec"), DG)


def norm(path):
    return CANON + path[len(ALIAS):] if isinstance(path, str) and path.startswith(ALIAS) else path


# ---------------------------------------------------------------- v2's verify_corpus.py (row re-check), unchanged logic
STOP = None  # (generation stop ids except the turn end, bos id)


def stop_ids(t):
    eos = json.loads(Path(H, "generation_config.json").read_text()).get("eos_token_id")
    ids = sorted(set(eos if isinstance(eos, list) else [eos]) - {t.eos_token_id})  # t.eos_token = '<|im_end|>'
    assert ids
    return ids, t.bos_token_id


def init():
    global tok, STOP
    transformers.logging.set_verbosity_error()
    tok = transformers.AutoTokenizer.from_pretrained(H)
    STOP = stop_ids(tok)


def tc(conv):
    tokens = np.array(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False, return_dict=False,
                                              chat_template=IDT))
    if tokens.ndim > 1:
        tokens = tokens[0]
    target = tokens.copy()
    idx = 0
    for ti, turn in enumerate(conv):
        if turn["role"] == "assistant":
            assert len(turn["content"]) > 0
            assert conv[ti - 1]["role"] in ("user", "tool")
        tt = tok.apply_chat_template([turn], tokenize=True, chat_template=IDT, return_dict=False)
        if isinstance(tt[0], list):
            tt = tt[0]
        if turn["role"] in ("system", "user", "tool"):
            target[idx:idx + len(tt)] = IGN
        assert np.array_equal(tokens[idx:idx + len(tt)], tt)
        idx += len(tt)
    assert idx == len(tokens)
    return tokens.tolist(), target.tolist()


def row(args):
    path, ri, msgs = args
    pad = tok.convert_tokens_to_ids("<unk>")
    convs, cur = [], []
    for m in msgs:
        m = {k: v for k, v in m.items() if v is not None}
        if m["role"] == "system":
            if cur:
                convs.append(cur)
            cur = [m]
        else:
            cur.append(m)
    convs.append(cur)
    P, W, T, expect, ids, lengths, lits = [], [], [], 0.0, [], {}, {}
    for c in convs:
        t, g = tc(c)
        s = c[0]
        ga = np.asarray(g)
        lits[s["step_weight_agent_run_id"]] = (int(np.isin(ga, STOP[0]).sum()), int((ga == STOP[1]).sum()))
        cnt = [int(x) for x in s["assistant_token_counts"]]
        w = [float(x) for x in s["assistant_weights"]]
        el = [i for i, x in enumerate(g) if x != IGN]
        assert sum(cnt) == len(el), (s["step_weight_agent_run_id"], sum(cnt), len(el))
        cw = [0.0] * len(g)
        cur_ = 0
        for k, ww in zip(cnt, w):
            for p in el[cur_:cur_ + k]:
                cw[p] = ww
            cur_ += k
        expect += sum(k * ww for k, ww in zip(cnt, w))
        P += t
        T += g
        W += cw
        r = len(P) % 32
        if r:
            n = 32 - r
            P += [pad] * n
            T += [pad] * n
            W += [0.0] * n
        assert len(P) < SEQ + 1, ("TRUNCATION", path, ri)
        ids.append(s["step_weight_agent_run_id"])
        lengths[s["step_weight_agent_run_id"]] = len(t)
    n = SEQ + 1 - len(P)
    P += [pad] * n
    T += [pad] * n
    W += [0.0] * n
    labels = np.array(T[1:])
    lm = np.array(W[1:], dtype=np.float32)
    lm[labels == pad] = 0
    lm[labels == IGN] = 0
    return path, ri, len(convs), float(lm.sum()), expect, ids, lengths, lits


# ---------------------------------------------------------------- v4 / v5 per-conversation re-check (worker)
_SNAPSHOT_FILES = {}


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _source(attempt: Path) -> Path:
    started = _read(attempt / "started.json", {}) or {}
    return Path(norm(started.get("agent_artifacts_from") or str(attempt)))


def _validity(attempt: Path):
    """ATP's cached generation-validity verdict of an attempt (in its agent_artifacts_from source dir), or None."""
    return _read(_source(attempt) / "generation-validity-v2-r7.json")


def _infra_signal(attempt: Path):
    """The evidence on which ATP rejects a generation BEFORE (and without) caching a verdict: an agent_interrupted
    capture event (SandboxPtyError, cancelled), generation_not_quiesced / agent_command_replay_prevented /
    agent_stream_writer_error events, quiescence not confirmed, a CancelledError failure, replay-prevented.log, or a
    stream-writer error. Returns the first signal found, or None."""
    source = _source(attempt)
    path = source / "capture-events.jsonl"
    for line in (path.read_text(errors="replace").splitlines() if path.exists() else []):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "agent_interrupted" and event.get("error_type") in ("SandboxPtyError", "cancelled"):
            return f"agent_interrupted {event.get('error_type')}"
        if event.get("event") in ("generation_not_quiesced", "agent_command_replay_prevented",
                                  "agent_stream_writer_error"):
            return event["event"]
    if (_read(source / "generation-quiescence.json", {}) or {}).get("confirmed") is False:
        return "quiescence not confirmed"
    if (_read(source / "failure.json", {}) or {}).get("error_type") == "CancelledError":
        return "CancelledError"
    if (source / "replay-prevented.log").exists():
        return "replay-prevented.log"
    writer = _read(source / "stream-writer-error.json", {}) or {}
    if any(writer.get(key) not in (None, 0) for key in ("stdout_exit_code", "stderr_exit_code")):
        return "stream-writer error"
    return None


def snapshot_problems(sel, meta, record):
    """Independent re-check of a SNAPSHOT record (campaign still running at the build) against the probe dirs: the
    attempt is the slot's (or, for a re-run, the planned re-run slot's) first completed attempt with a full-reward
    grade and ATP's cached validity verdict None (an earlier completed attempt must carry ATP's non-None verdict or
    an infrastructure signal, see _infra_signal); a results file written since must hold the same record."""
    problems = []
    root = V4 if sel["source"] == "v4" else V5
    attempt = Path(sel["attempt_dir"])
    slot_dir, camp_dir = attempt.parent, attempt.parents[4]
    task_index, repeat = (int(x) for x in slot_dir.name.split("-"))
    manifest = _read(camp_dir / "manifest.json", {}) or {}
    if {t["task_id"]: t["_ng_task_index"] for t in manifest.get("tasks", [])}.get(sel["task_id"]) != task_index:
        problems.append(f"slot dir {slot_dir.name} is not task {sel['task_id']} in {camp_dir.name}'s manifest")
    if "rerun_reason" in record:
        plan = _read(root / "reruns" / f"{meta['campaign']}.json", {}) or {}
        entry = next((e for e in plan.get("slots", []) if e["task_id"] == sel["task_id"] and e["slot"] == sel["slot"]),
                     None)
        if entry is None or (entry["rerun_campaign"], entry["rerun_slot"]) != (camp_dir.name, f"kimi-r{repeat}") \
                or record.get("rerun_campaign") != entry["rerun_campaign"]:
            problems.append(f"re-run attempt is not the planned re-run slot ({entry})")
    elif (camp_dir.name, f"kimi-r{repeat}") != (meta["campaign"], sel["slot"]):
        problems.append(f"attempt not in the slot dir of {meta['campaign']}/{sel['slot']}")
    verdict = _validity(attempt)
    if not verdict or verdict.get("version") != 2 or verdict.get("assessment_revision") != 7 \
            or verdict.get("failure") is not None:
        problems.append(f"no cached valid generation verdict ({None if not verdict else verdict.get('failure')})")
    number = int(attempt.name.split("-")[1])
    for earlier in slot_dir.glob("attempt-*"):
        if int(earlier.name.split("-")[1]) < number and \
                (_read(earlier / "terminal.json", {}) or {}).get("evaluation_completed") is True:
            ev = _validity(earlier)
            if ev is not None and ev.get("failure") is None:
                problems.append(f"earlier {earlier.name} is a valid completed grade: not the slot's first grade")
            elif ev is None and not _infra_signal(earlier):  # rejected by ATP without a cached verdict, or not?
                problems.append(f"earlier {earlier.name} is completed with neither ATP's verdict nor an infra signal")
    results = root / "results" / f"kimi-{meta['campaign']}.json"
    if results.exists():  # the campaign finished between the snapshot and this check
        final = next((r for r in json.loads(results.read_text())["records"].get(sel["task_id"], [])
                      if r["slot"] == sel["slot"]), None)
        if final != record:
            problems.append("the results file written since the snapshot holds another record for this slot")
    return problems


def v45_check(job):
    """(agent_run_id, problems, facts) for one v4 / v5 conversation, read-only, without sources_v4v5.py's code."""
    sel, meta, rescan = job
    problems = []
    rid = sel["agent_run_id"]
    snapshot = meta.get("record_source") == "snapshot"
    if snapshot:
        path = meta["snapshot_file"]
        if path not in _SNAPSHOT_FILES:
            _SNAPSHOT_FILES[path] = json.loads(Path(path).read_text())
        records = ((_SNAPSHOT_FILES[path]["campaigns"].get(meta["campaign"]) or {}).get("records") or {})
    else:
        records = json.loads(Path(meta["results_file"]).read_text())["records"]
    record = next((r for r in records.get(sel["task_id"], []) if r["slot"] == sel["slot"]), None)
    if record is None or record.get("status") != "pass" or record.get("reward") != 1:
        problems.append(f"{'snapshot' if snapshot else 'results'} record not a pass: "
                        f"{None if record is None else record.get('status')}")
    elif norm(record.get("attempt_dir")) != sel["attempt_dir"]:
        problems.append("attempt_dir differs from the record")
    elif snapshot:
        problems += snapshot_problems(sel, meta, record)
    attempt = Path(sel["attempt_dir"])
    generation = Path(meta.get("generation_dir") or attempt)
    terminal = json.loads((attempt / "terminal.json").read_text())
    if terminal.get("evaluation_completed") is not True or terminal.get("reward") != 1:
        problems.append("terminal.json not a completed full-reward grade")
    export = json.loads((generation / "export.json").read_bytes())
    users = compactions = parts = compacted = assistants = 0
    for m in export.get("messages") or []:
        info = m.get("info") or {}
        users += info.get("role") == "user"
        assistants += info.get("role") == "assistant"
        compactions += info.get("role") == "assistant" and (info.get("mode") == "compaction" or info.get("summary") is True)
        for p in m.get("parts") or []:
            parts += p.get("type") == "compaction"
            compacted += p.get("type") == "tool" and bool(((p.get("state") or {}).get("time") or {}).get("compacted"))
    if users != 1 or compactions or parts or compacted:
        problems.append(f"compaction / structure: users {users}, compaction messages {compactions}, parts {parts}, "
                        f"compacted outputs {compacted}")
    if assistants != sel["n_assistant"]:
        problems.append(f"assistant messages {assistants} != selection n_assistant {sel['n_assistant']}")
    facts = {"assistants": assistants}
    if rescan:
        _, replies = DG["scan"](str(generation))
        bad = [kind for _, kind, _ in replies if kind]
        facts["replies"] = len(replies)
        if bad:
            problems.append(f"degenerate replies {dict(Counter(bad))}")
    return rid, problems, facts


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-rescan", action="store_true", help="skip the http.jsonl degenerate re-scan of v4 / v5")
    ap.add_argument("--readme", action="store_true", help="write the result into README.md")
    args = ap.parse_args()
    out = args.out
    started = time.time()
    receipt = json.loads(Path(out, "DATA_PREP_COMPLETE.json").read_text())
    stats = receipt["build_stats"]
    result = {"out": out, "dry_run": stats["dry_run"]}

    # ---- v2's row re-check
    jobs = []
    shard_files = sorted(glob.glob(out + "/raw_sft/*.packed.512k"))
    assert [os.path.basename(f) for f in shard_files] == [f"part_{k:03d}.mat.jsonl.packed.512k" for k in range(6)]
    rows_per_shard = []
    for f in shard_files:
        ds = datasets.load_dataset("json", data_files=f, split="all")
        rows_per_shard.append(len(ds))
        print(os.path.basename(f), "rows", len(ds), flush=True)
        for i in range(len(ds)):
            jobs.append((os.path.basename(f), i, ds[i]["messages"]))
    allids, tot, exp, lengths, literal = [], 0, 0, {}, {}
    with Pool(args.workers, initializer=init) as p:
        for path, ri, nc, s, e, ids, lens, lit in p.imap_unordered(row, jobs):
            assert abs(s - e) < 0.5, (path, ri, s, e)
            tot += s
            exp += e
            allids += ids
            lengths.update(lens)
            literal.update(lit)
    assert len(allids) == len(set(allids))
    print("rows", len(jobs), "convs", len(allids), "loss_mask_sum", tot, "expected", exp, "OK", flush=True)
    result.update(rows=len(jobs), conversations=len(allids), loss_mask_sum=tot, expected=exp)
    del jobs

    # ---- selection <-> shards (v2's additions, generalized)
    sel = [json.loads(line) for line in open(out + "/selection/sft_v3.jsonl", encoding="utf-8")]
    assert sorted(allids) == sorted(s["agent_run_id"] for s in sel), "shard conversations != selection rows"
    assert all(lengths[s["agent_run_id"]] == int(s["token_length"]) for s in sel), "token_length mismatch"
    shas = {}
    for f in shard_files:
        for line in open(f, encoding="utf-8"):
            ms = json.loads(line)["messages"]
            cur = []
            for m in ms + [{"role": "system", "_end": 1}]:
                if m["role"] == "system" and cur:
                    shas[cur[0]["step_weight_agent_run_id"]] = hashlib.sha256(
                        json.dumps({"messages": cur}, ensure_ascii=False).encode()).hexdigest()
                    cur = []
                cur.append(m)
    assert all(shas[s["agent_run_id"]] == s["training_row_sha256"] for s in sel), "training_row_sha256 mismatch"
    by_source = Counter(s["source"] for s in sel)
    result["by_source"] = dict(by_source)
    stop_list, bos_id = stop_ids(transformers.AutoTokenizer.from_pretrained(H))
    src_of = {s["agent_run_id"]: s["source"] for s in sel}
    stop_hits = {k: v[0] for k, v in literal.items() if v[0]}
    bos_hits = {k: v[1] for k, v in literal.items() if v[1]}
    result["assistant_literal"] = {
        "stop_ids": stop_list, "bos_id": bos_id, "conversations_checked": len(literal),
        "stop_target_conversations": stop_hits,
        "bos_target_tokens": sum(bos_hits.values()), "bos_target_conversations": len(bos_hits),
        "bos_by_source": dict(Counter(src_of[k] for k in bos_hits))}

    # ---- receipt / blend
    assert receipt["selection_sha256"] == sha256_file(out + "/selection/sft_v3.jsonl")
    assert receipt["packed_rows"] == result["rows"] == sum(rows_per_shard)
    assert receipt["selected_trajectories"] == len(sel) == stats["conversations"]
    assert [s["packed_rows"] for s in receipt["shards"]] == rows_per_shard == stats["rows_per_shard"]
    assert [s["bytes"] for s in receipt["shards"]] == [os.path.getsize(f) for f in shard_files]
    assert [s["path"] for s in receipt["shards"]] == [f"{HSG_ROOT}/raw_sft/part_{k:03d}.mat.jsonl.packed.512k"
                                                       for k in range(6)]
    blend = json.loads(Path(out, "blends/raw_sft.json").read_text())
    assert blend["train"] == [x for s in receipt["shards"] for x in ("0.16666667", s["path"])]
    assert blend["valid"] is None and blend["test"] is None
    assert receipt["state"] == ("dry-run" if stats["dry_run"] else "complete")
    for src, ps in stats["per_source"].items():
        assert ps["conversations"] == by_source.get(src, 0), src

    # ---- v2 part
    v2sel = {r["agent_run_id"]: r for r in map(json.loads, open(V2 / "selection/kimi_pass_v2.jsonl", encoding="utf-8"))}
    v2_here = [s for s in sel if s["source"] in ("hengyuf", "baihe")]
    assert all(s["agent_run_id"] in v2sel for s in v2_here)
    assert all(shas[s["agent_run_id"]] == v2sel[s["agent_run_id"]]["training_row_sha256"] for s in v2_here), \
        "a v2 conversation changed"
    assert all(int(v2sel[s["agent_run_id"]]["token_length"]) <= V2_MAX for s in v2_here)
    expected_v2 = {k for k, r in v2sel.items() if int(r["token_length"]) <= V2_MAX}
    if not stats["dry_run"]:
        assert {s["agent_run_id"] for s in v2_here} == expected_v2, "v2 conversations <= 262,144 missing or extra"
    result["v2"] = {"present": len(v2_here), "expected_full": len(expected_v2),
                    "over_262144_in_v2": sum(1 for r in v2sel.values() if int(r["token_length"]) > V2_MAX),
                    "byte_identical": True}
    assert all(s["source"] in ("hengyuf", "baihe", "v4", "v5") for s in sel)

    # ---- v4 / v5 part
    metas = {}
    for line in open(out + "/metadata.jsonl", encoding="utf-8"):
        m = json.loads(line)
        metas[m["trajectory_id"]] = m
    v45 = [s for s in sel if s["source"] in ("v4", "v5")]
    problems = {}
    # build-stage drops for a literal stop token: confirm the literal in the attempt's own export.json (assistant text,
    # reasoning or tool input), independently of the converter
    build_drops = {tid: m for tid, m in metas.items()
                   if (m.get("corpus") or {}).get("dropped_reason") == "assistant_literal_eos"}
    drop_check = {}
    for tid, m in build_drops.items():
        gen = Path(m.get("generation_dir") or norm(m["attempt_dir"]))
        export = json.loads((gen / "export.json").read_bytes())
        n = 0
        for msg in export.get("messages") or []:
            if (msg.get("info") or {}).get("role") != "assistant":
                continue
            for part in msg.get("parts") or []:
                if part.get("type") in ("text", "reasoning"):
                    n += sum((part.get("text") or "").count(tk) for tk in ("</s>",))
                elif part.get("type") == "tool":
                    n += json.dumps((part.get("state") or {}).get("input"), ensure_ascii=False).count("</s>")
        drop_check[tid] = {"export_literal_eos": n, "build_literal_eos": m["corpus"].get("assistant_literal_eos")}
        if n == 0:
            problems[tid] = ["dropped for a literal stop token, but its export.json has no '</s>' in assistant parts"]
        if tid in src_of:
            problems[tid] = (problems.get(tid) or []) + ["dropped for a literal stop token but present in the selection"]
    result["assistant_literal"]["build_drops"] = drop_check
    replies = 0
    with Pool(args.workers) as p:
        for rid, probs, facts in p.imap_unordered(v45_check, [(s, metas[s["agent_run_id"]], not args.no_rescan)
                                                               for s in v45], chunksize=1):
            replies += facts.get("replies", 0)
            if probs:
                problems[rid] = probs
    result["v45"] = {"checked": len(v45), "problems": problems, "degenerate_rescan": not args.no_rescan,
                     "replies_scanned": replies}
    # completeness: every final pass of the processed results files is kept or excluded with a reason
    completeness = {}
    for batch, root in (("v4", V4), ("v5", V5)):
        rp = Path(out, "sources", batch, "selection_report.json")
        if not rp.exists():
            continue
        report = json.loads(rp.read_text())
        excluded = {e["trajectory_id"]: e["reason"] for e in report["exclusions"]}
        for tid, m in build_drops.items():  # build-stage exclusions (after selection)
            if (m.get("corpus") or {}).get("rollout_batch") == batch:
                assert tid not in excluded, tid
                excluded[tid] = "build:assistant_literal_eos"
        kept = {s["agent_run_id"] for s in v45 if s["source"] == batch}
        prefix = "kr4" if batch == "v4" else "kr5"
        missing = []
        considered = 0
        for campaign, info in report["results_files"].items():
            data = json.loads(Path(info["path"]).read_text())
            assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"], \
                f"{info['path']} changed since selection"
            for task_id in data["tasks"]:
                for r in data["records"][task_id]:
                    if report.get("limit_per_campaign"):
                        continue  # a --limit dry run considered only some slots
                    considered += 1
                    rid = f"{prefix}-{campaign}/{task_id}/{r['slot']}"
                    if rid not in kept and rid not in excluded:
                        missing.append(rid)
                    if r["status"] != "pass" and rid in kept:
                        missing.append(rid + " (kept but not pass)")
        # snapshot campaigns (still running at the build): every slot of the campaign's manifest is kept, excluded
        # with a reason, or pending at the snapshot -- exactly one of the three
        snap = report.get("snapshot") or None
        snapshot_check = None
        if snap and not report.get("limit_per_campaign"):
            body_raw = Path(snap["snapshot_file"]).read_bytes()
            assert hashlib.sha256(body_raw).hexdigest() == snap["snapshot_file_sha256"], "snapshot file changed"
            body = json.loads(body_raw)
            pending_ids = [p["trajectory_id"] for p in snap["pending"]]
            assert len(set(pending_ids)) == len(pending_ids)
            pending_set = set(pending_ids)
            snapshot_check = {"campaigns": {}, "pending": len(pending_set), "finished_since": {}}
            for campaign, c in body["campaigns"].items():
                ids = {f"{prefix}-{campaign}/{t}/kimi-r{r}" for t in c["tasks"] for r in range(c["repeats"])}
                manifest = json.loads(Path(c["probe_dir"], "manifest.json").read_text())
                assert [t["task_id"] for t in manifest["tasks"]] == c["tasks"], campaign
                assert manifest["settings"]["repeats"] == c["repeats"], campaign
                terminal = 0
                for rid in sorted(ids):
                    found = (rid in kept) + (rid in excluded) + (rid in pending_set)
                    terminal += rid not in pending_set
                    if found != 1:
                        missing.append(f"{rid} (in {found} of kept / excluded / pending)")
                for t, recs in c["records"].items():
                    for r in recs:
                        rid = f"{prefix}-{campaign}/{t}/{r['slot']}"
                        if rid in pending_set:
                            missing.append(rid + " (has a snapshot record and is pending)")
                        if r["status"] != "pass" and rid in kept:
                            missing.append(rid + " (kept but not pass)")
                considered += terminal
                snapshot_check["campaigns"][campaign] = {"slots": len(ids), "terminal": terminal,
                                                         "pending": len(ids) - terminal}
                results = root / "results" / f"kimi-{campaign}.json"
                if results.exists():  # finished between the snapshot and this check: how did it end?
                    data = json.loads(results.read_text())
                    same = differ = 0
                    pend_outcomes = Counter()
                    for t, recs in data["records"].items():
                        for r in recs:
                            rid = f"{prefix}-{campaign}/{t}/{r['slot']}"
                            snap_rec = next((x for x in c["records"].get(t, []) if x["slot"] == r["slot"]), None)
                            if rid in pending_set:
                                pend_outcomes[r["status"]] += 1
                            elif snap_rec == r:
                                same += 1
                            else:
                                differ += 1
                                missing.append(rid + " (results file written since the snapshot differs)")
                    snapshot_check["finished_since"][campaign] = {"records_same": same, "records_differ": differ,
                                                                  "pending_final_status": dict(pend_outcomes)}
            extra = pending_set - {f"{prefix}-{c}/{t}/kimi-r{r}" for c, v in body["campaigns"].items()
                                   for t in v["tasks"] for r in range(v["repeats"])}
            if extra:
                missing.append(f"pending ids outside the snapshot campaigns: {sorted(extra)[:3]}")
        completeness[batch] = {"considered": considered, "kept": len(kept), "excluded": len(excluded),
                               "pending_at_snapshot": len(snap["pending"]) if snap else 0,
                               "snapshot_check": snapshot_check,
                               "unaccounted": missing[:20], "n_unaccounted": len(missing),
                               "exclusion_counts": dict(Counter(excluded.values()))}
        assert not missing, f"{batch}: slots not accounted for exactly once: {missing[:5]}"
        if batch == "v5" and (root / "analysis/sft-usable-passes.jsonl").exists() and not report.get("partial") \
                and not report.get("limit_per_campaign"):
            usable = {f"kr5-{r['campaign'].split('-brk')[0]}/{r['task_id']}/{r['slot']}"
                      for r in map(json.loads, open(root / "analysis/sft-usable-passes.jsonl"))}
            kept_main = {k for k in kept if not k.startswith(f"{prefix}-pilot/")}  # the pilot is not in that list
            completeness[batch]["vs_sft_usable_passes"] = {"sft_usable": len(usable),
                                                           "only_sft_usable": sorted(usable - kept_main),
                                                           "only_here": sorted(kept_main - usable)}
    result["v45_completeness"] = completeness
    ok = not problems and not stop_hits
    result["ok"] = ok
    result["seconds"] = round(time.time() - started, 1)
    print(json.dumps(result, indent=1))
    if args.readme:
        readme = Path(out, "README.md")
        text = readme.read_text()
        v45c = ", ".join(f"{b} {c['kept']} kept / {c['excluded']} excluded of {c['considered']} slots"
                         + (f" + {c['pending_at_snapshot']} pending at the snapshot" if c.get("pending_at_snapshot")
                            else "") for b, c in completeness.items())
        n_snap = sum(1 for s in v45 if metas[s["agent_run_id"]].get("record_source") == "snapshot")
        since = {f"{b}/{k}": v for b, c in completeness.items() for k, v in
                 ((c.get("snapshot_check") or {}).get("finished_since") or {}).items()}
        line = (f"Result ({time.strftime('%Y-%m-%d %H:%M %Z')}): {'OK' if ok else 'FAILED'}. {result['rows']} rows, "
                f"{result['conversations']} conversations, no truncation, loss-mask sum {tot:,.0f} == "
                f"sum(counts x weights) {exp:,.0f}; selection == shards ({', '.join(f'{k} {v}' for k, v in by_source.items())}); "
                f"every training_row_sha256 and token_length matches; v2: {len(v2_here)} conversations byte-identical "
                f"to v2, all <= 262,144"
                + ("" if stats["dry_run"] else f" (= all {len(expected_v2)} such v2 conversations)")
                + f"; v4/v5: {len(v45)} conversations re-checked (record pass, no compaction, one user message"
                + (f", {replies:,} hub replies re-scanned with degenerate.py's scan(): none degenerate" if not args.no_rescan
                   else "")
                + (f"; {n_snap} from snapshot campaigns re-checked against their probe slot dirs and re-run plans"
                   if n_snap else "")
                + f"); completeness (every slot kept, excluded with a reason, or pending at the snapshot, exactly once): "
                f"{v45c}"
                + (f"; campaigns finished between the snapshot and this check: {json.dumps(since)}" if since else "")
                + f"; assistant targets: {len(stop_hits)} of {len(literal)} conversations contain a generation stop id "
                f"{stop_list}, literal BOS (id {bos_id}) targets {sum(bos_hits.values())} in {len(bos_hits)} conversations "
                f"({', '.join(f'{k} {v}' for k, v in result['assistant_literal']['bos_by_source'].items()) or 'none'}); "
                f"build-stage drops for a literal stop token, confirmed in their export.json: "
                + (", ".join(f"{k} ({v['export_literal_eos']} x </s>)" for k, v in drop_check.items()) or "none")
                + ".")
        tmp = readme.with_name(".README.md.tmp")
        tmp.write_text(text.replace("VERIFY_RESULT_PLACEHOLDER", line))
        os.replace(tmp, readme)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
