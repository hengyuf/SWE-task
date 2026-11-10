#!/usr/bin/env python3
"""Baihe's Kimi-K3 passes on his accepted synthetic DeepSWE tasks -> a batch in kimi_pass_batch_20261021's format.

Source (read-only): B = /lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu (the original of hengyuf's port).
Output: <out>/sources/baihe/{trajectories.jsonl.gz, harness.json, selection_report.json} with the same line layout as
the v1 batch ({"meta", "trajectory": <opencode export.json, redacted>, "final_patch"}), so build_corpus.py renders both
sources with one converter.

Logic reused from hengyuf's pipeline (S/synth) and kimi_pass_batch_20261021/build_batch.py:
- redaction, harness extraction (first 'request' event BODY of http.jsonl; headers never read), trajectory stats:
  the exact source text of build_batch.py is exec'd below (no copy can drift);
- scored candidate: the unique round-*/cand-*/metrics.json whose task_id is the task (recover_slots.locate);
- probe: the probe of that round whose kimi records reproduce the scored rollouts/<slot>.md header (first 2 lines as
  rollouts.render writes them) and rollouts/<slot>.patch (grading_repair.scored_probe); attempts dir <k>-<r>, k =
  manifest _ng_task_index, slot kimi-r<r>;
- attempt: ATP slot_state's rule, the first attempt (numeric order) whose terminal.json has evaluation_completed true
  and whose generation-validity cache says failure null (ATP's generation_failure is NOT run: it would write caches);
  record fields as rollouts._record; export from agent_artifacts_from when the attempt has none (build_batch.py).

Run: <vllm python> -I tools/baihe_source.py [--out DIR] [--dry-run]
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

B = Path("/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu")
RUNS = B / "synth" / "runs"
OUTPUTS = B / "outputs"
BUILD_BATCH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021/build_batch.py")
V1_HARNESS = BUILD_BATCH.parent / "harness.json"
OUT_DEFAULT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023")
TEACHER = "kimi"
RETRIEVAL = "Answer retrieval by action"  # recover_slots.RETRIEVAL
PATH_ALIASES = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/",
                "/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/baiheh/")
PATH_CANON = "/lustre/fsw/portfolios/nemotron/users/baiheh/"
READERS = 6

# ---------------------------------------------------------------- build_batch.py helpers, exec'd from its source text
_src = BUILD_BATCH.read_text()
_start = _src.index("def _entropy(text: str)")
_end = _src.index("# ============================================================================================ selection")
BB = {"re": re, "math": math, "json": json, "Counter": Counter, "Path": Path, "hashlib": hashlib,
      "ENV_MARK": "Here is some useful information about the environment you are running in:",
      "THINK": re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)}
exec(compile(_src[_start:_end], str(BUILD_BATCH), "exec"), BB)
BUILD_BATCH_HELPERS_SHA256 = hashlib.sha256(_src[_start:_end].encode()).hexdigest()
redact_text, redact_obj, PATTERN_NAMES = BB["redact_text"], BB["redact_obj"], BB["PATTERN_NAMES"]


def norm(path):
    text = str(path)
    for alias in PATH_ALIASES:
        if text.startswith(alias):
            return PATH_CANON + text[len(alias):]
    return text


def read_json(path, default=None):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_json(value):  # build_batch.sha256_json
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        .encode("utf-8", "surrogatepass"))


def md_header(path):
    """grading_repair.md_header + the second line: (slot, status, line2)."""
    if not path.exists():
        return None, None, None
    with path.open(errors="replace") as stream:
        first, second = stream.readline().strip(), stream.readline().rstrip("\n")
    m = re.match(r"^# (\S+) — (\S+)$", first)
    return (m.group(1), m.group(2), second) if m else (None, None, second)


def earliest(scored, name):  # grading_repair.earliest
    for suffix in (".pre-regrade", ".pre-gate-fix"):
        if (scored / f"{name}{suffix}").exists():
            return scored / f"{name}{suffix}"
    return scored / name


def attempts_for(directory):  # deepswe_campaign.attempts_for
    return sorted((p for p in Path(directory).glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))


def completed_attempt(slot_dir):
    """ATP slot_state: the first attempt whose terminal.json has evaluation_completed and a valid generation.

    ATP's generation_failure() is not run (it refreshes caches inside the probe). Its cached verdict
    (generation-validity-v2-r7.json "failure") is used instead; an attempt with a non-null cached failure, or with no
    cache at all (ATP writes it for every assessed attempt), does not count as completed and the next one is tried. The
    caller then requires the chosen attempt to reproduce score()'s rendered rollouts/<slot>.md head and .patch."""
    attempts = attempts_for(slot_dir)
    why = []
    for attempt in attempts:
        terminal = read_json(attempt / "terminal.json")
        if terminal and terminal.get("evaluation_completed") is True:
            started = read_json(attempt / "started.json") or {}
            source = Path(norm(started.get("agent_artifacts_from") or attempt))
            validity = read_json(source / "generation-validity-v2-r7.json")
            vfail = validity.get("failure") if isinstance(validity, dict) else "no-validity-cache"
            if vfail is None:
                return attempt, terminal, len(attempts), None
            why.append(f"{attempt.name}: generation {vfail}")
        else:
            why.append(f"{attempt.name}: evaluation not completed")
    return None, None, len(attempts), "; ".join(why) or "no attempt"


def generation_of(attempt, terminal):  # rollouts.probe_records / _generation
    if (terminal.get("response") or {}).get("output"):
        return terminal
    for earlier in sorted(attempt.parent.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]), reverse=True):
        for name in ("terminal.json", "native-result.json"):
            data = read_json(earlier / name, {})
            if (data.get("response") or {}).get("output"):
                return data
    return {}


def render_head(slot, status, terminal, generation):
    """rollouts.render: the first two lines."""
    agent_error = generation.get("opencode_run_error_type") or generation.get("failure_reason")
    return [f"# {slot} — {status}",
            f"reward {terminal.get('reward')}; F2P {terminal.get('f2p_passed')}/{terminal.get('f2p_total')}; "
            f"P2P {terminal.get('p2p_passed')}/{terminal.get('p2p_total')}; agent_error {agent_error}; "
            f"verifier_error {terminal.get('verifier_error')}"]


def diff_paths(patch):
    out = []
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            out.append(m.group(2))
    return out


TESTY = re.compile(r"(^|/)(tests?|__tests__|spec|testdata|e2e)(/|$)|(_test\.|\.test\.|\.spec\.|(^|/)test_[^/]*$)")


def kimi_slots(probe, task_index):
    manifest = read_json(probe / "manifest.json") or {}
    repeats = manifest["settings"]["repeats"]
    out = {}
    for r in range(repeats):
        slot_dir = probe / "models" / TEACHER / "attempts" / f"{int(task_index)}-{r}"
        attempt, terminal, n_attempts, why = completed_attempt(slot_dir)
        slot = f"{TEACHER}-r{r}"
        if attempt is None:
            out[slot] = {"slot": slot, "status": "infra", "why": why, "n_attempts": n_attempts}
            continue
        generation = generation_of(attempt, terminal)
        status = "pass" if terminal.get("reward") == 1 else "fail"
        out[slot] = {"slot": slot, "status": status, "attempt": attempt, "terminal": terminal, "n_attempts": n_attempts,
                     "head": render_head(slot, status, terminal, generation),
                     "patch": terminal.get("model_patch") or generation.get("model_patch") or ""}
    return out


def language_of(target, repo):
    for line in (B / "top100.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("task_id") == target and row.get("repo") == repo:
                return row.get("language")
    return None


def select(log=print):
    problems, exclusions, candidates, tasks = [], [], [], []
    out_tasks = sorted(p.name for p in OUTPUTS.iterdir()
                       if p.is_dir() and p.name.startswith("syn-") and not p.name.endswith(".workspace"))
    # accepted lists of every run
    accepted_lists = {}
    for path in sorted(RUNS.glob("*/accepted/tasks.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                tid = json.loads(line)["task_id"]
                if tid in accepted_lists:
                    problems.append(f"{tid} accepted in two runs: {accepted_lists[tid]} and {path.parent.parent.name}")
                accepted_lists[tid] = path.parent.parent.name
    out_rows = {json.loads(l)["task_id"] for l in (OUTPUTS / "tasks.jsonl").read_text().splitlines() if l.strip()}
    check = {"outputs_task_dirs": len(out_tasks), "accepted_tasks_jsonl_rows": len(accepted_lists),
             "outputs_tasks_jsonl_rows": len(out_rows),
             "in_outputs_not_in_accepted_lists": sorted(set(out_tasks) - set(accepted_lists)),
             "in_accepted_lists_not_in_outputs": sorted(set(accepted_lists) - set(out_tasks)),
             "outputs_dirs_vs_outputs_tasks_jsonl": sorted(set(out_tasks) ^ out_rows),
             "accepted_dir_without_row": [], "provenance_problems": []}
    grading_scan = Counter()
    for tid in out_tasks:
        prov = read_json(OUTPUTS / tid / "provenance.json") or {}
        if prov.get("task_id") != tid or prov.get("accept") is not True:
            check["provenance_problems"].append(f"{tid}: provenance task_id {prov.get('task_id')} accept {prov.get('accept')}")
        task_dir = Path(norm(prov["task_dir"]))
        rel = task_dir.relative_to(RUNS).parts  # run/build/target/repo/cand-x/task/tid
        run, target, repo_dir, cand = rel[0], rel[2], rel[3], rel[4]
        assert rel[1] == "build" and rel[5] == "task" and rel[6] == tid, rel
        if accepted_lists.get(tid) not in (None, run):
            problems.append(f"{tid}: provenance run {run} != accepted list run {accepted_lists.get(tid)}")
        if not (RUNS / run / "accepted" / tid).is_dir():
            check["accept_dir_missing"] = check.get("accept_dir_missing", []) + [tid]
        root = RUNS / run / "build" / target / repo_dir
        found = [p.parent for p in sorted(root.glob("round-*/cand-*/metrics.json"))
                 if (read_json(p) or {}).get("task_id") == tid]
        base = {"source": "baihe", "run": run, "task_id": tid, "deepswe_target": target}
        if len(found) != 1:
            exclusions.append(dict(base, slot=None, reason="scored_candidate_not_unique", detail=str(found)))
            continue
        scored = found[0]
        metrics = read_json(scored / "metrics.json")
        if scored.name != cand:
            problems.append(f"{tid}: scored {scored} is not provenance's cand {cand}")
        if metrics.get("accept") is not True:
            problems.append(f"{tid}: scored metrics accept={metrics.get('accept')}")
        diff_keys = sorted(k for k in set(metrics) | set(prov) if metrics.get(k) != prov.get(k)
                           and k not in ("task_dir",))
        if diff_keys:
            problems.append(f"{tid}: provenance differs from scored metrics.json in {diff_keys}")
        # withdrawn / grading-repair / regrade evidence
        withdrawn = bool(metrics.get("withdrawn") or prov.get("withdrawn")
                         or (RUNS / run / "withdrawn").exists())
        repair_files = sorted(str(p) for p in [RUNS / run / "audit", RUNS / run / "posthoc.jsonl",
                                               RUNS / run / "withdrawn"] if p.exists())
        repair_files += sorted(str(p) for p in scored.iterdir() if re.search(r"\.pre-|regrade|repair", p.name))
        grading_scan["tasks_with_repair_or_regrade_files"] += bool(repair_files)
        if withdrawn:
            exclusions.append(dict(base, slot=None, reason="withdrawn_task", detail=str(repair_files)))
            continue
        rollouts_dir = earliest(scored, "rollouts")
        teacher = read_json(scored / "teacher_audit.json") or {}
        # the probe score() rendered
        matches, probe_notes = [], []
        for probe in sorted(scored.parent.glob("probe-*")):
            manifest = read_json(probe / "manifest.json") or {}
            idx = next((t["_ng_task_index"] for t in manifest.get("tasks", []) if t.get("task_id") == tid), None)
            if idx is None:
                continue
            slots = kimi_slots(probe, idx)
            mismatch = []
            for slot, rec in slots.items():
                md = rollouts_dir / f"{slot}.md"
                if rec["status"] == "infra":  # score() renders lost slots with an '— infra' header
                    if md.exists() and md_header(md)[1] != "infra":
                        mismatch.append(slot)
                    continue
                lines = md.read_text(errors="replace").split("\n")[:2] if md.exists() else None
                pf = rollouts_dir / f"{slot}.patch"
                if lines != rec["head"] or (pf.exists() and pf.read_text() != rec["patch"]):
                    mismatch.append(slot)
            if mismatch:
                probe_notes.append(f"{probe.name}: {mismatch} differ")
            else:
                matches.append((probe, idx, slots))
        if len(matches) != 1:
            exclusions.append(dict(base, slot=None, reason="scored_probe_not_unique",
                                   detail=f"{len(matches)} matching probes; {probe_notes}"))
            continue
        probe, idx, slots = matches[0]
        test_patch = probe / "tasks" / tid / "tests" / "test.patch"
        reset = set(diff_paths(test_patch.read_text(errors="replace"))) if test_patch.exists() else set()
        lang = language_of(target, read_json(root / "summary.json", {}).get("repo") or repo_dir.replace("__", "/", 1))
        task = {"run": run, "task_id": tid, "cand": str(scored), "round": scored.parent.name, "probe": str(probe),
                "task_index": idx, "deepswe_target": target, "host_repo": repo_dir.replace("__", "/", 1),
                "language": lang, "similarity": metrics.get("similarity"),
                "student_rate": metrics.get("student_rate"), "teacher_clean_rate": metrics.get("teacher_clean_rate"),
                "repair_files": repair_files, "kimi_slots": {s: r["status"] for s, r in slots.items()}}
        tasks.append(task)
        md_slots = sorted(p.stem for p in rollouts_dir.glob(f"{TEACHER}-r*.md"))
        for slot in sorted(set(slots) | set(md_slots)):
            rec = slots.get(slot, {"status": "missing"})
            b = dict(base, slot=slot)
            md_slot, md_status, line2 = md_header(rollouts_dir / f"{slot}.md")
            final = md_status if md_slot == slot else rec["status"]
            if final != "pass":
                exclusions.append(dict(b, reason=f"final_status_{final}", detail=(line2 or rec.get("why") or "")[:200]))
                continue
            reward = rec["terminal"].get("reward")
            if reward != 1 or not (line2 or "").startswith("reward 1"):
                exclusions.append(dict(b, reason="reward_not_full", detail=line2))
                continue
            verdict = teacher.get(slot) if isinstance(teacher.get(slot), dict) else None
            if verdict is None:
                exclusions.append(dict(b, reason="no_audit_verdict", detail="teacher_audit.json has no verdict"))
                continue
            if verdict.get("cheated") is True:
                exclusions.append(dict(b, reason="cheated", detail=f"categories {verdict.get('categories')}"))
                continue
            if RETRIEVAL in " ".join(str(c) for c in verdict.get("categories") or []):
                exclusions.append(dict(b, reason="answer_retrieval_category", detail=str(verdict.get("categories"))))
                continue
            leak = verdict.get("environment_leak")
            if isinstance(leak, dict) and leak.get("found"):
                exclusions.append(dict(b, reason="environment_leak", detail=str(leak.get("source"))[:200]))
                continue
            if verdict.get("cheated") is not False:
                exclusions.append(dict(b, reason="audit_verdict_unreadable", detail=str(verdict.get("cheated"))))
                continue
            attempt = rec["attempt"]
            generation = attempt
            if not (attempt / "export.json").exists():
                src = (read_json(attempt / "started.json") or {}).get("agent_artifacts_from")
                if src and (Path(norm(src)) / "export.json").exists():
                    generation = Path(norm(src))
            rollout_patch = rollouts_dir / f"{slot}.patch"
            patch_path = attempt / "model.patch" if (attempt / "model.patch").exists() else \
                generation / "model.patch" if (generation / "model.patch").exists() else \
                rollout_patch if rollout_patch.exists() else None
            missing = [n for n, ok in (("export.json", (generation / "export.json").exists()),
                                       ("http.jsonl", (generation / "http.jsonl").exists()),
                                       ("patch", patch_path is not None)) if not ok]
            if missing:
                exclusions.append(dict(b, reason="files_missing", detail=f"missing {missing} under {attempt}"))
                continue
            t = rec["terminal"]
            touched = diff_paths(rec["patch"])
            candidates.append({
                "source": "baihe", "run": run, "task_id": tid, "cand": str(scored), "round": scored.parent.name,
                "deepswe_target": target, "host_repo": task["host_repo"], "language": lang,
                "included_because": "accepted", "accepted": True, "accepted_posthoc": False,
                "similarity": metrics.get("similarity"),
                "slot": slot, "trajectory_id": f"{run}/{tid}/{slot}",
                "probe": str(probe), "attempt_dir": str(attempt),
                "generation_dir": str(generation) if generation != attempt else None, "n_attempts": rec["n_attempts"],
                "recovered": False, "status_source": "candidate_scoring", "final_status": "pass",
                "recorded_status": md_status,
                "grader": "deepswe-recorded",
                "grader_detail": "DeepSWE's grader at probe time (Baihe's pipeline: no grading_repair/regrade/posthoc "
                                 "record exists for this task)",
                "sanitizer_noop": None,
                "patch_test_like_paths_outside_test_patch": [p for p in touched if TESTY.search(p) and p not in reset],
                "grading_note": None,
                "f2p": f"{t.get('f2p_passed')}/{t.get('f2p_total')}", "p2p": f"{t.get('p2p_passed')}/{t.get('p2p_total')}",
                "f2p_passed": t.get("f2p_passed"), "f2p_total": t.get("f2p_total"),
                "p2p_passed": t.get("p2p_passed"), "p2p_total": t.get("p2p_total"), "counts_from": "record",
                "teacher_audit": {"cheated": verdict.get("cheated"), "categories": verdict.get("categories") or [],
                                  "rubric_score": verdict.get("rubric_score"),
                                  "environment_leak_found": (leak or {}).get("found") if isinstance(leak, dict) else None},
                "_export_path": str(generation / "export.json"), "_http_path": str(generation / "http.jsonl"),
                "_patch_path": str(patch_path), "_rollout_patch": str(rollout_patch),
                "_record_patch_sha256": sha256_bytes(rec["patch"].encode("utf-8", "surrogateescape")),
            })
        log(f"{run}/{tid}: {sum(c['task_id'] == tid for c in candidates)} kept")
    return {"check": check, "problems": problems, "exclusions": exclusions, "candidates": candidates, "tasks": tasks,
            "grading_scan": dict(grading_scan)}


def read_files(c):
    export = Path(c["_export_path"]).read_bytes()
    patch = Path(c["_patch_path"]).read_bytes()
    rpatch = Path(c["_rollout_patch"]).read_bytes() if Path(c["_rollout_patch"]).exists() else None
    payload = BB["first_request_payload"](Path(c["_http_path"]))  # request BODY only; reading stops before responses
    return export, patch, rpatch, (BB["harness_of"](payload) if payload else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.umask(0o027)
    sel = select()
    cands = sel["candidates"]
    print(json.dumps({"check": sel["check"], "problems": sel["problems"], "kept": len(cands),
                      "exclusions": dict(Counter(e["reason"] for e in sel["exclusions"])),
                      "grading_scan": sel["grading_scan"]}, indent=1))
    if args.dry_run:
        return
    v1h = json.loads(V1_HARNESS.read_text())
    out = args.out / "sources" / "baihe"
    out.mkdir(parents=True, exist_ok=True)
    harness = {"system_prompt_variants": {}, "tool_sets": {}, "request_params": {}, "missing": [],
               "first_user_mismatch": [], "first_user_matches": 0}
    totals = Counter()
    harness_redactions = Counter()
    metas = []
    tmp = out / ".trajectories.jsonl.gz.tmp"
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="trajectories.jsonl", mode="wb", fileobj=raw, mtime=0) as gz, \
            ThreadPoolExecutor(READERS) as pool:
        for c, (export_b, patch_b, rpatch_b, h) in zip(cands, pool.map(read_files, cands)):
            counts = Counter()
            export = redact_obj(json.loads(export_b), counts)
            patch_text = redact_text(patch_b.decode("utf-8", errors="replace"), counts)
            meta = {k: v for k, v in c.items() if not k.startswith("_")}
            stats = BB["trajectory_stats"](export)
            meta.update(chars=stats["chars"], est_tokens=round(stats["chars"] / 4), chars_breakdown=stats["chars_breakdown"],
                        oc_tokens=BB["oc_tokens"](export), n_messages=stats["n_messages"],
                        n_tool_calls=stats["n_tool_calls"], tools_used=stats["tools_used"],
                        session_id=(export.get("info") or {}).get("id"),
                        opencode_version=(export.get("info") or {}).get("version"),
                        patch_chars=len(patch_text), patch_sha256=sha256_bytes(patch_b),
                        patch_matches_record=sha256_bytes(patch_b) == c["_record_patch_sha256"],
                        patch_matches_rollouts=(rpatch_b == patch_b) if rpatch_b is not None else None)
            if h is None:
                harness["missing"].append(c["trajectory_id"])
                meta.update(harness_system_sha256=None, harness_tools_sha256=None, harness_params_sha256=None)
            else:
                system_messages = redact_obj(h["system_messages"], harness_redactions)
                tools = redact_obj(h["tools"], harness_redactions)
                ssha, tsha, psha = sha256_json(system_messages), sha256_json(tools), sha256_json(h["params"])
                meta.update(harness_system_sha256=ssha, harness_tools_sha256=tsha, harness_params_sha256=psha,
                            harness_system_in_v1=ssha in v1h["system_prompt_variants"],
                            harness_tools_in_v1=tsha in v1h["tool_sets"],
                            harness_params_in_v1=psha in v1h["request_params"])
                text = "\n".join(BB["_text"](m) for m in system_messages)
                parts = BB["split_system"](text)
                v = harness["system_prompt_variants"].setdefault(ssha, {
                    "system_messages": system_messages, "chars": len(text),
                    "base_sha256": sha256_bytes(parts["base"].encode("utf-8", "surrogatepass")), "env": parts["env"],
                    "rest_sha256": sha256_bytes(parts["rest"].encode("utf-8", "surrogatepass")),
                    "rest_chars": len(parts["rest"]), "runs": [], "trajectories": 0, "in_v1": ssha in v1h["system_prompt_variants"]})
                v["trajectories"] += 1
                if c["run"] not in v["runs"]:
                    v["runs"].append(c["run"])
                te = harness["tool_sets"].setdefault(tsha, {"tools": tools, "runs": [], "trajectories": 0,
                                                            "in_v1": tsha in v1h["tool_sets"]})
                te["trajectories"] += 1
                if c["run"] not in te["runs"]:
                    te["runs"].append(c["run"])
                pe = harness["request_params"].setdefault(psha, {"params": h["params"], "runs": [], "trajectories": 0,
                                                                 "in_v1": psha in v1h["request_params"]})
                pe["trajectories"] += 1
                if c["run"] not in pe["runs"]:
                    pe["runs"].append(c["run"])
                if (h.get("first_user") or "").strip() == BB["first_user_text"](export).strip():
                    harness["first_user_matches"] += 1
                else:
                    harness["first_user_mismatch"].append(c["trajectory_id"])
            meta["redactions"] = dict(sorted(counts.items()))
            totals.update(counts)
            metas.append(meta)
            line = json.dumps({"meta": meta, "trajectory": export, "final_patch": patch_text}, ensure_ascii=False)
            try:
                gz.write((line + "\n").encode("utf-8"))
            except UnicodeEncodeError:
                gz.write((json.dumps({"meta": meta, "trajectory": export, "final_patch": patch_text},
                                     ensure_ascii=True) + "\n").encode("ascii"))
            print(f"  read {len(metas)}/{len(cands)} {c['trajectory_id']}", flush=True)
    os.replace(tmp, out / "trajectories.jsonl.gz")
    base_v1 = {v["base_sha256"] for v in v1h["system_prompt_variants"].values()}
    harness["comparison_with_v1"] = {
        "distinct_system_prompts": len(harness["system_prompt_variants"]),
        "system_prompts_also_in_v1": sum(v["in_v1"] for v in harness["system_prompt_variants"].values()),
        "distinct_system_prompt_bases": len({v["base_sha256"] for v in harness["system_prompt_variants"].values()}),
        "bases_not_in_v1": sorted({v["base_sha256"] for v in harness["system_prompt_variants"].values()} - base_v1),
        "distinct_tool_sets": len(harness["tool_sets"]),
        "tool_sets_not_in_v1": [k for k, v in harness["tool_sets"].items() if not v["in_v1"]],
        "distinct_request_params": len(harness["request_params"]),
        "request_params_not_in_v1": [k for k, v in harness["request_params"].items() if not v["in_v1"]],
    }
    harness["redactions"] = dict(harness_redactions)
    (out / "harness.json").write_text(json.dumps(harness, indent=1, ensure_ascii=False) + "\n")
    report = {"source_tree": str(B), "build_batch_helpers_sha256": BUILD_BATCH_HELPERS_SHA256,
              "check": sel["check"], "problems": sel["problems"], "grading_scan": sel["grading_scan"],
              "tasks": sel["tasks"], "kept": len(metas),
              "exclusion_counts": dict(Counter(e["reason"] for e in sel["exclusions"])),
              "exclusions": sel["exclusions"],
              "redactions_trajectories": {n: totals.get(n, 0) for n in PATTERN_NAMES},
              "redactions_harness": {n: harness_redactions.get(n, 0) for n in PATTERN_NAMES},
              "harness_comparison_with_v1": harness["comparison_with_v1"],
              "harness_missing": harness["missing"], "first_user_mismatch": harness["first_user_mismatch"],
              "patch_mismatch_record": [m["trajectory_id"] for m in metas if not m["patch_matches_record"]],
              "patch_mismatch_rollouts": [m["trajectory_id"] for m in metas if m["patch_matches_rollouts"] is False],
              "metas": metas}
    (out / "selection_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("kept", "exclusion_counts", "redactions_trajectories",
                                             "redactions_harness", "harness_comparison_with_v1", "harness_missing",
                                             "first_user_mismatch", "patch_mismatch_record",
                                             "patch_mismatch_rollouts")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
