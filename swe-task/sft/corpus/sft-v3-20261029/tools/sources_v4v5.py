#!/usr/bin/env python3
"""SFT-v3 (2026-10-29) sources v4 / v5: the Kimi-K3 passes of kimi-rollouts-v4 (all 194 v2 tasks x 5) and
kimi-rollouts-v5 (33 official2 tasks x 5), both OpenCode 1.17.11 with its STOCK system prompt and a 262,144-token
window, extracted into the v1 batch line layout so that build_sft_v3.py renders them with v2's converter.

User request (hengyuf, 2026-10-29, verbatim): "v4 rollout 结束之后, 我想造一个新的SFT dataset (命名为SFT-v3), 他包含v2
所有长度在256k以内的rollout, 以及v4, v5 里成功且没有compaction 的rollout,帮我build 一个新的SFT dataset 并用data mover
transfer 到HSG 上."

Snapshot build (user, 2026-10-29 ~16:25, verbatim: "不等这些长尾rollout了,直接开始构建SFT-v3的 sft data 并传到HSG"):
--snapshot-incomplete reads every campaign that has no results file yet (its kr4- driver job still running) from its
probe dirs, read-only, at the snapshot time, with the driver's own record logic (section "snapshot" below). Its
TERMINAL slots enter the selection like results-file slots; the rest is excluded and listed as "pending at snapshot".

Inputs (read-only): <V4|V5>/results/kimi-<campaign>.json, the attempt dirs they name, each probe campaign's
tasks/<task_id>/tests/ (test.patch, grader.py), v2's metadata.jsonl (task facts of the v2 tasks) and v5's
official2-tasks.json; with --snapshot-incomplete also probes/kimi/<campaign>[-brk<n>]/ (manifest.json, attempt dirs)
and reruns/<campaign>.json of the unfinished campaigns. No ATP code is CALLED (ATP's slot_state writes validity caches
into attempt dirs); the snapshot exec's ATP's read-only pieces from its source text (below).
Output: <out>/sources/<batch>/{trajectories.jsonl.gz, harness.json, selection_report.json[, snapshot_records.json]} and
<out>/sources/task_tests.json. Each line is {"meta", "trajectory": <opencode export.json, redacted>, "final_patch"}.

Snapshot (--snapshot-incomplete), per campaign without a results file (the driver writes results/kimi-<name>.json only
when every slot and every broken-reply re-run of the campaign is terminal):
  - slot status = ATP's slot_state "completed" test (deepswe_campaign.outcome): the slot's FIRST attempt whose
    terminal.json has evaluation_completed true and whose generation_failure() is None. ATP's generation_failure is
    exec'd from its source text up to its cache lookup (its early returns + the generation-validity-v2-r7.json cache,
    signatures checked); its full assessment would WRITE that cache, so an attempt without a valid cache is "unassessed"
    (slot pending). read_jsonl is ATP's minus its one write (the .interrupted-tail copy of a torn last line).
  - record = synth/rollouts.py probe_records -> _record (exec'd) -> driver.compact (exec'd from driver.py, with
    brokenreply.py loaded by path as driver.py loads it): the same compact record the driver writes.
  - broken-reply re-runs as driver.main: when reruns/<name>.json exists it must equal driver.broken_plan() of the
    snapshot records; a graded re-run (<name>-brk<n>) replaces the original (original_attempt kept, rerun_* fields).
  - TERMINAL = graded (ATP "completed"), as progress.py counts graded slots. PENDING AT SNAPSHOT (excluded, listed in
    selection_report.json "snapshot" and snapshot_records.json): in_flight / not_started / ended_ungraded (awaiting ATP
    retry or review) / unassessed slots, and graded slots ENDED by a broken reply whose approved re-run has no grade
    yet (broken_ended_rerun_pending) or is not planned yet (broken_ended_rerun_not_planned: main slots still running).
  - logic check: the same code replays every campaign of the batch that HAS a results file (finished=True: a slot
    without a grade is the driver's infra record) and must reproduce its records and first_attempt_records exactly;
    any difference aborts.

Selection, per slot of the campaigns' results files. `records` holds each slot's FINAL record: driver.py has already
put the one approved re-run in place of a slot that a broken reply ended, when that re-run was graded. The first rule a
slot fails is its exclusion reason (the counts are also reported independently):
  1. final_status_<s>      the final record's status is not "pass" (fail, or infra = ATP exhausted / fatal /
                           needs_review)
  2. not_clean             terminal.json not evaluation_completed or reward != 1, a cached generation-validity
                           failure, or export.json / http.jsonl / model.patch missing
  3. compaction            an OpenCode compaction: an assistant message with mode "compaction" or summary true (the
                           count of progress.py / features.py), a "compaction" part, or a tool output marked compacted;
     user_messages_not_1   (no compaction signal but) not exactly one user message
  4. degenerate_reply      a broken / degenerate reply ANYWHERE in the attempt: a status-200 reply of http.jsonl (main
                           session and task subagents) that degenerate.py's classify() labels pad / special / empty
                           (degenerate.py's definition: classify() and scan() are exec'd from its source text, which is
                           byte-identical in v4 and v5)
  5. harness_unexpected    no first request body, or a system prompt whose base is not v1's opencode base
Pilot campaigns (outside the 970 / 165 slots) are left out unless --include-pilots.

Per kept slot: trajectory = the export.json of the attempt (or of its agent_artifacts_from generation), final patch =
its model.patch, harness = the BODY of the first "request" event of its http.jsonl (build_batch.first_request_payload /
harness_of: headers are never read), redaction = build_batch.py's patterns. build_batch.py's helpers are exec'd from
its source text exactly as v2's baihe_source.py does (same sha256, asserted).

Flags (informational, never used to exclude), meta["flags"] (flags_of / journal_of):
  webfetch_calls / _completed     OpenCode webfetch calls / those that returned content
  net_fetch_cmds                  bash commands that fetch code from the network: curl / wget naming a non-local URL or
                                  a public code host, git fetch / clone / pull / ls-remote / submodule update|add,
                                  pip download, npm view / pack / info, go get / go mod download, cargo fetch / search,
                                  gem fetch; _failed = a network-failure message in the output (the sandboxes have no
                                  egress: "Could not resolve host", curl exit 6, ...); _ok = completed with non-empty
                                  output and no such message (a weak heuristic: silent commands cannot be judged)
  git_history_cmds                git commands reaching beyond the checked-out history (--all, --branches, --remotes,
                                  reflog, fsck, cat-file, lost-found)
  hidden_artifact_refs            bash commands and read / grep / glob / list paths naming test.patch / solution.patch /
                                  grader.py / hidden tests, or using /tests, /solution, /oracle, /verifier at the
                                  file-system root (weak: an agent probing for them; write / edit contents ignored)
  hidden_test_seen_before_touch   a path that the task's tests/test.patch CREATES appears in a tool output before the
                                  agent itself named or wrote that path (the file existed in the sandbox)
  hidden_test_first_touch_reads_ok  the read tool returned such a path's content on the agent's first touch of it
  hidden_test_read_attempts / hidden_test_bash_first_mentions   weaker: failed reads, bash commands naming such a path
  subagent_calls                  task tool calls (their sessions are not in export.json)
  journal_*                       net fetch / webfetch / hidden refs / git history over the tool calls of EVERY hub
                                  reply in http.jsonl (task subagents included)
  strong                          webfetch_completed, hidden_test_seen_before_touch or hidden_test_first_touch_reads_ok
                                  > 0 (evidence that content was actually obtained)
  any                             strong, or any attempt above (webfetch / net fetch / git history / hidden refs / hidden
                                  test reads or mentions, or subagent fetches beyond the export); subagent_calls alone
                                  does not count

  <python3 >= 3.10> -I -B tools/sources_v4v5.py [--out DIR] [--batches v4 v5] [--workers 16]
      [--allow-partial | --snapshot-incomplete] [--include-pilots] [--campaigns NAME ...] [--limit N] [--select-only]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from itertools import zip_longest  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
V2 = S / "sft/corpus/kimi-pass-v2-20261023"
OUT_DEFAULT = S / "sft/corpus/sft-v3-20261029"
BUILD_BATCH = S / "export/kimi_pass_batch_20261021/build_batch.py"
BUILD_BATCH_HELPERS_SHA256 = "9b436db30cddf0018557f4326ffa2f7b8b0fcd9be6802a4ba85272838269dd53"  # = v2 baihe_source
DEGENERATE_BLOCK_SHA256 = "1114c513f188a8b5a4502f103ba5bacc20de8f73de34bf112ea8bd60a8b1c1ad"  # classify() + scan()
V1_HARNESS = BUILD_BATCH.parent / "harness.json"
V2_BAIHE_HARNESS = V2 / "sources/baihe/harness.json"
BATCHES = {
    "v4": {"dir": S / "sft/kimi-rollouts-v4-stock256k-20261028", "prefix": "kr4",
           "campaigns": [f"sel50-g{g}" for g in range(5)] + [f"rest47-g{g}" for g in range(5)]
                        + [f"baihe-g{g}" for g in range(8)],
           "pilots": ["pilot", "baihe-pilot"], "slots": 970},
    "v5": {"dir": S / "sft/kimi-rollouts-v5-official2-20261028", "prefix": "kr5",
           "campaigns": [f"official2-g{g}" for g in range(4)], "pilots": ["pilot"], "slots": 165},
}
OFFICIAL2_TASKS = BATCHES["v5"]["dir"] / "official2-tasks.json"
V5_USABLE = BATCHES["v5"]["dir"] / "analysis/sft-usable-passes.jsonl"
PATH_ALIASES = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/",)
PATH_CANON = "/lustre/fsw/portfolios/nemotron/users/hengyuf/"
COMPACT_AT = 262144 - 20000   # OpenCode compacts at limit.input 262,144 - 20,000
OUTPUT_CAP = 131072
WORKDIR = "/app/"             # the agent's working directory (the <env> block of every v4 / v5 system prompt)
READERS = 6                   # Lustre readers of the write phase
CHUNK = 24                    # trajectories read ahead at most (bounds memory)

# ---------------------------------------------------------------- build_batch.py helpers, exec'd from its source text
_src = BUILD_BATCH.read_text()
_start = _src.index("def _entropy(text: str)")
_end = _src.index("# ============================================================================================ selection")
BB = {"re": re, "math": math, "json": json, "Counter": Counter, "Path": Path, "hashlib": hashlib,
      "ENV_MARK": "Here is some useful information about the environment you are running in:",
      "THINK": re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)}
exec(compile(_src[_start:_end], str(BUILD_BATCH), "exec"), BB)
assert hashlib.sha256(_src[_start:_end].encode()).hexdigest() == BUILD_BATCH_HELPERS_SHA256, "build_batch.py changed"
redact_text, redact_obj, PATTERN_NAMES = BB["redact_text"], BB["redact_obj"], BB["PATTERN_NAMES"]


# ---------------------------------------------------------------- degenerate.py's classify() / scan(), exec'd
def _degenerate_source(path: Path):
    text = path.read_text()
    special = re.search(r'^SPECIAL = re\.compile\(r"(.*)"\)$', text, re.M).group(1)
    return special, text[text.index("def classify(body):"):text.index("def fisher(")]


_SPECIAL, DEGENERATE_BLOCK = _degenerate_source(BATCHES["v4"]["dir"] / "degenerate.py")
assert _degenerate_source(BATCHES["v5"]["dir"] / "degenerate.py") == (_SPECIAL, DEGENERATE_BLOCK), "v4/v5 differ"
assert hashlib.sha256(DEGENERATE_BLOCK.encode()).hexdigest() == DEGENERATE_BLOCK_SHA256, "degenerate.py changed"
DG = {"json": json, "re": re, "Path": Path, "SPECIAL": re.compile(_SPECIAL)}
exec(compile(DEGENERATE_BLOCK, str(BATCHES["v4"]["dir"] / "degenerate.py"), "exec"), DG)
classify = DG["classify"]


# ---------------------------------------------------------------- small helpers
def norm(path):
    if path is None:
        return None
    text = str(path)
    for alias in PATH_ALIASES:
        if text.startswith(alias):
            return PATH_CANON + text[len(alias):]
    return text


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value) -> str:  # build_batch.sha256_json
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        .encode("utf-8", "surrogatepass"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def diff_paths(patch: str):  # v2 baihe_source.diff_paths
    return [m.group(2) for m in re.finditer(r"(?m)^diff --git a/(\S+) b/(\S+)", patch)]


TESTY = re.compile(r"(^|/)(tests?|__tests__|spec|testdata|e2e)(/|$)|(_test\.|\.test\.|\.spec\.|(^|/)test_[^/]*$)")


def patch_files(text: str):
    """(paths a unified git diff touches, paths it creates) from `diff --git` and `new file mode` lines."""
    files, new, current = [], set(), None
    for line in text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            current = m.group(2)
            files.append(current)
        elif current and line.startswith("new file mode"):
            new.add(current)
    return files, sorted(new)


def rel(path):
    if not isinstance(path, str):
        return None
    if path.startswith(WORKDIR):
        path = path[len(WORKDIR):]
    while path.startswith("./"):
        path = path[2:]
    return path


# ---------------------------------------------------------------- flags (informational)
CURL = re.compile(r"(?<![\w./-])(?:curl|wget|aria2c)\b")
URL_HOST = re.compile(r"(?i)\b(?:https?|ftp|git|ssh)://(?:[^@/\s'\"`]*@)?(\[[0-9a-f:.]+\]|[^/\s'\"`:?#]+)|\bgit@([^:\s'\"`]+):")
LOCAL_HOST = re.compile(r"(?i)^(?:localhost|127(?:\.\d+){3}|0\.0\.0\.0|\[::1?\]|::1|host\.docker\.internal)$")
PUBLIC_HOST = re.compile(r"(?i)github\.com|githubusercontent\.com|gitlab\.com|bitbucket\.org|codeberg\.org|pypi\.org|"
                         r"pythonhosted\.org|npmjs\.(?:org|com)|yarnpkg\.com|crates\.io|proxy\.golang\.org|pkg\.go\.dev|"
                         r"rubygems\.org|sourceforge\.net|googlesource\.com|unpkg\.com|jsdelivr\.net|go\.dev")
GIT_REMOTE = re.compile(r"\bgit\s+(?:-[cC]\s+\S+\s+)*(?:fetch|clone|pull|ls-remote|submodule\s+(?:update|add))\b")
PKG_DOWNLOAD = re.compile(r"\b(?:pip3?|uv\s+pip|python3?\s+-m\s+pip)\s+download\b|\bnpm\s+(?:view|pack|info|show)\b|"
                          r"\bgo\s+(?:get|mod\s+download)\b|\bcargo\s+(?:fetch|search)\b|\bgem\s+fetch\b")
GIT_HISTORY = re.compile(r"\bgit\s+(?:-[cC]\s+\S+\s+)*(?:log|rev-list|show|branch|for-each-ref|grep)\b[^\n|;&]*?"
                         r"\s--(?:all|branches|remotes)\b|\bgit\s+(?:-[cC]\s+\S+\s+)*(?:reflog|fsck|cat-file)\b|"
                         r"lost-found")
NET_FAIL = re.compile(r"(?i)could not resolve|couldn't resolve|failed to connect|temporary failure in name resolution|"
                      r"network is unreachable|connection refused|connection timed out|operation timed out|"
                      r"name or service not known|unable to access|could not read from remote|no route to host|"
                      r"failed to establish|getaddrinfo|dial tcp|no such host|i/o timeout|enotfound|eai_again|"
                      r"econnrefused|etimedout|proxyconnect|proxy error|ssl_connect|connection reset|"
                      r"no matching distribution|could not find a version|newconnectionerror|max retries exceeded|"
                      r"lookup disabled|no such file or directory|"
                      r"\bexit(?:\s*code)?\s*[:=]?\s*(?:6|7|28|35|52|56)\b")
NO_OUTPUT = re.compile(r"^\s*(?:\(no output\))?\s*$")
# hidden tests: their names anywhere, or a file-system-ROOT path /tests, /solution, /oracle, /verifier used as a path
# (start of the value, or after whitespace / ; & | ( " ' =, and followed by / or whitespace or the end); only in what
# reads or lists files: the bash command, and read / grep / glob / list paths (never write / edit / task contents)
HIDDEN_NAME = re.compile(r"(?i)\btest\.patch\b|\bsolution\.patch\b|\bgrader\.py\b|\bhidden[ _-]?tests?\b")
HIDDEN_ROOT = re.compile(r"(?:^|[\s;&|(\"'=])/(?:tests?|solution|oracle|verifier)(?=/|\s|$)")


def hidden_ref(name, given):
    """(text, match) when a read / list tool input refers to the hidden tests (see HIDDEN_NAME / HIDDEN_ROOT)."""
    if not isinstance(given, dict):
        return None
    if name == "bash":
        texts = [given.get("command")]
    elif name in ("read", "grep", "glob", "list"):
        texts = [given.get("filePath"), given.get("path")] + ([given.get("pattern")] if name == "glob" else [])
    else:
        return None
    for text in texts:
        if isinstance(text, str):
            found = HIDDEN_NAME.search(text) or HIDDEN_ROOT.search(text)
            if found:
                return text, found
    return None
EXAMPLES = 5


def net_kinds(command: str):
    kinds = []
    if CURL.search(command):
        hosts = [a or b for a, b in URL_HOST.findall(command)]
        if any(not LOCAL_HOST.match(h) for h in hosts) or PUBLIC_HOST.search(command):
            kinds.append("curl_wget")
    if GIT_REMOTE.search(command):
        kinds.append("git_remote")
    if PKG_DOWNLOAD.search(command):
        kinds.append("package_download")
    return kinds


def _text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _short(text: str, n=200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n] + "..."


def _around(text: str, match, n=70) -> str:
    return _short(("..." if match.start() > n else "") + text[max(0, match.start() - n):match.end() + n], 2 * n + 40)


def flags_of(export: dict, tests: dict) -> dict:
    """Informational flags of one (redacted) OpenCode export, main session only (see the module docstring)."""
    new_files = [f for f in (tests or {}).get("new_test_files") or [] if f]
    new_rx = [(f, re.compile(r"(?<![\w/.-])(?:/app/)?" + re.escape(f) + r"(?![\w.-])")) for f in new_files]
    counts, examples = Counter(), defaultdict(list)
    touched = set()

    def example(name, value):
        if len(examples[name]) < EXAMPLES:
            examples[name].append(_short(value))

    for message in export.get("messages") or []:
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        for part in message.get("parts") or []:
            if part.get("type") != "tool":
                continue
            name = part.get("tool")
            state = part.get("state") or {}
            status = state.get("status")
            given = state.get("input") if isinstance(state.get("input"), dict) else {}
            given_text = _text(given)
            output = _text(state.get("output") if status == "completed" else state.get("error"))
            if name == "task":
                counts["subagent_calls"] += 1
            if name == "webfetch":
                counts["webfetch_calls"] += 1
                counts["webfetch_completed"] += status == "completed"
                example("webfetch", f"{status} {given.get('url')}")
            command = given.get("command") if name == "bash" and isinstance(given.get("command"), str) else ""
            if command:
                kinds = net_kinds(command)
                if kinds:
                    failed = bool(NET_FAIL.search(output[-6000:]))
                    ok = status == "completed" and not failed and not NO_OUTPUT.match(output)
                    counts["net_fetch_cmds"] += 1
                    counts["net_fetch_failed"] += failed
                    counts["net_fetch_ok"] += ok
                    example("net_fetch", f"{'no-failure-text' if ok else 'failed' if failed else 'silent'} "
                                         f"{'+'.join(kinds)}: {command}")
                if GIT_HISTORY.search(command):
                    counts["git_history_cmds"] += 1
                    example("git_history", command)
            found = hidden_ref(name, given)
            if found:
                counts["hidden_artifact_refs"] += 1
                example("hidden_artifact", f"{name}: {_around(*found)}")
            named = {f for f, rx in new_rx if rx.search(given_text)}
            if name == "read" and rel(given.get("filePath")) in new_files:
                path = rel(given.get("filePath"))
                if path not in touched and status == "completed":
                    counts["hidden_test_first_touch_reads_ok"] += 1
                    example("hidden_test_read", path)
                elif path not in touched:
                    counts["hidden_test_read_attempts"] += 1
            elif name == "bash" and named - touched:
                counts["hidden_test_bash_first_mentions"] += 1
                example("hidden_test_bash", command)
            if status == "completed" and output:
                seen = [f for f, rx in new_rx if f not in touched and f not in named and rx.search(output)]
                if seen:
                    counts["hidden_test_seen_before_touch"] += 1
                    example("hidden_test_seen", f"{name} output shows {seen[0]}")
            touched |= named
            if name in ("write", "edit") and rel(given.get("filePath")):
                touched.add(rel(given.get("filePath")))
    keys = ("webfetch_calls", "webfetch_completed", "net_fetch_cmds", "net_fetch_failed", "net_fetch_ok",
            "git_history_cmds", "hidden_artifact_refs", "hidden_test_seen_before_touch",
            "hidden_test_first_touch_reads_ok", "hidden_test_read_attempts", "hidden_test_bash_first_mentions",
            "subagent_calls")
    out = {k: counts.get(k, 0) for k in keys}
    out["hidden_new_test_files"] = len(new_files)
    out["examples"] = {k: v for k, v in sorted(examples.items())}
    return out


def finish_flags(flags: dict, journal_flags: dict) -> dict:
    """Merge the journal counts (every hub reply, subagents included) and set any / strong."""
    out = dict(flags)
    for key, value in sorted((journal_flags or {}).items()):
        out[f"journal_{key}"] = value
    beyond = (out.get("journal_net_fetch_calls", 0) > out.get("net_fetch_cmds", 0)
              or out.get("journal_webfetch_calls", 0) > out.get("webfetch_calls", 0))
    out["subagent_beyond_export"] = bool(beyond)
    out["strong"] = bool(out.get("webfetch_completed") or out.get("hidden_test_seen_before_touch")
                         or out.get("hidden_test_first_touch_reads_ok"))
    out["any"] = bool(out["strong"] or beyond or any(out.get(k) for k in (
        "webfetch_calls", "net_fetch_cmds", "git_history_cmds", "hidden_artifact_refs", "hidden_test_read_attempts",
        "hidden_test_bash_first_mentions", "journal_hidden_refs", "journal_git_history_calls")))
    return out


def compaction_of(export: dict) -> dict:
    out = {"assistant_compaction_messages": 0, "compaction_parts": 0, "user_messages": 0, "compacted_tool_outputs": 0}
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") == "user":
            out["user_messages"] += 1
        if info.get("role") == "assistant" and (info.get("mode") == "compaction" or info.get("summary") is True):
            out["assistant_compaction_messages"] += 1
        for part in message.get("parts") or []:
            if part.get("type") == "compaction":
                out["compaction_parts"] += 1
            if part.get("type") == "tool" and (((part.get("state") or {}).get("time") or {}).get("compacted")):
                out["compacted_tool_outputs"] += 1
    out["compacted"] = bool(out["assistant_compaction_messages"] or out["compaction_parts"]
                            or out["compacted_tool_outputs"])
    return out


def journal_of(path: Path) -> dict:
    """degenerate.py's scan() over the whole journal (same head test, same status / body filters, same classify()),
    plus usage and the tool calls of every reply (flags)."""
    replies, degenerate, statuses = [], [], Counter()
    jflags = Counter({"tool_calls": 0, "webfetch_calls": 0, "net_fetch_calls": 0, "git_history_calls": 0,
                      "hidden_refs": 0})
    prompt_max = over = length_cut = 0
    with open(path, "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            try:
                event = json.loads(raw)
                statuses[str(event.get("status"))] += 1
                if event.get("status") != 200:
                    continue
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            if not isinstance(body, dict):  # scan() would raise in classify(); counted (and reported) instead
                statuses["200_body_not_object"] += 1
                continue
            kind, tool = classify(body)
            replies.append((kind, tool))
            choice = (body.get("choices") or [{}])[0] or {}
            usage = body.get("usage") or {}
            prompt = usage.get("prompt_tokens") or 0
            prompt_max = max(prompt_max, prompt)
            over += prompt > COMPACT_AT
            length_cut += choice.get("finish_reason") == "length" or (usage.get("completion_tokens") or 0) >= OUTPUT_CAP
            if kind:
                degenerate.append({"index": len(replies), "kind": kind, "tool": tool,
                                   "finish": choice.get("finish_reason"),
                                   "completion_tokens": usage.get("completion_tokens")})
            for call in ((choice.get("message") or {}).get("tool_calls") or []):
                function = call.get("function") or {}
                name, arguments = function.get("name"), function.get("arguments") or ""
                arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
                jflags["tool_calls"] += 1
                try:
                    given = json.loads(arguments)
                except ValueError:
                    given = {"command": arguments} if name == "bash" else {}
                given = given if isinstance(given, dict) else {}
                if name == "webfetch":
                    jflags["webfetch_calls"] += 1
                if name == "bash":
                    command = given.get("command") if isinstance(given.get("command"), str) else ""
                    jflags["net_fetch_calls"] += bool(net_kinds(command))
                    jflags["git_history_calls"] += bool(GIT_HISTORY.search(command))
                jflags["hidden_refs"] += bool(hidden_ref(name, given))
    last = replies[-1] if replies else None
    return {"replies": len(replies), "degenerate_n": len(degenerate), "degenerate": degenerate[:20],
            "kinds": dict(Counter(d["kind"] for d in degenerate)), "ended": bool(last and last[0] and not last[1]),
            "statuses": dict(statuses), "prompt_max": prompt_max, "over_compact_point": over,
            "length_cut": length_cut, "flags": dict(jflags)}


# ---------------------------------------------------------------- task facts
def task_facts() -> dict:
    """{task_id: facts}: v4's tasks are v2's 194 (v2 metadata.jsonl, one value per task), v5's the official2 list."""
    out = {}
    with open(V2 / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            out.setdefault(m["task_id"], {"run": m["run"], "deepswe_target": m.get("deepswe_target"),
                                          "host_repo": m.get("host_repo"), "language": m.get("language"),
                                          "included_because": m.get("included_because"),
                                          "accepted": m.get("accepted"), "task_origin": m["source"]})
    for t in (read_json(OFFICIAL2_TASKS, {}) or {}).get("tasks", []):
        parts = Path(t["candidate_metrics"][0]).parts
        i = parts.index("build")
        out.setdefault(t["task_id"], {"run": t["run"], "deepswe_target": parts[i + 1],
                                      "host_repo": parts[i + 2].replace("__", "/", 1), "language": t.get("language"),
                                      "included_because": t.get("why"), "accepted": t.get("why") in ("accepted", "both"),
                                      "task_origin": "hengyuf-official2"})
    return out


_TESTS = {}


def task_tests(campaign_dir: Path, task_id: str) -> dict:
    key = (str(campaign_dir), task_id)
    if key not in _TESTS:
        tdir = campaign_dir / "tasks" / task_id / "tests"
        data = (tdir / "test.patch").read_bytes() if (tdir / "test.patch").exists() else b""
        files, new = patch_files(data.decode("utf-8", "replace"))
        _TESTS[key] = {"tests_dir": str(tdir), "test_patch_sha256": sha256_bytes(data) if data else None,
                       "test_files": files, "new_test_files": new,
                       "grader_sha256": sha256_file(tdir / "grader.py") if (tdir / "grader.py").exists() else None}
    return _TESTS[key]


# ---------------------------------------------------------------- per slot (worker)
def examine(job: dict) -> dict:
    record = job["record"]
    out = {"key": job["key"], "status": record.get("status")}
    if record.get("status") != "pass":
        return out
    attempt = Path(norm(record["attempt_dir"]))
    started = read_json(attempt / "started.json", {}) or {}
    generation = Path(norm(started["agent_artifacts_from"])) if started.get("agent_artifacts_from") else attempt
    terminal = read_json(attempt / "terminal.json", {}) or {}
    validity = read_json(generation / "generation-validity-v2-r7.json")
    patch_path = next((d / "model.patch" for d in (attempt, generation) if (d / "model.patch").exists()), None)
    out.update(attempt_dir=str(attempt), generation_dir=str(generation) if generation != attempt else None,
               probe_campaign=attempt.parents[4].name,
               files={"export": (generation / "export.json").exists(), "http": (generation / "http.jsonl").exists(),
                      "patch": patch_path is not None},
               evaluation_completed=terminal.get("evaluation_completed"), reward=terminal.get("reward"),
               f2p_passed=terminal.get("f2p_passed"), f2p_total=terminal.get("f2p_total"),
               p2p_passed=terminal.get("p2p_passed"), p2p_total=terminal.get("p2p_total"),
               validity_cached=isinstance(validity, dict),
               validity_failure=validity.get("failure") if isinstance(validity, dict) else None,
               patch_path=str(patch_path) if patch_path else None)
    tests = task_tests(attempt.parents[4], job["task_id"])
    out["task_tests"] = {"test_patch_sha256": tests["test_patch_sha256"], "grader_sha256": tests["grader_sha256"],
                         "new_test_files": len(tests["new_test_files"]), "tests_dir": tests["tests_dir"]}
    flags = None
    if out["files"]["export"]:
        export = json.loads((generation / "export.json").read_bytes())
        out["compaction"] = compaction_of(export)
        flags = flags_of(redact_obj(export, Counter()), tests)
        del export
    if out["files"]["http"]:
        journal = journal_of(generation / "http.jsonl")
        payload = BB["first_request_payload"](generation / "http.jsonl")
        if payload:
            h = BB["harness_of"](payload)
            system_messages = redact_obj(h["system_messages"], Counter())
            text = "\n".join(BB["_text"](m) for m in system_messages)
            out["harness"] = {"system_sha256": sha256_json(system_messages),
                              "tools_sha256": sha256_json(redact_obj(h["tools"], Counter())),
                              "params_sha256": sha256_json(h["params"]),
                              "base_sha256": sha256_bytes(BB["split_system"](text)["base"].encode("utf-8", "surrogatepass"))}
        out["journal"] = {k: v for k, v in journal.items() if k != "flags"}
        if flags is not None:
            flags = finish_flags(flags, journal["flags"])
    out["flags"] = flags
    if patch_path:
        data = patch_path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        touched = diff_paths(text)
        reset = set(tests["test_files"])
        out.update(patch_chars_file=len(text), patch_sha256=sha256_bytes(data),
                   patch_test_like_paths_outside_test_patch=[p for p in touched if TESTY.search(p) and p not in reset])
    return out


def decide(fact: dict, v1_bases: set) -> str:
    """The first selection rule the slot fails (module docstring), or None when kept."""
    if fact["status"] != "pass":
        return f"final_status_{fact['status']}"
    if not all(fact["files"].values()):
        return "not_clean"
    if fact["evaluation_completed"] is not True or fact["reward"] != 1 or fact["validity_failure"] is not None:
        return "not_clean"
    if fact["compaction"]["compacted"]:
        return "compaction"
    if fact["compaction"]["user_messages"] != 1:
        return "user_messages_not_1"
    if fact["journal"]["degenerate_n"]:
        return "degenerate_reply"
    if not fact.get("harness") or fact["harness"]["base_sha256"] not in v1_bases:
        return "harness_unexpected"
    return None


# ---------------------------------------------------------------- snapshot of unfinished campaigns (--snapshot-incomplete)
SNAPSHOT_REQUEST = "hengyuf, 2026-10-29 ~16:25 (verbatim): 不等这些长尾rollout了,直接开始构建SFT-v3的 sft data 并传到HSG"
SNAPSHOT_RULE = ("campaigns without a results file are read from their probe dirs at the snapshot time with the driver's "
                 "record logic (rollouts.probe_records -> driver.compact; ATP slot_state's 'completed' test from ATP's "
                 "cached generation-validity assessment, read-only). Only TERMINAL slots are considered (graded = ATP "
                 "completed); a graded slot ENDED by a broken reply counts only through its graded approved re-run "
                 "(reruns/<campaign>.json, <campaign>-brk<n>). Pending at snapshot (excluded, listed): in flight, not "
                 "started, ended without a grade (awaiting ATP retry / review), not yet assessed by ATP, and broken-ended "
                 "slots whose re-run has no grade yet or is not planned yet.")
ATP_CAMPAIGN = S / "atp/scripts/deepswe_campaign.py"  # what the drivers' probe_states() imports (synth/common.py path)
SYNTH_ROLLOUTS = S / "synth/rollouts.py"
SYNTH_COMMON = S / "synth/common.py"
SNAPSHOT_BLOCKS_SHA256 = {  # the exact source blocks exec'd below (any change upstream aborts)
    "atp.read_json": "2f9c44d08ae9a1446e456938b0516cba6311a07151cb534db0ae3d8f639678d2",
    "atp.read_jsonl": "00da64226f2b8ce98d086be74256859affb42b45b5ff3b9c355ffbef540e1f83",
    "atp.helpers": "1c8d0154a387509b9c65de35bab558a52961afc067e63de701313da86e0e7631",
    "atp.generation_failure_head": "4ff95b58584ca1df0204dc4ea32a80d7886a3dcaefe35954324d0b950facd812",
    "synth.read_json": "a958007113c3a56002d5e9e8e672e12ffbfc91600cdba16beafed201255d74f7",
    "synth._record": "33b3e17b9952dacfd1cc63df17786e0a6594b540514669715b07783ec4c74f7f",
    "synth._generation": "29bf019627b0091bc6939d1d05af09e611a16a209358fcf78f01bde05d6f68cf",
    "synth.probe_records": "bd20d0437958c944284c4f8bc110037d86495f644b68981bd98a724944a4994c",  # checked, replicated
    "driver.compact_stats_broken_plan": "bf4a42fc752dc65ec81f9b7335ad8527480d1cf8fe89bbd7b282746581517b91",
}
ATP_TAIL_WRITE = ('            if not tail.exists():\n'
                  '                with tail.open("xb") as stream:\n'
                  '                    stream.write(line)\n'
                  '                    stream.flush()\n'
                  '                    os.fsync(stream.fileno())\n')


class _Unassessed:
    def __repr__(self):
        return "UNASSESSED"


UNASSESSED = _Unassessed()  # generation_failure(): ATP has no valid cached assessment (it would assess and write now)
_SNAP = {}


def _block(text: str, start: str, end: str, include_end: bool = False) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j + (len(end) if include_end else 0)]


def _checked(name: str, block: str) -> str:
    assert hashlib.sha256(block.encode()).hexdigest() == SNAPSHOT_BLOCKS_SHA256[name], f"{name} changed upstream"
    return block


def _snapshot_code() -> dict:
    """ATP's and synth's pieces of the record logic, exec'd once from their source texts (block sha256 asserted)."""
    if "atp" in _SNAP:
        return _SNAP
    atp = ATP_CAMPAIGN.read_text()
    jsonl = _checked("atp.read_jsonl", _block(atp, "def read_jsonl(path):", "def load_manifest(campaign):"))
    assert jsonl.count(ATP_TAIL_WRITE) == 1
    jsonl = jsonl.replace(ATP_TAIL_WRITE, "")  # read-only: a torn last line is skipped as ATP does, but not saved
    head = _checked("atp.generation_failure_head", _block(atp, "def generation_failure(directory, result=None):",
                                                          '        return cached["failure"]\n', True))
    ns = {"json": json, "Path": Path, "os": os, "UNASSESSED": UNASSESSED}
    for code in (_checked("atp.read_json", _block(atp, "def read_json(path, default=None):", "def read_jsonl(path):")),
                 jsonl,
                 _checked("atp.helpers", _block(atp, "def finalized_candidate(directory):",
                                                "def generation_failure(directory, result=None):")),
                 head + "    return UNASSESSED  # no valid cached assessment: ATP would assess (and write it) here\n"):
        exec(compile(code, str(ATP_CAMPAIGN), "exec"), ns)
    syn_text = SYNTH_ROLLOUTS.read_text()
    _checked("synth.probe_records", _block(syn_text, "def probe_records(probe_dir) -> dict:",
                                           "# Verifier infrastructure faults"))
    assert 'TERMINAL_STATES = {"completed", "exhausted", "fatal", "needs_review"}' in syn_text
    syn = {"json": json, "Path": Path}
    exec(compile(_checked("synth.read_json", _block(SYNTH_COMMON.read_text(), "def read_json(path, default=None):",
                                                    "def write_json(path, value):")), str(SYNTH_COMMON), "exec"), syn)
    for name, start, end in (("synth._record", "def _record(model, repeat, status, generation, grade, attempt_dir) -> dict:",
                              "_INDEX_CHECKED = []"),
                             ("synth._generation", "def _generation(attempt: Path) -> dict:",
                              "def probe_states(probe_dir) -> list:")):
        exec(compile(_checked(name, _block(syn_text, start, end)), str(SYNTH_ROLLOUTS), "exec"), syn)
    _SNAP.update(atp=ns, synth=syn)
    return _SNAP


def _driver(batch: str) -> dict:
    """driver.py's compact() / stats() / broken_plan() of the batch (exec'd from its text) with brokenreply.py loaded by
    path as driver.py loads it, and its MODEL / RERUN_REASON / RERUN_RULE constants."""
    key = "driver." + batch
    if key not in _SNAP:
        kr = BATCHES[batch]["dir"]
        text = (kr / "driver.py").read_text()
        consts = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("MODEL", "RERUN_REASON", "RERUN_RULE"):
                consts[node.targets[0].id] = ast.literal_eval(node.value)
        assert set(consts) == {"MODEL", "RERUN_REASON", "RERUN_RULE"}, consts
        spec = importlib.util.spec_from_file_location(f"{BATCHES[batch]['prefix']}_brokenreply", kr / "brokenreply.py")
        broken = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(broken)
        ns = {"json": json, "Path": Path, "BROKEN": broken, **consts}
        exec(compile(_checked("driver.compact_stats_broken_plan",
                              _block(text, "def compact(record):", "def write_json_atomic(path: Path, value) -> None:")),
                     str(kr / "driver.py"), "exec"), ns)
        _SNAP[key] = ns
    return _SNAP[key]


def _slot_view(slot_dir: Path) -> dict:
    """ATP slot_state's "completed" test, read-only: {"state": "graded", "attempt"} = the slot's FIRST attempt whose
    terminal.json has evaluation_completed true and whose generation_failure() is None; "unassessed" = such an attempt
    whose validity ATP has not assessed (no valid cache) or whose files do not parse; otherwise progress.py's split of
    the other states: "not_started" (no attempt), "in_flight" (latest attempt without terminal / failure / interrupted /
    dispatch-failure file), "ended_ungraded" (latest attempt ended without a grade: ATP retry, review or infra)."""
    atp = _snapshot_code()["atp"]
    attempts = sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
    if not attempts:
        return {"state": "not_started", "attempts": 0, "attempt": None}
    for attempt in attempts:
        try:
            terminal = atp["read_json"](attempt / "terminal.json")
            if not (terminal and terminal.get("evaluation_completed") is True):
                continue
            failure = atp["generation_failure"](attempt, terminal)
        except (OSError, ValueError, UnicodeError) as error:
            return {"state": "unassessed", "attempts": len(attempts), "attempt": attempt,
                    "error": f"{type(error).__name__}: {str(error)[:200]}"}
        if failure is UNASSESSED:
            return {"state": "unassessed", "attempts": len(attempts), "attempt": attempt}
        if failure is None:
            return {"state": "graded", "attempts": len(attempts), "attempt": attempt}
    latest = attempts[-1]
    ended = any((latest / name).exists() for name in ("terminal.json", "failure.json", "interrupted.json",
                                                       "dispatch-failure.json"))
    return {"state": "ended_ungraded" if ended else "in_flight", "attempts": len(attempts), "attempt": latest}


def _record_of(batch: str, view: dict, repeat: int, graded: bool) -> dict:
    """driver.compact(rollouts._record(...)) of one slot, as rollouts.probe_records builds it from slot_state."""
    syn = _snapshot_code()["synth"]
    drv = _driver(batch)
    attempt = view["attempt"]
    grade = syn["read_json"](attempt / "terminal.json", {}) if attempt else {}
    generation = grade if (grade.get("response") or {}).get("output") else (syn["_generation"](attempt) if attempt else {})
    status = ("pass" if grade.get("reward") == 1 else "fail") if graded else "infra"
    return drv["compact"](syn["_record"](drv["MODEL"], repeat, status, generation, grade, attempt))


def campaign_snapshot(batch: str, name: str, finished: bool = False, workers: int = 16) -> dict:
    """The driver's records of one campaign from its probe dirs (read-only). finished=False: snapshot of a campaign
    still running (terminal slots get a record, the rest is pending); finished=True: the logic-check replay of a
    campaign whose results file exists (a slot without a grade is the driver's infra record)."""
    conf = BATCHES[batch]
    drv = _driver(batch)
    model = drv["MODEL"]
    kr = conf["dir"].resolve()  # the drivers' KR = Path(__file__).resolve().parent: the record paths are built on it
    probe = kr / "probes" / model / name
    scanned_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest = read_json(probe / "manifest.json")
    if not manifest:
        raise SystemExit(f"{batch} {name}: no manifest.json in {probe} (campaign not created yet): cannot snapshot it")
    if any(p.is_dir() for p in (kr / "probes" / model).glob(f"{name}-rec*")):  # driver step 1, not replicated here
        raise SystemExit(f"{batch} {name}: infra-recovery campaigns ({name}-rec<n>) exist: not replicated by the snapshot")
    assert list(manifest["settings"]["models"]) == [model], probe
    repeats = manifest["settings"]["repeats"]
    tasks = [t["task_id"] for t in manifest["tasks"]]
    slots = [(t["task_id"], t["_ng_task_index"], r) for t in manifest["tasks"] for r in range(repeats)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        views = list(pool.map(lambda s: _slot_view(probe / "models" / model / "attempts" / f"{s[1]}-{s[2]}"), slots))
    plan_path = kr / "reruns" / f"{name}.json"
    plan_raw = plan_path.read_bytes() if plan_path.exists() else None
    plan = json.loads(plan_raw) if plan_raw else None
    main_terminal = finished or plan is not None  # the driver writes the plan only once every main slot is terminal
    pending = []

    def pend(task_id, slot, state, view, attempt_dir=None, **detail):
        attempt_dir = attempt_dir or (str(view["attempt"]) if view and view.get("attempt") else None)
        pending.append({"trajectory_id": f"{conf['prefix']}-{name}/{task_id}/{slot}", "campaign": name,
                        "task_id": task_id, "slot": slot, "state": state,
                        "attempts": view.get("attempts") if view else None, "attempt_dir": norm(attempt_dir),
                        **({"error": view["error"]} if view and view.get("error") else {}),
                        **({"detail": detail} if detail else {})})

    items = []
    for i, ((task_id, _, repeat), view) in enumerate(zip(slots, views)):
        if view["state"] == "graded" or (main_terminal and view["state"] == "ended_ungraded"):
            items.append((i, view, repeat, view["state"] == "graded"))
        else:
            pend(task_id, f"{model}-r{repeat}", view["state"], view,
                 **({"note": "the re-run plan exists, so the driver saw this slot terminal"} if main_terminal else {}))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        made = dict(pool.map(lambda item: (item[0], _record_of(batch, item[1], item[2], item[3])), items))
    first = {}
    for i, (task_id, _, _) in enumerate(slots):
        if i in made:
            first.setdefault(task_id, []).append(made[i])
    final = {task_id: [dict(r) for r in recs] for task_id, recs in first.items()}
    computed = drv["broken_plan"](name, first)
    plan_check = None
    if plan is not None:
        if len(made) == len(slots):  # every main slot terminal: exactly the driver's own plan check
            plan_check = "equal" if computed["slots"] == plan["slots"] else "differs"
        else:
            keys = {(e["task_id"], e["slot"], e["original_attempt"], e["original_status"], e["last_reply_kind"])
                    for e in plan["slots"]}
            plan_check = "subset" if all((e["task_id"], e["slot"], e["original_attempt"], e["original_status"],
                                          e["last_reply_kind"]) in keys for e in computed["slots"]) else "differs"
        if plan_check == "differs":
            raise SystemExit(f"{plan_path} differs from driver.broken_plan() of the {name} records read now "
                             f"({len(plan['slots'])} vs {len(computed['slots'])} slots): refusing, as the driver would")
    reruns = []
    if plan is not None:
        for entry in plan["slots"]:
            task_id, slot = entry["task_id"], entry["slot"]
            index = next((k for k, r in enumerate(final.get(task_id, [])) if r is not None and r["slot"] == slot), None)
            if index is None:  # the original itself has no record (pending above)
                continue
            original = first[task_id][index]
            brk = kr / "probes" / model / entry["rerun_campaign"]
            brk_manifest = read_json(brk / "manifest.json")
            repeat = int(entry["rerun_slot"].rsplit("-r", 1)[1])
            view = None
            if brk_manifest:
                index_of = {t["task_id"]: t["_ng_task_index"] for t in brk_manifest["tasks"]}
                view = _slot_view(brk / "models" / model / "attempts" / f"{index_of[task_id]}-{repeat}")
            if view is not None and (view["state"] == "graded" or (finished and view["state"] == "ended_ungraded")):
                rerun = _record_of(batch, view, repeat, view["state"] == "graded")
                graded = rerun["status"] in ("pass", "fail")
                outcome = "infra" if not graded else "broken_again" if rerun["ended_by_broken_reply"] else rerun["status"]
                provenance = {"rerun_reason": drv["RERUN_REASON"], "rerun_campaign": entry["rerun_campaign"],
                              "rerun_slot": rerun["slot"]}
                if graded:  # driver.main: the re-run replaces the original; the original stays in the record
                    final[task_id][index] = dict(rerun, slot=slot, original_attempt=original, **provenance)
                else:
                    final[task_id][index] = dict(original, rerun_attempt=rerun, rerun_replaced=False, **provenance)
                reruns.append({"task_id": task_id, "slot": slot, "rerun_campaign": entry["rerun_campaign"],
                               "rerun_slot": entry["rerun_slot"], "outcome": outcome,
                               "rerun_attempt": norm(rerun["attempt_dir"])})
            else:
                final[task_id][index] = None
                pend(task_id, slot, "broken_ended_rerun_pending", None, attempt_dir=original["attempt_dir"],
                     original_status=original["status"], last_reply_kind=original["last_reply_kind"],
                     rerun_campaign=entry["rerun_campaign"], rerun_slot=entry["rerun_slot"],
                     rerun_state=view["state"] if view else "rerun_campaign_not_created",
                     rerun_attempts=view.get("attempts") if view else 0,
                     rerun_attempt_dir=norm(str(view["attempt"])) if view and view.get("attempt") else None)
    elif not finished:  # no plan yet (main slots still running): every broken-ended graded slot waits for its re-run
        for task_id, recs in final.items():
            for index, r in enumerate(recs):
                if r["ended_by_broken_reply"]:
                    final[task_id][index] = None
                    pend(task_id, r["slot"], "broken_ended_rerun_not_planned", None, attempt_dir=r["attempt_dir"],
                         original_status=r["status"], last_reply_kind=r["last_reply_kind"])
    records = {task_id: [r for r in final.get(task_id, []) if r is not None] for task_id in tasks}
    return {"campaign": name, "batch": batch, "finished": finished, "scanned_at": scanned_at,
            "probe_dir": norm(str(probe)), "tasks": tasks, "repeats": repeats, "slots": len(slots),
            "slot_states": dict(Counter(v["state"] for v in views)),
            "plan_file": str(plan_path) if plan is not None else None,
            "plan_sha256": sha256_bytes(plan_raw) if plan_raw else None,
            "plan_slots": len(plan["slots"]) if plan is not None else None, "plan_check": plan_check,
            "broken_ended_graded": len(computed["slots"]),
            "terminal_slots": sum(len(v) for v in records.values()), "pending_slots": len(pending),
            "stats_first_attempts": drv["stats"](first), "stats_terminal": drv["stats"](records),
            "rerun_outcomes": dict(Counter(r["outcome"] for r in reruns)), "reruns": reruns,
            "pending": pending, "records": records, "first_attempt_records": first}


def snapshot_logic_check(batch: str, files: dict, workers: int = 16) -> dict:
    """Replay campaign_snapshot(finished=True) on every campaign of the batch that has a results file: its records and
    first_attempt_records must equal the file's exactly (same dicts), with nothing pending."""
    out = {"campaigns": [], "slots": 0, "equal": 0, "n_mismatches": 0, "mismatches": []}
    for name, info in files.items():
        data = json.loads(Path(info["path"]).read_text())
        snap = campaign_snapshot(batch, name, finished=True, workers=workers)
        out["campaigns"].append(name)
        for field in ("records", "first_attempt_records"):
            for task_id in data["tasks"]:
                for want, got in zip_longest(data[field].get(task_id, []), snap[field].get(task_id, [])):
                    if field == "records":
                        out["slots"] += 1
                        out["equal"] += want == got
                    if want != got:
                        out["n_mismatches"] += 1
                        if len(out["mismatches"]) < 20:
                            out["mismatches"].append({"campaign": name, "field": field, "task_id": task_id,
                                                      "results": want, "snapshot": got})
        if snap["pending"]:
            out["n_mismatches"] += len(snap["pending"])
            out["mismatches"].append({"campaign": name, "pending": snap["pending"][:5]})
    return out


# ---------------------------------------------------------------- results files
def load_jobs(batch: str, args) -> tuple:
    conf = BATCHES[batch]
    names = list(conf["campaigns"]) + (list(conf["pilots"]) if args.include_pilots else [])
    if args.campaigns:
        names = [n for n in names if n in args.campaigns]
    files, missing, jobs, snapshots = {}, [], [], {}
    snapshot_file = os.path.abspath(args.out / "sources" / batch / "snapshot_records.json")
    for name in names:
        path = conf["dir"] / "results" / f"kimi-{name}.json"
        if not path.exists() and args.snapshot_incomplete:  # the campaign's driver is still running: snapshot it
            snap = campaign_snapshot(batch, name, finished=False, workers=args.workers)
            snapshots[name] = snap
            print(f"{batch}: {name} snapshot at {snap['scanned_at']}: {snap['terminal_slots']} terminal / "
                  f"{snap['pending_slots']} pending of {snap['slots']} slots (states {snap['slot_states']}, plan "
                  f"{snap['plan_slots']}, re-runs {snap['rerun_outcomes']})", flush=True)
            taken = 0
            for task_id in snap["tasks"]:
                for record in snap["records"][task_id]:
                    if args.limit and taken >= args.limit:
                        break
                    taken += 1
                    jobs.append({"batch": batch, "campaign": name, "results_file": None, "record_source": "snapshot",
                                 "snapshot_at": snap["scanned_at"], "snapshot_file": snapshot_file,
                                 "task_id": task_id, "slot": record["slot"], "record": record,
                                 "key": f"{conf['prefix']}-{name}/{task_id}/{record['slot']}"})
            continue
        if not path.exists():
            missing.append(name)
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data["model"] == "kimi" and data["context_window"] == 262144 and data["prompt"] == "stock", path
        tasks, repeats, records = data["tasks"], data["repeats"], data["records"]
        assert sorted(records) == sorted(tasks), path
        files[name] = {"path": str(path), "sha256": sha256_bytes(raw), "finished_at": data.get("finished_at"),
                       "stats_first_attempts": data.get("stats_first_attempts"), "stats_final": data.get("stats_final"),
                       "broken_reply_rerun_outcomes": data.get("broken_reply_rerun_outcomes"),
                       "recovered_budget_refusals": sum(map(len, (data.get("recovered_budget_refusals") or {}).values())),
                       "tasks": len(tasks), "repeats": repeats}
        taken = 0
        for task_id in tasks:
            slots = records[task_id]
            assert [r["slot"] for r in slots] == [f"kimi-r{i}" for i in range(repeats)], (path, task_id)
            for record in slots:
                if args.limit and taken >= args.limit:
                    break
                taken += 1
                jobs.append({"batch": batch, "campaign": name, "results_file": str(path),
                             "record_source": "results_file", "task_id": task_id,
                             "slot": record["slot"], "record": record,
                             "key": f"{conf['prefix']}-{name}/{task_id}/{record['slot']}"})
    return files, missing, jobs, snapshots


# ---------------------------------------------------------------- write phase
def read_ahead(items, fn):
    with ThreadPoolExecutor(max_workers=READERS) as pool:
        pending, position = [], 0
        while pending or position < len(items):
            while len(pending) < CHUNK and position < len(items):
                pending.append((items[position], pool.submit(fn, items[position])))
                position += 1
            item, future = pending.pop(0)
            yield item, future.result()


def read_files(item):
    job, fact = item
    generation = Path(fact["generation_dir"] or fact["attempt_dir"])
    export = (generation / "export.json").read_bytes()
    patch = Path(fact["patch_path"]).read_bytes()
    payload = BB["first_request_payload"](generation / "http.jsonl")  # request BODY only
    return export, patch, (BB["harness_of"](payload) if payload else None)


def meta_of(job: dict, fact: dict, task: dict) -> dict:
    record = job["record"]
    original = record.get("original_attempt") or {}
    return {
        "source": job["batch"], "rollout_batch": job["batch"],
        "run": task["run"], "task_id": job["task_id"], "deepswe_target": task.get("deepswe_target"),
        "host_repo": task.get("host_repo"), "language": task.get("language"), "task_origin": task.get("task_origin"),
        "included_because": task.get("included_because"), "accepted": task.get("accepted"),
        "slot": job["slot"], "trajectory_id": job["key"], "campaign": job["campaign"],
        "probe_campaign": fact["probe_campaign"], "results_file": job["results_file"],
        "record_source": job["record_source"], "snapshot_at": job.get("snapshot_at"),
        "snapshot_file": job.get("snapshot_file"),
        "attempt_dir": fact["attempt_dir"], "generation_dir": fact["generation_dir"],
        "rerun": "rerun_reason" in record, "rerun_reason": record.get("rerun_reason"),
        "rerun_campaign": record.get("rerun_campaign"), "rerun_slot": record.get("rerun_slot"),
        "original_attempt_dir": norm(original.get("attempt_dir")), "original_status": original.get("status"),
        "original_last_reply_kind": original.get("last_reply_kind"),
        "status_source": f"{job['batch']}_results" if job["record_source"] == "results_file" else f"{job['batch']}_snapshot",
        "final_status": "pass", "recorded_status": record["status"],
        "grader": "task-installed",
        "grader_detail": "graded at rollout time by ATP with the task's installed tests/ (grader.py sha256 in "
                         "grader_sha256); no LLM teacher audit exists for v4 / v5 (flags only)",
        "grader_sha256": fact["task_tests"]["grader_sha256"], "test_patch_sha256": fact["task_tests"]["test_patch_sha256"],
        "sanitizer_noop": None, "grading_note": None,
        "f2p": record.get("f2p"), "p2p": record.get("p2p"),
        "f2p_passed": fact.get("f2p_passed"), "f2p_total": fact.get("f2p_total"),
        "p2p_passed": fact.get("p2p_passed"), "p2p_total": fact.get("p2p_total"), "counts_from": "record",
        "teacher_audit": None,
        "patch_test_like_paths_outside_test_patch": fact.get("patch_test_like_paths_outside_test_patch"),
        "compaction": fact["compaction"],
        "journal": {k: fact["journal"][k] for k in ("replies", "degenerate_n", "statuses", "prompt_max",
                                                     "over_compact_point", "length_cut")},
        "flags": fact["flags"],
    }


def write_batch(batch: str, kept: list, facts: dict, tasks: dict, out: Path, v1h: dict, v2h: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    harness = {"system_prompt_variants": {}, "tool_sets": {}, "request_params": {}, "missing": [],
               "first_user_mismatch": [], "first_user_matches": 0}
    totals, harness_redactions, metas = Counter(), Counter(), []
    tmp = out / ".trajectories.jsonl.gz.tmp"
    items = [(job, facts[job["key"]]) for job in kept]
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="trajectories.jsonl", mode="wb", fileobj=raw, mtime=0) as gz:
        for (job, fact), (export_b, patch_b, h) in read_ahead(items, read_files):
            counts = Counter()
            export = redact_obj(json.loads(export_b), counts)
            patch_text = redact_text(patch_b.decode("utf-8", errors="replace"), counts)
            meta = meta_of(job, fact, tasks[job["task_id"]])
            stats = BB["trajectory_stats"](export)
            meta.update(chars=stats["chars"], est_tokens=round(stats["chars"] / 4), chars_breakdown=stats["chars_breakdown"],
                        oc_tokens=BB["oc_tokens"](export), n_messages=stats["n_messages"],
                        n_tool_calls=stats["n_tool_calls"], tools_used=stats["tools_used"],
                        session_id=(export.get("info") or {}).get("id"),
                        opencode_version=(export.get("info") or {}).get("version"),
                        patch_chars=len(patch_text), patch_sha256=sha256_bytes(patch_b),
                        patch_matches_record=len(patch_b.decode("utf-8", errors="replace")) == job["record"].get("patch_chars"))
            if h is None:
                harness["missing"].append(job["key"])
                meta.update(harness_system_sha256=None, harness_tools_sha256=None, harness_params_sha256=None)
            else:
                system_messages = redact_obj(h["system_messages"], harness_redactions)
                tools = redact_obj(h["tools"], harness_redactions)
                ssha, tsha, psha = sha256_json(system_messages), sha256_json(tools), sha256_json(h["params"])
                meta.update(harness_system_sha256=ssha, harness_tools_sha256=tsha, harness_params_sha256=psha,
                            harness_system_in_v1=ssha in v1h["system_prompt_variants"],
                            harness_system_in_v2=ssha in v1h["system_prompt_variants"] or ssha in v2h["system_prompt_variants"],
                            harness_tools_in_v1=tsha in v1h["tool_sets"], harness_params_in_v1=psha in v1h["request_params"])
                text = "\n".join(BB["_text"](m) for m in system_messages)
                parts = BB["split_system"](text)
                v = harness["system_prompt_variants"].setdefault(ssha, {
                    "system_messages": system_messages, "chars": len(text),
                    "base_sha256": sha256_bytes(parts["base"].encode("utf-8", "surrogatepass")), "env": parts["env"],
                    "rest_sha256": sha256_bytes(parts["rest"].encode("utf-8", "surrogatepass")),
                    "rest_chars": len(parts["rest"]), "runs": [], "trajectories": 0,
                    "in_v1": ssha in v1h["system_prompt_variants"],
                    "in_v2": ssha in v1h["system_prompt_variants"] or ssha in v2h["system_prompt_variants"]})
                v["trajectories"] += 1
                if meta["run"] not in v["runs"]:
                    v["runs"].append(meta["run"])
                te = harness["tool_sets"].setdefault(tsha, {"tools": tools, "runs": [], "trajectories": 0,
                                                            "in_v1": tsha in v1h["tool_sets"]})
                te["trajectories"] += 1
                if meta["run"] not in te["runs"]:
                    te["runs"].append(meta["run"])
                pe = harness["request_params"].setdefault(psha, {"params": h["params"], "runs": [], "trajectories": 0,
                                                                 "in_v1": psha in v1h["request_params"]})
                pe["trajectories"] += 1
                if meta["run"] not in pe["runs"]:
                    pe["runs"].append(meta["run"])
                if (h.get("first_user") or "").strip() == BB["first_user_text"](export).strip():
                    harness["first_user_matches"] += 1
                else:
                    harness["first_user_mismatch"].append(job["key"])
            meta["redactions"] = dict(sorted(counts.items()))
            totals.update(counts)
            metas.append(meta)
            line = {"meta": meta, "trajectory": export, "final_patch": patch_text}
            try:
                gz.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            except UnicodeEncodeError:  # lone surrogates: keep them as JSON escapes (build_batch.dumps_line)
                gz.write((json.dumps(line, ensure_ascii=True) + "\n").encode("ascii"))
            if len(metas) % 25 == 0:
                print(f"  {batch}: wrote {len(metas)}/{len(kept)}", flush=True)
    os.replace(tmp, out / "trajectories.jsonl.gz")
    base_v1 = {v["base_sha256"] for v in v1h["system_prompt_variants"].values()}
    harness["comparison_with_v1"] = {
        "distinct_system_prompts": len(harness["system_prompt_variants"]),
        "system_prompts_also_in_v1": sum(v["in_v1"] for v in harness["system_prompt_variants"].values()),
        "system_prompts_also_in_v2": sum(v["in_v2"] for v in harness["system_prompt_variants"].values()),
        "distinct_system_prompt_bases": len({v["base_sha256"] for v in harness["system_prompt_variants"].values()}),
        "bases_not_in_v1": sorted({v["base_sha256"] for v in harness["system_prompt_variants"].values()} - base_v1),
        "distinct_tool_sets": len(harness["tool_sets"]),
        "tool_sets_not_in_v1": [k for k, v in harness["tool_sets"].items() if not v["in_v1"]],
        "distinct_request_params": len(harness["request_params"]),
        "request_params_not_in_v1": [k for k, v in harness["request_params"].items() if not v["in_v1"]],
    }
    harness["redactions"] = dict(harness_redactions)
    (out / "harness.json").write_text(json.dumps(harness, indent=1, ensure_ascii=False) + "\n")
    return {"metas": metas, "redactions": totals, "harness_redactions": harness_redactions, "harness": harness}


# ---------------------------------------------------------------- main
FLAG_KEYS = ("any", "strong", "webfetch_calls", "webfetch_completed", "net_fetch_cmds", "net_fetch_failed",
             "net_fetch_ok", "git_history_cmds", "hidden_artifact_refs", "hidden_test_seen_before_touch",
             "hidden_test_first_touch_reads_ok", "hidden_test_read_attempts", "hidden_test_bash_first_mentions",
             "subagent_calls", "subagent_beyond_export", "journal_net_fetch_calls", "journal_webfetch_calls",
             "journal_hidden_refs", "journal_git_history_calls")


def flag_counts(flag_rows: list) -> dict:
    """Conversations with each flag set (> 0 / true), and the total count of each counter."""
    out = {"conversations": len(flag_rows)}
    for key in FLAG_KEYS:
        values = [row.get(key) or 0 for row in flag_rows]
        out[key] = {"conversations": sum(1 for v in values if v), "total": int(sum(values))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="writes <out>/sources/")
    ap.add_argument("--batches", nargs="+", choices=sorted(BATCHES), default=sorted(BATCHES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-partial", action="store_true", help="dry run: skip campaigns without a results file")
    ap.add_argument("--snapshot-incomplete", action="store_true",
                    help="campaigns without a results file: take their terminal slots from the probe dirs now "
                         "(the driver's record logic, read-only); the rest is listed as pending at snapshot")
    ap.add_argument("--include-pilots", action="store_true")
    ap.add_argument("--campaigns", nargs="+", help="dry run: only these campaigns")
    ap.add_argument("--limit", type=int, default=0, help="dry run: first N slots per campaign")
    ap.add_argument("--select-only", action="store_true", help="selection and report only, no trajectories")
    args = ap.parse_args()
    if args.allow_partial and args.snapshot_incomplete:
        ap.error("--allow-partial and --snapshot-incomplete exclude each other")
    os.umask(0o027)
    started = time.time()
    v1h = json.loads(V1_HARNESS.read_text())
    v2h = json.loads(V2_BAIHE_HARNESS.read_text())
    v1_bases = {v["base_sha256"] for v in v1h["system_prompt_variants"].values()}
    tasks = task_facts()
    sources = args.out / "sources"
    summary, all_tests = {}, {}
    for batch in args.batches:
        conf = BATCHES[batch]
        snapshot_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        files, missing, jobs, snapshots = load_jobs(batch, args)
        if missing and not args.allow_partial:
            raise SystemExit(f"{batch}: no results file for {missing} (campaigns not finished); wait, or --allow-partial"
                             " / --snapshot-incomplete")
        unknown = sorted({j["task_id"] for j in jobs} - set(tasks))
        assert not unknown, f"{batch}: tasks without facts {unknown}"
        print(f"{batch}: {len(files)} results files, {len(snapshots)} snapshots, {len(jobs)} final slots, "
              f"missing {missing}", flush=True)
        snapshot = None
        pending = [p for s in snapshots.values() for p in s["pending"]]
        if snapshots:
            # the snapshot code must reproduce every results file of this batch exactly before it is trusted
            logic = snapshot_logic_check(batch, files, args.workers)
            print(f"{batch}: snapshot logic check on {len(logic['campaigns'])} finished campaigns: {logic['equal']}/"
                  f"{logic['slots']} records equal, {logic['n_mismatches']} mismatches", flush=True)
            if logic["n_mismatches"]:
                raise SystemExit(f"{batch}: the snapshot logic does not reproduce the results files: "
                                 f"{json.dumps(logic['mismatches'][:3], default=str)[:4000]}")
            snap_path = Path(os.path.abspath(sources / batch / "snapshot_records.json"))
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            body = {"batch": batch, "request": SNAPSHOT_REQUEST, "rule": SNAPSHOT_RULE,
                    "snapshot_started_at": snapshot_started,
                    "snapshot_at": {name: s["scanned_at"] for name, s in snapshots.items()},
                    "code": {"atp_campaign": str(ATP_CAMPAIGN), "atp_campaign_sha256": sha256_file(ATP_CAMPAIGN),
                             "synth_rollouts": str(SYNTH_ROLLOUTS), "synth_rollouts_sha256": sha256_file(SYNTH_ROLLOUTS),
                             "driver": str(conf["dir"] / "driver.py"),
                             "driver_sha256": sha256_file(conf["dir"] / "driver.py"),
                             "brokenreply_sha256": sha256_file(conf["dir"] / "brokenreply.py"),
                             "blocks_sha256": SNAPSHOT_BLOCKS_SHA256},
                    "logic_check": logic, "campaigns": snapshots}
            tmp = snap_path.with_name("." + snap_path.name + ".tmp")
            tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
            os.replace(tmp, snap_path)
            snapshot = {
                "request": SNAPSHOT_REQUEST, "rule": SNAPSHOT_RULE, "snapshot_started_at": snapshot_started,
                "snapshot_file": str(snap_path), "snapshot_file_sha256": sha256_file(snap_path),
                "campaigns": {name: {k: s[k] for k in ("scanned_at", "probe_dir", "slots", "slot_states", "terminal_slots",
                                                       "pending_slots", "plan_file", "plan_sha256", "plan_slots",
                                                       "plan_check", "broken_ended_graded", "stats_first_attempts",
                                                       "stats_terminal", "rerun_outcomes", "reruns")}
                              for name, s in snapshots.items()},
                "logic_check": {k: logic[k] for k in ("campaigns", "slots", "equal", "n_mismatches")},
                "pending_at_snapshot": len(pending),
                "pending_by_state": dict(Counter(p["state"] for p in pending)),
                "pending_by_campaign": dict(Counter(p["campaign"] for p in pending)),
                "pending": pending}
        record_sources = {**{name: {"source": "results_file", "path": info["path"], "sha256": info["sha256"],
                                    "finished_at": info["finished_at"]} for name, info in files.items()},
                          **{name: {"source": "snapshot", "scanned_at": s["scanned_at"], "terminal_slots":
                                    s["terminal_slots"], "pending_slots": s["pending_slots"]}
                             for name, s in snapshots.items()}}
        record_sources = {name: record_sources[name] for name in list(conf["campaigns"]) + list(conf["pilots"])
                          if name in record_sources}
        facts = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, fact in enumerate(pool.map(examine, jobs, chunksize=1), 1):
                facts[fact["key"]] = fact
                if n % 50 == 0:
                    print(f"  {batch}: examined {n}/{len(jobs)} ({time.time() - started:.0f} s)", flush=True)
        kept, exclusions = [], []
        independent = Counter()
        for job in jobs:
            fact = facts[job["key"]]
            reason = decide(fact, v1_bases)
            if fact["status"] == "pass":
                independent["final_pass"] += 1
                if not all(fact["files"].values()) or fact["evaluation_completed"] is not True or fact["reward"] != 1 \
                        or fact["validity_failure"] is not None:
                    independent["not_clean"] += 1
                compacted = bool((fact.get("compaction") or {}).get("compacted"))
                degenerate = bool((fact.get("journal") or {}).get("degenerate_n"))
                independent["compaction"] += compacted
                independent["user_messages_not_1"] += (fact.get("compaction") or {}).get("user_messages", 1) != 1
                independent["degenerate_reply"] += degenerate
                independent["compaction_and_degenerate"] += compacted and degenerate
                independent["rerun_passes"] += "rerun_reason" in job["record"]
            if reason is None:
                kept.append(job)
            else:
                detail = None
                if reason == "degenerate_reply":
                    detail = {"kinds": fact["journal"]["kinds"], "first": fact["journal"]["degenerate"][:3],
                              "ended": fact["journal"]["ended"]}
                elif reason == "compaction":
                    detail = fact["compaction"]
                elif reason.startswith("final_status"):
                    detail = {k: job["record"].get(k) for k in ("f2p", "p2p", "agent_error", "verifier_error",
                                                                 "ended_by_broken_reply", "last_reply_kind")}
                exclusions.append({"trajectory_id": job["key"], "task_id": job["task_id"], "slot": job["slot"],
                                   "campaign": job["campaign"], "reason": reason,
                                   "attempt_dir": norm(job["record"].get("attempt_dir")), "detail": detail})
        kept.sort(key=lambda j: j["key"])
        final_status = Counter(f["status"] for f in facts.values())
        main_considered = sum(1 for j in jobs if j["campaign"] in conf["campaigns"])
        if not (args.limit or args.campaigns or missing):  # every slot of the batch: considered or pending
            assert main_considered + len(pending) == conf["slots"], (batch, main_considered, len(pending), conf["slots"])
        report = {
            "batch": batch, "rollouts_dir": str(conf["dir"]), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rules": "final pass (approved broken-reply re-run in place) -> clean completion -> no compaction -> "
                     "exactly one user message -> no degenerate reply anywhere (degenerate.py classify) -> v1 harness base",
            "build_batch_helpers_sha256": BUILD_BATCH_HELPERS_SHA256, "degenerate_block_sha256": DEGENERATE_BLOCK_SHA256,
            "partial": bool(missing), "campaigns_missing": missing, "pilots_included": args.include_pilots,
            "limit_per_campaign": args.limit or None, "results_files": files,
            "snapshot_mode": bool(args.snapshot_incomplete), "record_sources": record_sources,
            "snapshot_campaigns": sorted(snapshots), "pending_at_snapshot": len(pending), "snapshot": snapshot,
            "slots_expected": conf["slots"], "considered": len(jobs), "considered_main": main_considered,
            "final_status": dict(final_status),
            "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
            "independent_counts_among_final_passes": dict(independent),
            "kept": len(kept), "kept_reruns": sum("rerun_reason" in j["record"] for j in kept),
            "kept_by_record_source": dict(Counter(j["record_source"] for j in kept)),
            "kept_by_campaign": dict(Counter(j["campaign"] for j in kept)),
            "kept_tasks": len({j["task_id"] for j in kept}),
            "tasks_considered": len({j["task_id"] for j in jobs}),
            "tasks_without_kept": sorted({j["task_id"] for j in jobs} - {j["task_id"] for j in kept}),
            "flag_counts_kept": flag_counts([facts[j["key"]]["flags"] for j in kept]),
            "exclusions": exclusions,
        }
        if batch == "v5" and V5_USABLE.exists() and not (args.limit or args.campaigns):
            usable = {(r["task_id"], r["slot"]) for r in map(json.loads, V5_USABLE.read_text().splitlines())}
            mine = {(j["task_id"], j["slot"]) for j in kept if j["campaign"] in conf["campaigns"]}  # pilots: not listed
            report["v5_sft_usable_crosscheck"] = {"sft_usable_passes": len(usable), "kept": len(mine),
                                                 "only_in_sft_usable": sorted(map(list, usable - mine)),
                                                 "only_kept_here": sorted(map(list, mine - usable))}
        out = sources / batch
        if not args.select_only and kept:
            built = write_batch(batch, kept, facts, tasks, out, v1h, v2h)
            report.update({
                "redactions_trajectories": {n: built["redactions"].get(n, 0) for n in PATTERN_NAMES},
                "redactions_harness": {n: built["harness_redactions"].get(n, 0) for n in PATTERN_NAMES},
                "harness_comparison_with_v1": built["harness"]["comparison_with_v1"],
                "harness_missing": built["harness"]["missing"],
                "first_user_mismatch": built["harness"]["first_user_mismatch"],
                "patch_mismatch_record": [m["trajectory_id"] for m in built["metas"] if not m["patch_matches_record"]],
                "metas": built["metas"]})
        else:
            out.mkdir(parents=True, exist_ok=True)
        (out / "selection_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        summary[batch] = {k: report[k] for k in ("partial", "campaigns_missing", "snapshot_campaigns",
                                                 "pending_at_snapshot", "considered", "considered_main", "final_status",
                                                 "exclusion_counts", "independent_counts_among_final_passes", "kept",
                                                 "kept_reruns", "kept_tasks", "tasks_considered", "tasks_without_kept")}
        if snapshot:
            summary[batch]["snapshot"] = {k: snapshot[k] for k in ("snapshot_started_at", "logic_check",
                                                                   "pending_by_state", "pending_by_campaign")}
            summary[batch]["snapshot"]["campaigns"] = {
                name: {k: c[k] for k in ("scanned_at", "slot_states", "terminal_slots", "pending_slots", "plan_slots",
                                         "plan_check", "rerun_outcomes", "stats_terminal")}
                for name, c in snapshot["campaigns"].items()}
        summary[batch]["kept_by_record_source"] = dict(Counter(j["record_source"] for j in kept))
        summary[batch]["flags_any"] = report["flag_counts_kept"]["any"]
        summary[batch]["flags_strong"] = report["flag_counts_kept"]["strong"]
        for key in ("harness_comparison_with_v1", "harness_missing", "first_user_mismatch", "patch_mismatch_record",
                    "v5_sft_usable_crosscheck"):
            if key in report:
                summary[batch][key] = report[key]
    # every task of the processed campaigns (all main-campaign task dirs, not only those with passes): the hidden-test
    # facts that build_sft_v3.py uses for the v2 baseline flags (v2's tasks are v4's)
    for batch in args.batches:
        for name in BATCHES[batch]["campaigns"]:
            campaign = BATCHES[batch]["dir"] / "probes" / "kimi" / name
            if args.campaigns and name not in args.campaigns:
                continue
            for tdir in sorted((campaign / "tasks").glob("*")) if (campaign / "tasks").is_dir() else []:
                tt = task_tests(campaign, tdir.name)
                all_tests.setdefault(tdir.name, {"batch": batch, "campaign": name, **tt})
    sources.mkdir(parents=True, exist_ok=True)
    (sources / "task_tests.json").write_text(json.dumps(dict(sorted(all_tests.items())), indent=1) + "\n")
    summary["seconds"] = round(time.time() - started, 1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
