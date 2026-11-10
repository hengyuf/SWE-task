#!/usr/bin/env python3
"""Repair and re-score tasks whose RUN TESTS block exits early (S/README.md, "Fifth issue (2026-10-18)").

A RUN TESTS block that gates the suites on a build, typecheck or install step with `exit`, `|| exit` or `set -e`
writes no report when that step fails, so the verifier leaves no /logs/verifier/reward.json and the rollout counts as
an ungraded infra slot instead of a FAIL. The repair keeps the failing step's output in the log, logs one extra line
instead of exiting, and lets the suites run, so the grader always scores the run.

  scan     --run R [--json]                  built tasks (cand-*/stage.json ok) whose packaged tests/test.sh RUN TESTS
                                             block can exit early, with the offending lines (read-only)
  repair   --run R --cand C [C ...]          rewrite those lines (nothing else changes) in every live copy of the task's
                                             test script: cand/final/tests/test.sh, cand/task/<id>/tests/test.sh,
                                             accepted/<id>/tests/test.sh and the author's run_tests.sh
  validate --run R --cand C [C ...]          real grader (sandbox.grade, task snapshot, repaired final tests): reference
                                             solution -> 1, empty patch -> 0, reference + syntax error -> 0 (graded)
  regrade  --run R --cand C --slots 6-0,...  grade the listed slots' final patches with the repaired task, save
                                             regrade.json, then re-score with task_builder.score() (and accept());
                                             the teacher audit keeps the verdicts of unchanged transcripts (--rejudge:
                                             judge them all again)

Gate mode (--gate; hengyuf 2026-10-18: a solution that does not build or typecheck must count as FAILED, the semantics
of the DeepSWE frame). Removing an exit let a failed rebuild/typecheck go unnoticed; gate mode records it as a test:
  repair   --gate --cand C [--add-typecheck "yarn build:ts"]
                       on a repaired block, the build/typecheck steps the removed exits guarded keep their exit status
                       and a recorder writes the CTRF test "[gate] <cmd> && <cmd>" (passed iff every status is 0) into
                       the base report, whose id joins config.json's p2p_node_ids in every live copy -- the convention of
                       the sibling blocks c01-r1b0, c01-r1b3, c04-r1b3, c05-r1b3 and of the vitest-duration-sharding
                       frame. --add-typecheck adds that step when no guarded step runs the type checker (jest's
                       `yarn build:js` only strips types). Never an early exit: a broken build fails a P2P test.
  validate --gate      also grades a type-only broken solution (valid JavaScript, a TypeScript type error); all four
                       checks look at the gate test: reference 1 (gate passed), empty 0 (gate passed: a P2P test),
                       syntax error and type error graded 0 with the gate FAILED
  regrade  --gate      as regrade, with the gated task; the outputs of an earlier re-score are kept as *.pre-gate-fix.
                       --slots passing --confirm grades the passing slots and re-scores only if a grade changed
  restore  --gate --cand C                   put the pre-gate copies back (a task that cannot pass validation)
  tidy-accepted --run R                      move *.pre-early-exit-fix / *.pre-gate-fix files out of accepted/<id>/
                                             (copies under <R>/early-exit-fix-backups/accepted/) and check accepted/

R: a build root (runs/<run>/build/<target>/<owner__repo>) or a run name with one build. repair, validate, regrade,
restore and tidy-accepted take --dry-run: no file writes, no sandboxes, no LLM calls (regrade still reads the probe
through rollouts.probe_records). Every write is logged to <R>/early-exit-fix.jsonl (gate mode: "mode": "gate"); each
changed file keeps its first version as <file>.pre-early-exit-fix (gate mode: <file>.pre-gate-fix); re-scoring moves
metrics.json, rollouts/, teacher_audit.json and failure_analysis/ aside as *.pre-regrade (gate mode: *.pre-gate-fix).
Run from S/synth with the pipeline venv, e.g.
  ../atp/runtime/controller/bin/python3 -B tools/regrade.py scan --run official1-vitest
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import difflib  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import uuid  # noqa: E402

SYNTH = Path(__file__).resolve().parents[1]
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from common import DEEPSWE, JUDGE, RUNS, S, STUDENT, TEACHER, read_json, write_json  # noqa: E402
import rollouts  # noqa: E402
import sandbox  # noqa: E402
import task_builder  # noqa: E402
from task_builder import RUN_END, RUN_START  # noqa: E402

FIX = "early-exit-fix"                   # per-task outputs: cand/early-exit-fix/{repair.json, validate.json, validate/}
BACKUP = ".pre-early-exit-fix"           # <file>.pre-early-exit-fix: a repaired file before its first repair
ASIDE = ".pre-regrade"                   # the score() outputs a re-score replaces
LOG = "early-exit-fix.jsonl"             # <build root>/early-exit-fix.jsonl: one JSON line per change
MARK = "[verifier] early-exit-fix:"      # the log line written instead of a removed exit
HUB_BUDGET = S / "tools/hub_budget.py"
# score()'s outputs, moved aside before a re-score; metrics.json last, so its .pre-regrade means "all moved"
SCORE_OUTPUTS = ("rollouts", "teacher_audit.json", "failure_analysis", "metrics.json")

# Gate mode (see the docstring). Per-task outputs: cand/early-exit-fix/gate/{repair.json, validate.json, validate/}.
GATE_DIR = "gate"
GATE_BACKUP = ".pre-gate-fix"            # <file>.pre-gate-fix: a file before its gate rewrite; also the suffix of the
                                         # score outputs (and regrade.json, regrade/) that a gate re-score replaces
GATE_TAG = "# gate-fix:"                 # comment on every line gate mode adds to a block
GATE_MARK = "[verifier] gate-fix:"       # the recorder's log line
GATE_PREFIX = "[gate] "                  # test-id convention of the sibling blocks and the DeepSWE frame
GATE_KINDS = frozenset({"build", "typecheck"})
GATE_RC = "gate_rc_"                     # variables gate mode adds for exit statuses nobody captured
TYPECHECKS = re.compile(r"\btsc\b|\bbuild:ts\b|\bbuildTs\.m?js\b|\btypecheck\b|\btype-check\b|\bvue-tsc\b|\bmypy\b|"
                        r"\bpyright\b")  # a step that runs a type checker
RC_CAPTURE = re.compile(r'^([A-Za-z_]\w*)=("?)(\$\?|\$\{PIPESTATUS\[0\]\})\2$')  # build_rc=${PIPESTATUS[0]}, rc=$?
BACKUPS_DIR = "early-exit-fix-backups"   # <build root>/early-exit-fix-backups/{accepted,unaccepted}/


class FixError(RuntimeError):
    pass


# ============================================================================================ bash: masking
# The analysis works on a same-length copy of the script in which quoted text, ${...}/$(...)/`...`, comments and
# heredoc bodies are blanked (newlines kept): every keyword left in it is real shell syntax, and offsets map 1:1.

HEREDOC_OP = re.compile(r"<<(-?)[ \t]*('[^'\n]*'|\"[^\"\n]*\"|\\?[^\s;&|<>()]+)")


def _skip_squote(text, i):
    j = text.find("'", i + 1)
    return len(text) if j < 0 else j + 1


def _skip_backtick(text, i):
    j = i + 1
    while j < len(text):
        if text[j] == "\\":
            j += 2
        elif text[j] == "`":
            return j + 1
        else:
            j += 1
    return len(text)


def _skip_dquote(text, i):
    j = i + 1
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
        elif c == '"':
            return j + 1
        elif text.startswith("$(", j):
            j = _skip_group(text, j + 1, "(", ")")
        elif text.startswith("${", j):
            j = _skip_group(text, j + 1, "{", "}")
        elif c == "`":
            j = _skip_backtick(text, j)
        else:
            j += 1
    return len(text)


def _skip_group(text, i, opener, closer):
    """Index just after the bracket that closes text[i] (quotes and nested substitutions respected)."""
    j, depth = i, 0
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "'" and opener == "(":
            j = _skip_squote(text, j)
            continue
        if c == '"':
            j = _skip_dquote(text, j)
            continue
        if c == "`":
            j = _skip_backtick(text, j)
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)


def mask_shell(text: str):
    """(masked, heredocs). heredocs: [{"op": offset of <<, "delim", "strip", "body"}] in source order."""
    n, out, heredocs, pending = len(text), list(text), [], []

    def fill(a, b, char="_"):
        for k in range(max(a, 0), min(b, n)):
            if out[k] != "\n":
                out[k] = char

    i = 0
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
        elif c == "'":
            j = _skip_squote(text, i)
            fill(i + 1, j - 1)
            i = j
        elif c == '"':
            j = _skip_dquote(text, i)
            fill(i + 1, j - 1)
            i = j
        elif c == "$" and text.startswith(("$(", "${", "$'", '$"'), i):
            nxt = text[i + 1]
            if nxt == "(":
                j = _skip_group(text, i + 1, "(", ")")
            elif nxt == "{":
                j = _skip_group(text, i + 1, "{", "}")
            elif nxt == "'":
                j = i + 2
                while j < n and text[j] != "'":
                    j += 2 if text[j] == "\\" else 1
                j = min(j + 1, n)
            else:
                j = _skip_dquote(text, i + 1)
            fill(i + 1, j)
            i = j
        elif c == "`":
            j = _skip_backtick(text, i)
            fill(i, j)
            i = j
        elif c == "#" and (i == 0 or text[i - 1] in " \t\n;&|()<>"):
            j = text.find("\n", i)
            j = n if j < 0 else j
            fill(i, j, " ")
            i = j
        elif text.startswith("<<<", i):
            i += 3
        elif text.startswith("<<", i) and (match := HEREDOC_OP.match(text, i)):
            word = match.group(2)
            pending.append({"op": i, "strip": match.group(1) == "-", "delim": re.sub(r"[\"'\\]", "", word)})
            i = match.end()
        elif c == "\n":
            i += 1
            for doc in pending:  # bodies follow the line that holds their operators, in order
                start, body_end = i, None
                while i < n:
                    j = text.find("\n", i)
                    end = n if j < 0 else j + 1
                    line = text[i:end].rstrip("\n")
                    if (line.lstrip("\t") if doc["strip"] else line) == doc["delim"]:
                        body_end, i = i, end
                        break
                    i = end
                doc["body"] = text[start:n if body_end is None else body_end]
                fill(start, i, " ")
                heredocs.append(doc)
            pending = []
        else:
            i += 1
    for doc in pending:
        doc["body"] = ""
        heredocs.append(doc)
    return "".join(out), heredocs


# ============================================================================================ bash: commands

TOKEN = re.compile(r"(?P<nl>\n)|(?P<ws>[ \t\r]+|\\\n)|(?P<op>&&|\|\||;;&?|;&|\|&|[;|&()])"
                   r"|(?P<redir>\d*(?:&>>|&>|>>|>&|<&|<<<|<<-?|<>|>\||[<>]))"
                   r"|(?P<word>(?:\\.|[^\s;&|()<>\\])+)", re.S)
ASSIGN = re.compile(r"^[A-Za-z_]\w*(\[[^\]]*\])?\+?=")
NAME = re.compile(r"^[A-Za-z_][\w.:-]*$")


@dataclasses.dataclass
class Command:
    """One simple command of the script (offsets into the text)."""
    start: int
    end: int
    words: list                                          # [(start, end)]; redirection targets are not words
    prev: str = "\n"                                     # separator before it
    sep: str = ""                                        # separator after it (&& || | |& ; ;; & ) \n or "")
    cond: bool = False                                   # inside an if/elif/while/until condition
    negated: bool = False
    function: str = ""                                   # name of the function whose body holds it
    subshell: int = 0
    if_cond: tuple = None                                # condition span of the innermost enclosing if branch
    group_gate: int = None                               # index of X in `X || { ...; }` for commands in the group
    heredocs: list = dataclasses.field(default_factory=list)


def parse_shell(text: str):
    """(commands, functions, masked): the simple commands of a bash script in source order; functions maps each
    function defined in it to the indexes of its body's commands. A pragmatic parser for verifier blocks: if/elif/
    else, while/until/for/select, case, { } groups, ( ) subshells, NAME() { } functions, arrays and heredocs."""
    masked, heredocs = mask_shell(text)
    by_op = {doc["op"]: doc for doc in heredocs}
    tokens = list(TOKEN.finditer(masked))
    commands, functions, frames = [], {}, []
    state = {"cur": None, "last": "\n", "negate": False, "redirect": False, "func": None, "func_kw": False}

    def innermost(*kinds):
        return next((frame for frame in reversed(frames) if frame["kind"] in kinds), None)

    def begin(pos):
        if state["cur"] is None:
            loop = innermost("if", "while")
            branch = next((f for f in reversed(frames) if f["kind"] == "if" and not f["cond"]), None)
            group = innermost("group")
            state["cur"] = Command(
                start=pos, end=pos, words=[], prev=state["last"], negated=state["negate"],
                cond=bool(loop and loop["cond"]), function=next((f["name"] for f in reversed(frames)
                                                                 if f["kind"] == "func"), ""),
                subshell=sum(f["kind"] == "subshell" for f in frames),
                if_cond=branch.get("span") if branch else None, group_gate=group.get("gate") if group else None)
            state["negate"] = False
        return state["cur"]

    def finish(sep):
        cur = state["cur"]
        if cur is not None:
            cur.sep = sep
            if cur.function:
                functions.setdefault(cur.function, []).append(len(commands))
            commands.append(cur)
        state["cur"] = None
        state["last"] = sep

    def pop(*kinds):
        for k in range(len(frames) - 1, -1, -1):
            if frames[k]["kind"] in kinds:
                del frames[k]
                return

    k = 0
    while k < len(tokens):
        token = tokens[k]
        k += 1
        kind, start, end = token.lastgroup, token.start(), token.end()
        value = masked[start:end]
        if kind == "ws":
            continue
        if kind == "word":
            if ASSIGN.match(value) and value.endswith("=") and masked.startswith("(", end):  # NAME=( ... )
                end = _skip_group(masked, end, "(", ")")
                while k < len(tokens) and tokens[k].start() < end:
                    k += 1
            if state["redirect"]:
                state["redirect"] = False
                state["cur"].end = end
                continue
            frame = frames[-1] if frames else None
            if state["cur"] is None or not state["cur"].words:
                if frame and frame["kind"] == "for" and frame["header"]:
                    if value == "do":
                        frame["header"] = False
                    continue
                if frame and frame["kind"] == "case":
                    if frame["header"]:
                        if value == "in":
                            frame.update(header=False, pattern=True)
                        continue
                    if frame["pattern"]:
                        if value == "esac":
                            pop("case")
                        continue
                if state["func_kw"]:
                    state.update(func=value, func_kw=False)
                    continue
                if value in ("if", "while", "until"):
                    frames.append({"kind": "if" if value == "if" else "while", "cond": True, "cstart": end})
                    continue
                if value == "elif":
                    frame = innermost("if")
                    if frame:
                        frame.update(cond=True, cstart=end)
                    continue
                if value in ("then", "do"):
                    frame = innermost("if", "while", "for")
                    if frame and frame.get("cond"):
                        frame.update(cond=False, span=(frame["cstart"], start))
                    continue
                if value == "else":
                    continue
                if value in ("fi", "done"):
                    pop("if", "while", "for")
                    continue
                if value in ("for", "select"):
                    frames.append({"kind": "for", "header": True, "cond": False})
                    continue
                if value == "case":
                    frames.append({"kind": "case", "header": True, "pattern": False})
                    continue
                if value == "!":
                    state["negate"] = True
                    continue
                if value == "{":
                    gate = len(commands) - 1 if state["last"] in ("&&", "||") and commands else None
                    frames.append({"kind": "func", "name": state["func"]} if state["func"]
                                  else {"kind": "group", "gate": gate})
                    state.update(func=None, last="{")
                    continue
                if value == "}":
                    finish("}")
                    pop("func", "group")
                    continue
                if value == "function":
                    state["func_kw"] = True
                    continue
                if value == "time":
                    continue
            cur = begin(start)
            cur.words.append((start, end))
            cur.end = end
            continue
        if kind == "redir":
            cur = begin(start)
            cur.end = end
            digits = len(value) - len(value.lstrip("0123456789"))
            if value[digits:] in ("<<", "<<-") and (start + digits) in by_op:
                cur.heredocs.append(by_op[start + digits])
            state["redirect"] = True
            continue
        if kind == "nl":
            if state["cur"] is None and state["last"] in ("&&", "||", "|", "|&"):
                continue  # a list or pipeline continues on the next line
            finish("\n")
            continue
        if value == "(":
            cur = state["cur"]
            nxt = next((t for t in tokens[k:] if t.lastgroup != "ws"), None)
            if cur is not None and len(cur.words) == 1 and nxt is not None and masked[nxt.start():nxt.end()] == ")" \
                    and NAME.match(masked[cur.words[0][0]:cur.words[0][1]]):
                state["func"] = text[cur.words[0][0]:cur.words[0][1]]  # NAME() { ...; }
                state["cur"] = None
                k = tokens.index(nxt) + 1
                continue
            if cur is None:
                frames.append({"kind": "subshell"})
                state["last"] = "("
                continue
            finish("(")
            continue
        if value == ")":
            frame = frames[-1] if frames else None
            if frame and frame["kind"] == "case" and frame.get("pattern"):
                state["cur"] = None
                frame["pattern"] = False
                state["last"] = ")"
                continue
            finish(")")
            if frame and frame["kind"] == "subshell":
                frames.pop()
            continue
        if value in (";;", ";&", ";;&"):
            finish(value)
            frame = innermost("case")
            if frame:
                frame["pattern"] = True
            continue
        finish(value)
    finish("")
    return commands, functions, masked


def _words(text, cmd):
    return [text[s:e] for s, e in cmd.words]


def _line_of(text, offset):
    return text.count("\n", 0, offset) + 1


def _line_text(text, offset):
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start:len(text) if end < 0 else end]


# ============================================================================================ bash: classification

SAFE = frozenset("echo printf export unset local declare typeset readonly set shopt trap true : cd pushd popd mkdir rm "
                 "touch ln chmod chown tee sleep umask ulimit alias hash type shift return continue break wait pwd "
                 "date basename dirname realpath readlink hostname nproc id whoami uname".split())
WRAPPERS = frozenset("time nice nohup command stdbuf env timeout xargs sudo".split())
INTERPRETERS = frozenset("node nodejs python python3 ruby perl bash sh deno bun".split())
KINDS = (  # first match names the step
    ("install", re.compile(r"^(?:yarn|npm|pnpm|bun)(?:\s+(?:install|ci|i|add)\b.*|\s+--\S+)*\s*$|"
                           r"^(?:pip3?|uv|poetry|pipenv|bundle|composer|gem)\b.*\b(?:install|sync)\b|"
                           r"^go\s+mod\s+(?:download|tidy)\b|^cargo\s+fetch\b|^apt(?:-get)?\s+install\b", re.I)),
    ("typecheck", re.compile(r"\btsc\b|\btypecheck\b|type-check|check-types|types:check|\bmypy\b|\bpyright\b|"
                             r"vue-tsc|\bflow\s+check\b|\bpytype\b", re.I)),
    ("report", re.compile(r"junit-to-ctrf|\bctrf\b", re.I)),
    ("suite", re.compile(r"\bjest\b|\bvitest\b|\bmocha\b|\bava\b|\bpytest\b|py\.test|-m\s+unittest|\bnosetests\b|"
                         r"\bgo\s+test\b|\bcargo\s+(?:test|nextest)\b|\brspec\b|\bphpunit\b|test-types|\btstyche\b|"
                         r"node\s+--test|\b(?:yarn|npm|pnpm)\s+(?:run\s+)?test\b", re.I)),
    ("build", re.compile(r"\bbuild\b|\bbundle\b|\bcompile\b|^make\b|\bcmake\b|\bninja\b|\bwebpack\b|\brollup\b|"
                         r"\besbuild\b|\bbabel\b|\bswc\b|setup\.py|^mvn\b|^gradle|gradlew|\bprepack\b", re.I)),
)
# A step whose failure must still let the suites run and the grader score: these are early exits under errexit.
DEFINITE = frozenset({"install", "typecheck", "build", "suite", "report", "check", "gate"})
RISKY = frozenset({"script", "other"})  # reported as risks (e.g. a tolerant report post-processor under set -e)
FAILS = re.compile(r"\bthrow\b|process\.exit\s*\(\s*(?!0?\s*\))|process\.exitCode\s*=\s*[1-9]|"
                   r"sys\.exit\s*\(\s*(?!0?\s*\))|\braise\b|^\s*assert\b|\bexit\s+[1-9]|\bdie\b", re.M)


def classify(words, *, heredocs=(), functions=None, text="", commands=(), depth=0) -> str:
    """'safe' or the kind of step a simple command runs (install, typecheck, build, suite, report, check, gate,
    script, other). Wrappers (env, timeout, run_log-style functions) are looked through."""
    functions = functions or {}
    words = list(words)
    if not words:
        return "safe"
    assigns = [w for w in words if ASSIGN.match(w)]
    while words and ASSIGN.match(words[0]):
        words.pop(0)
    if not words:  # assignments only: their status is that of their last command substitution
        inner = [m.group(1) or m.group(2) for w in assigns for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", w)]
        kinds = [classify(c.split(), functions=functions, depth=depth + 1) for c in inner]
        return next((kind for kind in kinds if kind in DEFINITE | RISKY), "safe")
    while words and (words[0] in WRAPPERS or (words[0] in functions and depth < 3)):
        head = words.pop(0)
        if head in functions:
            if words:
                kind = classify(words, heredocs=heredocs, functions=functions, depth=depth + 1)
                if kind != "other":
                    return kind
            body = [classify(_words(text, commands[i]), functions=functions, text=text, commands=commands,
                             heredocs=commands[i].heredocs, depth=depth + 1) for i in functions[head]]
            for kind in ("install", "typecheck", "build", "suite", "report", "check", "gate", "script", "other"):
                if kind in body:
                    return kind
            return "safe"
        while words and (words[0].startswith("-") or ASSIGN.match(words[0])
                         or (head == "timeout" and re.fullmatch(r"\d+[smhd]?", words[0]))):
            words.pop(0)
    if not words:
        return "safe"
    name = words[0].strip("'\"")
    base = name.rsplit("/", 1)[-1]
    joined = " ".join(words)
    if base in INTERPRETERS and (heredocs or "-" in words[1:2] or "-e" in words[1:2] or "-c" in words[1:2]):
        bodies = "\n".join(doc.get("body", "") for doc in heredocs)
        return "check" if FAILS.search(bodies) else "script"
    if base in ("[", "[[", "test"):
        return "gate"
    if base == "cat":
        return "safe" if all(w.startswith("-") for w in words[1:]) else "other"
    if base in ("cp", "mv"):
        return "report" if "/logs/verifier" in joined else "safe"
    if base == "exit":
        return "safe"  # reported on its own
    if base in SAFE:
        return "safe"
    for kind, pattern in KINDS:
        if pattern.search(joined):
            return kind
    return "other"


# ============================================================================================ analysis

def _errexit_change(args):
    """True (`set` turns errexit on), False (off), None (unchanged)."""
    state, i = None, 0
    while i < len(args):
        arg = args[i]
        if arg in ("-", "--") or arg[:1] not in "-+" or len(arg) < 2:
            break
        on, flags = arg[0] == "-", arg[1:]
        if "e" in flags:
            state = on
        if "o" in flags:
            if i + 1 < len(args) and args[i + 1] == "errexit":
                state = on
            i += 1
        i += 1
    return state


def neutralize_set(args):
    """`set` arguments without any errexit-on flag (-e, -o errexit); `set +e` if nothing else is left."""
    out, i = [], 0
    while i < len(args):
        arg = args[i]
        if arg in ("-", "--") or arg[:1] not in "-+" or len(arg) < 2:
            out += args[i:]
            break
        sign, flags = arg[0], arg[1:]
        if sign == "-":
            flags = flags.replace("e", "")
        if "o" in arg[1:] and i + 1 < len(args):
            name = args[i + 1]
            if sign == "-" and name == "errexit":
                flags = flags.replace("o", "", 1)
                if flags:
                    out.append(sign + flags)
            else:
                out += [sign + flags, name] if flags else [name]
            i += 2
            continue
        if flags:
            out.append(sign + flags)
        i += 1
    return out or ["+e"]


def _pipelines(commands):
    """Lists of command indexes joined by | or |&."""
    groups = []
    for index, cmd in enumerate(commands):
        if groups and cmd.prev in ("|", "|&") and commands[groups[-1][-1]].sep in ("|", "|&"):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _rc_steps(text, commands, upto, names):
    """For each `$name` in a gate condition: the step whose status `name=$?` / `name=${PIPESTATUS[..]}` captured."""
    steps = []
    for name in names:
        for index in range(upto - 1, -1, -1):
            words = _words(text, commands[index])
            if words and re.match(rf"^{re.escape(name)}=\"?(\$\?|\$\{{PIPESTATUS)", words[0]):
                step = index - 1  # the last command of the step's pipeline: walk back to its first command
                while step - 1 >= 0 and commands[step - 1].sep in ("|", "|&"):
                    step -= 1
                if step >= 0:
                    steps.append(step)
                break
    return steps


def _step(text, commands, functions, index):
    cmd, end = commands[index], index
    while end + 1 < len(commands) and commands[end].sep in ("|", "|&"):
        end += 1
    whole = " ".join(re.sub(r"\\\n", " ", text[cmd.start:commands[end].end]).split())
    return {"line": _line_of(text, cmd.start), "text": whole[:200],
            "kind": classify(_words(text, cmd), heredocs=cmd.heredocs, functions=functions, text=text,
                             commands=commands)}


def analyze(text: str) -> list:
    """Early exits in a RUN TESTS block. Each finding: kind exit|errexit|exec|trap, severity early-exit|risk,
    line (1-based in `text`), line_text, detail, and what it gates; exit/exec/errexit findings carry the span to edit."""
    commands, functions, masked = parse_shell(text)
    findings, errexit = [], None
    pipeline_of = {}
    for group in _pipelines(commands):
        for index in group:
            pipeline_of[index] = group
    for index, cmd in enumerate(commands):
        words = _words(text, cmd)
        if not words:
            continue
        line = _line_of(text, cmd.start)
        base = {"line": line, "line_text": _line_text(text, cmd.start)}
        group = pipeline_of[index]
        last = commands[group[-1]]
        exempt = cmd.cond or last.sep in ("&&", "||", "&") or commands[group[0]].negated
        if words[0] == "exit" and not cmd.subshell:
            if re.match(r"^\s*cd\s+([\"']?)/app/?\1\s*(\|\||&&)", base["line_text"]):
                continue  # `cd /app || exit 6`: the repository root is missing, an infrastructure failure
            gates = []
            if cmd.prev in ("||", "&&") and index > 0:
                head = pipeline_of[index - 1][0]
                tested = _words(text, commands[head])
                named = re.findall(r"\$\{?([A-Za-z_]\w*)", " ".join(tested)) \
                    if tested[:1] in (["["], ["[["], ["test"]) else []
                gates = (_rc_steps(text, commands, head, named) if named else []) or [head]
            elif cmd.group_gate is not None:
                gates = [pipeline_of[cmd.group_gate][0]]
            elif cmd.if_cond:
                condition = masked[cmd.if_cond[0]:cmd.if_cond[1]]
                inside = [i for i, c in enumerate(commands) if c.cond and cmd.if_cond[0] <= c.start < cmd.if_cond[1]]
                named = re.findall(r"\$\{?([A-Za-z_]\w*)", text[cmd.if_cond[0]:cmd.if_cond[1]])
                gates = _rc_steps(text, commands, inside[0] if inside else index, named) if named else []
                if not gates and inside and not condition.strip().startswith(("[", "test")):
                    gates = inside
            if cmd.function:  # die() { ...; exit 1; }: the steps its call sites gate (`step || die ...`)
                gates = []
                for i, call in enumerate(commands):
                    if not call.function and _words(text, call)[:1] == [cmd.function]:
                        gates.append(pipeline_of[i - 1][0] if call.prev in ("||", "&&") and i > 0 else i)
            findings.append(dict(base, kind="exit", severity="early-exit", span=(cmd.start, cmd.end),
                                 arg=words[1] if len(words) > 1 else "", function=cmd.function,
                                 condition=text[cmd.if_cond[0]:cmd.if_cond[1]].strip().rstrip(";").strip()
                                 if cmd.if_cond else "",
                                 gates=[_step(text, commands, functions, i) for i in gates],
                                 detail="exit " + " ".join(words[1:]) if len(words) > 1 else "exit"))
            continue
        if cmd.function:
            continue  # runs when called; its call sites are classified through `functions`
        if words[0] == "set" and not cmd.cond and not cmd.subshell:
            change = _errexit_change(words[1:])
            if change is True:
                errexit = cmd
            elif change is False:
                errexit = None
            continue
        if words[0] == "exec" and len(words) > 1:
            findings.append(dict(base, kind="exec", severity="early-exit", span=(cmd.words[0][0], cmd.words[1][0]),
                                 detail="exec replaces the verifier shell: grading never runs"))
            continue
        if words[0] == "trap" and "ERR" in words and re.search(r"\bexit\b", text[cmd.start:cmd.end]):
            findings.append(dict(base, kind="trap", severity="early-exit", detail="an ERR trap that exits"))
            continue
        if errexit is None or exempt:
            continue
        kind = classify(words, heredocs=cmd.heredocs, functions=functions, text=text, commands=commands)
        if kind == "safe":
            continue
        set_words = _words(text, errexit)
        findings.append(dict(base, kind="errexit", severity="early-exit" if kind in DEFINITE else "risk", step=kind,
                             span=(errexit.start, errexit.end), set_line=_line_of(text, errexit.start),
                             set_text=" ".join(set_words), set_args=set_words[1:],
                             detail=f"`{' '.join(set_words)}` is in force: a failing {kind} step here ends the "
                                    "verifier"))
    return sorted(findings, key=lambda f: (f["line"], f["kind"]))


# ============================================================================================ repair

def _shown_status(arg: str) -> str:
    """The exit status as text for a double-quoted log line (never fails under set -u)."""
    if not arg:
        return ""
    if match := re.fullmatch(r'"?\$\{?([A-Za-z_]\w*)\}?"?', arg):
        return "${%s-}" % match.group(1)
    if re.fullmatch(r'"?\$\?"?', arg):
        return "$?"
    if re.fullmatch(r"\d+", arg):
        return arg
    return re.sub(r'([\\"$`])', r"\\\1", arg)


def exit_replacement(finding: dict) -> str:
    """The command that replaces a removed `exit`: one log line (stdout and $RUN_LOG), status 0."""
    kinds = [g["kind"] for g in finding.get("gates") or [] if g["kind"] not in ("safe", "other", "gate")]
    status = _shown_status(finding.get("arg", ""))
    was = f"exit {status}" if status else "exit"
    if finding.get("gates") or finding.get("condition"):
        message = (f"{'/'.join(dict.fromkeys(kinds)) or 'gating'} step failed ({was} removed); the suites still run "
                   "and the grader scores them")
    else:
        message = f"{was} removed; the grader scores this run"
    return f'echo "{MARK} {message}" | tee -a "${{RUN_LOG:-/logs/verifier/run.log}}"'


def _set_replacement(text, finding):
    start, end = finding["span"]
    if "\n" in text[start:end]:
        raise FixError(f"line {finding['set_line']}: a multi-line `set` command is not rewritten automatically")
    new = "set " + " ".join(neutralize_set(finding["set_args"]))
    rest = text[end:text.find("\n", end) if text.find("\n", end) >= 0 else len(text)]
    if not rest.strip():  # nothing else on the line: say what it was
        new += f"  # {FIX}: was '{text[start:end]}'"
    return new


def _apply(text, edits):
    out, last = text, None
    for (start, end), new in sorted(edits.items(), reverse=True):
        if last is not None and end > last:
            raise FixError("overlapping edits")
        out = out[:start] + new + out[end:]
        last = start
    return out


def repair_block(block: str, *, include_risks=False):
    """(repaired block, edits). Only the offending commands change; a block without an early exit comes back as is.
    exit -> one logged line (MARK); an errexit `set` covering a failing step -> the same `set` without -e; exec X -> X."""
    wanted = {"early-exit", "risk"} if include_risks else {"early-exit"}
    current = block
    for _ in range(50):
        findings = [f for f in analyze(current) if f["severity"] in wanted]
        if not findings:
            break
        edits = {}
        for finding in findings:
            if finding["kind"] == "exit":
                edits[finding["span"]] = exit_replacement(finding)
            elif finding["kind"] == "errexit":
                edits[finding["span"]] = _set_replacement(current, finding)
            elif finding["kind"] == "exec":
                edits[finding["span"]] = ""
            else:
                raise FixError(f"line {finding['line']}: cannot rewrite {finding['kind']}: {finding['line_text'].strip()}")
        current = _apply(current, edits)
    else:
        raise FixError("the repair did not converge")
    if current.count("\n") != block.count("\n"):
        raise FixError("the repair changed the number of lines")
    before, after = block.split("\n"), current.split("\n")
    edits = [{"line": n + 1, "before": a, "after": b} for n, (a, b) in enumerate(zip(before, after)) if a != b]
    return current, edits


def split_test_sh(text: str):
    """(head, block, tail) with head + block + tail == text; block = the RUN TESTS block as test_sh() inserted it."""
    if text.count(RUN_START) != 1 or text.count(RUN_END) != 1:
        raise FixError("test.sh must hold exactly one RUN TESTS block")
    head_end = text.index(RUN_START) + len(RUN_START) + 1
    tail_start = text.index("\n" + RUN_END)
    if text[head_end - 1] != "\n" or tail_start < head_end - 1:
        raise FixError("malformed RUN TESTS markers")
    return text[:head_end], text[head_end:tail_start], text[tail_start:]


# ============================================================================================ gate: block rewrite
# The sibling blocks (official1-vitest c01-r1b0, c01-r1b3, c04-r1b3, c05-r1b3) and the vitest-duration-sharding frame
# keep a failed rebuild in the grade without stopping: the status of the build step(s) becomes one CTRF test,
# {"name": "[gate] yarn build:js && yarn build:ts", "status": "passed"|"failed", "duration": 0}, added to the base
# report after the suites ran (an absent or unreadable report is written with that test alone), and its id is a P2P
# test in tests/config.json. A failed build therefore fails a P2P test: reward 0, with a report and a grade.

GATE_RECORDER = r'''@@TAG@@ the build/typecheck step(s) above are the P2P test @@SHOWN@@ in the base report
# (the "[gate]" convention of the DeepSWE frame): if one failed, the run fails with a report and a grade; the suites ran.
python3 - @@REPORT@@ @@NAME@@ @@STATUSES@@ <<'GATE' 2>&1 | tee -a "${RUN_LOG:-/logs/verifier/run.log}"
import json
import sys
from pathlib import Path

path, name, codes = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
try:
    report = json.loads(path.read_text())
    tests = report["results"]["tests"]
    if not isinstance(tests, list):
        raise TypeError("results.tests is not a list")
except Exception as error:  # no report (no suite started) or an unreadable one: the gate test still counts
    print(f"[verifier] gate-fix: {path.name} unusable ({type(error).__name__}); writing it with the gate test only")
    report = {"reportFormat": "CTRF", "specVersion": "1.0.0",
              "results": {"tool": {"name": "junit-to-ctrf"}, "tests": []}}
    tests = report["results"]["tests"]
status = "passed" if codes and all(code == "0" for code in codes) else "failed"
tests.append({"name": name, "status": status, "duration": 0})
summary = report["results"].get("summary")
summary = summary if isinstance(summary, dict) else {}
for key in ("passed", "failed", "skipped", "pending"):
    summary[key] = sum(isinstance(test, dict) and test.get("status") == key for test in tests)
summary["other"] = len(tests) - sum(summary[key] for key in ("passed", "failed", "skipped", "pending"))
summary["tests"] = len(tests)
report["results"]["summary"] = summary
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(f"[verifier] gate-fix: {name}: {status} (exit statuses: {' '.join(codes)})")
GATE'''


def gate_recorder(name: str, statuses: list, report: str) -> str:
    """The lines that add the gate test `name` (passed iff every status variable is 0; an unset one counts as 1) to
    the CTRF report `report`. A python3 heredoc: the frame itself needs python3 (tests/grader.py)."""
    if not name.startswith(GATE_PREFIX) or "\n" in name:
        raise FixError(f"bad gate test name {name!r}")
    return (GATE_RECORDER.replace("@@TAG@@", GATE_TAG).replace("@@SHOWN@@", json.dumps(name))
            .replace("@@REPORT@@", shlex.quote(report)).replace("@@NAME@@", shlex.quote(name))
            .replace("@@STATUSES@@", " ".join(f'"${{{var}-1}}"' for var in statuses)))


def _first_on_line(text, commands, line):
    """Index of the first command (with words) that starts a pipeline on `line` of `text`, else None."""
    for index, cmd in enumerate(commands):
        if cmd.words and _line_of(text, cmd.start) == line and not (cmd.prev in ("|", "|&") and index
                                                                    and commands[index - 1].sep in ("|", "|&")):
            return index
    return None


def _step_command(text, commands, functions, index) -> str:
    """The command a step runs, as its gate test names it: the words of its first command without leading assignments
    and wrapper functions (`run_log yarn build:js` -> `yarn build:js`)."""
    words = _words(text, commands[index])
    while words and ASSIGN.match(words[0]):
        words.pop(0)
    while len(words) > 1 and (words[0] in functions or words[0] == "run_log"):
        words.pop(0)
    return " ".join(" ".join(words).split())


def _command_span(text, commands, functions, index):
    """(start, end) of the step command _step_command() names, in `text`."""
    cmd, skip = commands[index], 0
    words = _words(text, cmd)
    while skip < len(words) and ASSIGN.match(words[skip]):
        skip += 1
    while len(words) - skip > 1 and (words[skip] in functions or words[skip] == "run_log"):
        skip += 1
    return cmd.words[skip][0], cmd.words[-1][1]


def gated_steps(block: str) -> list:
    """The build/typecheck steps that the early exits of an UNREPAIRED block guard, in order: [{line, kind, text}]
    (`line` in the block, `text` the whole step as _step() shows it). An exit names the steps it gates; under errexit
    the failing step is the gate itself. Other guarded steps (e.g. a report check) are not part of the gate."""
    commands, functions, _ = parse_shell(block)
    steps = {}
    for finding in analyze(block):
        if finding["severity"] != "early-exit":
            continue
        found = []
        if finding["kind"] == "exit":
            found = [gate for gate in finding.get("gates") or [] if gate["kind"] in GATE_KINDS]
        elif finding["kind"] == "errexit" and finding.get("step") in GATE_KINDS:
            index = _first_on_line(block, commands, finding["line"])
            if index is not None:
                found = [dict(_step(block, commands, functions, index), kind=finding["step"])]
        for gate in found:
            steps.setdefault(gate["line"], {"line": gate["line"], "kind": gate["kind"], "text": gate["text"]})
    return [steps[line] for line in sorted(steps)]


def _reassigned(text, after: int, var: str) -> bool:
    """True if `var` may be assigned again after offset `after` (the gate must read the step's own status)."""
    return bool(re.search(rf"(^|[\s;&|(]){re.escape(var)}\+?=|(^|[;&|]\s*)\s*(read|local|declare|typeset|for)\b"
                          rf"[^\n;]*\b{re.escape(var)}\b", text[after:], re.M))


def _reads_status(text, commands, index) -> bool:
    """True if command `index` reads $? or PIPESTATUS (a line inserted before it would clobber them)."""
    if index >= len(commands):
        return False
    cmd = commands[index]
    return bool(re.search(r"\$\?|\$\{?PIPESTATUS\b", text[cmd.start:cmd.end]))


def gate_block(original: str, current: str, *, report: str, add_typecheck: str = None):
    """(gated block, info). `current` is the repaired block (no early exit left), `original` the block before the
    early-exit repair: its exits name the build/typecheck steps the gate covers (the repair kept every line in place).
    For each step its exit status is read from the author's own capture (`rc=$?`, `rc=${PIPESTATUS[0]}` right after
    it) or from a new `gate_rc_<n>=...` line; `add_typecheck` (e.g. "yarn build:ts") becomes one more gated step, in the
    form of the last one, when no step runs a type checker; the recorder goes before the block's closing `set -e`
    (else at its end), where errexit is off. Nothing else changes; the result must not exit early."""
    if GATE_TAG in current or GATE_MARK in current:
        raise FixError("the block is already gated")
    if [f for f in analyze(current) if f["severity"] == "early-exit"]:
        raise FixError("the block still exits early: run repair first")
    if re.search(rf"\b{GATE_RC}\d+=", current):
        raise FixError(f"the block already uses {GATE_RC}<n> variables")
    if current.count("\n") != original.count("\n"):
        raise FixError("the repaired block does not have the original's lines")
    steps = gated_steps(original)
    if not steps:
        raise FixError("no early exit of the original block guards a build or typecheck step: nothing to gate")
    commands, functions, _ = parse_shell(current)
    pipeline_of = {}
    for group in _pipelines(commands):
        for index in group:
            pipeline_of[index] = group
    lines = current.split("\n")
    inserts, statuses, names, done, counter = {}, [], [], [], [0]

    def new_var():
        counter[0] += 1
        return f"{GATE_RC}{counter[0]}"

    after = None
    for step in steps:
        index = _first_on_line(current, commands, step["line"])
        if index is None or _step(current, commands, functions, index)["text"] != step["text"]:
            raise FixError(f"block line {step['line']}: the repaired block no longer runs `{step['text']}` there")
        group = pipeline_of[index]
        cmd, last = commands[index], commands[group[-1]]
        if cmd.cond or cmd.function or cmd.subshell or cmd.if_cond or cmd.group_gate is not None or cmd.negated \
                or last.sep not in ("\n", ";", ""):
            raise FixError(f"block line {step['line']}: the status of `{step['text']}` is consumed by a condition, "
                           "list, function or subshell; gate it by hand")
        name = _step_command(current, commands, functions, index)
        nxt = group[-1] + 1
        words = _words(current, commands[nxt]) if nxt < len(commands) else []
        capture = RC_CAPTURE.match(words[0]) if len(words) == 1 and not commands[nxt].cond else None
        indent = re.match(r"[ \t]*", lines[step["line"] - 1]).group(0)
        if capture:
            var, after = capture.group(1), _line_of(current, commands[nxt].end)
            how = "captured by the block"
            if _reassigned(current, commands[nxt].end, var):
                if _reads_status(current, commands, nxt + 1):
                    raise FixError(f"block line {after + 1} reads $? right after `{words[0]}`: gate it by hand")
                copy_var = new_var()
                inserts.setdefault(after, []).append(f'{indent}{copy_var}="${{{var}-1}}"  {GATE_TAG} exit status '
                                                     f"of `{name}`")
                var, how = copy_var, f"copied from ${capture.group(1)} (reassigned later)"
        else:
            if _reads_status(current, commands, nxt):
                raise FixError(f"block line {_line_of(current, commands[nxt].start)} reads $? or PIPESTATUS after "
                               f"`{name}`: a capture line would clobber it; gate it by hand")
            var, after = new_var(), _line_of(current, last.end)
            source = "${PIPESTATUS[0]}" if len(group) > 1 else "$?"
            inserts.setdefault(after, []).append(f"{indent}{var}={source}  {GATE_TAG} exit status of `{name}`")
            how = "captured by gate-fix"
        statuses.append(var)
        names.append(name)
        done.append(dict(step, command=name, status=var, how=how, index=index, indent=indent, after=after))
    typecheck = None
    if add_typecheck and not any(TYPECHECKS.search(name) for name in names):
        last_step = done[-1]
        index, after = last_step["index"], last_step["after"]
        group = pipeline_of[index]
        following = next((i for i, c in enumerate(commands) if c.words and _line_of(current, c.start) > after), None)
        if following is not None and _reads_status(current, commands, following):
            raise FixError(f"block line {_line_of(current, commands[following].start)} reads $? or PIPESTATUS: "
                           f"adding `{add_typecheck}` before it would clobber them; gate it by hand")
        start, end = _command_span(current, commands, functions, index)
        rest = current[end:commands[group[-1]].end]  # e.g. ` 2>&1 | tee -a "$RUN_LOG"`
        one_line = _line_of(current, commands[index].start) == _line_of(current, commands[group[-1]].end)
        if one_line and "\n" not in rest and not re.sub(r"\d*>&\d+", "", rest).count(">"):
            # the same form as the last step: `run_log yarn build:ts`, `yarn build:ts 2>&1 | tee -a "$RUN_LOG"`
            body = current[current.rfind("\n", 0, start) + 1:start] + add_typecheck + rest
            piped = len(group) > 1
        else:  # a multi-line step or one that writes its own log file: the generic form
            body = f'{last_step["indent"]}{add_typecheck} 2>&1 | tee -a "${{RUN_LOG:-/logs/verifier/run.log}}"'
            piped = True
        var = new_var()
        inserts.setdefault(after, []).extend([
            f"{body}  {GATE_TAG} typecheck too: a solution that does not typecheck fails the gate",
            f"{last_step['indent']}{var}={'${PIPESTATUS[0]}' if piped else '$?'}  {GATE_TAG} exit status of "
            f"`{add_typecheck}`"])
        statuses.append(var)
        names.append(add_typecheck)
        typecheck = {"command": add_typecheck, "status": var, "after_block_line": after, "line": body.strip()}
    gate_id = GATE_PREFIX + " && ".join(names)
    recorder = gate_recorder(gate_id, statuses, report).split("\n")
    # where the recorder goes: before the closing top-level `set -e`, else after the last line
    tail = max((i for i, c in enumerate(commands) if c.words), default=None)
    closing = tail is not None and _words(current, commands[tail]) == ["set", "-e"] and not (
        commands[tail].function or commands[tail].subshell or commands[tail].cond or commands[tail].if_cond
        or commands[tail].group_gate is not None) and lines[_line_of(current, commands[tail].start) - 1].startswith("set")
    out = []
    for number, line in enumerate(lines, 1):
        if closing and number == _line_of(current, commands[tail].start):
            out += recorder
        out.append(line)
        out += inserts.get(number, [])
    if not closing:
        while out and not out[-1].strip():
            out.pop()
        out += recorder
    new = "\n".join(out)
    left = [f for f in analyze(new) if f["severity"] == "early-exit"]
    if left:
        raise FixError(f"the gated block would exit early (block line {left[0]['line']}): gate it by hand")
    if report not in current:
        raise FixError(f"the block never writes {report}: the gate test would have no base report")
    recorder_text = "\n".join(recorder)
    if Path(report).name in new[new.index(recorder_text) + len(recorder_text):]:
        raise FixError(f"the block touches {Path(report).name} after the recorder")
    return new, {"gate_id": gate_id, "report": report,
                 "steps": [{k: v for k, v in s.items() if k not in ("index", "indent", "after")} for s in done],
                 "typecheck_added": typecheck, "recorder": "before the closing `set -e`" if closing else "at the end",
                 "added_lines": len(new.split("\n")) - len(lines)}


def gated_config(config: dict, gate_id: str) -> dict:
    """config.json with the gate test among the P2P ids (sorted like calibrate() writes them; already there: as is)."""
    if gate_id in config.get("f2p_node_ids", []):
        raise FixError(f"{gate_id!r} is an F2P test")
    p2p = list(config.get("p2p_node_ids", []))
    if gate_id in p2p:
        return config
    return dict(config, p2p_node_ids=sorted(p2p + [gate_id]) if p2p == sorted(p2p) else [gate_id] + p2p)


def gate_ids(config: dict) -> list:
    return [n for n in config.get("p2p_node_ids", []) if str(n).startswith(GATE_PREFIX)]


# ============================================================================================ build tree

def resolve_root(value: str) -> Path:
    path = Path(value)
    if path.is_dir() and (list(path.glob("cand-*")) or (path / "summary.json").exists()):
        return path.resolve()
    builds = sorted(p for p in (RUNS / value / "build").glob("*/*") if p.is_dir())
    if len(builds) == 1:
        return builds[0].resolve()
    raise SystemExit(f"--run {value}: not a build root, and runs/{value}/build holds {len(builds)} builds")


def accepted_dir(root: Path) -> Path:
    return root.parents[2] / "accepted"  # runs/<run>/build/<target>/<owner__repo> -> runs/<run>/accepted


def built_cands(root: Path):
    for cand in sorted(root.glob("cand-*")):
        stage = read_json(cand / "stage.json")
        if isinstance(stage, dict) and stage.get("ok"):
            yield cand, stage


def cand_dir(root: Path, name: str) -> Path:
    name = name if name.startswith("cand-") else f"cand-{name}"
    cand = root / name
    stage = read_json(cand / "stage.json")
    if not (isinstance(stage, dict) and stage.get("ok")):
        raise SystemExit(f"{cand}: not a built task (stage.json ok)")
    return cand


def cand_names(values) -> list:
    names = []
    for value in values or []:
        names += [part for part in value.split(",") if part]
    return names


def scored_dir(root: Path, cand: str):
    """round-<k>/<cand> holding the task's metrics (or the metrics a re-score moved aside), else None."""
    found = [r / cand for r in sorted(root.glob("round-*"))
             if any((r / cand / f"metrics.json{suffix}").exists() for suffix in ("", ASIDE, GATE_BACKUP))]
    if len(found) > 1:
        raise FixError(f"{cand} is scored in several rounds: {found}")
    return found[0] if found else None


def sha256(data) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def original(path: Path) -> str:
    """A file as it was before its first repair."""
    backup = path.with_name(path.name + BACKUP)
    return (backup if backup.exists() else path).read_text()


def copies(root: Path, cand: Path, stage: dict) -> dict:
    """The live copies of a task's test script."""
    built = read_json(cand / "built.json") or {}
    files = {"final": cand / "final/tests/test.sh", "task": Path(stage["task_dir"]) / "tests/test.sh"}
    if (accepted := accepted_dir(root) / stage["task_id"] / "tests/test.sh").exists():
        files["accepted"] = accepted
    files["run_tests"] = cand / f"author-{built.get('attempt', 0)}" / "run_tests.sh"
    return files


def history_copies(root: Path, cand: Path, stage: dict) -> list:
    """Copies that record what already ran (left untouched): the gates' provisional tests and probe inputs."""
    found = [cand / "provisional/tests/test.sh"]
    found += sorted(root.glob(f"round-*/probe-*/tasks/{stage['task_id']}/tests/test.sh"))
    return [path for path in found if path.exists()]


def log_change(root: Path, entry: dict, dry_run=False):
    entry = {"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "tool": "tools/regrade.py",
             **entry}
    if dry_run:
        return entry
    with (root / LOG).open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


@contextlib.contextmanager
def locked(root: Path, dry_run=False):
    if dry_run:
        yield
        return
    with (root / f".{FIX}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another tools/regrade.py holds {root / f'.{FIX}.lock'}") from None
        yield


def atomic_write(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text)
    shutil.copymode(path, temporary)
    temporary.replace(path)


def move_aside(path: Path, suffix=ASIDE):
    target = path.with_name(path.name + suffix)
    if target.exists() or target.is_symlink():
        raise FixError(f"both {path} and {target} exist: inspect them")
    path.rename(target)
    return target


# ============================================================================================ scan

def scan_task(root: Path, cand: Path, stage: dict) -> dict:
    path = Path(stage["task_dir"]) / "tests/test.sh"
    path = path if path.exists() else cand / "final/tests/test.sh"
    text = path.read_text()
    head, block, _ = split_test_sh(text)
    offset = head.count("\n")
    findings = analyze(block)
    for finding in findings:
        finding["test_sh_line"] = finding["line"] + offset
        if "set_line" in finding:
            finding["set_test_sh_line"] = finding["set_line"] + offset
        for gate in finding.get("gates") or []:
            gate["test_sh_line"] = gate["line"] + offset
        finding.pop("span", None)
    files = copies(root, cand, stage)
    differ = [name for name, p in files.items() if name != "run_tests" and p.exists() and p.read_text() != text]
    scored = scored_dir(root, cand.name)
    metrics = (read_json(scored / "metrics.json") or read_json(scored / f"metrics.json{ASIDE}")) if scored else None
    return {"cand": cand.name, "task_id": stage["task_id"], "test_sh": str(path), "findings": findings,
            "early_exit": any(f["severity"] == "early-exit" for f in findings),
            "risk": any(f["severity"] == "risk" for f in findings), "repaired": MARK in block or FIX in block,
            "accepted": (accepted_dir(root) / stage["task_id"]).is_dir(), "copies_differ": differ,
            "metrics": {key: metrics.get(key) for key in ("student_rate", "teacher_clean_rate", "valid",
                                                          "grading_failures", "accept", "reason")} if metrics else None,
            "scored_in": str(scored.relative_to(root)) if scored else None}


def cmd_scan(args):
    root = resolve_root(args.run)
    tasks, unreadable = [], []
    for cand, stage in built_cands(root):
        try:
            tasks.append(scan_task(root, cand, stage))
        except (FixError, OSError) as error:
            unreadable.append(f"{cand.name}: {error}")
    if args.json:
        print(json.dumps({"build_root": str(root), "tasks": tasks, "unreadable": unreadable}, indent=2))
        return 1 if unreadable else 0
    print(f"build root {root}\n")
    for severity, title in (("early-exit", "EARLY EXIT"), ("risk", "RISK only (not an early exit by default)")):
        chosen = [t for t in tasks if t["early_exit"]] if severity == "early-exit" else \
            [t for t in tasks if t["risk"] and not t["early_exit"]]
        print(f"== {title}: {len(chosen)} task(s)")
        for task in chosen:
            metrics = task["metrics"] or {}
            notes = ["ACCEPTED" if task["accepted"] else "", f"scored in {task['scored_in']}" if task["scored_in"] else "",
                     f"grading failures {metrics.get('grading_failures')}" if metrics.get("grading_failures") else "",
                     f"copies differ: {task['copies_differ']}" if task["copies_differ"] else ""]
            print(f"{task['cand']}  {task['task_id']}  " + "  ".join(n for n in notes if n))
            for finding in task["findings"]:
                if finding["severity"] != severity:
                    continue
                print(f"  test.sh:{finding['test_sh_line']}: {finding['line_text'].strip()}")
                if finding["kind"] == "exit":
                    where = f" in function {finding['function']}()" if finding["function"] else ""
                    cond = f" under `if {finding['condition']}`" if finding["condition"] else ""
                    print(f"      exit{where}{cond}")
                    for gate in finding["gates"]:
                        print(f"      gates {gate['kind']} step test.sh:{gate['test_sh_line']}: {gate['text']}")
                elif finding["kind"] == "errexit":
                    print(f"      {finding['step']} step under `{finding['set_text']}` (test.sh:"
                          f"{finding['set_test_sh_line']}): if it fails, the verifier ends here")
                else:
                    print(f"      {finding['detail']}")
        print()
    clean = [t["cand"] for t in tasks if not t["findings"]]
    print(f"== clean: {len(clean)} task(s): {' '.join(clean)}")
    for line in unreadable:
        print(f"== NOT SCANNED: {line}")
    if any(t["early_exit"] for t in tasks):
        print(f"\nrepair: tools/regrade.py repair --run {args.run} --cand " +
              " ".join(t["cand"] for t in tasks if t["early_exit"]))
    return 0


# ============================================================================================ repair

def archive_copies(archive: Path, needles) -> dict:
    """Members of a workspace archive that hold the RUN TESTS block (or are named like a test script)."""
    hits, checked = [], 0
    needles = [n.encode() for n in needles if n]
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            checked += 1
            if member.name.rsplit("/", 1)[-1] in ("test.sh", "run_tests.sh"):
                hits.append(member.name)
                continue
            if member.size > 4_000_000:
                continue
            data = tar.extractfile(member).read()
            if any(needle in data for needle in needles):
                hits.append(member.name)
    return {"archive": str(archive), "files_checked": checked, "copies": hits}


def plan_repair(root: Path, cand: Path, *, include_risks=False) -> dict:
    stage = read_json(cand / "stage.json")
    files = copies(root, cand, stage)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FixError(f"missing: {missing}")
    originals = {name: original(path) for name, path in files.items()}
    scripts = {originals[name] for name in files if name != "run_tests"}
    if len(scripts) != 1:
        raise FixError("the test.sh copies (final, task, accepted) differ: " +
                       ", ".join(f"{n} {sha256(originals[n])[:12]}" for n in files if n != "run_tests"))
    text = originals["final"]
    head, block, tail = split_test_sh(text)
    if originals["run_tests"].strip() != block:
        raise FixError(f"{files['run_tests']} is not the RUN TESTS block of {files['final']}")
    new_block, edits = repair_block(block, include_risks=include_risks)
    new_text = head + new_block + tail
    frame = DEEPSWE / root.parent.name
    if (frame / "tests/test.sh").exists() and task_builder.test_sh(new_block, frame) != new_text:
        raise FixError(f"the frame of {files['final']} is not task_builder.test_sh()'s frame of {frame.name}")
    new_run_tests, _ = repair_block(originals["run_tests"], include_risks=include_risks)
    if new_run_tests.strip() != new_block:
        raise FixError("run_tests.sh would not get the same block as test.sh")
    offset = head.count("\n")
    changes = []
    for name, path in files.items():
        target = new_run_tests if name == "run_tests" else new_text
        current = path.read_text()
        if current == target:
            status = "unchanged" if target == originals[name] else "already repaired"
        elif current == originals[name]:
            status = "repair"
        else:
            raise FixError(f"{path} differs from both its original and its repaired version")
        changes.append({"copy": name, "path": str(path), "status": status, "sha256_before": sha256(current),
                        "sha256_after": sha256(target), "_target": target})
    archive = None
    if "accepted" in files and edits:
        workspace = accepted_dir(root) / f"{stage['task_id']}.workspace/workspace.tar.gz"
        needles = [RUN_START] + sorted((l.strip() for l in block.splitlines() if "/logs/verifier" in l
                                        and len(l.strip()) >= 30), key=len, reverse=True)[:3]
        archive = archive_copies(workspace, needles) if workspace.exists() else {"archive": str(workspace),
                                                                                  "missing": True, "copies": []}
    return {"cand": cand.name, "task_id": stage["task_id"], "findings": [
                {k: v for k, v in f.items() if k not in ("span",)} for f in analyze(block)],
            "edits": [dict(e, test_sh_line=e["line"] + offset) for e in edits], "changes": changes,
            "unified_diff": "".join(difflib.unified_diff(text.splitlines(True), new_text.splitlines(True),
                                                         "test.sh", "test.sh (repaired)")),
            "untouched_history": [str(p) for p in history_copies(root, cand, stage)], "archive": archive,
            "test_sh_sha256_after": sha256(new_text)}


def apply_repair(root: Path, cand: Path, plan: dict, *, dry_run=False):
    for change in plan["changes"]:
        if change["status"] != "repair":
            continue
        path = Path(change["path"])
        backup = path.with_name(path.name + BACKUP)
        if not dry_run:
            if not backup.exists():
                shutil.copy2(path, backup)
            atomic_write(path, change["_target"])
        log_change(root, {"action": "repair", "cand": cand.name, "task_id": plan["task_id"], "copy": change["copy"],
                          "path": str(path), "backup": str(backup), "sha256_before": change["sha256_before"],
                          "sha256_after": change["sha256_after"], "edits": plan["edits"]}, dry_run)
    record = {k: v for k, v in plan.items() if k != "changes"}
    record["changes"] = [{k: v for k, v in c.items() if k != "_target"} for c in plan["changes"]]
    record["at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if not dry_run:
        write_json(cand / FIX / "repair.json", record)
    return record


def cmd_repair(args):
    root = resolve_root(args.run)
    status = 0
    with locked(root, args.dry_run):
        for name in cand_names(args.cand):
            cand = cand_dir(root, name)
            try:
                plan = plan_repair(root, cand, include_risks=args.include_risks)
            except FixError as error:
                print(f"{cand.name}: NOT REPAIRED: {error}")
                status = 1
                continue
            print(f"== {cand.name} ({plan['task_id']}){' [dry run]' if args.dry_run else ''}")
            if not plan["edits"]:
                print("  no early exit: nothing to change")
                continue
            if not any(change["status"] == "repair" for change in plan["changes"]):
                print("  already repaired: " + ", ".join(change["path"] for change in plan["changes"]))
                continue
            for edit in plan["edits"]:
                print(f"  test.sh:{edit['test_sh_line']}\n    - {edit['before'].strip()}\n    + {edit['after'].strip()}")
            for change in plan["changes"]:
                verb = {"repair": "would rewrite" if args.dry_run else "rewrote"}.get(change["status"], change["status"])
                print(f"  {verb}: {change['path']}" + (f" (backup {Path(change['path']).name}{BACKUP})"
                                                        if change["status"] == "repair" else ""))
            for path in plan["untouched_history"]:
                print(f"  left as is (record of what ran): {path}")
            if plan["archive"]:
                archive = plan["archive"]
                if archive.get("missing"):
                    print(f"  workspace archive missing: {archive['archive']}")
                elif archive["copies"]:
                    print(f"  WARNING: workspace archive holds a copy of the test script: {archive['copies']}. The archive "
                          "is accept()'s backup of /app in the task snapshot, so regenerating it would reproduce the same "
                          "copy; it needs a new snapshot (not done here)")
                    status = 1
                else:
                    print(f"  workspace archive holds no copy of the test script ({archive['files_checked']} files "
                          f"checked): {archive['archive']}")
            apply_repair(root, cand, plan, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nlog: {root / LOG}")
    return status


# ============================================================================================ gate: repair

def gate_original(path: Path) -> str:
    """A file as it was before its gate rewrite."""
    backup = path.with_name(path.name + GATE_BACKUP)
    return (backup if backup.exists() else path).read_text()


def config_copies(root: Path, cand: Path, stage: dict) -> dict:
    """The live copies of a task's tests/config.json."""
    files = {"final": cand / "final/tests/config.json", "task": Path(stage["task_dir"]) / "tests/config.json"}
    if (accepted := accepted_dir(root) / stage["task_id"] / "tests/config.json").exists():
        files["accepted"] = accepted
    return files


def config_text(config: dict) -> str:
    return json.dumps(config, indent=2, sort_keys=True) + "\n"  # common.write_json's format (task_builder wrote it)


def plan_gate(root: Path, cand: Path, *, add_typecheck=None) -> dict:
    """The gate rewrite of every live copy: test.sh (final, task, accepted), the author's run_tests.sh and config.json
    (final, task, accepted). Starts from the pre-gate versions (<file>.pre-gate-fix if it exists), so a rerun finds
    the files already gated."""
    stage = read_json(cand / "stage.json")
    scripts, configs = copies(root, cand, stage), config_copies(root, cand, stage)
    missing = [str(p) for p in [*scripts.values(), *configs.values()] if not p.exists()]
    if missing:
        raise FixError(f"missing: {missing}")
    before = {name: gate_original(path) for name, path in scripts.items()}
    if len({before[name] for name in scripts if name != "run_tests"}) != 1:
        raise FixError("the test.sh copies (final, task, accepted) differ: " +
                       ", ".join(f"{n} {sha256(before[n])[:12]}" for n in scripts if n != "run_tests"))
    text = before["final"]
    head, block, tail = split_test_sh(text)
    run_tests = before["run_tests"]
    if run_tests.strip() != block:
        raise FixError(f"{scripts['run_tests']} is not the RUN TESTS block of {scripts['final']}")
    early = original(scripts["final"])  # test.sh before the early-exit repair: its exits name the gated steps
    config_before = {name: gate_original(path) for name, path in configs.items()}
    if len(set(config_before.values())) != 1:
        raise FixError("the config.json copies (final, task, accepted) differ")
    config = json.loads(config_before["final"])
    if config_text(config) != config_before["final"]:
        raise FixError("config.json is not in write_json's format: refusing to rewrite it")
    grade = config.get("grade") or {}
    if grade.get("format") != "ctrf" or not grade.get("reports"):
        raise FixError(f"gate mode writes a CTRF test into the base report; this task grades {grade.get('format')}")
    new_block, info = gate_block(split_test_sh(early)[1], block, report=grade["reports"][0],
                                 add_typecheck=add_typecheck)
    new_text = head + new_block + tail
    frame = DEEPSWE / root.parent.name
    if (frame / "tests/test.sh").exists() and task_builder.test_sh(new_block, frame) != new_text:
        raise FixError(f"the frame of {scripts['final']} is not task_builder.test_sh()'s frame of {frame.name}")
    lead, trail = run_tests[:len(run_tests) - len(run_tests.lstrip())], run_tests[len(run_tests.rstrip()):]
    new_config = gated_config(config, info["gate_id"])
    targets = {**{("script", name): lead + new_block + trail if name == "run_tests" else new_text
                  for name in scripts},
               **{("config", name): config_text(new_config) for name in configs}}
    changes = []
    for (kind, name), target in targets.items():
        path = (scripts if kind == "script" else configs)[name]
        was = before[name] if kind == "script" else config_before[name]
        current = path.read_text()
        if current == target:
            status = "unchanged" if target == was else "already gated"
        elif current == was:
            status = "gate"
        else:
            raise FixError(f"{path} differs from both its pre-gate and its gated version")
        changes.append({"copy": name if kind == "script" else f"config:{name}", "path": str(path), "status": status,
                        "sha256_before": sha256(current), "sha256_after": sha256(target), "_target": target})
    archive = None
    if "accepted" in scripts:
        workspace = accepted_dir(root) / f"{stage['task_id']}.workspace/workspace.tar.gz"
        needles = [RUN_START] + sorted((l.strip() for l in block.splitlines() if "/logs/verifier" in l
                                        and len(l.strip()) >= 30), key=len, reverse=True)[:3]
        archive = archive_copies(workspace, needles) if workspace.exists() else {"archive": str(workspace),
                                                                                  "missing": True, "copies": []}
    offset = head.count("\n")
    return {"cand": cand.name, "task_id": stage["task_id"], "mode": "gate", "gate_id": info["gate_id"],
            "report": info["report"], "steps": [dict(s, test_sh_line=s["line"] + offset) for s in info["steps"]],
            "typecheck_added": info["typecheck_added"], "recorder": info["recorder"],
            "added_lines": info["added_lines"], "p2p_added": new_config is not config, "changes": changes,
            "unified_diff": "".join(difflib.unified_diff(text.splitlines(True), new_text.splitlines(True),
                                                         "test.sh", "test.sh (gated)")),
            "untouched_history": [str(p) for p in history_copies(root, cand, stage)] +
                                 [str(p) for p in [cand / "provisional/tests/config.json",
                                                   *sorted(root.glob(f"round-*/probe-*/tasks/{stage['task_id']}"
                                                                     "/tests/config.json"))] if p.exists()],
            "archive": archive, "test_sh_sha256_after": sha256(new_text)}


def apply_gate(root: Path, cand: Path, plan: dict, *, dry_run=False):
    for change in plan["changes"]:
        if change["status"] != "gate":
            continue
        path = Path(change["path"])
        backup = path.with_name(path.name + GATE_BACKUP)
        if not dry_run:
            if not backup.exists():
                shutil.copy2(path, backup)
            atomic_write(path, change["_target"])
        log_change(root, {"action": "gate_repair", "mode": "gate", "cand": cand.name, "task_id": plan["task_id"],
                          "copy": change["copy"], "path": str(path), "backup": str(backup),
                          "sha256_before": change["sha256_before"], "sha256_after": change["sha256_after"],
                          "gate_id": plan["gate_id"], "typecheck_added": plan["typecheck_added"]}, dry_run)
    record = {k: v for k, v in plan.items() if k != "changes"}
    record["changes"] = [{k: v for k, v in c.items() if k != "_target"} for c in plan["changes"]]
    record["at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if not dry_run:
        write_json(cand / FIX / GATE_DIR / "repair.json", record)
    return record


def cmd_repair_gate(args):
    root = resolve_root(args.run)
    status = 0
    with locked(root, args.dry_run):
        for name in cand_names(args.cand):
            cand = cand_dir(root, name)
            try:
                plan = plan_gate(root, cand, add_typecheck=args.add_typecheck)
            except FixError as error:
                print(f"{cand.name}: NOT GATED: {error}")
                status = 1
                continue
            print(f"== {cand.name} ({plan['task_id']}){' [dry run]' if args.dry_run else ''}")
            print(f"  gate test (P2P, {Path(plan['report']).name}): {plan['gate_id']}")
            for step in plan["steps"]:
                print(f"  gated step test.sh:{step['test_sh_line']} ({step['kind']}): {step['command']} -> "
                      f"${step['status']} ({step['how']})")
            if plan["typecheck_added"]:
                print(f"  added step (no gated step typechecks): {plan['typecheck_added']['line']} -> "
                      f"${plan['typecheck_added']['status']}")
            print(f"  recorder: {plan['recorder']}; {plan['added_lines']} line(s) added, nothing else changes")
            if not any(change["status"] == "gate" for change in plan["changes"]):
                print("  already gated: " + ", ".join(change["path"] for change in plan["changes"]))
                continue
            if args.dry_run or args.show_diff:
                print("".join("    " + line for line in plan["unified_diff"].splitlines(True)))
            for change in plan["changes"]:
                verb = {"gate": "would rewrite" if args.dry_run else "rewrote"}.get(change["status"], change["status"])
                print(f"  {verb}: {change['path']}" + (f" (backup {Path(change['path']).name}{GATE_BACKUP})"
                                                        if change["status"] == "gate" else ""))
            for path in plan["untouched_history"]:
                print(f"  left as is (record of what ran): {path}")
            if plan["archive"]:
                archive = plan["archive"]
                if archive.get("missing"):
                    print(f"  workspace archive missing: {archive['archive']}")
                elif archive["copies"]:
                    print(f"  WARNING: workspace archive holds a copy of the test script: {archive['copies']}")
                    status = 1
                else:
                    print(f"  workspace archive holds no copy of the test script ({archive['files_checked']} files "
                          f"checked): {archive['archive']}")
            apply_gate(root, cand, plan, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nlog: {root / LOG}")
    return status


def cmd_restore_gate(args):
    """Put the pre-gate copies back (test.sh copies, run_tests.sh, config.json copies); the backups stay."""
    root = resolve_root(args.run)
    status = 0
    with locked(root, args.dry_run):
        for name in cand_names(args.cand):
            cand = cand_dir(root, name)
            stage = read_json(cand / "stage.json")
            print(f"== {cand.name} ({stage['task_id']}){' [dry run]' if args.dry_run else ''}")
            files = {**copies(root, cand, stage), **{f"config:{k}": v for k, v in config_copies(root, cand,
                                                                                                 stage).items()}}
            for copy, path in files.items():
                backup = path.with_name(path.name + GATE_BACKUP)
                if not backup.exists():
                    print(f"  no backup: {path}")
                    continue
                current, before = path.read_text(), backup.read_text()
                if current == before:
                    print(f"  already restored: {path}")
                    continue
                if not args.dry_run:
                    atomic_write(path, before)
                log_change(root, {"action": "gate_restore", "mode": "gate", "cand": cand.name,
                                  "task_id": stage["task_id"], "copy": copy, "path": str(path), "backup": str(backup),
                                  "sha256_before": sha256(current), "sha256_after": sha256(before),
                                  "reason": args.reason}, args.dry_run)
                print(f"  {'would restore' if args.dry_run else 'restored'}: {path} (from {backup.name})")
    return status


# ============================================================================================ validate

SOURCE_EXT = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".py", ".go", ".rs", ".java", ".kt", ".rb",
              ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".swift", ".scala")
NOT_SOURCE = re.compile(r"(^|/)(__tests__|__typetests__|tests?|spec|e2e|fixtures?|docs?|examples?)(/|$)|"
                        r"\.(test|spec)\.[^/]+$|\.d\.ts$")
DECLARATION = re.compile(r"^(export\s|import\s|async\s|function\s|class\s|const\s|let\s|var\s|def\s|type\s|interface\s|"
                         r"enum\s|func\s|fn\s|pub\s|public\s|private\s|protected\s|static\s|module\.exports)")
BROKEN_LINE = "<<<<<<< early-exit-fix validate: deliberate syntax error"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _sections(patch: str):
    lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("diff --git ")]
    return lines, [(a, b) for a, b in zip(starts, starts[1:] + [len(lines)])]


def break_patch(patch: str, prefer: str = None):
    """(the patch plus one syntax-error line in one source file it touches, info). The file: `prefer`, else the source
    file (not a test or doc) with the most added lines, a new file only if the patch imports it (otherwise it may be
    dead code). A new file gets the line at its end; a modified file just before an added line that starts a
    declaration (zero indentation first). Hunk counts are adjusted, so the result applies like the original."""
    return _insert_line(patch, BROKEN_LINE, prefer, extensions=SOURCE_EXT, option="--broken-file")


# Valid JavaScript (babel and webpack keep it, it does nothing at run time) but a TypeScript error: Number.parseInt
# takes a string, so the type checker reports TS2345 for the number argument (a wrong parameter type).
TYPE_ERROR_LINE = "void Number.parseInt(0); // gate-fix validate: deliberate type error (valid JavaScript)"
TS_SOURCE = (".ts", ".tsx", ".mts", ".cts")


def type_break_patch(patch: str, prefer: str = None):
    """(the patch plus TYPE_ERROR_LINE at the TOP LEVEL of one TypeScript source file it touches, info): at the end of a
    new file, or just before a zero-indentation declaration of a modified one -- an added one first, else one of the
    hunk's context lines whose preceding line ends a statement -- never inside a class or function body, so the file
    stays valid JavaScript once types are erased. The file: `prefer`, else the TypeScript source file with the most
    added lines (one that is imported, if new; a type checker sees every file of its project anyway). Files with
    @ts-nocheck are skipped."""
    return _insert_line(patch, TYPE_ERROR_LINE, prefer, extensions=TS_SOURCE, option="--typeonly-file",
                        top_level=True)


def _insert_line(patch: str, inserted: str, prefer: str = None, *, extensions, option, top_level=False):
    lines, sections = _sections(patch)
    options = []
    for a, b in sections:
        path = next((l[6:].rstrip("\n") for l in lines[a:b] if l.startswith("+++ b/")), None)
        new_file = any(l.startswith("--- /dev/null") for l in lines[a:b])
        added = sum(l.startswith("+") and not l.startswith("+++") for l in lines[a:b])
        if top_level and any("@ts-nocheck" in l for l in lines[a:b]):
            continue
        if path and added and path.endswith(extensions) and not NOT_SOURCE.search(path):
            stem = re.escape(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            imported = re.compile(rf"\b(import|require|from)\b.*\b{stem}\b")
            used = not new_file or any(imported.search(line) for i, line in enumerate(lines)
                                       if not a <= i < b and line.startswith("+") and not line.startswith("+++"))
            options.append((path, a, b, new_file, added, used))
    if prefer:
        options = [o for o in options if o[0] == prefer]
        if not options:
            raise FixError(f"{option} {prefer}: not a {'/'.join(extensions)} source file with added lines in the "
                           "solution")
    options.sort(key=lambda o: (not o[5], -o[4]))  # a new file nothing imports may be dead code: last
    for path, a, b, new_file, _, _ in options:
        hunks = [i for i in range(a, b) if lines[i].startswith("@@")]
        spot = None
        if new_file and hunks:
            end = b
            while end > hunks[-1] + 1 and not lines[end - 1].startswith(("+", "\\")):
                end -= 1
            spot = (hunks[-1], end - 1 if lines[end - 1].startswith("\\") else end)
        else:
            # added declarations, zero indentation first; top_level: zero indentation only, then a zero-indentation
            # declaration among the hunk's context lines whose preceding line ends a statement (or is blank)
            passes = [("+", True), (" ", True)] if top_level else [("+", True), ("+", False)]
            for prefix, indent_ok in passes:
                for h, hunk in enumerate(hunks):
                    stop = hunks[h + 1] if h + 1 < len(hunks) else b
                    for i in range(hunk + 1, stop):
                        body = lines[i][1:]
                        if not (lines[i].startswith(prefix) and DECLARATION.match(body.lstrip()) and
                                (not indent_ok or body[:1] not in (" ", "\t"))):
                            continue
                        if top_level:  # the line before it in the patched file, if the hunk shows it
                            before = next((lines[k][1:].strip() for k in range(i - 1, hunk, -1)
                                           if not lines[k].startswith(("-", "\\"))), None)
                            if before is not None and before.startswith("@"):  # a decorator's target
                                continue
                            if prefix == " " and (before is None or not (before == "" or before.endswith((";", "}")))):
                                continue
                        spot = (hunk, i)
                        break
                    if spot:
                        break
                if spot:
                    break
        if not spot:
            continue
        hunk, at = spot
        match = HUNK.match(lines[hunk].rstrip("\n"))
        if not match:
            continue
        old_start, old_count, new_start, new_count, rest = match.groups()
        new_count = 1 if new_count is None else int(new_count)
        out = list(lines)
        out[hunk] = f"@@ -{old_start}{',' + old_count if old_count is not None else ''} +{new_start},{new_count + 1} @@{rest}\n"
        for later in [i for i in range(hunk + 1, b) if lines[i].startswith("@@")]:
            m = HUNK.match(lines[later].rstrip("\n"))
            o1, o2, n1, n2, r = m.groups()
            out[later] = f"@@ -{o1}{',' + o2 if o2 is not None else ''} +{int(n1) + 1}{',' + n2 if n2 is not None else ''} @@{r}\n"
        if not out[at - 1].endswith("\n"):  # a patch whose text ends without a newline
            out[at - 1] += "\n"
        out.insert(at, "+" + inserted + "\n")
        return "".join(out), {"file": path, "new_file": new_file, "hunk": lines[hunk].strip(),
                              "before_line": "(end of file)" if new_file else lines[at].rstrip("\n"),
                              "inserted": inserted}
    raise FixError(f"no {'/'.join(extensions)} source file in the solution has a place for {inserted!r}; use {option}")


def _log_marks(log_text: str, mark: str) -> list:
    return list(dict.fromkeys(l.strip() for l in log_text.splitlines() if mark in l and not l.lstrip().startswith("+")))


def _grade_summary(result: dict, out_dir: Path, gate_id: str = None, config: dict = None) -> dict:
    log_text = (out_dir / "output.log").read_text(errors="replace") if (out_dir / "output.log").exists() else ""
    reward_json = result.get("reward_json")
    summary = {"reward": result.get("reward"), "reward_json_present": isinstance(reward_json, dict),
               "reward_json": reward_json, "apply_failed": bool((reward_json or {}).get("apply_failed")),
               "model_patch_applied": "model.patch applied" in log_text, "exit_code": result.get("exit_code"),
               "timed_out": result.get("timed_out"),
               "suite_sizes": {k: len(v) for k, v in result.get("suites", {}).items()},
               "repair_marks": _log_marks(log_text, MARK)[:5], "out_dir": str(out_dir)}
    if gate_id:
        base = Path(config["grade"]["reports"][0]).name
        summary.update(gate=(result.get("suites") or {}).get(base, {}).get(gate_id) or "missing",
                       gate_marks=_log_marks(log_text, GATE_MARK)[:5],
                       failing=failing_tests(result.get("suites") or {}, config))
    return summary


def single_gate_id(config: dict) -> str:
    ids = gate_ids(config)
    if len(ids) != 1:
        raise FixError(f"config.json lists {len(ids)} '{GATE_PREFIX}...' P2P tests (want exactly 1): run repair --gate")
    return ids[0]


async def validate_cand(root: Path, cand: Path, *, gate, dry_run=False, broken_file=None, gate_mode=False,
                        typeonly_file=None) -> dict:
    stage = read_json(cand / "stage.json")
    final = cand / "final"
    text = (final / "tests/test.sh").read_text()
    _, block, _ = split_test_sh(text)
    left = [f for f in analyze(block) if f["severity"] == "early-exit"]
    if left:
        raise FixError(f"final/tests/test.sh still exits early (line {left[0]['line']} of the block): repair first")
    config = json.loads((final / "tests/config.json").read_text())
    gate_id = None
    if gate_mode:
        gate_id = single_gate_id(config)
        if GATE_TAG not in block or gate_id not in text:
            raise FixError(f"final/tests/test.sh does not record {gate_id!r}: run repair --gate")
    built = read_json(cand / "built.json") or {}
    solution = (cand / f"author-{built.get('attempt', 0)}" / "solution.patch").read_text()
    packaged = Path(stage["task_dir"]) / "solution/solution.patch"
    if packaged.exists() and packaged.read_text() != solution:
        raise FixError(f"{packaged} differs from the author's solution.patch")
    broken, info = break_patch(solution, broken_file)
    patches = {"solution": solution, "empty": "", "broken": broken}
    typeonly_info = None
    if gate_mode:
        patches["typeonly"], typeonly_info = type_break_patch(solution, typeonly_file)
    fix_dir = cand / FIX / GATE_DIR if gate_mode else cand / FIX
    out = fix_dir / "validate"
    if dry_run:
        return {"cand": cand.name, "dry_run": True, "broken": info, "typeonly": typeonly_info, "gate_id": gate_id,
                "out": str(out), "snapshot": stage["snapshot"], "test_sh_sha256": sha256(text)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "broken.patch").write_text(broken)
    if gate_mode:
        (out / "typeonly.patch").write_text(patches["typeonly"])
    run = root.parents[2].name

    async def one(name):
        async with gate:
            shutil.rmtree(out / name, ignore_errors=True)  # this tool's own output of an earlier validate
            return await sandbox.grade(stage["snapshot"], final, patches[name], run=run, out_dir=out / name)

    names = list(patches)
    results = await asyncio.gather(*(one(name) for name in names), return_exceptions=True)
    graded = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            graded[name] = {"error": f"{type(result).__name__}: {str(result)[-500:]}"}
        else:
            graded[name] = _grade_summary(result, out / name, gate_id, config)
    sol, emp, brk = graded["solution"], graded["empty"], graded["broken"]

    def graded_0(result):
        return result.get("reward_json_present") is True and result.get("reward") == 0 \
            and not result.get("apply_failed") and result.get("model_patch_applied") is True

    checks = {
        "solution_scores_1": sol.get("reward_json_present") is True and sol.get("reward") == 1,
        "empty_scores_0": emp.get("reward_json_present") is True and emp.get("reward") == 0,
        "broken_graded_0": graded_0(brk),
    }
    if gate_mode:  # the gate passes for the reference and the starting state (a P2P test) and fails for both breaks
        typ = graded["typeonly"]
        checks = {"solution_scores_1": checks["solution_scores_1"] and sol.get("gate") == "passed",
                  "empty_scores_0": checks["empty_scores_0"] and emp.get("gate") == "passed",
                  "broken_graded_0": checks["broken_graded_0"] and brk.get("gate") == "failed",
                  "typeonly_graded_0": graded_0(typ) and typ.get("gate") == "failed"}
        if "failing" in typ:  # whether the gate alone decided it (every other whitelisted test passed)
            typ["only_gate_failed"] = typ["failing"] == [f"[p2p] {gate_id}"]
    record = {"cand": cand.name, "task_id": stage["task_id"], "snapshot": stage["snapshot"],
              "test_sh_sha256": sha256(text), "broken": info, "results": graded, "checks": checks,
              "all_ok": all(checks.values()), "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    if gate_mode:
        record.update(mode="gate", gate_id=gate_id, typeonly=typeonly_info,
                      config_sha256=sha256((final / "tests/config.json").read_text()))
    write_json(fix_dir / "validate.json", record)
    log_change(root, {"action": "gate_validate" if gate_mode else "validate", "cand": cand.name,
                      "task_id": stage["task_id"], "checks": checks, "test_sh_sha256": record["test_sh_sha256"],
                      "path": str(fix_dir / "validate.json"), **({"mode": "gate", "gate_id": gate_id}
                                                                  if gate_mode else {})})
    return record


def _where(info: dict) -> str:
    return "new file, at its end" if info["new_file"] else "before " + repr(info["before_line"].strip()[:80])


async def _validate_all(args):
    root = resolve_root(args.run)
    gate = asyncio.Semaphore(args.concurrency)
    with locked(root, args.dry_run):
        cands = [cand_dir(root, name) for name in cand_names(args.cand)]
        results = await asyncio.gather(*(validate_cand(root, cand, gate=gate, dry_run=args.dry_run,
                                                       broken_file=args.broken_file, gate_mode=args.gate,
                                                       typeonly_file=args.typeonly_file) for cand in cands),
                                       return_exceptions=True)
    status = 0
    for cand, result in zip(cands, results):
        if isinstance(result, BaseException):
            print(f"{cand.name}: NOT VALIDATED: {type(result).__name__}: {result}")
            status = 1
            continue
        info = result["broken"]
        print(f"== {cand.name}" + (" [dry run: no grading]" if result.get("dry_run") else ""))
        if result.get("gate_id"):
            print(f"  gate test: {result['gate_id']}")
        print(f"  broken solution: {info['inserted']!r} in {info['file']} ({_where(info)})")
        if result.get("typeonly"):
            print(f"  type-only broken solution: {result['typeonly']['inserted']!r} in {result['typeonly']['file']} "
                  f"({_where(result['typeonly'])})")
        if result.get("dry_run"):
            print(f"  would grade {' / '.join(['solution', 'empty', 'broken'] + (['typeonly'] if result.get('typeonly') else []))} "
                  f"on {result['snapshot']} with final/tests (test.sh sha256 {result['test_sh_sha256'][:16]}) into "
                  f"{result['out']}")
            continue
        for name, check in zip(result["results"], result["checks"]):
            graded = result["results"][name]
            reward = graded.get("reward_json") or {}
            gate_note = f" gate {graded.get('gate')}" if "gate" in graded else ""
            print(f"  {name:8s} reward {graded.get('reward')!s:5s} reward.json {'yes' if graded.get('reward_json_present') else 'NO '} "
                  f"F2P {reward.get('f2p_passed')}/{reward.get('f2p_total')} P2P {reward.get('p2p_passed')}/{reward.get('p2p_total')}"
                  f"{gate_note} exit {graded.get('exit_code')}{' ERROR ' + graded['error'] if 'error' in graded else ''}"
                  f"  -> {check}: {'OK' if result['checks'][check] else 'FAILED'}")
            if "only_gate_failed" in graded:
                print(f"           only the gate failed: {'yes' if graded['only_gate_failed'] else 'NO'} "
                      f"(failing: {', '.join(graded['failing'][:3])}{' ...' if len(graded['failing']) > 3 else ''})")
            for mark in (graded.get("repair_marks") or []) + (graded.get("gate_marks") or []):
                print(f"           log: {mark[:150]}")
        where = cand / FIX / (GATE_DIR + "/validate.json" if result.get("mode") == "gate" else "validate.json")
        print(f"  all checks {'OK' if result['all_ok'] else 'FAILED'}: {where}")
        status |= 0 if result["all_ok"] else 1
    return status


# ============================================================================================ regrade

RANK = {"passed": 0, "skipped": 1, "failed": 2}


def failing_tests(suites: dict, config: dict) -> list:
    """The grader's "[p2p] id"/"[f2p] id" names of whitelisted tests that did not pass (missing = failed), in its
    order: what rollouts._record reads from the grader's ctrf.json."""
    merged = {}
    for tests in suites.values():
        for node, status in tests.items():
            status = status if status in RANK else "failed"
            if node not in merged or RANK[status] > RANK[merged[node]]:
                merged[node] = status
    failing = []
    for bucket, key in (("p2p", "p2p_node_ids"), ("f2p", "f2p_node_ids")):
        for node in dict.fromkeys(str(n).strip() for n in config.get(key, []) if str(n).strip()):
            if merged.get(node) != "passed":
                failing.append(f"[{bucket}] {node}")
    return failing


def regraded_record(old: dict, result: dict, config: dict, info: dict) -> dict:
    """The slot's record after grading its unchanged patch with the repaired task (rollouts._record's fields)."""
    new = dict(old)
    reward_json = result.get("reward_json")
    if not isinstance(reward_json, dict) or result.get("reward") is None:
        new.update(status="infra", verifier_error=f"regrade produced no reward.json (exit {result.get('exit_code')}, "
                                                  f"timed out {result.get('timed_out')})", regrade=info)
        return new
    reward = result["reward"]
    new.update(status="pass" if reward == 1 else "fail", reward=float(reward),
               f2p=f"{reward_json.get('f2p_passed')}/{reward_json.get('f2p_total')}",
               p2p=f"{reward_json.get('p2p_passed')}/{reward_json.get('p2p_total')}",
               failing_tests=failing_tests(result.get("suites") or {}, config), verifier_error=None,
               regrade=info)
    return new


def replace_records(records: list, replacements: dict) -> list:
    """The record list with the given slots replaced (order and every other record unchanged)."""
    slots = [r["slot"] for r in records]
    missing = sorted(set(replacements) - set(slots))
    if missing:
        raise FixError(f"no such slots: {missing}")
    if len(set(slots)) != len(slots):
        raise FixError("duplicate slots in the record list")
    return [replacements.get(record["slot"], record) for record in records]


def resolve_slots(value: str, *, model: str, manifest: dict, task_id: str) -> list:
    """`<task index>-<repeat>` (ATP's attempts/<ti>-<r>, checked against the probe manifest) or `<model>-r<repeat>`."""
    index = next((t["_ng_task_index"] for t in manifest["tasks"] if t["task_id"] == task_id), None)
    slots = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if match := re.fullmatch(r"(\d+)-(\d+)", item):
            if int(match.group(1)) != index:
                raise FixError(f"slot {item}: {task_id} is task {index} of the probe, not {match.group(1)}")
            slots.append(f"{model}-r{int(match.group(2))}")
        elif re.fullmatch(r"[A-Za-z0-9_.]+-r\d+", item):
            slots.append(item)
        else:
            raise FixError(f"slot {item!r}: use <task index>-<repeat> or <model>-r<repeat>")
    if len(set(slots)) != len(slots):
        raise FixError(f"duplicate slots: {slots}")
    return slots


def find_probe(round_dir: Path, task_id: str) -> Path:
    found = [p for p in sorted(round_dir.glob("probe-*"))
             if task_id in {t["task_id"] for t in (read_json(p / "manifest.json") or {}).get("tasks", [])}]
    if len(found) != 1:
        raise FixError(f"{len(found)} probes of {round_dir} hold {task_id}: {found}; pass --probe")
    return found[0]


def make_build(root: Path):
    """task_builder.Build for a build root (its root must be this directory)."""
    summary = read_json(root / "summary.json") or {}
    repo = summary.get("repo") or root.name.replace("__", "/", 1)
    b = task_builder.Build(root.parents[2].name, root.parent.name, repo)
    if b.root.resolve() != root.resolve():
        raise FixError(f"Build({b.run}, {b.target}, {repo}).root is {b.root}, not {root}")
    return b


def hub_budget_ok(min_remaining=500) -> bool:
    """S/tools/hub_budget.py (exit 0 fine, 3 below the floor, 2 unknown); it prints one JSON line with the key NAME."""
    proc = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HUB_BUDGET), "--min-remaining", str(min_remaining)],
                          capture_output=True, text=True, timeout=600)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    print(f"hub budget guard: exit {proc.returncode} {line}")
    return proc.returncode == 0


def _check_records(records, rollouts_dir: Path) -> list:
    """Problems if probe_records no longer gives what score() rendered into rollouts/ (status line, patch)."""
    problems = []
    for record in records:
        md, patch = rollouts_dir / f"{record['slot']}.md", rollouts_dir / f"{record['slot']}.patch"
        if not md.exists():
            problems.append(f"{md} missing")
            continue
        if md.read_text().split("\n")[:3] != rollouts.render(record).split("\n")[:3]:
            problems.append(f"{record['slot']}: the record differs from {md.name}")
        if patch.exists() and patch.read_text() != record["patch"]:
            problems.append(f"{record['slot']}: the patch differs from {patch.name}")
    return problems


def _gate_marks(out_dir: Path) -> list:
    log_text = (out_dir / "output.log").read_text(errors="replace") if (out_dir / "output.log").exists() else ""
    return (_log_marks(log_text, MARK) + _log_marks(log_text, GATE_MARK))[:5]


def refresh_seeds(b, round_dir: Path, root: Path, suffix=ASIDE, extra=None):
    """<round>/seeds.json from task_builder.valid_seeds() over the round's metrics, as a resumed build writes it."""
    seeds = round_dir / "seeds.json"
    results = [read_json(round_dir / name / "metrics.json") for name in
               sorted(p.name for p in round_dir.iterdir() if (p / "metrics.json").exists())]
    backup = seeds.with_name(seeds.name + suffix)
    if seeds.exists() and not backup.exists():
        shutil.copy2(seeds, backup)
    before = read_json(seeds)
    task_builder.valid_seeds(b, round_dir.name, results)
    log_change(root, {"action": "seeds", "path": str(seeds), "backup": str(backup), "before": before,
                      "after": read_json(seeds), **(extra or {})})


def earliest(scored: Path, name: str) -> Path:
    """The FIRST scoring's copy of a score() output: <name>.pre-regrade (moved aside by the first re-score), else
    <name>.pre-gate-fix (a gate re-score of a task the first pass did not re-score), else the live one."""
    for suffix in (ASIDE, GATE_BACKUP):
        if (scored / f"{name}{suffix}").exists():
            return scored / f"{name}{suffix}"
    return scored / name


def teacher_verdicts(scored: Path, records: list, replaced) -> dict:
    """{slot: verdict} from the first scoring's teacher audit (teacher_audit.json.pre-regrade, or .pre-gate-fix) for
    every teacher pass the regrade leaves as it was (not regraded, and its transcript renders exactly as in the first
    scoring's rollouts/). The judge already saw exactly that transcript with the same instruction and rubric (the
    repair changes tests/ only), so the re-score keeps its verdict instead of drawing a new one: only the regraded
    slots change the scores."""
    audit = read_json(earliest(scored, "teacher_audit.json"))
    old = earliest(scored, "rollouts")
    kept = {}
    for record in records:
        verdict = audit.get(record["slot"]) if isinstance(audit, dict) else None
        md = old / f"{record['slot']}.md"
        if record["model"] == TEACHER and record["status"] == "pass" and record["slot"] not in replaced \
                and isinstance(verdict, dict) and isinstance(verdict.get("cheated"), bool) and md.exists() \
                and md.read_text() == rollouts.render(record):
            kept[record["slot"]] = verdict
    return kept


def build_order(name) -> tuple:
    """Where build() puts a task in summary.json's accepted list: round 0 first, then round 1, ...; inside a round the
    probe pool's order (lineages in seed order, branches in order), which is the names' natural order."""
    name = str(name or "")
    match = re.search(r"-r(\d+)b\d+$", name)
    return int(match.group(1)) if match else 0, [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def refresh_summary(root: Path, metrics: dict, suffix=ASIDE, extra=None):
    """summary.json's accepted list with the re-scored task's new metrics (added where build() would have put it,
    replaced in place, or removed)."""
    path = root / "summary.json"
    summary = read_json(path)
    if not isinstance(summary, dict) or not isinstance(summary.get("accepted"), list):
        return
    accepted = summary["accepted"]
    where = [i for i, m in enumerate(accepted) if m.get("task_id") == metrics["task_id"]]
    new = [m for m in accepted if m.get("task_id") != metrics["task_id"]]
    if metrics.get("accept"):
        later = (i for i, m in enumerate(new) if build_order(m.get("name")) > build_order(metrics.get("name")))
        new.insert(where[0] if where else next(later, len(new)), metrics)
    if [m.get("task_id") for m in new] == [m.get("task_id") for m in accepted] and not metrics.get("accept"):
        return
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    summary["accepted"] = new
    write_json(path, summary)
    log_change(root, {"action": "summary", "path": str(path), "backup": str(backup),
                      "accepted": [m.get("name") for m in new], **(extra or {})})


def unaccept(root: Path, task_id: str, extra=None) -> list:
    """Take a task that a re-score no longer accepts out of runs/<run>/accepted/: its row leaves tasks.jsonl (the
    previous tasks.jsonl is kept under <build root>/early-exit-fix-backups/accepted/), its task directory and
    workspace archive move to <build root>/early-exit-fix-backups/unaccepted/. Nothing is deleted."""
    accepted, backups = accepted_dir(root), root / BACKUPS_DIR
    rows_path = accepted / "tasks.jsonl"
    moved = []
    if rows_path.exists():
        lines = [l for l in rows_path.read_text().splitlines() if l]
        keep = [l for l in lines if json.loads(l).get("task_id") != task_id]
        if keep != lines:
            backup = backups / "accepted" / f"tasks.jsonl{GATE_BACKUP}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(rows_path, backup)
            atomic_write(rows_path, "".join(l + "\n" for l in keep))
            moved.append({"path": str(rows_path), "row_removed": task_id, "backup": str(backup)})
    for name in (task_id, f"{task_id}.workspace"):
        source = accepted / name
        if source.exists():
            target = backups / "unaccepted" / name
            if target.exists():
                raise FixError(f"{target} exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append({"path": str(source), "to": str(target)})
    log_change(root, {"action": "unaccept", "task_id": task_id, "moved": moved, **(extra or {})})
    return moved


async def regrade_cand(root: Path, cand: Path, args) -> int:
    gate_mode = bool(getattr(args, "gate", False))
    suffix = GATE_BACKUP if gate_mode else ASIDE      # what a re-score keeps of the outputs it replaces
    extra = {"mode": "gate"} if gate_mode else {}
    fix_dir = cand / FIX / GATE_DIR if gate_mode else cand / FIX
    stage = read_json(cand / "stage.json")
    task_id, task_dir = stage["task_id"], Path(stage["task_dir"])
    scored = scored_dir(root, cand.name)
    if scored is None:
        raise FixError(f"{cand.name} has no round-*/{cand.name}/metrics.json")
    round_dir = scored.parent
    metrics_path, aside = scored / "metrics.json", scored / f"metrics.json{suffix}"
    if metrics_path.exists() and aside.exists():
        print(f"{cand.name}: already re-scored ({metrics_path} and {aside.name} exist); nothing to do")
        return 0
    old_metrics = read_json(aside) if aside.exists() else read_json(metrics_path)
    first_metrics = read_json(earliest(scored, "metrics.json"))
    test_sh = task_dir / "tests/test.sh"
    text = test_sh.read_text()
    if text != (cand / "final/tests/test.sh").read_text():
        raise FixError("the task's tests/test.sh differs from final/tests/test.sh: repair both first")
    left = [f for f in analyze(split_test_sh(text)[1]) if f["severity"] == "early-exit"]
    if left:
        raise FixError(f"{test_sh} still exits early (block line {left[0]['line']}): run repair first")
    config_raw = (task_dir / "tests/config.json").read_text()
    config = json.loads(config_raw)
    gate_id = None
    if gate_mode:
        if config_raw != (cand / "final/tests/config.json").read_text():
            raise FixError("the task's tests/config.json differs from final/tests/config.json: run repair --gate")
        gate_id = single_gate_id(config)
        if GATE_TAG not in text or gate_id not in text:
            raise FixError(f"{test_sh} does not record {gate_id!r}: run repair --gate")
    validated = read_json(fix_dir / "validate.json") or {}
    if not args.no_validate_check and not (validated.get("all_ok") and validated.get("test_sh_sha256") == sha256(text)
                                           and (not gate_mode or validated.get("config_sha256") == sha256(config_raw))):
        raise FixError(f"no passing validate.json for the current tests ({fix_dir / 'validate.json'}); run "
                       f"validate{' --gate' if gate_mode else ''} first or pass --no-validate-check")
    probe = Path(args.probe).resolve() if args.probe else find_probe(round_dir, task_id)
    manifest = read_json(probe / "manifest.json")
    records = (rollouts.probe_records(probe) or {}).get(task_id) or []
    by_slot = {r["slot"]: r for r in records}
    passing = args.slots == "passing"
    if passing:  # confirm the passes: the gate can only turn a pass into a fail
        slots = [r["slot"] for r in records if r["status"] == "pass"]
    else:
        slots = resolve_slots(args.slots, model=args.model, manifest=manifest, task_id=task_id)
    for slot in slots:
        record = by_slot.get(slot)
        if record is None:
            raise FixError(f"{slot}: not a slot of {task_id} in {probe}")
        lost = record["status"] == "infra" and record["verifier_error"] and not record["agent_error"]
        if not lost and not args.force and not passing:
            raise FixError(f"{slot}: status {record['status']}, agent_error {record['agent_error']!r}, verifier_error "
                           f"{record['verifier_error']!r}: not a slot whose grade was lost (--force to grade anyway)")
    if not slots:
        print(f"== {cand.name} ({task_id}): no {'passing ' if passing else ''}slot to grade")
        return 0
    old_rollouts = earliest(scored, "rollouts")  # probe_records gives the probe's own grades: the first scoring's
    problems = _check_records(records, old_rollouts)
    if problems and not args.force:
        raise FixError("probe_records no longer matches the scored rollouts: " + "; ".join(problems[:5]))
    regrade_path = scored / "regrade.json"
    state = read_json(regrade_path) or {}
    retired = []
    if gate_mode and state and state.get("mode") != "gate":  # the first pass's regrade.json and regrade/ stay as
        for name in ("regrade.json", "regrade"):                # *.pre-gate-fix
            path = scored / name
            if path.exists():
                retired.append(str(path))
                if not args.dry_run:
                    target = move_aside(path, GATE_BACKUP)
                    log_change(root, {"action": "move_aside", "cand": cand.name, "path": str(path), "to": str(target),
                                      **extra})
        state = {}
    test_sha = sha256(text)
    done = {s: v for s, v in (state.get("slots") or {}).items()
            if v.get("test_sh_sha256") == test_sha and v.get("new", {}).get("status") in ("pass", "fail")}
    todo = [s for s in slots if not (s in done and done[s].get("patch_sha256") == sha256(by_slot[s]["patch"]))]
    teacher_passes = sum(r["model"] == TEACHER and r["status"] == "pass" for r in records)
    print(f"== {cand.name} ({task_id}) round {round_dir.name}, probe {probe.name}"
          f"{' [gate]' if gate_mode else ''}{' [dry run]' if args.dry_run else ''}")
    if gate_mode:
        print(f"  gate test: {gate_id}")
    for slot in slots:
        record = by_slot[slot]
        print(f"  {slot}: {record['status']} ({record['verifier_error']}); patch {len(record['patch'])} chars, "
              f"{record['attempt_dir']}" + ("" if slot in todo else "  [graded earlier with this test.sh]"))
    if args.dry_run:
        print((f"  would move the first pass's {' and '.join(Path(p).name for p in retired)} aside ({GATE_BACKUP}), "
               if retired else "  would ") +
              f"grade {len(todo)} slot(s) with sandbox.grade({stage['snapshot']}, {task_dir.name}) into "
              f"{scored / 'regrade'}/<slot>/, write {regrade_path}" +
              (", and stop there if no grade changed; else" if args.confirm else ",") +
              f" run the hub budget guard, move {', '.join(n for n in SCORE_OUTPUTS if (scored / n).exists())} aside "
              f"({suffix}) and call task_builder.score() ({teacher_passes} teacher pass(es): " +
              ("all judged again by " + JUDGE if args.rejudge else
               f"the verdicts of the first scoring for unchanged transcripts are kept, the others judged by {JUDGE}")
              + "), then accept() if it accepts, then refresh seeds.json/summary.json")
        return 0
    gate = asyncio.Semaphore(args.concurrency)

    async def one(slot):
        async with gate:
            shutil.rmtree(scored / "regrade" / slot, ignore_errors=True)  # this tool's own earlier attempt
            return await sandbox.grade(stage["snapshot"], task_dir, by_slot[slot]["patch"], run=root.parents[2].name,
                                       out_dir=scored / "regrade" / slot)

    results = await asyncio.gather(*(one(slot) for slot in todo), return_exceptions=True)
    entries = dict(done)
    for slot, result in zip(todo, results):
        record = by_slot[slot]
        info = {"tool": "tools/regrade.py", "test_sh_sha256": test_sha, "out_dir": str(scored / "regrade" / slot),
                "old_status": record["status"], "old_verifier_error": record["verifier_error"]}
        if isinstance(result, BaseException):
            entries[slot] = {"error": f"{type(result).__name__}: {str(result)[-500:]}", "test_sh_sha256": test_sha}
            continue
        new = regraded_record(record, result, config, info)
        base = Path(config["grade"]["reports"][0]).name
        entries[slot] = {"attempt_dir": record["attempt_dir"], "patch_sha256": sha256(record["patch"]),
                         "patch_chars": len(record["patch"]), "test_sh_sha256": test_sha,
                         "old": {k: record[k] for k in ("status", "reward", "f2p", "p2p", "verifier_error")},
                         "new": {k: new[k] for k in ("status", "reward", "f2p", "p2p", "verifier_error",
                                                     "failing_tests")},
                         "grade": {"reward_json": result.get("reward_json"), "exit_code": result.get("exit_code"),
                                   "timed_out": result.get("timed_out"), "out_dir": info["out_dir"],
                                   "gate_marks": _gate_marks(scored / "regrade" / slot),
                                   **({"gate": (result.get("suites") or {}).get(base, {}).get(gate_id) or "missing"}
                                      if gate_mode else {})}}
    state.update({"cand": cand.name, "task_id": task_id, "round": round_dir.name, "probe": str(probe),
                  "task": dict(stage, **({"lineage": old_metrics["lineage"]} if (old_metrics or {}).get("lineage")
                                         else {})),
                  "slots": entries, "old_metrics": {k: (old_metrics or {}).get(k) for k in (
                      "student_rate", "teacher_clean_rate", "valid", "grading_failures", "accept", "reason",
                      "similarity")}})
    if gate_mode:
        state.update(mode="gate", gate_id=gate_id, config_sha256=sha256(config_raw),
                     first_metrics={k: (first_metrics or {}).get(k) for k in (
                         "student_rate", "teacher_clean_rate", "valid", "grading_failures", "accept", "reason",
                         "similarity")})
    write_json(regrade_path, state)
    log_change(root, {"action": "regrade", "cand": cand.name, "task_id": task_id, "path": str(regrade_path),
                      "slots": {s: {"old": e.get("old"), "new": {k: v for k, v in (e.get("new") or {}).items()
                                                                  if k != "failing_tests"}, "error": e.get("error"),
                                    **({"gate": e.get("grade", {}).get("gate")} if gate_mode else {})}
                                for s, e in entries.items() if s in slots}, **extra})
    for slot in slots:
        entry = entries.get(slot) or {}
        if "error" in entry or entry.get("new", {}).get("status") not in ("pass", "fail"):
            print(f"  {slot}: NOT GRADED: {entry.get('error') or entry.get('new', {}).get('verifier_error')}")
            continue
        old, new = entry["old"], entry["new"]
        gate_note = f", gate {entry['grade'].get('gate')}" if gate_mode else ""
        print(f"  {slot}: {old['status']} -> {new['status']} (reward {new['reward']}, F2P {new['f2p']}, P2P "
              f"{new['p2p']}{gate_note})" + (f"; log: {entry['grade']['gate_marks'][-1][:140]}"
                                            if entry["grade"]["gate_marks"] else ""))
    if any(entries.get(s, {}).get("new", {}).get("status") not in ("pass", "fail") for s in slots):
        print(f"  stopping before the re-score: not every slot was graded (rerun to retry): {regrade_path}")
        return 1
    changed = [s for s in slots if entries[s]["new"]["status"] != entries[s]["old"]["status"]]
    if args.confirm and not changed:
        state["confirmed"] = {"slots": slots, "unchanged": True,
                              "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        write_json(regrade_path, state)
        log_change(root, {"action": "confirm", "cand": cand.name, "task_id": task_id, "slots": slots,
                          "unchanged": True, "path": str(regrade_path), **extra})
        print(f"  confirmed: every grade is unchanged under the current tests; no re-score ({regrade_path})")
        return 0
    replacements = {}
    for slot in (changed if args.confirm else slots):  # --confirm: the unchanged passes keep their probe records
        new = entries[slot]["new"]
        info = {"tool": "tools/regrade.py", "test_sh_sha256": test_sha, "out_dir": entries[slot]["grade"]["out_dir"],
                "old_status": entries[slot]["old"]["status"], "old_verifier_error": entries[slot]["old"]["verifier_error"]}
        replacements[slot] = dict(by_slot[slot], **new, regrade=info)
    corrected = replace_records(records, replacements)
    if args.no_score:
        print(f"  --no-score: graded only ({regrade_path})")
        return 0
    if not hub_budget_ok():
        print("  stopping: the hub budget guard did not pass (exit 3 = below $500 left, 2 = unknown)")
        return 1
    moved = []
    if aside.exists():  # an earlier re-score stopped inside score(): the originals are aside, its partial outputs are
        print("  resuming an interrupted re-score (score() rewrites or re-checks what it left)")  # rewritten/re-checked
    else:
        for name in SCORE_OUTPUTS:
            path = scored / name
            if path.exists() or path.is_symlink():
                target = move_aside(path, suffix)
                moved.append(str(target))
                log_change(root, {"action": "move_aside", "cand": cand.name, "path": str(path), "to": str(target),
                                  **extra})
    state["moved_aside"] = state.get("moved_aside", []) + moved
    state["replaced"] = sorted(replacements)
    write_json(regrade_path, state)
    b = make_build(root)
    task = state["task"]
    reuse = {} if args.rejudge else teacher_verdicts(scored, corrected, replacements)
    judged = sum(r["model"] == TEACHER and r["status"] == "pass" for r in corrected) - len(reuse)
    print(f"  re-scoring with task_builder.score() (teacher passes: {len(reuse)} verdict(s) kept from "
          f"{earliest(scored, 'teacher_audit.json').name}, {judged} judged by {JUDGE})")
    real_judge = task_builder.judge_pass

    async def judge_pass(record, instruction, rubric):
        if record["slot"] in reuse:
            return reuse[record["slot"]]
        return await real_judge(record, instruction, rubric)

    task_builder.judge_pass = judge_pass  # score() looks judge_pass up in task_builder at call time
    try:
        metrics = await task_builder.score(b, task, corrected, scored)
    finally:
        task_builder.judge_pass = real_judge
    state["teacher_verdicts"] = {"kept": sorted(reuse), "judged": judged}
    state.update(new_metrics={k: metrics.get(k) for k in ("student_rate", "teacher_clean_rate", "valid",
                                                          "grading_failures", "accept", "reason", "similarity")},
                 rescored_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    write_json(regrade_path, state)
    log_change(root, {"action": "score", "cand": cand.name, "task_id": task_id, "path": str(metrics_path),
                      "old": state["old_metrics"], "new": state["new_metrics"], **extra})
    print(f"  score: student {metrics.get('student_rate')} teacher {metrics.get('teacher_clean_rate')} valid "
          f"{metrics.get('valid')} -> {metrics.get('reason')} accept={metrics.get('accept')}")
    if metrics.get("accept"):
        await task_builder.accept(b, metrics)
        state["accepted"] = True
        write_json(regrade_path, state)
        log_change(root, {"action": "accept", "cand": cand.name, "task_id": task_id,
                          "path": str(accepted_dir(root) / task_id), **extra})
        print(f"  accepted: {accepted_dir(root) / task_id}")
    elif (old_metrics or {}).get("accept"):
        if getattr(args, "unaccept", False):
            moved_out = unaccept(root, task_id, extra)
            state["unaccepted"] = moved_out
            write_json(regrade_path, state)
            print(f"  NO LONGER ACCEPTED: {task_id} left {accepted_dir(root)} (moved to {root / BACKUPS_DIR}/unaccepted/)")
        else:
            print(f"  WARNING: {task_id} was accepted before and is not now; {accepted_dir(root) / task_id} is left as "
                  "is (--unaccept takes it out)")
    if not args.no_bookkeeping:
        refresh_seeds(b, round_dir, root, suffix, extra)
        refresh_summary(root, metrics, suffix, extra)
        print(f"  refreshed {round_dir / 'seeds.json'} and {root / 'summary.json'} (backups *{suffix})")
    return 0


async def _regrade(args):
    root = resolve_root(args.run)
    names = cand_names(args.cand)
    if len(names) != 1:
        raise SystemExit("regrade takes one --cand (slots belong to one task)")
    with locked(root, args.dry_run):
        try:
            return await regrade_cand(root, cand_dir(root, names[0]), args)
        except FixError as error:
            print(f"{names[0]}: NOT REGRADED: {error}")
            return 1
        except Exception as error:  # e.g. a judge or sandbox failure inside score()/accept(): rerunning resumes
            print(f"{names[0]}: FAILED (rerun the same command to resume): {type(error).__name__}: {str(error)[-800:]}")
            return 1


# ============================================================================================ tidy accepted/

def accepted_tasks(root: Path) -> list:
    """The task directories in runs/<run>/accepted/ (every directory but the *.workspace archives)."""
    accepted = accepted_dir(root)
    return sorted(p for p in accepted.iterdir() if p.is_dir() and not p.name.endswith(".workspace")) \
        if accepted.exists() else []


def check_accepted(root: Path) -> list:
    """Problems with runs/<run>/accepted/ for this build: tasks.jsonl rows vs task directories vs workspace archives vs
    summary.json; every accepted task's files equal its packaged cand/task/<id>/ (backups aside), its row equals
    dataset_row(), its provenance.json says accepted."""
    accepted = accepted_dir(root)
    problems = []
    rows = {}
    if (accepted / "tasks.jsonl").exists():
        for line in (accepted / "tasks.jsonl").read_text().splitlines():
            if line:
                row = json.loads(line)
                rows[row["task_id"]] = row
    dirs = {p.name for p in accepted_tasks(root)}
    if set(rows) != dirs:
        problems.append(f"tasks.jsonl rows {sorted(rows)} != task directories {sorted(dirs)}")
    summary = read_json(root / "summary.json") or {}
    listed = [m.get("task_id") for m in summary.get("accepted") or []]
    if sorted(listed) != sorted(rows):
        problems.append(f"summary.json accepted {sorted(listed)} != tasks.jsonl {sorted(rows)}")
    stages = {s["task_id"]: (cand, s) for cand, s in built_cands(root)}
    for task_id in sorted(set(rows) | dirs):
        task = accepted / task_id
        workspace = accepted / f"{task_id}.workspace"
        if not ((workspace / "manifest.json").exists() and (workspace / "workspace.tar.gz").exists()):
            problems.append(f"{task_id}: workspace archive incomplete ({workspace})")
        if task_id not in stages:
            problems.append(f"{task_id}: no built candidate in {root}")
            continue
        cand, stage = stages[task_id]
        packaged = Path(stage["task_dir"])
        names = lambda d: {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()
                           and not p.name.endswith((BACKUP, GATE_BACKUP, ASIDE))}
        ours, theirs = names(task) - {"provenance.json"}, names(packaged)
        if ours != theirs:
            problems.append(f"{task_id}: files differ from {packaged}: {sorted(ours ^ theirs)[:5]}")
        for name in sorted(ours & theirs):
            if (task / name).read_bytes() != (packaged / name).read_bytes():
                problems.append(f"{task_id}: {name} differs from {packaged / name}")
        stray = sorted(str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()
                       and p.name.endswith((BACKUP, GATE_BACKUP, ASIDE)))
        if stray:
            problems.append(f"{task_id}: stray backups {stray}")
        try:
            row = json.loads(json.dumps(rollouts.dataset_row(task)))
        except (OSError, KeyError, ValueError) as error:
            problems.append(f"{task_id}: no dataset row ({type(error).__name__}: {error})")
        else:
            if task_id in rows and rows[task_id] != row:
                problems.append(f"{task_id}: tasks.jsonl row differs from dataset_row()")
        provenance = read_json(task / "provenance.json") or {}
        if not provenance.get("accept"):
            problems.append(f"{task_id}: provenance.json does not say accepted")
        scored = scored_dir(root, cand.name)
        metrics = read_json(scored / "metrics.json") if scored else None
        if not (metrics or {}).get("accept"):
            problems.append(f"{task_id}: its metrics.json does not accept it")
    return problems


def cmd_tidy_accepted(args):
    """Copy every *.pre-early-exit-fix / *.pre-gate-fix file inside accepted/<id>/ to
    <build root>/early-exit-fix-backups/accepted/<id>/ (same relative path), check the copy, delete it from accepted/
    (the published tasks hold no stray files), then check accepted/ (check_accepted)."""
    root = resolve_root(args.run)
    accepted = accepted_dir(root)
    status = 0
    with locked(root, args.dry_run):
        task_dirs = accepted_tasks(root)
        for task in task_dirs:
            for path in sorted(p for p in task.rglob("*") if p.is_file() and p.name.endswith((BACKUP, GATE_BACKUP))):
                target = root / BACKUPS_DIR / "accepted" / path.relative_to(accepted)
                if target.exists() and target.read_bytes() != path.read_bytes():
                    print(f"  NOT MOVED: {target} exists with other content")
                    status = 1
                    continue
                if not args.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(path, target)
                    if target.read_bytes() != path.read_bytes():
                        raise FixError(f"copy of {path} differs")
                    path.unlink()
                log_change(root, {"action": "tidy_accepted", "path": str(path), "to": str(target),
                                  "sha256": sha256(path.read_bytes()) if path.exists() else sha256(target.read_bytes())},
                           args.dry_run)
                print(f"  {'would move' if args.dry_run else 'moved'}: {path} -> {target}")
    problems = check_accepted(root)
    for problem in problems:
        print(f"  accepted/ problem: {problem}")
    if not problems:
        print(f"  accepted/ consistent: {len(task_dirs)} task(s), tasks.jsonl, workspace archives and summary.json agree")
    return status or (1 if problems and not args.dry_run else 0)


# ============================================================================================ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("scan", help="list tasks whose RUN TESTS block can exit early (read-only)")
    sub.add_argument("--run", required=True, help="build root, or a run name with one build")
    sub.add_argument("--json", action="store_true")
    sub = commands.add_parser("repair", help="remove the early exits from every live copy of the test script "
                                             "(--gate: record the guarded build/typecheck as a P2P gate test)")
    sub.add_argument("--run", required=True)
    sub.add_argument("--cand", required=True, nargs="+", help="cand-c04-r1b1 or c04-r1b1 (several, or comma-separated)")
    sub.add_argument("--include-risks", action="store_true",
                     help="also take errexit off commands scan lists as risks (e.g. tolerant report scripts)")
    sub.add_argument("--gate", action="store_true", help="gate mode on an already repaired block (see above)")
    sub.add_argument("--add-typecheck", metavar="CMD",
                     help='gate mode: gate this command too when no guarded step runs a type checker (jest: '
                          '"yarn build:ts"; `yarn build:js` only strips types)')
    sub.add_argument("--show-diff", action="store_true", help="gate mode: print the test.sh diff when writing too")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("validate", help="grade reference / empty / broken (/ type-only with --gate) solutions "
                                               "with the repaired tests")
    sub.add_argument("--run", required=True)
    sub.add_argument("--cand", required=True, nargs="+")
    sub.add_argument("--gate", action="store_true", help="gate mode: four checks, each on the gate test too")
    sub.add_argument("--broken-file", help="solution file that gets the syntax error (default: the source file with "
                                           "the most added lines; a new file only if the patch imports it)")
    sub.add_argument("--typeonly-file", help="gate mode: TypeScript file that gets the type error (default: the TS "
                                             "source file with the most added lines)")
    sub.add_argument("--concurrency", type=int, default=6, help="sandboxes at a time (default 6)")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("regrade", help="grade lost slots with the repaired task, then re-score")
    sub.add_argument("--run", required=True)
    sub.add_argument("--cand", required=True, nargs="+")
    sub.add_argument("--gate", action="store_true", help="gate mode: the gated task; earlier re-score outputs are kept "
                                                         "as *.pre-gate-fix")
    sub.add_argument("--slots", required=True, help="6-0,6-1 (probe attempts/<task index>-<repeat>), sc14-r0,..., or "
                                                     "'passing' (every passing slot, both models)")
    sub.add_argument("--confirm", action="store_true", help="re-score only if a grade changed; then only the changed "
                                                            "slots are replaced")
    sub.add_argument("--unaccept", action="store_true", help="gate mode: a task no longer accepted leaves accepted/ "
                                                             "(moved to <R>/early-exit-fix-backups/unaccepted/)")
    sub.add_argument("--model", default=STUDENT, help=f"model of <ti>-<r> slots (default {STUDENT})")
    sub.add_argument("--probe", help="probe directory (default: the round's probe that holds the task)")
    sub.add_argument("--concurrency", type=int, default=4, help="sandboxes at a time (default 4)")
    sub.add_argument("--force", action="store_true", help="grade slots that are not lost grades / skip the record check")
    sub.add_argument("--no-validate-check", action="store_true", help="do not require a passing validate.json")
    sub.add_argument("--no-score", action="store_true", help="grade and write regrade.json only")
    sub.add_argument("--no-bookkeeping", action="store_true", help="leave seeds.json and summary.json as they are")
    sub.add_argument("--rejudge", action="store_true", help="judge every teacher pass again (default: keep the "
                                                            "verdicts of the unchanged transcripts)")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("restore", help="gate mode: put the pre-gate copies back (a task that cannot pass "
                                              "validation)")
    sub.add_argument("--run", required=True)
    sub.add_argument("--cand", required=True, nargs="+")
    sub.add_argument("--gate", action="store_true", required=True)
    sub.add_argument("--reason", default="", help="logged with the restore")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("tidy-accepted", help="move *.pre-early-exit-fix / *.pre-gate-fix files out of "
                                                    "accepted/<id>/ and check accepted/")
    sub.add_argument("--run", required=True)
    sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "repair":
        if args.add_typecheck and not args.gate:
            parser.error("--add-typecheck needs --gate")
        return cmd_repair_gate(args) if args.gate else cmd_repair(args)
    if args.command == "restore":
        return cmd_restore_gate(args)
    if args.command == "tidy-accepted":
        return cmd_tidy_accepted(args)
    if args.command == "validate":
        if args.typeonly_file and not args.gate:
            parser.error("--typeonly-file needs --gate")
        return asyncio.run(_validate_all(args))
    if args.unaccept and not args.gate:
        parser.error("--unaccept needs --gate")
    return asyncio.run(_regrade(args))


if __name__ == "__main__":
    sys.exit(main())
