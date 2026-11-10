#!/usr/bin/env python3
"""Queue-manager poll (read-only on the runs; updates only log offsets in state.json, atomically).

    /usr/bin/python3 -I -B qpoll.py [--runs a,b] [--no-advance] [--max-lines 40]

For every queue run that has runs/<run>/launch.jsonl: newest job (launch.jsonl last line), its squeue state (or sacct
state once it left the queue), summary.json under build/<task>/*/, a stage digest (analysis, env, understanding, plan,
candidates, probes, scores, seeds), new job-log lines since the last poll (minus 'sandbox ... up' and 'agent start'
noise) and alert counts (budget/422, verifier, PtyError/OOM, redeclared, ReadTimeout, tracebacks, incomplete probes).
"""
import argparse
import json
import os
import pwd
import re
import subprocess
import sys
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
RUNS = S / "synth/runs"
sys.path.insert(0, str(S / "synth/queue"))
import qstate  # noqa: E402

HOME = pwd.getpwuid(os.getuid()).pw_dir
ENV = {"HOME": HOME, "USER": "hengyuf", "LANG": "C.UTF-8",
       "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf",
       "PATH": "/cm/shared/apps/slurm/current/bin:/usr/bin:/bin"}
NOISE = re.compile(r"^\S+ (sandbox \S+ up \(|agent\[codex\] start )")
ALERTS = {
    "BUDGET/422": re.compile(r"budget_exceeded|\b422\b|Max budget"),
    "VERIFIER": re.compile(r"Verifier did not produce|verifier timed out|verifier timeout", re.I),
    "PTY/OOM": re.compile(r"PtyError|OOMKilled|out of memory|\bOOM\b"),
    "REDECLARED": re.compile(r"redeclared in this block"),
    "READTIMEOUT": re.compile(r"ReadTimeout"),
    "TRACEBACK": re.compile(r"Traceback \(most recent call last\)"),
    "INCOMPLETE": re.compile(r"incomplete", re.I),
    "FAILED": re.compile(r"\bFAILED\b"),
    "GATES": re.compile(r" gates: | attempt \d failed: "),
    "LEAK": re.compile(r"released copy|dependency tree|leak", re.I),
}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=90).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"ERR {type(error).__name__}"


def job_state(job):
    state = run(["squeue", "-h", "-j", job, "-o", "%T|%M|%P|%R"])
    if state and not state.startswith("ERR"):
        return "squeue:" + state, True
    acct = run(["sacct", "-n", "-P", "-X", "-j", job, "-o", "State,ExitCode,Elapsed,End"])
    return "sacct:" + (acct.splitlines()[0] if acct else "?"), False


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def short(text, run_name, task):
    text = re.sub(r"/lustre/fsw/portfolios/nemotron/(projects/nemotron_agents_dev/)?users/hengyuf/swe-task/synth/runs/"
                  + re.escape(run_name) + r"/build/" + re.escape(task) + r"/[^/ ]+/", "", text)
    text = re.sub(r"/lustre/fsw/portfolios/nemotron/(projects/nemotron_agents_dev/)?users/hengyuf/swe-task/", "S/", text)
    return text


def terminal_counts(probe: Path):
    out = {}
    for model_dir in sorted((probe / "models").glob("*")):
        slots = list((model_dir / "attempts").glob("*"))
        done = sum(1 for slot in slots if any(slot.glob("attempt-*/terminal.json")))
        retries = sum(max(0, len(list(slot.glob("attempt-*"))) - 1) for slot in slots)
        out[model_dir.name] = f"{done}/{len(slots)} terminal" + (f", {retries} retried" if retries else "")
    return out


def grep_count(path: Path, patterns):
    counts = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return counts
    for name, pattern in patterns.items():
        hits = len(pattern.findall(text))
        if hits:
            counts[name] = hits
    return counts


def cand_digest(cand: Path):
    stage = read_json(cand / "stage.json")
    if stage:
        if stage.get("ok"):
            built = read_json(cand / "built.json") or {}
            return f"OK F2P {stage.get('f2p')}/P2P {stage.get('p2p')} att {built.get('attempt')} {stage.get('task_id')}"
        return "FAILED: " + str(stage.get("problems"))[:260].replace("\n", " | ")
    parts = []
    authors = sorted(p.name for p in cand.glob("author-*"))
    if (cand / "evolve").exists() and not (cand / "evolve/stage.json").exists():
        parts.append("evolve-plan")
    if authors:
        parts.append(authors[-1])
    gates = sorted(p.name for p in cand.glob("gates-*.json"))
    if gates:
        parts.append(",".join(gates))
    if (cand / "built.json").exists():
        parts.append("built->instruction/package")
    return "in progress: " + (" ".join(parts) or "started")


QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\\\]|\\\\.)*\"")
EXIT_WORD = re.compile(r"(^|[;&|{(]|\bthen|\belse|\bdo)\s*exit\b")
CD_GUARD = re.compile(r"^\s*cd\s+\S+\s*\|\|\s*exit\b")
SET_E = re.compile(r"^\s*set\s+-[a-z]*e")
SET_PLUS_E = re.compile(r"^\s*set\s+\+[a-z]*e")
TIMEOUT = re.compile(r"(?<![-\w])timeout\s+(?:-\S+\s+)*(\d+)([smh]?)\b")
REQUIRE_CMD = re.compile(r"\brequire_cmd\b")
TEST_CMD = re.compile(r"\b(pytest|go\s+(test|build|vet)|jest|vitest|mocha|cargo\s+(test|build)|bun\s+(test|run)|"
                      r"(npm|yarn|pnpm)\s+(run\s+)?\w*(test|build)|tsc|make|node\s)")


def author_checks(root: Path):
    """RUN TESTS blocks (author-*/run_tests.sh): bash-level `exit` (other than `cd X || exit` and DeepSWE's
    `require_cmd` helper), commands that run while `set -e` is active (the frame runs the block without errexit; DeepSWE
    blocks end with a bare `set -e`), and `timeout N` wrappers shorter than the verifier's 1800 s (go test's own
    -timeout flag is fine)."""
    issues = []
    for script in sorted(root.glob("cand-*/author-*/run_tests.sh")):
        flagged, errexit, errexit_from, early = [], False, None, None
        for number, line in enumerate(script.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            code = QUOTED.sub("''", line).split(" #", 1)[0]
            if not code.strip():
                continue
            if EXIT_WORD.search(code) and not CD_GUARD.search(code) and not REQUIRE_CMD.search(code):
                flagged.append(f"L{number} exit: {line.strip()[:90]}")
            if SET_E.search(code):
                errexit, errexit_from = True, number
                continue
            if SET_PLUS_E.search(code):
                errexit = False
                continue
            if errexit and early is None and TEST_CMD.search(code):
                early = (errexit_from, number)
            for value, unit in TIMEOUT.findall(code):
                seconds = int(value) * {"": 1, "s": 1, "m": 60, "h": 3600}[unit]
                if seconds < 1800:
                    flagged.append(f"L{number} timeout {value}{unit}: {line.strip()[:70]}")
        if early:
            flagged.append(f"set -e at L{early[0]} active when a build/test command runs at L{early[1]}")
        if flagged:
            issues.append(f"  RUN-TESTS {script.parent.parent.name}/{script.parent.name}: " + " || ".join(flagged[:4]))
    return issues


def round_digest(root: Path, name: str):
    lines = []
    rdir = root / name
    if not rdir.exists():
        return lines
    for probe in sorted(rdir.glob("probe-*")):
        counts = terminal_counts(probe)
        alerts = grep_count(probe / "worker.log", {k: ALERTS[k] for k in ("BUDGET/422", "VERIFIER", "PTY/OOM")})
        lines.append(f"  {name}/{probe.name}: {counts} worker.log alerts {alerts or '-'}")
    for scored in sorted(p for p in rdir.glob("cand-*") if p.is_dir()):
        metrics = read_json(scored / "metrics.json")
        if metrics:
            lines.append(f"  {name}/{scored.name}: sc14 {metrics.get('student_rate')} kimi {metrics.get('teacher_clean_rate')}"
                         f" sim {metrics.get('similarity')} valid {metrics.get('valid')} accept {metrics.get('accept')}"
                         f" gf {len(metrics.get('grading_failures') or [])} uf {bool(metrics.get('uniform_failure'))}"
                         f" cheated {metrics.get('cheated')} | {str(metrics.get('reason'))[:90]}")
    seeds = read_json(rdir / "seeds.json")
    if seeds:
        lines.append(f"  {name}/seeds.json: valid {seeds.get('valid')} invalid {seeds.get('invalid')}")
    return lines


def digest(run_name, task):
    lines = []
    analysis = RUNS / run_name / "analysis" / task
    lines.append(f"  target FA: {'done' if (analysis / 'result.json').exists() else ('running' if analysis.exists() else '-')}")
    roots = [p for p in (RUNS / run_name / "build" / task).glob("*") if p.is_dir()]
    repo, summary = None, False
    for root in roots:
        repo = root.name
        env = read_json(root / "env/stage.json")
        a = read_json(root / "env/a.json")
        lines.append(f"  repo {root.name}: env " + (f"OK sha {env['sha'][:8]} baseline {env.get('baseline_passing')} flaky {len(env.get('flaky') or [])}" if env
                                                    else ("A built, gating" if a else ("building" if (root / 'env').exists() else "-"))))
        lines.append(f"  understanding {'done' if (root / 'understanding/stage.json').exists() else ('running' if (root / 'understanding').exists() else '-')}"
                     f"; plan {'done' if (root / 'plan/mask_plan.md').exists() else ('running' if (root / 'plan').exists() else '-')}")
        for cand in sorted(p for p in root.glob("cand-*") if p.is_dir()):
            lines.append(f"  {cand.name}: {cand_digest(cand)}")
        lines += author_checks(root)
        for rnd in sorted(p.name for p in root.glob("round-*")):
            lines += round_digest(root, rnd)
        if (root / "summary.json").exists():
            summary = True
            s = read_json(root / "summary.json") or {}
            lines.append(f"  SUMMARY: accepted {[m.get('task_id') for m in s.get('accepted', [])]} built {len(s.get('built', []))} rejected {s.get('rejected')}")
    return lines, repo, summary


def budget():
    """Both hub keys by NAME (tools/hub_budget.py never prints key values); recorded in state['global']['budget']."""
    out = {}
    for name in ("NVAPI_KEY_NEW", "NVAPI_KEY_OFFICIAL", "NVAPI_KEY_OFFICIAL_1"):
        text = run(["/usr/bin/python3", "-I", "-B", str(S / "tools/hub_budget.py"), "--key-name", name])
        try:
            data = json.loads(text.splitlines()[-1])
        except (ValueError, IndexError):
            data = {"error": text[-200:]}
        out[name] = data.get("remaining")
        out[name + "_status"] = data.get("status")
        if data.get("error"):
            out[name + "_error"] = str(data.get("error"))[:200]
        out["at"] = data.get("at")
    try:
        out["active_key_file"] = (S / "synth/hub_key_active.txt").read_text().strip()
    except OSError:
        out["active_key_file"] = None
    state = qstate.load()
    state["global"]["budget"] = out
    history = state["global"].setdefault("budget_history", [])
    history.append({"at": out.get("at"), "NEW": out.get("NVAPI_KEY_NEW"), "OFFICIAL": out.get("NVAPI_KEY_OFFICIAL"),
                    "OFFICIAL_1": out.get("NVAPI_KEY_OFFICIAL_1"), "active": out.get("active_key_file")})
    del history[:-60]
    qstate.save(state)
    print("BUDGET " + json.dumps(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", action="store_true", help="also check both hub keys and record them")
    parser.add_argument("--runs")
    parser.add_argument("--no-advance", action="store_true")
    parser.add_argument("--max-lines", type=int, default=40)
    parser.add_argument("--no-digest", action="store_true")
    args = parser.parse_args()
    if args.budget:
        budget()
    state = qstate.load()
    wanted = set(args.runs.split(",")) if args.runs else None
    offsets_update = {}
    for name in state["queue_order"]:
        if wanted and name not in wanted:
            continue
        rec = state["runs"][name]
        launch = RUNS / name / "launch.jsonl"
        if not launch.exists():
            continue
        last = json.loads(launch.read_text().splitlines()[-1])
        job = last["job_id"]
        jstate, queued = job_state(job)
        lines, repo, summary = digest(name, rec["task"]) if not args.no_digest else ([], None, False)
        print(f"== {name} (#{rec['index']} {rec['task']}) status={rec['status']} job {job} {jstate} repo={repo} summary={summary}")
        for line in lines:
            print(short(line, name, rec["task"]))
        log = Path(last["log"])
        offset = int(rec.get("log_offsets", {}).get(job, 0))
        new = ""
        if log.exists():
            with open(log, "rb") as handle:
                handle.seek(offset)
                data = handle.read()
            # only advance to the last complete line
            cut = data.rfind(b"\n") + 1
            new = data[:cut].decode(errors="replace")
            offsets_update[(name, job)] = offset + cut
        kept = [l for l in new.splitlines() if l.strip() and not NOISE.match(l)]
        counts = {k: len([l for l in new.splitlines() if p.search(l)]) for k, p in ALERTS.items()}
        counts = {k: v for k, v in counts.items() if v}
        print(f"  log: {len(new.splitlines())} new lines ({len(kept)} kept) alerts {counts or '-'}")
        for line in kept[-args.max_lines:]:
            print("   | " + short(line, name, rec["task"])[:420])
    if not args.no_advance and offsets_update:
        state = qstate.load()
        for (name, job), value in offsets_update.items():
            state["runs"][name].setdefault("log_offsets", {})[job] = value
        qstate.save(state)


if __name__ == "__main__":
    main()
