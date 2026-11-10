"""Phase-0 smoke checks (run inside a CPU job): python3 -B smoke.py --run R s1 s3 s4   (s2 = probe, ~1-3 h)

S1 snapshot semantics + real-grader golden/empty checks; S2 ATP probe glue (image- and snapshot-backed task;
student sc14 via the student router + teacher kimi); S3 codex drives the sandbox wrappers (Cursor dropped: user
decision, codex for every stage); S4 chat for sc14 (student router) and kimi/glm (hub), plus a long GLM input.
"""
from common import DEEPSWE, JUDGE, RUNS, SETTINGS, STUDENT, TEACHER, read_json, task_toml, write_json

import argparse
import asyncio
import json
from pathlib import Path
import re
import shutil

from llm_engine import agent, chat
from rollouts import probe, render, target_records
from sandbox import grade, owned, write_wrappers

TASK = "abs-stepped-slices"
PROBE = ("pwd; command -v go go-ctrf-json-reporter; git -C /app rev-parse HEAD; "
         "cat /root/synth-marker 2>/dev/null || echo NO-MARKER; echo ===ENV===; env | sort")
VOLATILE = re.compile(r"^(HOSTNAME|KUBERNETES_|OPENSANDBOX_ID=|HOME=|PWD=|OLDPWD=|SHLVL=|_=)")


def report(results: dict, name: str, ok: bool, detail) -> None:
    results[name] = {"ok": ok, "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(detail)[:600], flush=True)


async def s1(run: str, results: dict) -> None:
    task_dir = DEEPSWE / TASK
    image = task_toml(task_dir)["environment"]["docker_image"]
    out = RUNS / run / "smoke/s1"
    async with owned(image, run=run, purpose="smoke-s1") as box:
        before = (await box.sh(PROBE, cwd=None)).stdout
        await box.sh("echo synth-marker > /root/synth-marker", cwd=None)
        snapshot = await box.snapshot(f"{run}-smoke-s1")
    write_json(out / "snapshot.json", {"snapshot": snapshot, "image": image})
    async with owned(snapshot, run=run, purpose="smoke-s1-restore") as box:
        after = (await box.sh(PROBE, cwd=None)).stdout
    (out / "before.txt").write_text(before)
    (out / "after.txt").write_text(after)
    head_before, head_after = before.split("===ENV===")[0].splitlines(), after.split("===ENV===")[0].splitlines()
    env_before = {l for l in before.split("===ENV===")[1].splitlines() if not VOLATILE.match(l)}
    env_after = {l for l in after.split("===ENV===")[1].splitlines() if not VOLATILE.match(l)}
    report(results, "s1_default_cwd_is_app", head_before[0] == head_after[0] == "/app",
           {"image": head_before[0], "snapshot": head_after[0]})
    report(results, "s1_tools_and_head_survive", head_before[1:-1] == head_after[1:-1] and len(head_after) == 5,
           head_after[1:-1])
    report(results, "s1_marker_survives", head_before[-1] == "NO-MARKER" and head_after[-1] == "synth-marker",
           head_after[-1])
    report(results, "s1_env_survives", env_before == env_after,
           {"only_image": sorted(env_before - env_after)[:10], "only_snapshot": sorted(env_after - env_before)[:10]})
    solution = (task_dir / "solution/solution.patch").read_text()
    for label, source in (("image", image), ("snapshot", snapshot)):
        golden, empty = await asyncio.gather(
            grade(source, task_dir, solution, run=run, out_dir=out / f"grade-{label}-golden"),
            grade(source, task_dir, "", run=run, out_dir=out / f"grade-{label}-empty"))
        empty_json = empty["reward_json"] or {}
        report(results, f"s1_grade_{label}", golden["reward"] == 1 and empty["reward"] == 0
               and empty_json.get("p2p_passed") == empty_json.get("p2p_total"),
               {"golden": golden["reward_json"], "empty": empty_json})


async def s2(run: str, results: dict) -> None:
    snapshot = read_json(RUNS / run / "smoke/s1/snapshot.json")
    out = RUNS / run / "smoke/s2"
    task_dirs = []
    for name, image in (("smoke-abs-image", snapshot["image"]), ("smoke-abs-snapshot", snapshot["snapshot"])):
        task_dir = out / "tasks" / name
        if not task_dir.exists():
            shutil.copytree(DEEPSWE / TASK, task_dir)
            text = (task_dir / "task.toml").read_text()
            text = re.sub(r'(?m)^task_id = ".*"$', f'task_id = "{name}"', text)
            text = re.sub(r'(?m)^docker_image = ".*"$', f'docker_image = "{image}"', text)
            (task_dir / "task.toml").write_text(text)
        task_dirs.append(task_dir)
    records = await probe(out / "probe", task_dirs, repeats=1)
    for name, recs in records.items():
        for record in recs:
            attempt = Path(record["attempt_dir"]) if record["attempt_dir"] else None
            journal = (attempt / "http.jsonl").read_text() if attempt and (attempt / "http.jsonl").exists() else ""
            calls = sum(item.get("type") == "function_call" for item in record["output"])
            # source_payload = ATP's deepswe_http rewrote a request: kimi's upstream SSE, or canonicalize_reasoning of
            # Gym's two reasoning aliases. A local student keeps its thinking in the assistant content (the eval's
            # preserve_reasoning_in_assistant_content, set by atp build_config), so nothing is rewritten for it: the
            # check is informational for local models and gates the hosted ones as in Baihe's smoke.
            checks = {"graded": record["status"] in ("pass", "fail"), "tool_calls": calls,
                      "source_payload": "source_payload" in journal,
                      "cleanup": bool(attempt and (attempt / "cleanup-confirmed.json").exists())}
            if record["model"] == "kimi":
                checks["kimi_stream"] = "upstream_stream_chunk" in journal  # ATP's upstream SSE path for Kimi
            local = SETTINGS["models"].get(record["model"], {}).get("local", False)
            ok = checks["graded"] and calls >= 3 and (checks["source_payload"] or local) and checks.get("kimi_stream", True)
            report(results, f"s2_{name}_{record['slot']}", ok, {**checks, "status": record["status"],
                                                                 "reward": record["reward"]})


async def s3(run: str, results: dict) -> None:
    image = task_toml(DEEPSWE / TASK)["environment"]["docker_image"]
    out = RUNS / run / "smoke/s3"
    async with owned(image, run=run, purpose="smoke-s3") as box:
        head = (await box.sh("git rev-parse HEAD")).stdout.strip()
        for backend in ("codex",):  # Cursor dropped by user decision (2026-10-16): codex for every stage
            workdir = out / backend
            write_wrappers(workdir / "bin", box.sid)
            prompt = ("Run exactly this command from your working directory: bin/osb-exec 'git rev-parse HEAD'\n"
                      "Then write result.json in your working directory containing "
                      '{"status": "ok", "head": "<the commit hash it printed>"} and stop. Do nothing else.')
            try:
                result = await agent(prompt, workdir, backend=backend, timeout=1800,
                                     check=lambda r: [] if r.get("head") == head else [f"head {r.get('head')!r}"])
                report(results, f"s3_{backend}", True, result)
            except Exception as error:
                report(results, f"s3_{backend}", False, f"{type(error).__name__}: {error}")


async def s4(run: str, results: dict) -> None:
    for model in (STUDENT, TEACHER, JUDGE):  # STUDENT: this job's router (waits for a worker, $SYNTH_STUDENT_WAIT_S)
        try:
            reply = await chat(model, [{"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"ok": true}'}],
                               json_mode=True)
            report(results, f"s4_chat_{model}", isinstance(reply, dict) and reply.get("ok") is True, reply)
        except Exception as error:
            report(results, f"s4_chat_{model}", False, f"{type(error).__name__}: {error}")
    transcript = render(target_records(TASK)[0])  # the first sc14 slot
    try:
        reply = await chat(JUDGE, [{"role": "user", "content": transcript + "\n\nReply with JSON only: "
                                    '{"tool_calls": <number of "call" sections above>, "outcome": "<one sentence>"}'}],
                           json_mode=True)
        report(results, "s4_glm_long_input", isinstance(reply, dict) and "tool_calls" in reply,
               {"input_chars": len(transcript), "reply": reply})
    except Exception as error:
        report(results, "s4_glm_long_input", False, f"{type(error).__name__}: {error}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("checks", nargs="+", choices=("s1", "s2", "s3", "s4"))
    args = parser.parse_args()
    results = {}
    for check in args.checks:
        try:
            await {"s1": s1, "s2": s2, "s3": s3, "s4": s4}[check](args.run, results)
        except Exception as error:
            report(results, check, False, f"{type(error).__name__}: {error}")
    path = RUNS / args.run / "smoke" / f"summary-{'-'.join(args.checks)}.json"
    write_json(path, results)
    failed = [name for name, value in results.items() if not value["ok"]]
    print(f"{len(results) - len(failed)}/{len(results)} passed; summary {path}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
