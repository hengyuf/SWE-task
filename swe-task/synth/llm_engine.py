"""LLM engine: chat completions (hosted models on the NVIDIA inference hub; the local student through this job's
student router), and headless coding agents (codex; the cursor command line is kept but no stage uses it).

Student/teacher *rollouts* are not run here: they go through ATP's campaign code (rollouts.probe), which is
the only path that runs the models with the DeepSWE harness fixes (see synth README).
"""
from common import S, SETTINGS, api_keys, log, read_json, scratch, write_json

import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import httpx

CODEX = S / "tools/codex/rust-v0.163.0/bin/codex"  # codex-cli 0.163.0, provider nvhub: gpt-6-astra, xhigh
CODEX_CONFIG = S / "tools/codex/home/config.toml"  # seeds each job's CODEX_HOME (no credentials inside)
CURSOR = Path.home() / ".local/bin/cursor-agent"  # unused since the port (user decision: codex for every stage)
CURSOR_MODEL = "claude-sonnet-5-thinking-xhigh"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520}
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")  # matched against upper-cased variable names


class AgentError(RuntimeError):
    pass


def parse_json(text: str):
    """Parse a JSON object from a model reply (tolerates code fences and surrounding prose)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


async def chat(model: str, messages: list, *, json_mode=False, max_tokens=32768, timeout=1800, retries=6,
               student_wait=None):
    """One chat completion with the pipeline's model settings; returns the reply text (or parsed JSON).

    Hosted models (kimi, glm) use the hub base_url and key. A "local" model (the student, sc14) goes to this job's
    student router without credentials, after waiting up to `student_wait` s (default $SYNTH_STUDENT_WAIT_S or
    3600) for a routable worker; the router applies the student's serving semantics (cap, sampling, clamp).
    """
    spec = SETTINGS["models"][model]
    body = {"model": spec["id"], "messages": messages, "max_tokens": max_tokens,
            "temperature": SETTINGS["temperature"], "top_p": SETTINGS["top_p"],
            **spec.get("sampling_overrides", {})}
    if "chat_template_kwargs" in spec:
        body["chat_template_kwargs"] = spec["chat_template_kwargs"]
    if spec.get("local"):
        from student.client import ensure_router, wait_for_workers
        base_url, keys = await asyncio.to_thread(ensure_router), []  # starts (or reuses) this job's router
        patience = float(student_wait if student_wait is not None else os.environ.get("SYNTH_STUDENT_WAIT_S", 3600))
        await asyncio.to_thread(wait_for_workers, 1, timeout_s=patience)
    else:
        base_url, keys = SETTINGS["base_url"], api_keys()
    url = base_url.rstrip("/") + "/chat/completions"
    # The local router is on loopback: never route it through an environment proxy. Hosted calls are unchanged.
    async with httpx.AsyncClient(timeout=timeout, trust_env=not spec.get("local")) as client:
        for attempt in range(retries + 1):
            wait = min(60, 2 ** attempt) + random.random()
            headers = {"Authorization": "Bearer " + keys[attempt % len(keys)]} if keys else {}  # retries alternate keys
            try:
                response = await client.post(url, json=body, headers=headers)
            except httpx.TransportError as error:
                problem = type(error).__name__
            else:
                if response.status_code == 200:
                    text = response.json()["choices"][0]["message"].get("content") or ""
                    if text.strip():
                        try:
                            return parse_json(text) if json_mode else text
                        except json.JSONDecodeError:
                            problem = "reply is not JSON"
                    else:
                        problem = "empty reply"
                elif response.status_code in RETRY_STATUS:
                    problem = f"HTTP {response.status_code}"
                    if response.headers.get("retry-after", "").isdigit():
                        wait = max(wait, float(response.headers["retry-after"]))
                else:
                    raise RuntimeError(f"chat({model}) HTTP {response.status_code}: {response.text[:300]}")
            if attempt == retries:
                raise RuntimeError(f"chat({model}) failed after {retries} retries: {problem}")
            log(f"chat({model}) retry {attempt + 1}: {problem}")
            await asyncio.sleep(wait)


def agent_argv(backend: str, prompt: str, workdir: Path, add_dirs=(), state_dir: Path | None = None) -> list:
    """The fixed, non-interactive command line for each agent backend."""
    if backend == "codex":
        argv = [str(CODEX), "exec", "--ephemeral", "--skip-git-repo-check", "-s", "workspace-write", "--json",
                "-C", str(workdir)]
        for directory in add_dirs:
            argv += ["--add-dir", str(directory)]
        argv += ["-c", "sandbox_workspace_write.network_access=true",
                 "-c", f'sqlite_home="{state_dir}"', "-c", f'log_dir="{state_dir}"',
                 "-o", str(workdir / ".agent/last-message.txt"), prompt]
    elif backend == "cursor":
        argv = [str(CURSOR), "-p", "--output-format", "json", "--model", CURSOR_MODEL, "--force", "--trust",
                "--workspace", str(workdir)]
        for directory in add_dirs:
            argv += ["--add-dir", str(directory)]
        argv.append(prompt)
    else:
        raise ValueError(f"Unknown agent backend {backend!r}")
    return argv


# Appended to every agent prompt: the operating constraints of this project apply to the agents too.
RULES = """

## Ground rules (mandatory)
- Create, modify or delete files only inside your working directory ({workdir}). Everything else on the shared
  filesystem and in the home directory is read-only for you.
- Never run Slurm commands (sbatch, srun, salloc, scancel, scontrol, ...) and never touch any existing job or process.
- Do not run heavy commands locally on the shared Lustre filesystem: builds, test suites, dependency installs and
  large searches belong in the remote sandbox via bin/osb-exec (when your task provides one).
- Never read, print or copy credentials (.env, .api, auth.json, keys, tokens).
"""


def _codex_home() -> Path:
    """This job's CODEX_HOME on node-local scratch (concurrent codex runs must not share one on Lustre)."""
    home = scratch() / "codex-home"
    home.mkdir(mode=0o700, exist_ok=True)
    home.chmod(0o700)
    config, seeded = CODEX_CONFIG.read_bytes(), home / "config.toml"
    if not seeded.exists() or seeded.read_bytes() != config:
        temporary = home / f".config.toml.{os.getpid()}.tmp"
        temporary.write_bytes(config)
        temporary.chmod(0o600)
        temporary.replace(seeded)
    return home


def _agent_env(workdir: Path, backend: str) -> dict:
    """The agent's environment without any secret-like variable (NVAPI_KEY included) and without
    SYNTH_STUDENT_BASE_URL; codex alone gets its own CODEX_HOME and the hub key as NVHUB_API_KEY (config.toml keeps
    it out of the commands codex runs)."""
    env = {key: value for key, value in os.environ.items() if not SECRET_NAME.search(key.upper())}
    env.pop("SYNTH_STUDENT_BASE_URL", None)  # the student router is the pipeline's, not the agents'
    env["PATH"] = os.pathsep.join([str(workdir / "bin"), str(CODEX.parent), env.get("PATH", "/usr/bin:/bin")])
    if backend == "codex":
        env.update(CODEX_HOME=str(_codex_home()), NVHUB_API_KEY=api_keys()[0])
    return env


def _problems(check, result) -> list:
    if not isinstance(result, dict):
        return ["agent did not write a JSON result.json"]
    try:
        return list(check(result)) if check else []
    except Exception as error:  # a malformed result must fail validation, not wedge the stage
        return [f"result.json failed validation: {type(error).__name__}: {error}"]


async def agent(prompt: str, workdir, *, backend="codex", check=None, timeout=4 * 3600) -> dict:
    """Run one headless agent in `workdir`; it must leave a valid ./result.json (validated by `check`).

    `check(result) -> list[str]` returns problems; an empty list means the stage output is acceptable.
    Every prompt gets the project's ground rules (RULES) appended.
    """
    workdir = Path(workdir).resolve()
    existing = read_json(workdir / "result.json")
    if check is not None and not _problems(check, existing):
        log(f"agent[{backend}] reuse {workdir}")
        return existing
    logs = workdir / ".agent"
    logs.mkdir(parents=True, exist_ok=True)
    state = scratch() / f"agent-state-{backend}"
    state.mkdir(exist_ok=True)
    (workdir / "result.json").unlink(missing_ok=True)
    prompt += RULES.format(workdir=workdir)
    argv = agent_argv(backend, prompt, workdir, (), state)
    (logs / "prompt.md").write_text(prompt)
    write_json(logs / "launch.json", {"backend": backend, "argv": argv[:-1], "started_at": time.time(),
                                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    log(f"agent[{backend}] start {workdir}")
    with open(logs / "events.jsonl", "wb") as stdout, open(logs / "stderr.log", "wb") as stderr:
        process = await asyncio.create_subprocess_exec(*argv, cwd=workdir, env=_agent_env(workdir, backend),
                                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                                       start_new_session=True)
        try:
            exit_code = await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            exit_code = 124
    result = read_json(workdir / "result.json")
    problems = _problems(check, result)
    write_json(logs / "exit.json", {"exit_code": exit_code, "finished_at": time.time(), "problems": problems})
    log(f"agent[{backend}] exit {exit_code} {workdir} problems={len(problems)}")
    if problems:
        raise AgentError(f"{workdir}: " + "; ".join(problems))
    return result
