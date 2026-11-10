#!/usr/bin/env python3
"""Run the host-repo reselection dispatcher (codex_prompt.md) headless, in the foreground of the calling process.
hengyuf asked main to launch it in the background on 2026-10-22; main starts this in a detached tmux
session `reselect`. This is the same codex setup as synth/llm_engine.py:
- CODEX_HOME on node-local scratch, seeded from S/tools/codex/home/config.toml;
- the active hub key from common.api_keys() is passed as NVHUB_API_KEY, in the environment only (never on a command
  line or in a log);
- secret-like variables are dropped from the inherited environment.
Codex flags were checked by the 2026-10-22 review: workspace-write with network on, out/ writable, 8 subagents.
Events go to <scratch>/events.jsonl, the dispatcher's last message to <scratch>/dispatcher-last.txt, and stderr to
<scratch>/stderr.log."""
import os
import re
import sys
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
sys.path.insert(0, str(S / "synth"))
import common  # noqa: E402  (stdlib only)

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")  # = llm_engine.SECRET_NAME
SCRATCH = Path(f"/tmp/{os.environ.get('USER') or 'hengyuf'}-reselect")


def main() -> None:
    home = SCRATCH / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    SCRATCH.chmod(0o700)
    home.chmod(0o700)
    config = home / "config.toml"
    config.write_bytes((S / "tools/codex/home/config.toml").read_bytes())
    config.chmod(0o600)
    env = {key: value for key, value in os.environ.items() if not SECRET_NAME.search(key.upper())}
    env.update(CODEX_HOME=str(home), NVHUB_API_KEY=common.api_keys()[0])
    codex = S / "tools/codex/rust-v0.163.0/bin/codex"
    argv = [str(codex), "exec", "--strict-config", "--ephemeral", "--skip-git-repo-check",
            "-s", "workspace-write", "-C", str(S / "repo_reselect/out"),
            "-c", "sandbox_workspace_write.network_access=true",
            "-c", "agents.max_concurrent_threads_per_session=8",
            "--json", "-o", str(SCRATCH / "dispatcher-last.txt"), "-"]
    stdin = os.open(S / "repo_reselect/codex_prompt.md", os.O_RDONLY)
    stdout = os.open(SCRATCH / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    stderr = os.open(SCRATCH / "stderr.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(stdin, 0)
    os.dup2(stdout, 1)
    os.dup2(stderr, 2)
    (SCRATCH / "codex.pid").write_text(f"{os.getpid()}\n")
    os.execve(str(codex), argv, env)


if __name__ == "__main__":
    main()
