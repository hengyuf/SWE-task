"""kimi-rollouts-v5 (2026-10-28): synth/probe_worker.py with this batch's in-process overrides (kimi-rollouts-v4's
wrapper without its Baihe sandbox path: every official2 task was built by hengyuf's pipeline). The shared pipeline
files (synth/hub_key_active.txt, synth/settings.json, synth/common.py, synth/probe_worker.py, S/atp/.env, S/secrets/,
ATP's scripts) are NOT edited; this wrapper changes one thing in its own process and checks another, then runs the
unchanged synth/probe_worker.py. driver.py points rollouts.probe() here (rollouts.SYNTH = this directory) and passes
the NAMES below in the environment (never a secret value):

1. Hub key (KR3_HUB_KEY_NAME = NVAPI_KEY_NEW_2 | NVAPI_KEY_NEW_3, chosen by driver.py per leg): common.hub_key_name is
   rebound, so probe_worker's api_keys() -- ATP's single "existing" credential NVIDIA_API_KEY -- is that key.
2. System prompt: NOT changed (KR3_PROMPT must be "stock", as in v4: "prompt 保持不变(不要加reason before action
   之类的prompt)"). deepswe_stage.build_config is wrapped only to CHECK that the launch's gym.yaml gives the OpenCode
   agent no opencode_config (so no agent.build.prompt): OpenCode then uses SystemPrompt.provider() for "dummy_model" =
   prompt/opencode-1.17.11-default.txt (sha256 checked here), followed by its environment / instruction-file parts,
   exactly as for v4. The gym.yaml is not modified.
3. Sandbox credential: KR3_SANDBOX must be "hengyuf": S/atp/.env's credential (CELL3_SANDBOX_API_KEY) is used as staged,
   untouched (no override exists in this wrapper).
The module search path is made identical to running synth/probe_worker.py directly (this directory removed, synth first),
so nothing here can shadow a module that probe_worker or ATP imports. Appends {job, key, sandbox, prompt, context
window} to <campaign>/kr5-legs.jsonl.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
SYNTH = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth"
KEYS = ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
STOCK_PROMPT_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
SANDBOXES = {"hengyuf": "CELL3_SANDBOX_API_KEY"}  # S/atp/.env, as staged by ATP

sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

KEY_NAME = os.environ.get("KR3_HUB_KEY_NAME", "")
SANDBOX = os.environ.get("KR3_SANDBOX", "")
PROMPT = os.environ.get("KR3_PROMPT", "")
if PROMPT != "stock":
    raise SystemExit(f"kimi-rollouts-v5 runs OpenCode's stock prompt only: KR3_PROMPT must be 'stock', got {PROMPT!r}")
if KEY_NAME not in KEYS:
    raise SystemExit(f"KR3_HUB_KEY_NAME must be one of {KEYS}, got {KEY_NAME!r}")
if SANDBOX not in SANDBOXES:
    raise SystemExit(f"KR3_SANDBOX must be one of {sorted(SANDBOXES)}, got {SANDBOX!r}")
common.hub_key_name = lambda: KEY_NAME
if not common.env_value(common.USER_ENV, KEY_NAME):
    raise SystemExit(f"no {KEY_NAME} in {common.USER_ENV}")
if hashlib.sha256((HERE / "prompt" / "opencode-1.17.11-default.txt").read_bytes()).hexdigest() != STOCK_PROMPT_SHA256:
    raise SystemExit("prompt/opencode-1.17.11-default.txt changed")

import deepswe_stage  # noqa: E402  (the job-local staged copy: PYTHONPATH=<scratch>/atp/scripts, as for probe_worker)

_build_config = deepswe_stage.build_config


def build_config(staged, campaign, launch, model_alias, dataset_path, *, concurrency=1):
    """ATP's build_config, unchanged; refuses a gym.yaml that would override the OpenCode agent's prompt."""
    path = _build_config(staged, campaign, launch, model_alias, dataset_path, concurrency=concurrency)
    config = json.loads(Path(path).read_text())
    agent = config[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
    if "opencode_config" in agent:
        raise RuntimeError("build_config now sets opencode_config (a prompt override?); kimi-rollouts-v5 needs the "
                           "stock OpenCode prompt -- review before running")
    return path


deepswe_stage.build_config = build_config

campaign_dir = Path(sys.argv[1]).resolve()
try:
    context_window = json.loads((campaign_dir / "settings.json").read_text())["models"]["kimi"].get("context_window")
except (OSError, ValueError, KeyError):
    context_window = None
with open(campaign_dir / "kr5-legs.jsonl", "a") as stream:
    stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                             "hub_key": KEY_NAME, "sandbox": SANDBOX,
                             "sandbox_credential": SANDBOXES[SANDBOX],
                             "prompt": PROMPT, "system_prompt_sha256": STOCK_PROMPT_SHA256,
                             "context_window": context_window}) + "\n")
print(f"kimi-rollouts-v5 probe_worker wrapper: hub key {common.hub_key_name()}, sandbox {SANDBOX}, prompt {PROMPT} "
      f"(OpenCode stock, {STOCK_PROMPT_SHA256[:12]}), context window {context_window}", file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
