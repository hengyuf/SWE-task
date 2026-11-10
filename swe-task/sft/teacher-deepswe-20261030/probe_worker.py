"""teacher-deepswe (2026-10-30): synth/probe_worker.py with this campaign's in-process overrides (kimi-rollouts-v4's
wrapper with the Baihe sandbox path removed and this campaign's hub key only: NVAPI_KEY_NEW_2 for every leg started
from 2026-10-31 18:15 (OPERATOR_NOTE 18:15, user decision relayed by main; the legs of 2026-10-30 21:45 .. 10-31 10:40
ran on NVAPI_KEY_NEW_3 until its floor stop, the three pilot legs of 21:34 on NVAPI_KEY_NEW_2; NEW_3 is no longer
accepted here); driver.py picks the key per leg). The shared pipeline files
(synth/hub_key_active.txt, synth/settings.json, synth/common.py, synth/probe_worker.py, S/atp/.env, S/secrets/, ATP's
scripts) are NOT edited; this wrapper changes one thing in its own process and checks two, then runs the unchanged
synth/probe_worker.py. driver.py points rollouts.probe() here (rollouts.SYNTH = this directory) and passes the NAMES
below in the environment (never a secret value):

1. Hub key (TDW_HUB_KEY_NAME, chosen by driver.py per leg; must be one of KEYS): common.hub_key_name is rebound,
   so probe_worker's api_keys() -- ATP's single "existing" credential NVIDIA_API_KEY -- is that key.
2. System prompt: NOT changed (TDW_PROMPT must be "stock"). deepswe_stage.build_config is wrapped only to CHECK that the
   launch's gym.yaml gives the OpenCode agent no opencode_config (so no agent.build.prompt): OpenCode then uses
   SystemPrompt.provider() for "dummy_model" = prompt/opencode-1.17.11-default.txt (sha256 checked here), followed by its
   environment / instruction-file parts, exactly as KR4's stock-prompt Kimi rollouts. The gym.yaml is not modified.
3. Model (TDW_MODEL): the campaign's settings.json must hold exactly that one model alias (checked; recorded).
The sandbox credential is S/atp/.env's (CELL3_SANDBOX_API_KEY, ATP's own); there is no override path in this copy.
The module search path is made identical to running synth/probe_worker.py directly (this directory removed, synth first),
so nothing here can shadow a module that probe_worker or ATP imports. Appends {job, model, key, sandbox, prompt, context
window} to <campaign>/tdw-legs.jsonl.
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
KEYS = ("NVAPI_KEY_NEW_2",)  # every leg from 2026-10-31 18:15 (OPERATOR_NOTE 18:15); NEW_3 (10-30 21:45 .. its floor) no more
MODELS = ("kimi", "glm", "deepseek")
STOCK_PROMPT_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"

sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

KEY_NAME = os.environ.get("TDW_HUB_KEY_NAME", "")
PROMPT = os.environ.get("TDW_PROMPT", "")
MODEL = os.environ.get("TDW_MODEL", "")
if PROMPT != "stock":
    raise SystemExit(f"teacher-deepswe runs OpenCode's stock prompt only: TDW_PROMPT must be 'stock', got {PROMPT!r}")
if KEY_NAME not in KEYS:
    raise SystemExit(f"TDW_HUB_KEY_NAME must be one of {KEYS}, got {KEY_NAME!r}")
if MODEL not in MODELS:
    raise SystemExit(f"TDW_MODEL must be one of {MODELS}, got {MODEL!r}")
common.hub_key_name = lambda: KEY_NAME
if not common.env_value(common.USER_ENV, KEY_NAME):
    raise SystemExit(f"no {KEY_NAME} in {common.USER_ENV}")
if hashlib.sha256((HERE / "prompt" / "opencode-1.17.11-default.txt").read_bytes()).hexdigest() != STOCK_PROMPT_SHA256:
    raise SystemExit("prompt/opencode-1.17.11-default.txt changed")

campaign_dir = Path(sys.argv[1]).resolve()
models = json.loads((campaign_dir / "settings.json").read_text())["models"]
if list(models) != [MODEL]:
    raise SystemExit(f"{campaign_dir}/settings.json has models {sorted(models)}, expected only {MODEL!r}")
context_window = models[MODEL].get("context_window")

import deepswe_stage  # noqa: E402  (the job-local staged copy: PYTHONPATH=<scratch>/atp/scripts, as for probe_worker)

_build_config = deepswe_stage.build_config


def build_config(staged, campaign, launch, model_alias, dataset_path, *, concurrency=1):
    """ATP's build_config, unchanged; refuses a gym.yaml that would override the OpenCode agent's prompt."""
    path = _build_config(staged, campaign, launch, model_alias, dataset_path, concurrency=concurrency)
    config = json.loads(Path(path).read_text())
    agent = config[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
    if "opencode_config" in agent:
        raise RuntimeError("build_config now sets opencode_config (a prompt override?); teacher-deepswe needs the "
                           "stock OpenCode prompt -- review before running")
    return path


deepswe_stage.build_config = build_config

with open(campaign_dir / "tdw-legs.jsonl", "a") as stream:
    stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                             "model": MODEL, "model_id": models[MODEL].get("id"), "hub_key": KEY_NAME,
                             "sandbox": "hengyuf", "sandbox_credential": "CELL3_SANDBOX_API_KEY",
                             "prompt": PROMPT, "system_prompt_sha256": STOCK_PROMPT_SHA256,
                             "context_window": context_window}) + "\n")
print(f"teacher-deepswe probe_worker wrapper: model {MODEL} ({models[MODEL].get('id')}), hub key "
      f"{common.hub_key_name()}, sandbox hengyuf, prompt {PROMPT} (OpenCode stock, {STOCK_PROMPT_SHA256[:12]}), "
      f"context window {context_window}", file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
