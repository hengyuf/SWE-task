"""kimi-rollouts-v3 (2026-10-26): synth/probe_worker.py with this campaign's in-process overrides. The shared pipeline
files (synth/hub_key_active.txt, synth/settings.json, synth/common.py, synth/probe_worker.py, S/atp/.env, S/secrets/,
ATP's scripts) are NOT edited; this wrapper changes three things in its own process, then runs the unchanged
synth/probe_worker.py. driver.py points rollouts.probe() here (rollouts.SYNTH = this directory) and passes the NAMES
below in the environment (never a secret value):

1. Hub key (KR3_HUB_KEY_NAME = NVAPI_KEY_NEW_2 | NVAPI_KEY_NEW_3, chosen by driver.py per leg): common.hub_key_name is
   rebound, so probe_worker's api_keys() -- ATP's single "existing" credential NVIDIA_API_KEY -- is that key.
2. System prompt: deepswe_stage.build_config (run_deepswe.run_model imports it at every launch) is wrapped so that the
   launch's gym.yaml gives the OpenCode agent `opencode_config.agent.build.prompt` = OpenCode 1.17.11's stock prompt for
   this model (prompt/opencode-1.17.11-default.txt, what SystemPrompt.provider() returns for "dummy_model") + "\n" +
   ONE sentence + "\n" (prompt/build-prompt.txt, sha256 checked). OpenCode uses agent.prompt in place of the provider
   prompt and appends environment / instruction files / skills as before, so the system message the hub receives is
   the previous one with the sentence inserted before "You are powered by the model named ...". Gym merges the key into
   the agent's opencode_config (permission and tools unchanged; checked offline). Nothing else in gym.yaml changes.
3. Sandbox credential (KR3_SANDBOX=baihe only): deepswe_stage.stage is wrapped so that the staged environment's
   OPENSANDBOX_API_KEY comes from secrets/baihe-opensandbox.env (BAIHE_SANDBOX_API_KEY, make_baihe_secrets.py) instead
   of S/atp/.env (CELL3_SANDBOX_API_KEY); the domain must be the same. Every sandbox of the campaign (agent, verifier,
   cleanup) then uses that one credential. KR3_SANDBOX=hengyuf leaves S/atp/.env's credential untouched.
4. Control arm (KR3_PROMPT=stock, campaign ctrl-old; user, 2026-10-26 ~13:40): override 2 is NOT applied, so the
   OpenCode agent keeps its stock system prompt (the previous Kimi configuration); KR3_PROMPT unset = "sentence".
The module search path is made identical to running synth/probe_worker.py directly (this directory removed, synth first),
so nothing here can shadow a module that probe_worker or ATP imports. Appends {job, key, sandbox, prompt} to
<campaign>/kr3-legs.jsonl.
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
BUILD_PROMPT_SHA256 = "727afb569248fd450b6e716dc66e793cfe9f85f1013ee0fe88fe86b42f107d67"
SANDBOXES = {"hengyuf": None, "baihe": HERE / "secrets" / "baihe-opensandbox.env"}

sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

KEY_NAME = os.environ.get("KR3_HUB_KEY_NAME", "")
SANDBOX = os.environ.get("KR3_SANDBOX", "")
PROMPT = os.environ.get("KR3_PROMPT", "sentence")  # "stock" = the control arm ctrl-old: OpenCode's own prompt, unchanged
if PROMPT not in ("sentence", "stock"):
    raise SystemExit(f"KR3_PROMPT must be sentence or stock, got {PROMPT!r}")
if KEY_NAME not in KEYS:
    raise SystemExit(f"KR3_HUB_KEY_NAME must be one of {KEYS}, got {KEY_NAME!r}")
if SANDBOX not in SANDBOXES:
    raise SystemExit(f"KR3_SANDBOX must be one of {sorted(SANDBOXES)}, got {SANDBOX!r}")
common.hub_key_name = lambda: KEY_NAME
if not common.env_value(common.USER_ENV, KEY_NAME):
    raise SystemExit(f"no {KEY_NAME} in {common.USER_ENV}")

BUILD_PROMPT = (HERE / "prompt" / "build-prompt.txt").read_text()
if hashlib.sha256(BUILD_PROMPT.encode()).hexdigest() != BUILD_PROMPT_SHA256:
    raise SystemExit("prompt/build-prompt.txt changed")

import deepswe_stage  # noqa: E402  (the job-local staged copy: PYTHONPATH=<scratch>/atp/scripts, as for probe_worker)

_build_config = deepswe_stage.build_config


def build_config(staged, campaign, launch, model_alias, dataset_path, *, concurrency=1):
    path = _build_config(staged, campaign, launch, model_alias, dataset_path, concurrency=concurrency)
    config = json.loads(Path(path).read_text())
    agent = config[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
    if "opencode_config" in agent:
        raise RuntimeError("build_config now sets opencode_config itself; review the system-prompt override")
    agent["opencode_config"] = {"agent": {"build": {"prompt": BUILD_PROMPT}}}
    temporary = Path(path).with_name("." + Path(path).name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(temporary, path)
    return path


if PROMPT == "sentence":
    deepswe_stage.build_config = build_config

SANDBOX_FILE = SANDBOXES[SANDBOX]
if SANDBOX_FILE is not None:
    def _literal(path: Path) -> dict:
        values = {}
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and key.startswith("OPENSANDBOX_"):
                value = value.strip()
                values[key] = value[1:-1] if value[:1] == "'" and value.endswith("'") else value
        return values

    if (os.stat(SANDBOX_FILE).st_mode & 0o077) or not _literal(SANDBOX_FILE).get("OPENSANDBOX_API_KEY"):
        raise SystemExit(f"{SANDBOX_FILE} must exist with mode 600 and an OPENSANDBOX_API_KEY")
    _stage = deepswe_stage.stage

    def stage(app, scratch):
        staged = _stage(app, scratch)
        values = _literal(SANDBOX_FILE)
        env = dict(staged["env"])
        for name in ("OPENSANDBOX_DOMAIN", "OPENSANDBOX_TLS_VERIFY"):
            if env.get(name) != values.get(name):
                raise RuntimeError(f"{SANDBOX_FILE} {name} differs from the ATP app's; refusing a mixed sandbox setup")
        if env.get("OPENSANDBOX_API_KEY") == values["OPENSANDBOX_API_KEY"]:
            raise RuntimeError("the Baihe sandbox credential equals the ATP app's; refusing")
        env["OPENSANDBOX_API_KEY"] = values["OPENSANDBOX_API_KEY"]
        return dict(staged, env=env)

    deepswe_stage.stage = stage

campaign_dir = Path(sys.argv[1]).resolve()
with open(campaign_dir / "kr3-legs.jsonl", "a") as stream:
    stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                             "hub_key": KEY_NAME, "sandbox": SANDBOX,
                             "sandbox_credential": "BAIHE_SANDBOX_API_KEY" if SANDBOX_FILE else "CELL3_SANDBOX_API_KEY",
                             "prompt": PROMPT,
                             "build_prompt_sha256": BUILD_PROMPT_SHA256 if PROMPT == "sentence" else None}) + "\n")
print(f"kimi-rollouts-v3 probe_worker wrapper: hub key {common.hub_key_name()}, sandbox {SANDBOX}, "
      f"prompt {PROMPT}" + (f" (build prompt {BUILD_PROMPT_SHA256[:12]})" if PROMPT == "sentence" else ""),
      file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
