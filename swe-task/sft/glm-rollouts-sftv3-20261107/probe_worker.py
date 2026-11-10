"""glm-rollouts-sftv3 (2026-11-07): synth/probe_worker.py for the self-hosted GLM-5.3 campaigns, with in-process overrides
only (the shared pipeline files -- synth/settings.json, synth/common.py, synth/probe_worker.py, S/atp/.env, S/secrets,
ATP's scripts -- are NOT edited). driver.py points rollouts.probe() here (rollouts.SYNTH = this directory) and passes
NAMES only in the environment:

1. No hub key: every model of the campaign must be a local-endpoint spec (own non-hub base_url, api_key "EMPTY"), and
   common.api_keys is rebound to a placeholder, so ATP's single "existing" credential (NVIDIA_API_KEY in the staged
   environment) is never the hub key (as glm53-serve/e2e/probe_worker.py).
2. System prompt: OpenCode's stock prompt, unchanged (GR_PROMPT must be "stock"): deepswe_stage.build_config is wrapped
   only to CHECK that the launch's gym.yaml gives the OpenCode agent no opencode_config (as kimi-rollouts-v4).
3. Sandbox credential (GR_SANDBOX=hengyuf | baihe; kimi-rollouts-v4's mechanism): baihe campaigns stage
   OPENSANDBOX_API_KEY from secrets/baihe-opensandbox.env (BAIHE_SANDBOX_API_KEY, make_baihe_secrets.py); hengyuf
   campaigns keep S/atp/.env's CELL3_SANDBOX_API_KEY.
The module search path is made identical to running synth/probe_worker.py directly. Appends one line per ATP worker
start to <campaign>/gr-legs.jsonl.
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
STOCK_PROMPT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028/prompt/"
                    "opencode-1.17.11-default.txt")
STOCK_PROMPT_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
SANDBOXES = {"hengyuf": None, "baihe": HERE / "secrets" / "baihe-opensandbox.env"}

sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

SANDBOX = os.environ.get("GR_SANDBOX", "")
PROMPT = os.environ.get("GR_PROMPT", "")
if PROMPT != "stock":
    raise SystemExit(f"glm-rollouts-sftv3 runs OpenCode's stock prompt only: GR_PROMPT must be 'stock', got {PROMPT!r}")
if SANDBOX not in SANDBOXES:
    raise SystemExit(f"GR_SANDBOX must be one of {sorted(SANDBOXES)}, got {SANDBOX!r}")
if hashlib.sha256(STOCK_PROMPT.read_bytes()).hexdigest() != STOCK_PROMPT_SHA256:
    raise SystemExit(f"{STOCK_PROMPT} changed")

campaign_dir = Path(sys.argv[1]).resolve()
settings = json.loads((campaign_dir / "settings.json").read_text())
for alias, spec in settings["models"].items():
    base_url = str(spec.get("base_url") or "")
    if (not base_url or spec.get("api_key") != "EMPTY" or "nvidia.com" in base_url
            or base_url.rstrip("/") == settings["base_url"].rstrip("/")):
        raise SystemExit(f"{campaign_dir}: model {alias!r} is not a local-endpoint spec; this wrapper is for local models")
common.api_keys = lambda: ["placeholder-not-a-hub-key"]

import deepswe_stage  # noqa: E402  (the job-local staged copy: PYTHONPATH=<scratch>/atp/scripts, as for probe_worker)

_build_config = deepswe_stage.build_config


def build_config(staged, campaign, launch, model_alias, dataset_path, *, concurrency=1):
    """ATP's build_config, unchanged; refuses a gym.yaml that would override the OpenCode agent's prompt."""
    path = _build_config(staged, campaign, launch, model_alias, dataset_path, concurrency=concurrency)
    config = json.loads(Path(path).read_text())
    agent = config[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
    if "opencode_config" in agent:
        raise RuntimeError("build_config now sets opencode_config (a prompt override?); glm-rollouts-sftv3 needs the "
                           "stock OpenCode prompt -- review before running")
    return path


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

# 4. ATP's capture layer fsyncs every OpenCode output chunk to Lustre inside the Gym event loop; a stalled fsync made
#    aiohttp drop every PTY websocket of the process (SandboxPtyError bursts, 2026-11-07). capture_patch.py moves that
#    write + fsync to a worker thread in this job's STAGED copy only (scratch/scripts/deepswe_capture.py).
import importlib.util as _ilu  # noqa: E402

_cp_spec = _ilu.spec_from_file_location("gr_capture_patch", HERE / "capture_patch.py")
CAPTURE_PATCH = _ilu.module_from_spec(_cp_spec)
_cp_spec.loader.exec_module(CAPTURE_PATCH)
_stage_inner = deepswe_stage.stage


def _stage_and_patch(app, scratch):
    staged = _stage_inner(app, scratch)
    roots = {Path(str(staged.get("scratch") or "")), Path(str(scratch))}
    targets = [root / "scripts" / "deepswe_capture.py" for root in roots if str(root) not in ("", ".")]
    targets = [t for t in targets if t.is_file()]
    if not targets:
        raise RuntimeError(f"staged deepswe_capture.py not found under {sorted(map(str, roots))}")
    for target in targets:
        outcome = CAPTURE_PATCH.patch_file(target)
        print(f"glm-rollouts-sftv3: capture patch {outcome}: {target}", file=sys.stderr, flush=True)
    return staged


deepswe_stage.stage = _stage_and_patch

spec = settings["models"].get("glmlocal") or {}
with open(campaign_dir / "gr-legs.jsonl", "a") as stream:
    stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                             "sandbox": SANDBOX,
                             "sandbox_credential": "BAIHE_SANDBOX_API_KEY" if SANDBOX_FILE else "CELL3_SANDBOX_API_KEY",
                             "prompt": PROMPT, "system_prompt_sha256": STOCK_PROMPT_SHA256,
                             "base_url": os.environ.get("GLM53_BASE_URL"), "context_window": spec.get("context_window"),
                             "max_output_tokens": spec.get("max_output_tokens")}) + "\n")
print(f"glm-rollouts-sftv3 probe_worker wrapper: local models {sorted(settings['models'])}, no hub key, sandbox {SANDBOX}, "
      f"prompt {PROMPT} ({STOCK_PROMPT_SHA256[:12]})", file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
