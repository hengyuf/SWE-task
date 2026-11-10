"""TEMPLATE, NOT RUN (2026-11-06, swe-task-harness subagent): synth/probe_worker.py for LOCAL-endpoint campaigns only,
with a placeholder instead of the NVIDIA hub key.

glm_local_driver.py points rollouts.probe() here (rollouts.SYNTH = this directory), as teacher-rollouts-v2 and
teacher-deepswe point it at their key-by-name wrappers. synth/probe_worker.py does `from common import ATP, api_keys`
and puts api_keys()[0] into the staged environment as NVIDIA_API_KEY (ATP's single "existing" credential); rebinding
common.api_keys before running it unchanged means the Gym / OpenCode processes of this campaign never hold the hub key.
(The key still sits in S/atp/.env, which deepswe_stage.runtime_env reads for its own uv / gym-prefetch subprocesses;
those never call a model.) The wrapper refuses a campaign that has any model without its own non-hub base_url and
api_key "EMPTY". The module search path is made identical to running synth/probe_worker.py directly.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import runpy  # noqa: E402

HERE = Path(__file__).resolve().parent
SYNTH = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth"
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

campaign = Path(sys.argv[1]).resolve()
settings = json.loads((campaign / "settings.json").read_text())
for alias, spec in settings["models"].items():
    base_url = str(spec.get("base_url") or "")
    if (not base_url or spec.get("api_key") != "EMPTY" or "nvidia.com" in base_url
            or base_url.rstrip("/") == settings["base_url"].rstrip("/")):
        raise SystemExit(f"{campaign}: model {alias!r} is not a local-endpoint spec; this wrapper is for local models")
common.api_keys = lambda: ["placeholder-not-a-hub-key"]
print(f"local-model probe_worker wrapper: models {sorted(settings['models'])}, no hub key", file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
