"""teacher-rollouts-v2 (2026-10-25): synth/probe_worker.py with the hub key chosen BY NAME for this campaign leg.
Key name: $TR_HUB_KEY_NAME, which driver.py sets for the leg it runs (2026-10-26: --key-names, i.e. NVAPI_KEY_NEW_2
and, once NEW_2 is used up, NVAPI_KEY_NEW_3 -- user, 2026-10-26 ~00:45, for every leg submitted after that). Unset means
NVAPI_KEY_NEW_1, the key of every leg whose driver predates the knob (user decision relayed by main 2026-10-25;
NVAPI_KEY_OFFICIAL_2 stays reserved for the r4 RL judge), so a job that started before this file changed and spawns
this wrapper later keeps its key. The shared pipeline files (synth/hub_key_active.txt, synth/common.py,
synth/probe_worker.py) are NOT edited: this wrapper only rebinds common.hub_key_name in its own process before running
the unchanged synth/probe_worker.py. probe_worker does `from common import api_keys` and api_keys() looks up
hub_key_name() at call time, so ATP's single "existing" credential (NVIDIA_API_KEY in the staged environment, read by
deepswe_credentials.bind_credential) is the named key. driver.py points rollouts.probe() here (rollouts.SYNTH = this
directory; probe() uses SYNTH only for this path). The module search path is made identical to running
synth/probe_worker.py directly (this directory removed, synth first), so nothing in this directory can shadow a module
that probe_worker or ATP imports. Only key NAMES pass through the environment; values stay in the credentials file.
"""
import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from pathlib import Path  # noqa: E402
import runpy  # noqa: E402

HERE = Path(__file__).resolve().parent
SYNTH = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth"
KEY_NAMES = ("NVAPI_KEY_NEW_1", "NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
KEY_NAME = os.environ.pop("TR_HUB_KEY_NAME", "") or "NVAPI_KEY_NEW_1"
if KEY_NAME not in KEY_NAMES:
    raise SystemExit(f"TR_HUB_KEY_NAME {KEY_NAME!r} is not one of {KEY_NAMES}")

sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, SYNTH)
import common  # noqa: E402

common.hub_key_name = lambda: KEY_NAME
if not common.env_value(common.USER_ENV, KEY_NAME):
    raise SystemExit(f"no {KEY_NAME} in {common.USER_ENV}")
print(f"teacher-rollouts probe_worker wrapper: hub key {common.hub_key_name()}", file=sys.stderr, flush=True)
sys.argv[0] = SYNTH + "/probe_worker.py"
runpy.run_path(SYNTH + "/probe_worker.py", run_name="__main__")
