"""Run one probe campaign with ATP's own campaign code (no ATP edits). Spawned by rollouts.probe().

Runs under the job-local staged controller python with PYTHONPATH=<scratch>/atp/scripts. Mirrors
run_deepswe.worker(), except that the task set, settings and source snapshot belong to the probe campaign.
A local student (sc14) is reached through the parent's router: SYNTH_STUDENT_BASE_URL is inherited.
"""
import sys

sys.dont_write_bytecode = True

import asyncio
import json
import os
from pathlib import Path
import signal
import time

from common import ATP, api_keys  # ATP = S/atp (hengyuf's app copy); api_keys(): NVAPI_KEY_NEW from ~hengyuf/.env
# Lets task.toml docker_image = "snapshot:<id>" start sandboxes from an OpenSandbox snapshot (job-local Gym copy only).
OLD = """        if spec.image is not None:
            kwargs["image"] = _to_image_spec(spec.image, options.image_auth)
"""
NEW = """        if spec.image is not None and spec.image.startswith("snapshot:"):
            kwargs["snapshot_id"] = spec.image.removeprefix("snapshot:")
        elif spec.image is not None:
            kwargs["image"] = _to_image_spec(spec.image, options.image_auth)
"""


def patch_provider(gym: Path) -> None:
    path = gym / "nemo_gym/sandbox/providers/opensandbox/provider.py"
    text = path.read_text()
    if NEW in text:
        return
    if text.count(OLD) != 1:
        raise RuntimeError(f"{path} changed; the snapshot-image patch no longer applies")
    path.write_text(text.replace(OLD, NEW))


async def main(probe_dir: Path) -> int:
    import deepswe_campaign as campaign
    import run_deepswe
    from deepswe_source import freeze_source
    from deepswe_stage import stage

    settings = json.loads((probe_dir / "settings.json").read_text())
    campaign.create_campaign(probe_dir, probe_dir / "rows.jsonl", settings, shards=1)
    source = freeze_source(ATP, probe_dir)
    os.environ.pop("ATP_DEEPSWE_SOURCE", None)  # stage once per job against the app's own pinned settings
    staged = stage(ATP, Path(os.environ["SYNTH_ATP_SCRATCH"]))
    patch_provider(staged["gym"])
    manifest = campaign.load_manifest(probe_dir)
    # Probes use this pipeline's single hub key: ATP binds its "existing" credential (NVIDIA_API_KEY) to every
    # hosted call; no ATP_DEEPSWE_PRIORITY_* pool. The local student gets no credential.
    credentials = {"NVIDIA_API_KEY": api_keys()[0]}
    for name in ("ATP_DEEPSWE_PRIORITY_1_API_KEY", "ATP_DEEPSWE_PRIORITY_2_API_KEY"):
        staged["env"].pop(name, None)
        os.environ.pop(name, None)
    # Historical approvals and attempt records must point at this campaign's frozen source.
    staged = dict(staged, tasks_dir=probe_dir / "tasks", settings=manifest["settings"],
                  env=dict(staged["env"], ATP_DEEPSWE_SOURCE=str(source), **credentials))
    os.environ.update(staged["env"])
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(probe_dir)
    end = run_deepswe.job_end_time(os.environ)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    timer = loop.call_later(max(0, end - time.time() - manifest["settings"]["shutdown_grace_seconds"]), stopped.set)
    try:
        results = await asyncio.gather(*(run_deepswe.run_model(campaign, staged, probe_dir, model, 0, end, stopped)
                                         for model in manifest["settings"]["models"]), return_exceptions=True)
    finally:
        timer.cancel()
    failed = [result for result in results if isinstance(result, BaseException)]
    for error in failed:
        print("probe model failed: " + run_deepswe.redact(str(error)), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(Path(sys.argv[1]).resolve())))
