#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28): offline validation of this batch before launch. No rollout, no hub call, no Slurm job,
no write outside analysis/validate-offline.json (and a scratch dir under /tmp, removed at the end). Read-only on
kimi-rollouts-v4 (its campaign settings, frozen ATP source manifest and a launch gym.yaml are the reference).

Checks (each "*_ok" is a bool; "all_ok" is their conjunction):
  task_list_ok        official2_tasks.py (check mode) derives the same 33 tasks / digests as official2-tasks.json,
                      no problems, no withdrawn task
  groups_ok           driver.py --dry-run resolves every group 0-3 (task dirs, frozen digests, rollouts.dataset_row,
                      stock prompt sha, settings.json kimi spec); the groups are disjoint and cover the task list
  settings_ok         each group's would-be campaign settings.json equals v4's sel50-g0 settings.json except
                      expected_tasks (same model spec with context_window 262144, max_output_tokens 131072, OpenCode
                      1.17.11, Gym / dataset pins, concurrency caps 8 / 8, repeats 5)
  atp_source_ok       every ATP file frozen in v4's campaign source (sel50-g0/source/source-manifest.json; scripts,
                      overlays, deepswe.sbatch) has the same sha256 in S/atp now (configs/deepswe.json is the settings
                      dump, compared above), and the pins are equal
  shared_code_ok      synth/{probe_worker.py, rollouts.py, common.py, settings.json} and ATP's Gym.tar were last
                      modified before v4's first leg started (so v5 runs the code v4 runs), Gym.tar's sha256 = its pin
  gym_config_ok       ATP's own build_config (S/atp/scripts/deepswe_stage.py, loaded by path) on group 0's settings
                      gives the same gym.yaml as v4's sel50-g0 launch once hosts, ports and paths are dropped:
                      policy kimi-k3 at the hub, max_tokens 131072, reasoning_effort max, opencode_version 1.17.11,
                      opencode_max_context_window 262144, sandbox_timeout 10800, and no opencode_config (the check
                      probe_worker.py applies: no agent.build.prompt override, i.e. OpenCode's stock prompt). The
                      OpenCode config is built from this agent section by the same staged Gym, so it is v4's:
                      limit.context = limit.input = 262144, limit.output = 131072 (v4 pilot verify_pilot: all OK)
  prompt_ok           prompt/ is byte-identical to v4's; the stock prompt sha256 is 962fbf3c...
  code_vs_v4          (info) per-file sha256 status of the copied code against v4

    <ATP controller python> -I -B validate_offline.py
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
V4 = S / "sft/kimi-rollouts-v4-stock256k-20261028"
CONTROLLER = S / "atp/runtime/controller/bin/python3"
STOCK_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
GROUPS = 4
PER_CAMPAIGN = ("expected_tasks",)  # the only setting that may differ from v4's sel50-g0 (10 tasks there)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_json(argv, cwd=None) -> tuple:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "OPENSANDBOX_"))}
    out = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=900, cwd=cwd, env=env)
    try:
        return out.returncode, json.loads(out.stdout)
    except ValueError:
        return out.returncode, {"stdout_tail": out.stdout[-1500:], "stderr_tail": out.stderr[-1500:]}


def normalize(config: dict) -> dict:
    """A gym.yaml without hosts, ports, paths and per-launch names."""
    config = json.loads(json.dumps(config))
    config["config_paths"] = [p.split("/atp/Gym/", 1)[-1] for p in config["config_paths"]]
    for key in ("head_server", "cache_dir", "results_dir", "uv_cache_dir", "uv_venv_dir", "nemo_gym_log_dir",
                "model_call_capture_dir"):
        config.pop(key, None)
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    for key in ("host", "port"):
        model.pop(key, None)
    resources = config["deepswe_opencode_resources_server"]["resources_servers"]["deepswe"]
    for key in ("host", "port", "tasks_dir", "logs_dir", "expected_task_count"):
        resources.pop(key, None)
    agent = config["deepswe_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
    for key in ("host", "port"):
        agent.pop(key, None)
    for dataset in agent.get("datasets", []):
        dataset.pop("jsonl_fpath", None)
    config["sandbox"]["default_metadata"].pop("nemo-gym.nvidia.com/run", None)
    return config


def diff(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += diff(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        return out
    return [] if a == b else [{"path": path, "v4": a, "v5": b}]


def main() -> int:
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # 1. task list
    code, tasks_check = run_json([CONTROLLER, "-I", "-B", HERE / "official2_tasks.py"])
    frozen = json.loads((HERE / "official2-tasks.json").read_text())
    task_ids = [t["task_id"] for t in frozen["tasks"]]
    report["task_list"] = {"exit": code, "counts": tasks_check.get("counts"), "excluded": tasks_check.get("excluded"),
                           "problems": tasks_check.get("problems"),
                           "frozen_matches_runs": tasks_check.get("frozen_matches_runs"),
                           "sha256": sha256(HERE / "official2-tasks.json")}
    report["task_list_ok"] = code == 0 and tasks_check.get("frozen_matches_runs") is True and \
        len(task_ids) == 33 == len(set(task_ids)) and not tasks_check.get("problems") and not tasks_check.get("excluded")

    # 2. dry runs of every group
    dry = {}
    for group in range(GROUPS):
        code, out = run_json([CONTROLLER, "-B", HERE / "driver.py", "--taskset", "official2", "--group", group,
                              "--max-concurrency", 8, "--dry-run"], cwd=S / "synth")
        dry[group] = {"exit": code, **out}
    covered = [t for group in range(GROUPS) for t in dry[group].get("tasks", [])]
    report["groups"] = {f"official2-g{g}": {"exit": d["exit"], "tasks": len(d.get("tasks", [])), "slots": d.get("slots"),
                                            "sandbox_credential": d.get("sandbox_credential"), "prompt": d.get("prompt"),
                                            "worker": d.get("worker"), "campaign_exists": d.get("campaign_exists")}
                        for g, d in dry.items()}
    report["groups_ok"] = all(d["exit"] == 0 and d.get("dry_run") for d in dry.values()) and \
        sorted(covered) == sorted(task_ids) and len(covered) == len(set(covered)) and \
        all(d.get("sandbox_credential") == "CELL3_SANDBOX_API_KEY" and d.get("prompt") == "stock" and
            d.get("system_prompt_sha256") == STOCK_SHA256 and d.get("repeats") == 5 and
            d.get("worker") == str(HERE / "probe_worker.py") and not d.get("campaign_exists") for d in dry.values())

    # 3. campaign settings vs v4
    v4_settings = json.loads((V4 / "probes/kimi/sel50-g0/settings.json").read_text())
    settings_diffs = {}
    for group, d in dry.items():
        found = d.get("settings") or {}
        settings_diffs[f"official2-g{group}"] = {k: {"v4": v4_settings.get(k), "v5": found.get(k)}
                                                for k in sorted(set(v4_settings) | set(found))
                                                if v4_settings.get(k) != found.get(k)}
    report["settings_vs_v4_sel50_g0"] = settings_diffs
    report["settings_ok"] = all(d.get("settings") for d in dry.values()) and \
        all(set(diffs) <= set(PER_CAMPAIGN) for diffs in settings_diffs.values()) and \
        all(d["settings"]["models"] == {"kimi": {"id": "nvidia/moonshotai/kimi-k3", "context_window": 262144,
                                                 "sampling_overrides": {"reasoning_effort": "max"}}}
            for d in dry.values())

    # 4. ATP source = v4's frozen source
    manifest = json.loads((V4 / "probes/kimi/sel50-g0/source/source-manifest.json").read_text())
    changed = {name: {"v4": digest, "now": sha256(S / "atp" / name) if (S / "atp" / name).exists() else None}
               for name, digest in manifest["files"].items() if name != "configs/deepswe.json"
               and ((S / "atp" / name).exists() is False or sha256(S / "atp" / name) != digest)}
    pins = {key: dry[0]["settings"].get(key) for key in ("dataset_revision", "gym_commit", "opencode_version",
                                                         "python_version")} if dry[0].get("settings") else {}
    report["atp_source"] = {"files_checked": len(manifest["files"]) - 1, "changed": changed,
                            "pins_v4": manifest["pins"], "pins_v5_settings": pins}
    report["atp_source_ok"] = not changed and all(manifest["pins"].get(k) == v for k, v in pins.items())

    # 5. shared synth code and Gym unchanged since v4's first leg
    first_leg = min(json.loads(line)["at"] for line in (V4 / "legs.jsonl").read_text().splitlines() if line.strip())
    first_epoch = time.mktime(time.strptime(first_leg[:19], "%Y-%m-%dT%H:%M:%S"))
    shared = {}
    for path in (S / "synth/probe_worker.py", S / "synth/rollouts.py", S / "synth/common.py", S / "synth/settings.json",
                 S / "atp/vendor/Gym.tar"):
        mtime = path.stat().st_mtime
        shared[str(path)] = {"sha256": sha256(path), "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime)),
                             "before_v4_first_leg": mtime < first_epoch}
    gym_pin = (S / "atp/vendor/Gym.tar.sha256").read_text().split()[0]
    report["shared_code"] = {"v4_first_leg_start": first_leg, "files": shared, "gym_tar_pin": gym_pin}
    report["shared_code_ok"] = all(v["before_v4_first_leg"] for v in shared.values()) and \
        shared[str(S / "atp/vendor/Gym.tar")]["sha256"] == gym_pin

    # 6. gym.yaml from ATP's build_config vs v4's launch
    spec = importlib.util.spec_from_file_location("kr5_deepswe_stage", S / "atp/scripts/deepswe_stage.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    scratch = Path(tempfile.mkdtemp(prefix="kr5-validate-"))
    try:
        campaign = scratch / "official2-g0"
        launch = campaign / "models/kimi/shards/0/launch-offline-validate"
        tls = next((line.split("=", 1)[1].strip().strip("'\"").lower()
                    for line in (S / "atp/.env").read_text().splitlines()
                    if line.strip().removeprefix("export ").startswith("OPENSANDBOX_TLS_VERIFY=")), "true")
        staged = {"settings": dry[0]["settings"], "gym": scratch / "atp/Gym", "scratch": scratch / "atp",
                  "env": {"UV_CACHE_DIR": str(scratch / "atp/cache/uv"), "NEMO_GYM_USER": "hengyuf",
                          "ATP_OS_TLS_VERIFY": tls},  # as stage() derives it from S/atp/.env (only this line is read)
                  "tasks_dir": campaign / "tasks"}
        path = stage.build_config(staged, campaign, launch, "kimi", launch / "tasks.jsonl", concurrency=8)
        v5_config = json.loads(Path(path).read_text())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    v4_launch = sorted((V4 / "probes/kimi/sel50-g0/models/kimi/shards/0").glob("launch-*/gym.yaml"))[0]
    v4_config = json.loads(v4_launch.read_text())
    agent = v5_config[stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
    model = v5_config["policy_model"]["responses_api_models"]["vllm_model"]
    gym_diffs = diff(normalize(v4_config), normalize(v5_config))
    report["gym_config"] = {"v4_reference": str(v4_launch), "differences_after_normalization": gym_diffs,
                            "policy_model_name": v5_config["policy_model_name"],
                            "policy_base_url": v5_config["policy_base_url"],
                            "sampling_overrides": model["sampling_overrides"],
                            "opencode_version": agent["opencode_version"],
                            "opencode_max_context_window": agent["opencode_max_context_window"],
                            "sandbox_timeout": agent["sandbox_timeout"],
                            "opencode_config_present": "opencode_config" in agent,
                            "v4_pilot_opencode_limit": json.loads((V4 / "analysis/verify-pilot-0-0-attempt-0.json")
                                                                  .read_text()).get("opencode_limit"),
                            "v4_pilot_verify_all_ok": json.loads((V4 / "analysis/verify-pilot-0-0-attempt-0.json")
                                                                 .read_text()).get("all_ok")}
    report["gym_config_ok"] = not gym_diffs and "opencode_config" not in agent and \
        agent["opencode_max_context_window"] == 262144 and model["sampling_overrides"].get("max_tokens") == 131072 and \
        model["sampling_overrides"].get("reasoning_effort") == "max" and \
        v5_config["policy_model_name"] == "nvidia/moonshotai/kimi-k3"

    # 7. prompt files
    prompt = {p.name: sha256(p) == sha256(V4 / "prompt" / p.name) for p in sorted((HERE / "prompt").iterdir())}
    report["prompt"] = {"identical_to_v4": prompt, "stock_sha256": sha256(HERE / "prompt/opencode-1.17.11-default.txt")}
    report["prompt_ok"] = all(prompt.values()) and sorted(prompt) == sorted(p.name for p in (V4 / "prompt").iterdir()) \
        and report["prompt"]["stock_sha256"] == STOCK_SHA256

    # 8. code vs v4 (information)
    names = sorted({p.name for p in HERE.glob("*.py")} | {p.name for p in V4.glob("*.py")} |
                   {"driver.sbatch", "submit.sh", "prices.json", "README.md"})
    report["code_vs_v4"] = {name: ("identical" if (HERE / name).exists() and (V4 / name).exists()
                                   and sha256(HERE / name) == sha256(V4 / name)
                                   else "adapted" if (HERE / name).exists() and (V4 / name).exists()
                                   else "new in v5" if (HERE / name).exists() else "dropped in v5")
                            for name in names}
    checks = {k: v for k, v in report.items() if k.endswith("_ok")}
    report["checks"] = checks
    report["all_ok"] = all(checks.values())
    out = HERE / "analysis" / "validate-offline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": checks, "all_ok": report["all_ok"], "written": str(out)}, indent=1))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
