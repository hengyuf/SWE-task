#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): the per-alias configuration ATP produces for this campaign, from ATP's own code,
without starting a campaign (no Slurm, hub or sandbox call). Writes analysis/config-check/<alias>/.../gym.yaml (ATP's
build_config output for a stand-in launch dir; it holds only ${oc.env:...} placeholders, no credential) and
analysis/config-check.json.

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <TD>/setup/config_check.py)

For each alias (kimi, glm, deepseek):
  1. the campaign settings rollouts.probe() would write (synth/settings.json + the alias's in-memory spec from driver.py,
     expected_tasks / repeats / max_concurrency_per_model of a wave-1 group);
  2. ATP's deepswe_stage.build_config(...) on those settings -> gym.yaml: policy_model_name, sampling_overrides
     (temperature, top_p, max_tokens, reasoning_effort), opencode_max_context_window, sandbox default_metadata;
  3. ATP's run_deepswe.gym_command(...) -> the gym eval CLI flags (temperature, top-p, max-output-tokens);
  4. run_deepswe.run_model's ATP_DEEPSWE_MAX_OUTPUT_TOKENS expression and ATP_DEEPSWE_UPSTREAM_STREAM default;
  5. the patched Gym OpenCode agent (overlays/deepswe-capture.patch on responses_api_agents/opencode_sandboxed_agent/
     app.py): limit.context = limit.input = opencode_max_context_window, limit.output = int(ATP_DEEPSWE_MAX_OUTPUT_TOKENS)
     -- the patch lines are quoted from the file and the limit is evaluated with the values of steps 2 and 4;
  6. cross-check: kimi-rollouts-v4's pilot attempt (same code path, alias kimi) recorded exactly that limit in its
     agent-command.json (OPENCODE_CONFIG_CONTENT), and no "agent" key (no prompt override).
"""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import shlex  # noqa: E402

TD = Path(__file__).resolve().parents[1]
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
ATP = S / "atp"
KR4 = S / "sft/kimi-rollouts-v4-stock256k-20261028"
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != TD / "setup"]
sys.path.insert(0, str(ATP / "scripts"))  # ATP's run_deepswe imports deepswe_http / deepswe_upstream by name


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    driver_spec = importlib.util.spec_from_file_location("tdw_driver_for_check", TD / "driver.py")
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)  # module level only: constants, synth imports (no main())
    synth_settings = json.loads((S / "synth" / "settings.json").read_text())
    stage = load("atp_deepswe_stage_check", ATP / "scripts" / "deepswe_stage.py")
    run = load("atp_run_deepswe_check", ATP / "scripts" / "run_deepswe.py")
    patch = (ATP / "overlays" / "deepswe-capture.patch").read_text().splitlines()
    patch_lines = [line for line in patch if "opencode_max_context_window" in line or "ATP_DEEPSWE_MAX_OUTPUT_TOKENS" in line]
    out = {"atp_scripts": str(ATP / "scripts"), "patch_lines": patch_lines, "aliases": {}}
    for alias, base in driver.BASE_SPECS.items():
        spec = dict(base, context_window=driver.CONTEXT_WINDOW)
        settings = dict(synth_settings, expected_tasks=27, repeats=5, max_concurrency_per_model=8,
                        initial_concurrency_per_model=8, models={alias: spec})
        campaign = TD / "analysis" / "config-check" / alias / "campaign"
        launch = campaign / "models" / alias / "shards" / "0" / "launch-0-configcheck"
        staged = {"settings": settings, "gym": Path("/nonexistent/gym"), "scratch": Path("/nonexistent/scratch"),
                  "tasks_dir": Path("/nonexistent/tasks"), "gym_cli": Path("/nonexistent/gym-cli"),
                  "env": {"UV_CACHE_DIR": "/nonexistent/uv", "NEMO_GYM_USER": "hengyuf", "ATP_OS_TLS_VERIFY": "true"}}
        path = stage.build_config(staged, campaign, launch, alias, Path("/nonexistent/tasks.jsonl"), concurrency=8)
        config = json.loads(Path(path).read_text())
        model = config["policy_model"]["responses_api_models"]["vllm_model"]
        agent = config[stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
        command = run.gym_command(staged, path, launch, settings, 8, alias)
        flags = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i] in
                 ("--temperature", "--top-p", "--max-output-tokens", "--num-repeats", "--concurrency")}
        max_output_env = str(settings["models"][alias].get("max_output_tokens", settings["max_output_tokens"]))
        stream_default = "1" if alias == "kimi" else "0"  # run_model: environment.setdefault(ATP_DEEPSWE_UPSTREAM_STREAM, ...)
        limit = {"context": agent["opencode_max_context_window"], "input": agent["opencode_max_context_window"],
                 "output": int(max_output_env)}
        out["aliases"][alias] = {
            "spec": spec, "gym_yaml": str(path), "policy_model_name": config["policy_model_name"],
            "policy_api_key": config["policy_api_key"], "policy_base_url": config["policy_base_url"],
            "sampling_overrides": model["sampling_overrides"],
            "opencode_max_context_window": agent["opencode_max_context_window"],
            "agent_has_opencode_config": "opencode_config" in agent, "sandbox_timeout": agent["sandbox_timeout"],
            "sandbox_default_metadata": config["sandbox"]["default_metadata"],
            "gym_eval_flags": flags, "ATP_DEEPSWE_MAX_OUTPUT_TOKENS": max_output_env,
            "ATP_DEEPSWE_UPSTREAM_STREAM_default": stream_default, "opencode_limit": limit,
            "ok": (limit == {"context": 262144, "input": 262144, "output": 131072}
                   and model["sampling_overrides"] == {"temperature": 1.0, "top_p": 0.95, "max_tokens": 131072,
                                                       "reasoning_effort": "max"}
                   and config["policy_model_name"] == base["id"] and "opencode_config" not in agent
                   and flags["--max-output-tokens"] == "131072" and flags["--temperature"] == "1.0"
                   and flags["--top-p"] == "0.95")}
    command = json.loads((KR4 / "probes/kimi/pilot/models/kimi/attempts/0-0/attempt-0/agent-command.json").read_text())
    text = command["command"]
    at = text.index("OPENCODE_CONFIG_CONTENT=") + len("OPENCODE_CONFIG_CONTENT=")
    recorded = json.loads(shlex.split(text[at:])[0])
    out["kr4_pilot_recorded"] = {"limit": recorded["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"],
                                 "has_agent_key": "agent" in recorded, "model": recorded.get("model")}
    out["kr4_pilot_matches_kimi"] = out["kr4_pilot_recorded"]["limit"] == out["aliases"]["kimi"]["opencode_limit"]
    out["all_ok"] = all(entry["ok"] for entry in out["aliases"].values()) and out["kr4_pilot_matches_kimi"] \
        and not out["kr4_pilot_recorded"]["has_agent_key"]
    (TD / "analysis" / "config-check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
