#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): a few more small priced calls (x-litellm-response-cost) on NVAPI_KEY_NEW_1 to pin
deepseek-v4-pro's per-token prices with an over-determined fit (hub_smoke.json gave exactly 3 points): a ~3k-token
prompt (uncached), the same prompt again (prefix-cache hits), and a longer answer. The harness's sampling fields
(max_tokens 131072, temperature 1.0, top_p 0.95, reasoning_effort max). Key read by NAME, never printed.
Writes price_calib.json (same shape as hub_smoke.json; prices.py reads both). Cost well under $0.05.

    /usr/bin/python3 -I -B price_calib.py [--models deepseek]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(HERE))
import hub_smoke  # noqa: E402  (call(), MODELS; its own key check is not used here)
from common import SETTINGS, USER_ENV, env_value  # noqa: E402

KEY_NAME = "NVAPI_KEY_NEW_1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["deepseek"])
    args = parser.parse_args()
    key = env_value(USER_ENV, KEY_NAME)
    if not key:
        raise SystemExit(f"no {KEY_NAME}")
    filler = (Path(hub_smoke.__file__).read_text() * 3)[:12000]  # ~3k tokens of fixed text (this repo's own code)
    results = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": KEY_NAME, "models": {}}
    for alias in args.models:
        spec = hub_smoke.MODELS[alias]
        base = {"model": spec["id"], "max_tokens": SETTINGS["max_output_tokens"], "temperature": SETTINGS["temperature"],
                "top_p": SETTINGS["top_p"], **spec.get("sampling_overrides", {})}
        long_prompt = [{"role": "system", "content": "You review Python code.\n\n" + filler},
                       {"role": "user", "content": "In one sentence: what does the call() function return?"}]
        runs = {"long_uncached": hub_smoke.call(key, dict(base, messages=long_prompt))}
        time.sleep(2)
        runs["long_cached"] = hub_smoke.call(key, dict(base, messages=long_prompt))
        runs["long_answer"] = hub_smoke.call(key, dict(base, messages=[
            {"role": "user", "content": "Write about 250 words explaining how HTTP retries with exponential backoff work."}]))
        for run in runs.values():
            run.pop("_message", None)
        results["models"][alias] = {"id": spec["id"], "calls": runs}
    (HERE / "price_calib.json").write_text(json.dumps(results, indent=1) + "\n")
    for alias, entry in results["models"].items():
        for kind, run in entry["calls"].items():
            usage = run.get("usage") or {}
            print(alias, kind, run.get("status"), f"{run.get('elapsed_s')}s",
                  {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                  "cached", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                  "hdr_cost", (run.get("headers") or {}).get("x-litellm-response-cost"),
                  "err", json.dumps(run.get("error"))[:200] if run.get("error") else "")


if __name__ == "__main__":
    main()
