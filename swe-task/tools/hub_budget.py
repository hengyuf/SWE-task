#!/usr/bin/env python3
"""Budget of a swe-task hub key (default: the active one, synth/common.hub_key_name()); keys cannot call the hub's
/key/info themselves, so: one minimal
glm call (max_tokens 4, about $0.00003), then LiteLLM's x-litellm-key-spend / x-litellm-key-max-budget response
headers (or, once the key is refused, the "Current cost: X, Max budget: Y" of the 422 body). Never prints the key.

    /usr/bin/python3 -I -B tools/hub_budget.py [--min-remaining 500] [--key-name NVAPI_KEY_OFFICIAL]

Prints one JSON line {"at", "key", "status", "spend", "max_budget", "remaining", "error"} ("key" is the NAME). Exit 0: fine; 3: the remaining
budget is below --min-remaining or the hub refuses the key for budget (stop the builds before calls start failing:
ATP makes 4xx slots fatal and a failed author attempt is recorded, see README "Evolution seeds"); 2: unknown (network
or another HTTP error). Why: 2026-10-17 the old key hit its $5000 cap mid-build and the 422s had to be cleaned up.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synth"))
from common import HUB_KEYS, SETTINGS, USER_ENV, env_value, hub_key_name  # noqa: E402  (stdlib-only module)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-remaining", type=float, default=500.0, help="exit 3 below this many dollars")
    parser.add_argument("--key-name", choices=HUB_KEYS, help="check this key instead of the active one")
    args = parser.parse_args()
    name = args.key_name or hub_key_name()
    key = env_value(USER_ENV, name)
    if not key:
        print(json.dumps({"key": name, "error": f"no {name} in the credentials file"}))
        return 2
    body = {"model": SETTINGS["models"]["glm"]["id"], "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 4}  # the glm deployment rejects < 2
    request = urllib.request.Request(SETTINGS["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": name, "status": None, "spend": None, "max_budget": None,
           "remaining": None, "error": None}
    text = ""
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            out["status"], headers = response.status, response.headers
    except urllib.error.HTTPError as error:
        out["status"], headers = error.code, error.headers
        text = error.read()[:600].decode(errors="replace").replace(key, "<key>")
        out["error"] = re.sub(r"sk-[A-Za-z0-9_.-]+", "<redacted>", text)[:300]
    except (urllib.error.URLError, OSError) as error:
        out["error"] = f"{type(error).__name__}: {error}"[:300]
        print(json.dumps(out))
        return 2
    out["spend"], out["max_budget"] = number(headers.get("x-litellm-key-spend")), number(
        headers.get("x-litellm-key-max-budget"))
    refused = re.search(r"Current cost: ([0-9.]+), Max budget: ([0-9.]+)", text)
    if refused:
        out["spend"], out["max_budget"] = float(refused.group(1)), float(refused.group(2))
    if out["spend"] is not None and out["max_budget"] is not None:
        out["remaining"] = round(out["max_budget"] - out["spend"], 2)
    print(json.dumps(out))
    if refused or "budget_exceeded" in text or (out["remaining"] is not None and out["remaining"] < args.min_remaining):
        return 3
    return 0 if out["status"] == 200 else 2


if __name__ == "__main__":
    sys.exit(main())
