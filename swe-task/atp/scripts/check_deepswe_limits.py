#!/usr/bin/env python3
"""Read quota metadata without recording credentials or other users' sandboxes."""
import asyncio
import json
import os
from pathlib import Path
import sys

import httpx

from deepswe_stage import read_dotenv


async def main():
    app = Path(sys.argv[1])
    values = read_dotenv(app / ".env")
    settings = json.loads((app / "configs/deepswe.json").read_text())
    models = {model["id"] for model in settings["models"].values()}
    base = settings["base_url"].removesuffix("/v1")
    allowed = {"id", "model_name", "model", "tpm", "rpm", "max_parallel_requests",
               "max_budget", "budget_duration", "rate_limit", "rate_limits", "limits",
               "model_info", "litellm_params", "model_group", "max_input_tokens",
               "max_output_tokens", "max_tokens", "supported_openai_params"}
    def filtered(item):
        if isinstance(item, dict):
            return {k: filtered(v) for k, v in item.items() if k in allowed}
        if isinstance(item, list):
            return [filtered(v) for v in item]
        return item
    async with httpx.AsyncClient(timeout=30) as client:
        for path in ("/v1/models", "/model/info", "/key/info"):
            response = await client.get(base + path,
                headers={"Authorization": "Bearer " + values["NVIDIA_API_KEY"]})
            result = {"service": "inference", "path": path, "status": response.status_code}
            if response.is_success:
                data = response.json()
                records = data.get("data", [])
                result["models"] = [filtered(row) for row in records
                    if row.get("id") in models or row.get("model_name") in models]
                if path == "/key/info":
                    result["limits"] = filtered(data.get("info", data))
            print(json.dumps(result), flush=True)
    domain = values["OPENSANDBOX_DOMAIN"].rstrip("/")
    if "://" not in domain:
        domain = "https://" + domain
    async with httpx.AsyncClient(timeout=30,
            verify=values.get("OPENSANDBOX_TLS_VERIFY", "true").lower() == "true") as client:
        response = await client.get(domain + "/openapi.json",
            headers={"OPEN-SANDBOX-API-KEY": values["OPENSANDBOX_API_KEY"]})
        result = {"service": "opensandbox", "path": "/openapi.json", "status": response.status_code}
        if response.is_success:
            result["quota_paths"] = [path for path in response.json().get("paths", {})
                                     if any(word in path.lower() for word in ("quota", "limit", "capacity"))]
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this check inside the CPU validation allocation")
    asyncio.run(main())
