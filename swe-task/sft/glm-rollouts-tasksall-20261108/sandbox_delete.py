#!/usr/bin/env python3
"""Delete (terminate) one OpenSandbox sandbox of a glm-rollouts campaign by id: the lifecycle API's
DELETE /v1/sandboxes/<id>, the call the OpenSandbox SDK's kill() makes, with the campaign's own credential (read from
its secrets file by NAME, never printed). For a sandbox that no longer answers commands, which blocks ATP's recovery
(fail-closed: one unrecoverable attempt stops the whole shard); a missing sandbox counts as already stopped.

    /usr/bin/python3 -I -B sandbox_delete.py --sandbox baihe|hengyuf --id <sandbox uuid> [--dry-run]
"""
import argparse
import json
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
SECRETS = {"baihe": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/secrets/"
                         "baihe-opensandbox.env"),
           "hengyuf": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/secrets/opensandbox.env")}


def read_env(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key.startswith("OPENSANDBOX_"):
            value = value.strip()
            values[key] = value[1:-1] if value[:1] in "'\"" and value[-1:] == value[:1] else value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", choices=sorted(SECRETS), required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", args.id):
        raise SystemExit("--id must be a sandbox uuid")
    env = read_env(SECRETS[args.sandbox])
    domain = env["OPENSANDBOX_DOMAIN"].rstrip("/")
    base = domain if "://" in domain else "https://" + domain
    url = f"{base}/v1/sandboxes/{args.id}"
    print(json.dumps({"method": "DELETE", "url": url, "credential": f"{args.sandbox} (OPENSANDBOX_API_KEY)"}))
    if args.dry_run:
        return
    context = ssl.create_default_context()
    if env.get("OPENSANDBOX_TLS_VERIFY", "true").lower() == "false":
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(url, method="DELETE", headers={"OPEN-SANDBOX-API-KEY": env["OPENSANDBOX_API_KEY"]})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
    try:
        with opener.open(request, timeout=120) as response:
            print(json.dumps({"status": response.status, "body": response.read(500).decode("utf-8", "replace")}))
    except urllib.error.HTTPError as error:
        print(json.dumps({"status": error.code, "body": error.read(500).decode("utf-8", "replace"),
                          "note": "404 = already gone (counts as stopped)" if error.code == 404 else ""}))


if __name__ == "__main__":
    main()
