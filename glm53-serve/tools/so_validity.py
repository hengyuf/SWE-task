#!/usr/bin/env python3
"""Structured-output validity + MTP acceptance under grammar-constrained decoding (stdlib only).

    python3 tools/so_validity.py --url http://IP:8000/v1 [-c 16] [-n 64] [--label X]

Sends -n chat requests (-c at a time) with response_format = a strict JSON schema (nested object, enum, integer
bounds, array of objects), reasoning_effort low, temperature 1.0, and checks that every reply's content parses and
satisfies the schema (a grammar violation shows up as invalid JSON or a schema miss). Prints the MTP spec-decode
counters of the run (acceptance length) next to the validity count.
"""
import argparse
import concurrent.futures
import json
import re
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country_code": {"type": "string", "enum": ["US", "FR", "JP", "BR", "IN", "DE", "CN", "OTHER"]},
        "population_millions": {"type": "integer", "minimum": 0, "maximum": 100},
        "landmarks": {"type": "array", "minItems": 2, "maxItems": 4, "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "year_built": {"type": "integer"}, "kind": {
                "type": "string", "enum": ["building", "bridge", "park", "museum", "monument", "other"]}},
            "required": ["name", "year_built", "kind"], "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["city", "country_code", "population_millions", "landmarks", "summary"],
    "additionalProperties": False,
}
CITIES = ["Paris", "Tokyo", "Rio de Janeiro", "Mumbai", "Berlin", "Shanghai", "Chicago", "Kyoto", "Lyon", "Munich",
          "Sao Paulo", "Delhi", "Osaka", "San Francisco", "Marseille", "Hamburg"]
LINE = re.compile(r"^(vllm:spec_decode_[a-z_]+?)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def counters(url):
    base = url.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    with OPENER.open(base + "/metrics", timeout=30) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if match and "per_pos" not in match.group(1):
            out[match.group(1)] = out.get(match.group(1), 0.0) + float(match.group(2))
    return out


def check(obj):
    if not isinstance(obj, dict) or set(obj) != set(SCHEMA["required"]):
        return "keys"
    props = SCHEMA["properties"]
    if not isinstance(obj["city"], str) or not isinstance(obj["summary"], str):
        return "string"
    if obj["country_code"] not in props["country_code"]["enum"]:
        return "enum"
    if not isinstance(obj["population_millions"], int) or not 0 <= obj["population_millions"] <= 100:
        return "integer bounds"
    marks = obj["landmarks"]
    if not isinstance(marks, list) or not 2 <= len(marks) <= 4:
        return "array length"
    for mark in marks:
        if not isinstance(mark, dict) or set(mark) != {"name", "year_built", "kind"}:
            return "item keys"
        if not isinstance(mark["year_built"], int) or mark["kind"] not in props["landmarks"]["items"]["properties"]["kind"]["enum"]:
            return "item values"
    return None


def one(url, i):
    city = CITIES[i % len(CITIES)]
    body = {"model": "glm-5.3", "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": f"Describe {city} as JSON: country code, population in millions "
                                                     f"(integer), 2-4 landmarks with the year built and kind, and a "
                                                     f"two-sentence summary."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "strict": True,
                                                                       "schema": SCHEMA}}}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=1800) as response:
        data = json.load(response)
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if not content.strip():
        # the reply never left the reasoning section, so the grammar never applied (not a grammar violation)
        problem = "no content (reasoning only, %d chars)" % len(reasoning)
    else:
        try:
            problem = check(json.loads(content))
        except ValueError:
            problem = "INVALID JSON"
    return problem, data["usage"]["completion_tokens"], data["choices"][0].get("finish_reason"), (content or reasoning[-200:])[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("-c", type=int, default=16)
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("--label", default="")
    a = parser.parse_args()
    before = counters(a.url)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.c) as pool:
        results = list(pool.map(lambda i: one(a.url, i), range(a.n)))
    after = counters(a.url)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    bad = [r for r in results if r[0]]
    violations = [r for r in bad if not r[0].startswith("no content")]
    finishes = {}
    for r in results:
        finishes[r[2]] = finishes.get(r[2], 0) + 1
    print(f"[{a.label}] json_schema n{a.n} c{a.c}: valid {a.n - len(bad)}/{a.n} (grammar violations "
          f"{len(violations)}, reasoning-only {len(bad) - len(violations)}), finish {finishes}, "
          f"completion tokens {sum(r[1] for r in results)}, {time.time() - started:.0f} s; MTP drafts {drafts:.0f}, "
          f"acceptance rate {accepted / draft_tokens if draft_tokens else float('nan'):.3f}, "
          f"mean acceptance length {1 + accepted / drafts if drafts else float('nan'):.2f}")
    for r in bad[:5]:
        print("   NOT VALID:", r[0], "|", r[3].replace("\n", " "))


if __name__ == "__main__":
    main()
