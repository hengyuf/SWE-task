# Using the sc14 student inference workers

This guide shows how to send requests to hengyuf's **sc14 student workers** from your own jobs. These are vLLM servers of the SWE checkpoint `swonlysc-step0014`. The workers are shared on a **best-effort** basis. Their main consumer is the swe-task synthetic-task pipeline. For heavier use, read [Rules of use](#rules-of-use) first.

| | |
|---|---|
| Model | `swonlysc-step0014` (alias **sc14**), a Super 3.5 (NemotronH) checkpoint. Weights: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` |
| Served model name | `policy` (OpenAI-compatible `/v1/chat/completions`) |
| Engine | vLLM 0.29, TP8, one replica per 8-GPU node, at most 64 concurrent sequences per worker |
| Context | 262,144 tokens (prompt + output); at most 131,072 output tokens per call |
| Output | reasoning model: `message.reasoning` holds the thinking, `message.content` the answer; tool calls are parsed (`qwen3_coder` format, `tool_choice` auto) |
| Fleet | about 10 workers on `batch` as 4 h jobs, replaced automatically by a maintainer; sometimes extra "bridge" workers on other partitions |
| Owner | hengyuf |

## 1. Before you start

- **Access.** Your account must be in the `llmservice` Unix group, because the project directory is group-only. Check with `id -nG | tr ' ' '\n' | grep -x llmservice`. Without the group you cannot read the worker registry; ask hengyuf.
- **Where to run.** Any login or compute node of this cluster. The workers sit on the internal network (`http://10.65.x.y:8000`).
- **No authentication.** The endpoints are plain HTTP without an API key. Anyone on the cluster network can reach them, so do not send secrets or confidential data.
- **No SLA.** Workers are 4 h jobs and move to new nodes all the time. When the `batch` queue is slow there can be gaps with few or no workers. The fleet will be shut down when the project ends.

## 2. Recommended: run your own router

The router is a small local proxy (`student/router.py`, the same one the pipeline uses). It:
- reads the worker registry and sends each call to a healthy worker;
- keeps each conversation on the same worker while that worker has free capacity, so the worker's prefix cache is reused (it reads each worker's vLLM `/metrics` every 5 s and sends calls to workers with room first);
- retries on another worker after connection errors, 429/502/503/504, one 500 and generations aborted by a stopping worker (`finish_reason: abort`) (up to 4 attempts);
- caps `max_tokens` at 131,072 and, on a "maximum context length" error, retries with the largest output that fits;
- when no worker is up, holds a call for up to 1 h, then answers `429` with `Retry-After: 120`.

The router listens on `127.0.0.1` only. A multi-node job needs one router per node.

### Shell

```bash
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
PY=$S/atp/runtime/controller/bin/python        # Python 3.13 with aiohttp; any Python with aiohttp works
LOG=$HOME/student-router                        # any directory you own (small logs: one line per call)
mkdir -p "$LOG" && rm -f "$LOG/port"

setsid nohup "$PY" -B "$S/student/router.py" --registry "$S/student/registry" \
    --host 127.0.0.1 --port 0 --port-file "$LOG/port" --log-dir "$LOG" \
    > "$LOG/stdout.log" 2>&1 < /dev/null &
for i in $(seq 60); do [ -s "$LOG/port" ] && break; sleep 1; done

export STUDENT_BASE_URL="http://127.0.0.1:$(cat "$LOG/port")/v1"
curl -s "$STUDENT_BASE_URL/models"                                   # -> {"data":[{"id":"policy",...}]}
curl -s "http://127.0.0.1:$(cat "$LOG/port")/router/status"          # workers it sees, calls in flight, counters
```

To stop the router:

```bash
kill "$(curl -s "http://127.0.0.1:$(cat "$LOG/port")/router/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pid"])')"
```

### Python (starts and stops the router for you)

```python
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
from student.client import ensure_router, wait_for_workers

wait_for_workers(n=1, timeout_s=3600)                     # block until at least one worker is routable
base_url = ensure_router(log_dir=os.path.expanduser("~/student-router"))
# base_url is "http://127.0.0.1:<port>/v1". The router stops when this process exits.
```

## 3. Sending requests

### curl

```bash
curl -s "$STUDENT_BASE_URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "policy",
  "max_tokens": 8192,
  "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]
}'
```

### OpenAI Python SDK, with a tool

```python
from openai import OpenAI

client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)   # long generations can take many minutes
r = client.chat.completions.create(
    model="policy",
    max_tokens=16384,
    messages=[{"role": "user", "content": "Use the bash tool to list the files in /tmp."}],
    tools=[{"type": "function", "function": {
        "name": "bash", "description": "Run a bash command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
)
msg = r.choices[0].message
print(r.choices[0].finish_reason)                    # "tool_calls"
print(msg.tool_calls[0].function.arguments)          # {"command": "ls -la /tmp"}
print(msg.model_extra.get("reasoning"))              # the thinking text
```

### What the router does to your request

It applies the same serving settings as the DeepSWE eval. **You cannot change these through the router:**
- `model` is set to `policy`, `temperature` to 1.0 and `top_p` to 1.0.
- `chat_template_kwargs` is set to `{"enable_thinking": true, "truncate_history_thinking": false}`.
- Hosted-only parameters such as `reasoning_effort` are dropped.
- `max_tokens` / `max_completion_tokens` become min(requested, 131072), or 131072 if you set neither. **Set `max_tokens` yourself** if you don't need very long outputs.

Streaming (`"stream": true`) passes through. A streamed call can only move to another worker before its first byte arrives.

Generations can take many minutes, because thinking can run to tens of thousands of tokens. Use long client timeouts.

## 4. Without the router: call a worker directly

Use this when you need other sampling settings (e.g. `temperature: 0`, or thinking off). You then have to handle worker rotation, failover and the length limits yourself.

Pick a routable worker from the registry. A worker is routable when its status is `ready` or `draining`, its heartbeat is at most 120 s old, and its job has not ended. `draining` means the job's last 45 min.

Note (2026-10-20): the registry is on Lustre, and a Lustre stall can leave every heartbeat stale for a few minutes while the workers serve normally. The router therefore keeps a worker whose heartbeat went stale as long as its own `GET /health` checks of that worker keep passing. If you call workers directly, check `GET /health` yourself before you drop a worker for a stale heartbeat.

```python
import glob, json, time
REG = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/registry"
now = time.time()
workers = [w for w in (json.load(open(p)) for p in glob.glob(f"{REG}/*.json"))
           if w.get("status") in ("ready", "draining") and now - w.get("heartbeat_at", 0) <= 120
           and now < w.get("end_epoch", 0)]
workers.sort(key=lambda w: (w["status"] != "ready", -w["end_epoch"]))    # ready first, longest remaining first
print(workers[0]["base_url"], "ends", time.strftime("%H:%M", time.localtime(workers[0]["end_epoch"])))
```

```bash
curl -s http://<worker-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "policy", "max_tokens": 8192, "temperature": 1.0, "top_p": 1.0,
  "chat_template_kwargs": {"enable_thinking": true, "truncate_history_thinking": false},
  "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]
}'
```

Rules when calling a worker directly:
- Keep `max_tokens` ≤ 131072 and prompt + `max_tokens` ≤ 262144. Otherwise vLLM answers 400.
- A worker's IP is only valid until its job ends (see `end_epoch` in its registry file). Look it up again for every new batch of work.
- To get the same model behaviour as the eval, send the sampling settings and `chat_template_kwargs` shown above.

## 5. Checking capacity

```bash
python3 -B /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student/fleet.py status   # read-only
```

This prints every worker job with its state, time left, registry status and base URL, plus queued replacements and their estimated start.

For the current load on one worker, read vLLM's metrics:

```bash
curl -s http://<worker-ip>:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)'
```

## Rules of use

1. **Keep concurrency modest.** Use at most **32 requests in flight per person** unless you have agreed more with hengyuf. Each worker serves at most 64 sequences, and the pipeline needs its share.
2. **Read-only.** Only use `fleet.py status`. Never submit, cancel or modify fleet jobs. Never write anything under `swe-task/student/`.
3. **Expect gaps.** Build in retries. Through the router you get a `429` with `Retry-After: 120` after 1 h without any worker.
4. **Tell hengyuf** before starting a long batch job, such as an eval over hundreds of tasks.

## Troubleshooting

| Symptom | Meaning / what to do |
|---|---|
| `429` with `Retry-After: 120` from the router | No worker was routable for an hour (fleet gap). Check `fleet.py status` and retry later. |
| Connection refused / timeout on a worker IP | That worker's job ended or moved. Use the router, or look the worker up again in the registry. |
| `400 ... maximum context length` (direct calls) | Prompt + `max_tokens` > 262,144. Lower `max_tokens` or shorten the prompt. The router does this for you. |
| Answer in `reasoning`, `content` empty, `finish_reason: "length"` | `max_tokens` ran out while the model was still thinking. Raise `max_tokens`. |
| Permission denied on `swe-task/...` | You are not in the `llmservice` group. |
| Something else | Send hengyuf the matching line of `<your log dir>/student-router-requests.jsonl`. It has the worker id, status and error, and no content. |
