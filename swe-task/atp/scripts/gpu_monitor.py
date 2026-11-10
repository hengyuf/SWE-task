"""Monitor only the allocated GPU and stop evaluations that make no inference progress."""

import json
import math
import re
import subprocess
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


METRICS = tuple("vllm:" + name for name in (
    "prompt_tokens_total", "generation_tokens_total", "request_success_total",
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total",
))
COUNTERS = METRICS[:3]
GPU_FIELDS = "timestamp,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def parse_metrics(text):
    """Sum each selected Prometheus metric across its label combinations."""
    values = dict.fromkeys(METRICS, 0.0)
    seen = set()
    for line in text.splitlines():
        match = re.match(r"^([^\s{]+)(?:\{.*\})?\s+(\S+)(?:\s.*)?$", line)
        if not match or match[1] not in values:
            continue
        value = float(match[2])
        if not math.isfinite(value):
            raise ValueError("Nonfinite inference metric")
        values[match[1]] += value
        seen.add(match[1])
    # Labelled success/preemption series may not exist until the first event.
    required = {METRICS[0], METRICS[1], METRICS[3], METRICS[4]}
    if not required.issubset(seen):
        raise ValueError("Inference metrics are missing token counters or request gauges")
    return values


class GpuMonitor:
    def __init__(self, attempt: Path, env: dict, idle_seconds=600):
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be positive")
        self.attempt, self.env, self.idle_seconds = Path(attempt), dict(env), idle_seconds
        self.process = None
        self._gpu_file = self._log_file = self._metrics_file = None
        self._last_sample = self._last_progress = self._unavailable_since = None
        self._previous = None
        self._http = build_opener(ProxyHandler({}))

    def start(self):
        gpu = self.env.get("SLURM_JOB_GPUS", "").strip()
        if not re.fullmatch(r"(?:[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+)", gpu):
            raise RuntimeError("GPU monitoring requires exactly one explicit SLURM_JOB_GPUS ID")
        if self.process is not None:
            raise RuntimeError("GPU monitor is already started")
        try:
            self._gpu_file = (self.attempt / "gpu.csv").open("w")
            self._gpu_file.write(GPU_FIELDS + "\n")
            self._gpu_file.flush()
            self._log_file = (self.attempt / "gpu-monitor.log").open("w")
            self._metrics_file = (self.attempt / "inference-metrics.jsonl").open("w")
            # Slurm's global index can be remapped inside the allocation's device cgroup.
            visible = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                env=self.env, capture_output=True, text=True, check=True, timeout=10,
            )
            identifiers = [line.strip() for line in visible.stdout.splitlines() if line.strip()]
            uuid_pattern = rf"(?:(?:GPU|MIG)-{UUID_PATTERN}|MIG-GPU-{UUID_PATTERN}/[0-9]+/[0-9]+)"
            if len(identifiers) != 1 or not re.fullmatch(uuid_pattern, identifiers[0]):
                raise RuntimeError("GPU monitoring requires exactly one valid visible GPU UUID in this allocation")
            gpu_uuid = identifiers[0]
            self._log_file.write(json.dumps({"slurm_job_gpus": gpu,
                                            "cuda_visible_devices": self.env.get("CUDA_VISIBLE_DEVICES"),
                                            "resolved_uuid": gpu_uuid}) + "\n")
            self._log_file.flush()
            self.process = subprocess.Popen(
                ["nvidia-smi", "--id", gpu_uuid, "--query-gpu=" + GPU_FIELDS,
                 "--format=csv,noheader,nounits", "--loop=30"],
                env=self.env, stdout=self._gpu_file, stderr=self._log_file, start_new_session=True,
            )
            deadline = time.monotonic() + 10
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("Allocated-GPU monitor exited at startup; inspect gpu-monitor.log")
                if (self.attempt / "gpu.csv").stat().st_size > len(GPU_FIELDS) + 1:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Allocated-GPU monitor produced no initial sample within 10 seconds")
                time.sleep(0.1)
        except BaseException:
            self.stop()
            raise

    def _record(self, **record):
        self._metrics_file.write(json.dumps({"timestamp": time.time(), **record}) + "\n")
        self._metrics_file.flush()

    def observe(self, model_url):
        """Called after model readiness; samples at most once every 30 seconds."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Allocated-GPU monitor is not running; inspect gpu-monitor.log")
        now = time.monotonic()
        if self._last_sample is not None and now - self._last_sample < 30:
            return
        self._last_sample = now
        if self._last_progress is None:
            self._last_progress = now
        base = model_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            with self._http.open(base + "/metrics", timeout=10) as response:
                metrics = parse_metrics(response.read().decode("utf-8"))
        except (OSError, ValueError, HTTPException) as error:
            if self._unavailable_since is None:
                self._unavailable_since = now
            elapsed = now - self._unavailable_since
            self._record(state="metrics_unavailable", error_type=type(error).__name__, unavailable_seconds=elapsed)
            if elapsed >= 120:
                raise RuntimeError("Inference metrics unavailable for at least 120 seconds") from error
            return
        self._unavailable_since = None
        if self._previous is not None and any(metrics[key] > self._previous[key] for key in COUNTERS):
            self._last_progress = now
        self._previous = metrics
        idle = now - self._last_progress
        queued = metrics[METRICS[3]] + metrics[METRICS[4]] > 0
        state = "observing"
        if idle >= self.idle_seconds:
            state = "inference_stall" if queued else "model_idle_sandbox_tool_wait"
        self._record(state=state, no_progress_seconds=idle, metrics=metrics)
        if state == "inference_stall":
            raise RuntimeError(f"Inference stall: no token or request progress for {idle:.0f}s with requests running/waiting")
        if state == "model_idle_sandbox_tool_wait":
            raise RuntimeError(f"Model idle during sandbox/tool wait: no inference progress for {idle:.0f}s and no queued requests")

    def stop(self):
        """Stop only this monitor's nvidia-smi child and close its files."""
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                else:
                    self.process.wait()
        finally:
            self.process = None
            for name in ("_gpu_file", "_log_file", "_metrics_file"):
                stream = getattr(self, name)
                if stream is not None:
                    stream.close()
                    setattr(self, name, None)
