#!/usr/bin/env python3
"""teacher-rollouts-v2: which hub key NAME the processes of this user on this node carry in NVIDIA_API_KEY (and any
other *_API_KEY variable), by comparing values with the .env entries in memory. Prints names and counts only, never a
value. Run inside a job step on the probe node (srun --overlap)."""
import sys
sys.dont_write_bytecode = True
import collections, os
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
from common import USER_ENV, env_value
names = ("NVAPI_KEY_NEW", "NVAPI_KEY_OFFICIAL", "NVAPI_KEY_OFFICIAL_1", "NVAPI_KEY_OFFICIAL_2", "NVAPI_KEY_NEW_1", "NVAPI_KEY")
values = {}
for name in names:
    value = env_value(USER_ENV, name)
    if value:
        values.setdefault(value, name)
found = collections.Counter()
uid = os.getuid()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        if os.stat(f"/proc/{pid}").st_uid != uid:
            continue
        with open(f"/proc/{pid}/environ", "rb") as stream:
            env = stream.read().split(b"\0")
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            cmd = stream.read().split(b"\0")[0:3]
    except OSError:
        continue
    for item in env:
        key, _, value = item.partition(b"=")
        key = key.decode(errors="replace")
        if key.endswith("_API_KEY") or key.startswith("NVAPI"):
            match = values.get(value.decode(errors="replace"), "unknown/other" if value else "empty")
            found[(key, match)] += 1
for (key, match), count in sorted(found.items()):
    print(f"{key} = {match}: {count} processes")
