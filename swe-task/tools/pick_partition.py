"""Pick the CPU partition where a synth job would start first.

`sbatch --test-only` (never submits) the exact job on each candidate partition and takes the earliest estimated
start; estimates within 60 s of the earliest are a tie, broken by more idle nodes, then by the listed order. A
partition that rejects the job shape (e.g. a walltime above its limit) is skipped, and so is one whose QoS would
hold the job (--test-only ignores QoS limits: e.g. p_batch_short's partition-wide GrpTRES node=20 kept bridge jobs
pending as QOSGrpNodeLimit although --test-only said "start now"). Prints one JSON line:
{"partition": <chosen>, "probes": [{partition, start, idle_nodes, error}, ...], "probed_at": ...}.

Usage: pick_partition.py [--partitions cpu_datamover,cpu_dataprocessing,cpu_interactive] -- <sbatch args>
  (the sbatch arguments of the real submission, without "sbatch", "--parsable" and "--partition")
"""
import argparse
import getpass
import json
import re
import subprocess
import sys
import time

DEFAULT = "cpu_datamover,cpu_dataprocessing,cpu_interactive"
TIE_S = 60


def _nodes(spec: str) -> int:
    for item in (spec or "").split(","):
        key, _, value = item.partition("=")
        if key == "node" and value.isdigit():
            return int(value)
    return 0


def _running_nodes(*args) -> int:
    out = subprocess.run(["squeue", "-h", "-t", "R", "-o", "%D", *args], capture_output=True, text=True, timeout=60).stdout
    return sum(int(word) for word in out.split() if word.isdigit())


def qos_block(partition: str, nodes: int = 1):
    """Why the partition's QoS would hold a new `nodes`-node job right now (per-user or partition-wide node cap),
    else None. Pending jobs do not count against these limits; running ones do. A partition QoS covers every job in
    the partitions that use it (p_cpu_short: cpu_interactive and cpu_short), whatever the job's own QoS."""
    try:
        listing = subprocess.run(["scontrol", "show", "partition", "-o"], capture_output=True, text=True,
                                 timeout=60).stdout
        qos_of = {m.group(1): q.group(1) for line in listing.splitlines()
                  if (m := re.search(r"PartitionName=(\S+)", line)) and (q := re.search(r"\bQoS=(\S+)", line))}
        qos = qos_of.get(partition)
        if qos in (None, "N/A", "(null)"):
            return None
        sharing = ",".join(name for name, value in qos_of.items() if value == qos)
        limits = subprocess.run(["sacctmgr", "-nP", "show", "qos", qos, "format=GrpTRES,MaxTRESPU"],
                                capture_output=True, text=True, timeout=60).stdout.strip().split("|")
        group_cap, user_cap = _nodes(limits[0]), _nodes(limits[1] if len(limits) > 1 else "")
        if user_cap and _running_nodes("-u", getpass.getuser(), "-p", sharing) + nodes > user_cap:
            return "QoS %s per-user node limit %d reached" % (qos, user_cap)
        if group_cap and _running_nodes("-p", sharing) + nodes > group_cap:
            return "QoS %s partition-wide node limit %d reached" % (qos, group_cap)
    except (OSError, subprocess.TimeoutExpired, AttributeError, IndexError):
        return None
    return None


def probe(partition: str, sbatch_args: list) -> dict:
    try:
        run = subprocess.run(["sbatch", "--test-only", "--partition=" + partition, *sbatch_args],
                             capture_output=True, text=True, timeout=120)
        text = (run.stdout + run.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        text = "%s: %s" % (type(error).__name__, error)
    match = re.search(r"to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", text)
    try:
        idle = subprocess.run(["sinfo", "-h", "-p", partition, "-t", "idle", "-o", "%D"],
                              capture_output=True, text=True, timeout=60).stdout
        idle_nodes = sum(int(word) for word in idle.split() if word.isdigit())
    except (OSError, subprocess.TimeoutExpired):
        idle_nodes = 0
    blocked = qos_block(partition)
    return {"partition": partition, "start": match.group(1) if match else None,
            "start_epoch": time.mktime(time.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")) if match and not blocked
            else None,
            "idle_nodes": idle_nodes,
            "error": blocked or (None if match else (text.splitlines() or ["no estimate"])[-1][:200])}


def choose(probes: list, order: list) -> str:
    valid = [item for item in probes if item["start_epoch"] is not None]
    if not valid:
        return order[0]  # no estimate anywhere: keep the first (Baihe's cpu_datamover by default)
    earliest = min(item["start_epoch"] for item in valid)
    tied = [item for item in valid if item["start_epoch"] - earliest <= TIE_S]
    return min(tied, key=lambda item: (-item["idle_nodes"], order.index(item["partition"])))["partition"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partitions", default=DEFAULT)
    parser.add_argument("sbatch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sbatch_args = args.sbatch_args[1:] if args.sbatch_args[:1] == ["--"] else args.sbatch_args
    if not sbatch_args or any(arg.startswith(("--partition", "-p")) for arg in sbatch_args):
        parser.error("give the job's sbatch arguments after --, without --partition")
    order = [name for name in args.partitions.split(",") if name]
    probes = [probe(name, sbatch_args) for name in order]
    chosen = choose(probes, order)
    for item in probes:
        item.pop("start_epoch")
    print(json.dumps({"partition": chosen, "probes": probes, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))


if __name__ == "__main__":
    sys.exit(main())
