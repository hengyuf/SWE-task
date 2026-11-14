#!/usr/bin/env python3
"""Compact progress note for the GitHub issue thread (markdown on stdout), from progress.json (progress.py),
keeper-status.json and TIER-PAUSE.json of glm-rollouts-tasksall. Read-only.

    /usr/bin/python3 -I -B issue_progress.py
"""
import json
from pathlib import Path
import time

GR = Path(__file__).resolve().parents[1]


def load(name):
    try:
        return json.loads((GR / name).read_text())
    except (OSError, ValueError):
        return {}


def main():
    progress, keeper, pause = load("progress.json"), load("keeper-status.json"), load("TIER-PAUSE.json")
    plan = load("tasks.json")
    tiers = progress.get("tiers") or {}
    names = {"1": "accepted/strong", "2": "partial", "3": "weak", "4": "none/unlabeled"}
    camps = {str(t): 0 for t in range(1, 5)}
    for c in plan.get("campaigns", []):
        camps[str(c["tier"])] += 1
    lines = [f"**Progress {time.strftime('%Y-%m-%d %H:%M %Z')}**", ""]
    if pause:
        lines.append(f"Tier pause: ON since {pause.get('paused_at')} (only tiers {pause.get('allowed_tiers')} run; the "
                     f"others wait for your go-ahead).")
    lines += ["", "| tier | campaigns done | slots | graded | pass | rate | running |", "|---|---|---|---|---|---|---|"]
    total = {"slots": 0, "graded": 0, "passed": 0, "running": 0}
    gate = (keeper.get("campaigns") or {}).get("leg_gate") or {}
    # campaigns whose driver leg is running rollouts now (leg capacity gate: legs follow the ready servers)
    live = (set(gate.get("working") or []) if gate else
            {a["name"] for a in (keeper.get("campaigns") or {}).get("active") or [] if a.get("driver")})
    running = {}
    for row in progress.get("rows") or []:  # in-flight attempts of campaigns that have a driver leg now
        if row.get("tier") is not None and row["campaign"].split("-rec")[0] in live:
            running[str(row["tier"])] = running.get(str(row["tier"]), 0) + row.get("running", 0)
    for tier in ("1", "2", "3", "4"):
        t = dict(tiers.get(tier) or {}, running=running.get(tier, 0))
        for key in total:
            total[key] += t.get(key, 0)
        rate = f"{100 * t.get('passed', 0) / t['graded']:.1f}%" if t.get("graded") else "-"
        lines.append(f"| {tier} {names[tier]} | {t.get('complete', 0)}/{camps[tier]} | {t.get('slots', 0):,} | "
                     f"{t.get('graded', 0):,} | {t.get('passed', 0):,} | {rate} | {t.get('running', 0)} |")
    rate = f"{100 * total['passed'] / total['graded']:.1f}%" if total["graded"] else "-"
    lines.append(f"| all | | {total['slots']:,} | {total['graded']:,} | {total['passed']:,} | {rate} | {total['running']} |")
    sched = keeper.get("campaigns") or {}
    router = progress.get("router") or {}
    lines += ["",
              f"Servers: {keeper.get('ready_servers', '?')} ready, {len(keeper.get('running') or [])} running, "
              f"{len(keeper.get('pending') or [])} pending; generation {router.get('gen_tok_per_s_last_30min', '?')} tok/s "
              f"(30 min). Active campaigns: {', '.join(a['name'] for a in sched.get('active') or []) or 'none'}; parked: "
              f"{', '.join(p['name'] for p in sched.get('parked') or []) or 'none'}."]
    if gate:
        waiting = len(gate.get("held_back") or [])
        lines.append(f"Driver legs follow the ready GPU capacity ({gate.get('capacity', 0)} concurrent rollouts): "
                     f"{len(gate.get('working') or [])} campaign(s) running rollouts (load {gate.get('working_load', 0)}), "
                     f"{len(gate.get('admitted') or [])} starting, {waiting} released campaign(s) waiting for servers.")
    hosts = keeper.get("leg_hosts") or {}
    if any(hosts.values()):  # driver CPU jobs are packed on a few nodes (per-user node limits count distinct nodes)
        lines.append("Driver CPU jobs: " + "; ".join(
            f"{sum(nodes.values())} on {len(nodes)} `{partition}` node(s)" for partition, nodes in hosts.items() if nodes)
            + " (the keeper job is one of the cpu_interactive ones).")
    if sched.get("leg_wait"):
        lines.append(f"Some campaigns wait for room on the CPU host nodes since {keeper.get('leg_wait_since')}: "
                     f"{sched['leg_wait']}.")
    if keeper.get("idle_since"):
        lines.append(f"No rollout work left since {keeper['idle_since']}: the GPU servers are released "
                     f"{keeper.get('config', {}).get('idle_release_minutes', 60)} min after that"
                     + (" (released)." if keeper.get("idle_released") else "."))
    lines.append("(Graded counts come from the live attempt dirs; re-runs of infra-hit slots replace the originals in the "
                 "final report.)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
