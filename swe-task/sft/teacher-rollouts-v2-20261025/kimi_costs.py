#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): hub usage and cost of the existing Kimi-K3 build-probe rollouts on the 50 tasks
(context for the teacher cost comparison): repeat 0 of every task (the final attempt of that slot), from its http.jsonl
(kimi's streamed replies carry LiteLLM's usage.cost). Read-only. Writes costs/kimi-r0.json.

    /usr/bin/python3 -I -B kimi_costs.py
"""
import sys

sys.dont_write_bytecode = True

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
sys.path.insert(0, str(HERE))
import costs  # noqa: E402
import sources  # noqa: E402


def main():
    slots = [s for s in sources.slots("kimi") if s["repeat"] == 0 and s.get("attempt_dir")]
    price = costs.prices()
    with ThreadPoolExecutor(8) as pool:
        usages = list(pool.map(lambda s: costs.attempt_usage(Path(s["attempt_dir"]), price, "kimi"), slots))
    rows = [{"task_id": s["task_id"], "status": s["status"], **{k: u[k] for k in ("calls", "ok", "prompt", "cached",
            "completion", "reasoning_reported", "cost", "cost_reported", "first", "last")}} for s, u in zip(slots, usages)]
    (HERE / "costs").mkdir(exist_ok=True)
    (HERE / "costs" / "kimi-r0.json").write_text(json.dumps(rows, indent=1) + "\n")
    reported = [r["cost_reported"] for r in rows if r["cost_reported"]]
    print(f"kimi r0 slots {len(rows)}; reported-cost slots {len(reported)}; mean ${statistics.fmean(reported):.2f} "
          f"median ${statistics.median(reported):.2f}; prompt Mtok mean {statistics.fmean([r['prompt'] for r in rows])/1e6:.1f}; "
          f"cached {100*sum(r['cached'] for r in rows)/max(1,sum(r['prompt'] for r in rows)):.0f}%; "
          f"completion ktok mean {statistics.fmean([r['completion'] for r in rows])/1e3:.1f}; calls mean "
          f"{statistics.fmean([r['calls'] for r in rows]):.0f}")


if __name__ == "__main__":
    main()
