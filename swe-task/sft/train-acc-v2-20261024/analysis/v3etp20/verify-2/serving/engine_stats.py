#!/usr/bin/env python3
"""verify-2: vLLM 10 s engine stats per fleet (samples with >= 1 running request): mean running requests per worker,
generation throughput per worker, per running sequence (= sum gen tput / sum running, and mean of per-sample ratio)."""
import re
import sys
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
RE = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")

for model in sys.argv[1:]:
    run, gen, wait, ratio = [], [], [], []
    for f in sorted((TA / "fleets" / model / "logs").glob("vllm-*.log")):
        for line in open(f, errors="replace"):
            if "Avg generation throughput" in line:
                m = RE.search(line)
                if m and int(m.group(2)) >= 1:
                    g, r, w = float(m.group(1)), int(m.group(2)), int(m.group(3))
                    run.append(r)
                    gen.append(g)
                    wait.append(w)
                    ratio.append(g / r)
    n = len(run)
    print(f"{model}: samples {n}; running/worker {sum(run) / n:.2f}; waiting {sum(wait) / n:.2f}; gen tok/s/worker {sum(gen) / n:.1f}; "
          f"per sequence (sum gen / sum running) {sum(gen) / sum(run):.1f}; mean per-sample ratio {sum(ratio) / n:.1f}")
