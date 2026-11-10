#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): behaviour features of the 10- and 20-epoch SFT checkpoints, computed with the
teacher-rollouts-v2 extractor (TR/features.py, imported read-only, never changed) on each slot's final attempt.

    /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B \
        run_features.py [--sources sftv2e10 sftv2e20] [--workers 16]
      -> TA/analysis/features/<source>.jsonl (one line per slot; same schema as TR/features/<source>.jsonl)

Slots come from TA/results/<source>-group-<K>.json exactly as TR/sources.py reads sc14 and sftv2 (_from_results).
(The vLLM env python is used because TR/features.py needs `tokenizers`; the tokenizer is sc14's, as in TR.)

2026-11-04: copied unchanged to analysis/v3etp20/features (HERE = this directory) for v3etp20; v3etp10 re-run with
--suffix -recheck to check that it reproduces analysis/features/v3etp10.jsonl.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")  # unresolved, as TR/sources.py
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
sys.path.insert(0, str(TR))
import features as tr_features  # noqa: E402  (TR/features.py)
import sources as tr_sources  # noqa: E402  (TR/sources.py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["sftv2e10", "sftv2e20"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--suffix", default="", help="output name suffix (e.g. -recheck for a reproduction check)")
    args = parser.parse_args()
    for source in args.sources:
        slots = tr_sources._from_results(sorted(TA.glob(f"results/{source}-group-*.json")), source)
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(tr_features.features, slots, chunksize=4))
        path = HERE / f"{source}{args.suffix}.jsonl"
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        tmp.replace(path)
        print(source, len(rows), "slots;", sum(bool(r.get("export")) for r in rows), "with export", flush=True)


if __name__ == "__main__":
    main()
