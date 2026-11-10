#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): input run dir for the DeepSWE empty-patch classifier
(eval/deepswe-v1.1/analysis/empty-patch-20261018/base-step0-rep5-out131k/{analyze,summarize}.py, used read-only),
built the way analysis/empty-patch/{sftv2,sc14}/run was: one row per slot = the terminal.json of the slot's final
attempt (TA/results/<model>-group-<K>.json attempt_dir), with _ng_task_index = 100 * K + the campaign's task index
(so the classifier's (task index, rollout index) dedup keeps all 8 groups) and opencode_results_fpath = the attempt's
export.json.

    python3 -I -B build_run.py MODEL [--check-against sftv2]
      -> analysis/empty-patch/<MODEL>/run/shards/shard_0/rollouts.jsonl
    then: python3 -B $CLS/analyze.py <MODEL>/run <MODEL> --procs 16 > <MODEL>/analyze.log
          python3 -B $CLS/summarize.py <MODEL> <MODEL>/run > <MODEL>/summary.txt

2026-11-04: copied to analysis/v3etp20/empty-patch; only HERE changed. Run for v3etp20 (and v3etp10 as a reproduction
check against analysis/empty-patch/v3etp10/all.jsonl) with python3 -I -B.
"""
import argparse
import json
import os
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = TA / "analysis/v3etp20/empty-patch"  # 2026-11-04 copy for v3etp20 (the original analysis/empty-patch is read-only)


def rows(model):
    for path in sorted(TA.glob(f"results/{model}-group-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        data = json.loads(path.read_text())
        group = data["group"]
        for task_id, records in data["records"].items():
            for record in records:
                attempt = record.get("attempt_dir")
                if not attempt or not (Path(attempt) / "terminal.json").exists():
                    print(f"skip {model} g{group} {task_id} {record['slot']}: no terminal.json ({attempt})")
                    continue
                row = json.loads((Path(attempt) / "terminal.json").read_text())
                assert row.get("task_id") == task_id, (row.get("task_id"), task_id)
                row["_ng_task_index"] = 100 * group + int(row["_ng_task_index"])
                row["opencode_results_fpath"] = str(Path(attempt) / "export.json")
                yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out-root", default=str(HERE))
    args = parser.parse_args()
    shard = Path(args.out_root) / args.model / "run/shards/shard_0"
    shard.mkdir(parents=True, exist_ok=True)
    out = shard / "rollouts.jsonl"
    tmp = out.with_name(".rollouts.jsonl.tmp")
    n = 0
    with open(tmp, "w") as handle:
        for row in rows(args.model):
            handle.write(json.dumps(row) + "\n")
            n += 1
    os.replace(tmp, out)
    print(args.model, n, "rows ->", out)


if __name__ == "__main__":
    main()
