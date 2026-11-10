"""Offline check of the env-exclusion rule (gap 5): recompute calibrate() for every built candidate of a run from the
reports its gates saved (cand/grade/{empty,solution-0,1,2}), without and with env_unreliable(); read-only for runs/.
Run from S/synth:  ../atp/runtime/controller/bin/python3 -B repair/gates-1020/calibrate_check.py RUN [RUN...]"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import task_builder as tb  # noqa: E402
from validate import build_of  # noqa: E402


def suites(directory: Path, b) -> dict:
    out = {}
    for report in b.reports:
        name = Path(report).name
        path = directory / name
        out[name] = tb.parse_report(path, b.grade_cfg) if path.exists() else {}
    return {"suites": out}


def main():
    total = {}
    for run in sys.argv[1:]:
        b = build_of(tb, run)
        unreliable = tb.env_unreliable(b)
        print(f"== {run}: env-unreliable tests: {sorted(unreliable)}")
        for cand in sorted(p for p in b.root.glob("cand-*") if (p / "built.json").exists()):
            grade = cand / "grade"
            empty, solved = suites(grade / "empty", b), [suites(grade / f"solution-{i}", b) for i in range(3)]
            f2p0, p2p0, problems0 = tb.calibrate(empty, solved, b.reports)
            f2p1, p2p1, problems1 = tb.calibrate(empty, solved, b.reports, unreliable)
            built = tb.read_json(cand / "built.json")
            recorded = len(built["gate"]["p2p"])
            dropped = sorted(set(p2p0) - set(p2p1))
            same = (f2p0 == f2p1, problems0 == problems1, len(p2p0) == recorded)
            total[run] = total.get(run, 0) + (1 if dropped else 0)
            print(f"   {cand.name}: P2P {len(p2p0)} -> {len(p2p1)} (recorded {recorded}); dropped {dropped}; "
                  f"f2p/problems unchanged {same[:2]}; recomputed = recorded {same[2]}")
    print("candidates with dropped tests:", total)


if __name__ == "__main__":
    main()
