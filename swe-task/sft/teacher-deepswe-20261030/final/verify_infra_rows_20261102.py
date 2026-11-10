"""One-off patch (collect phase, 2026-11-02 PST): final/verify_collection.py's row-status check (check 2) assumed every
dataset row is graded (true for the 2026-10-31 partial collection). A complete campaign's dataset also holds its final
infra slots as driver.py's results record them (status "infra"; here deepseek w1-g0 langchain-request-coalescing r4,
the verifier killed at its 1,800 s timeout, ATP needs_review). Such a row is accepted only when its campaign is complete
and its terminal.json has no completed evaluation; it is counted in rows.final_infra_rows.

    python3 verify_infra_rows_20261102.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
old = '''    status_checked = 0
    for row in index:
        terminal = read_json(Path(row["attempt_dir"]) / "terminal.json", {}) or {}
        want = "pass" if terminal.get("reward") == 1 else "fail"
        if terminal.get("evaluation_completed") is not True or row["status"] != want:
            problems.append(f"status {row['status']} vs terminal reward {terminal.get('reward')} "
                            f"(evaluation_completed {terminal.get('evaluation_completed')}): {row['attempt_dir']}")
        status_checked += 1
'''
new = '''    status_checked = 0
    final_infra_rows = []
    for row in index:
        terminal = read_json(Path(row["attempt_dir"]) / "terminal.json", {}) or {}
        if row["status"] == "infra":  # a complete campaign's final infra slot (driver.py's record, ATP not graded)
            if not row.get("campaign_complete") or terminal.get("evaluation_completed") is True:
                problems.append(f"infra row outside a complete campaign or with a completed evaluation "
                                f"(evaluation_completed {terminal.get('evaluation_completed')}): {row['attempt_dir']}")
            final_infra_rows.append(row["attempt_dir"])
            status_checked += 1
        else:
            want = "pass" if terminal.get("reward") == 1 else "fail"
            if terminal.get("evaluation_completed") is not True or row["status"] != want:
                problems.append(f"status {row['status']} vs terminal reward {terminal.get('reward')} "
                                f"(evaluation_completed {terminal.get('evaluation_completed')}): {row['attempt_dir']}")
            status_checked += 1
'''
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new)
old2 = '''    out["rows"] = {"gz_lines": gz_lines, "index_rows": len(index), "status_checked": status_checked}
'''
new2 = '''    out["rows"] = {"gz_lines": gz_lines, "index_rows": len(index), "status_checked": status_checked,
                   "final_infra_rows": final_infra_rows}
'''
assert text.count(old2) == 1, text.count(old2)
text = text.replace(old2, new2)
dst.write_text(text)
print("ok", dst)
