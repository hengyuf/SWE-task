"""Re-apply the final stale-oracle rules (removed_source_items' kinds + one name or two literals) to the saved
validate/oracles-*.json scans (offline; the scan output itself does not depend on the rules)."""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import task_builder as tb  # noqa: E402
from validate import build_of, candidate  # noqa: E402

rows = []
for path in sorted((HERE / "validate").glob("oracles-official1-*.json")):
    saved = json.loads(path.read_text())
    if not saved.get("stale"):
        rows.append((saved["run"], saved["cand"], "quiet", {}))
        continue
    b = build_of(tb, saved["run"])
    c = candidate(tb, b, saved["cand"])
    items = tb.removed_source_items(c["start_patch"], c["globs"])
    stale = {}
    for file, found in saved["stale"].items():
        kept = [item for item in found if item in items]
        if len(kept) >= 2 or any(items[item]["kind"] == "name" or " " in item for item in kept):
            stale[file] = {item: items[item]["kind"] for item in kept}
    rows.append((saved["run"], saved["cand"], "FLAGGED" if stale else "quiet (was flagged)", stale or saved["stale"]))
for run, cand, verdict, detail in rows:
    if verdict != "quiet":
        print(f"{run} {cand}: {verdict}: {json.dumps(detail)[:400]}")
print(f"{len(rows)} scans; flagged now: {sum(r[2] == 'FLAGGED' for r in rows)}; "
      f"no longer flagged: {sum(r[2].startswith('quiet (') for r in rows)}")
