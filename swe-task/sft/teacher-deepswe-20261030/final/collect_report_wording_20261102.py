"""One-off patch (collect phase, 2026-11-02 PST), collect.py's FINAL-REPORT.md cost section only (no number changes):
1. a bullet that splits "all attempts" = graded slots' final attempts + ungraded slots' attempts + the earlier attempts
   of slots ATP retried (the table's columns do not add up without it);
2. the NEW_3 / NEW_2 "rest" of the key's spend is other use of the key (the r6 RL rubric judge, key_budget.py's own
   checks) plus any gap between the journal's cost and LiteLLM's spend figure, not only the r6 judge.

    python3 collect_report_wording_20261102.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()


def sub(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)


sub('''    lines.append("- Pilots (NVAPI_KEY_NEW_2): " + ", ".join(
''', '''    lines.append("- all attempts = the graded slots' final attempts + the ungraded slots' attempts + the earlier attempts "
                 "of slots that ATP retried before grading them: " + ", ".join(
                     f"{m} {money(c[m]['all_attempts'] - c[m]['graded_final_attempts'] - c[m]['ungraded_attempts'])}"
                     for m in MODELS) + " in earlier attempts.")
    lines.append("- Pilots (NVAPI_KEY_NEW_2): " + ", ".join(
''')
sub('''                     f"campaign's journals account for {money(mine3)}; the rest is the r6 RL run's rubric judge and any "
                     "other user of the key. LiteLLM's spend figure lags, and replicas differ by about $20.")
''', '''                     f"campaign's journals account for {money(mine3)}. The rest is other use of the key (the r6 RL "
                     "run's rubric judge, key_budget.py's own checks) plus any gap between the journal's cost and "
                     "LiteLLM's spend figure. That figure lags, and replicas differ by about $20.")
''')
sub('''                     f"({money(since - key_totals.get('NVAPI_KEY_NEW_2', 0.0))}) is the r6 RL rubric judge, which "
                     "uses NEW_2 since 10-31 18:15, and any other user of the key.")
''', '''                     f"({money(since - key_totals.get('NVAPI_KEY_NEW_2', 0.0))}) is other use of the key: the r6 RL "
                     "rubric judge (on NEW_2 since 10-31 18:15) and key_budget.py's own checks. It also holds any gap "
                     "between the journal's cost and LiteLLM's spend figure.")
''')
dst.write_text(text)
print("ok", dst)
