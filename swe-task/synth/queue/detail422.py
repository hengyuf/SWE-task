#!/usr/bin/env python3
"""For recent 422/budget_exceeded responses in probe http journals: time, slot, error type, key-alias suffix (8 chars),
reported cost / max budget. Never prints bodies, headers or key fragments. Usage: detail422.py MINUTES RUN [RUN...]"""
import glob, json, os, re, sys, time
S = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs"
mins = float(sys.argv[1]); cut = time.time() - mins * 60
for run in sys.argv[2:]:
    rows = []
    for path in glob.glob(f"{S}/{run}/build/*/*/round-*/probe-*/models/*/attempts/*/attempt-*/http.jsonl"):
        st = os.stat(path)
        if st.st_mtime < cut:
            continue
        with open(path, "rb") as fh:
            fh.seek(max(0, st.st_size - 3 * 1024 * 1024)); data = fh.read()
        for line in data.split(b"\n"):
            if b'"status":422' not in line and b'budget_exceeded' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = d.get("timestamp") or 0
            if ts < cut:
                continue
            body = json.dumps(d.get("body"))
            etype = re.search(r'\\?"type\\?": ?\\?"([a-z_]+)', body)
            alias = re.search(r'Key=([A-Za-z0-9_.-]+)', body)
            cost = re.search(r'Current cost: ([0-9.]+)', body)
            mx = re.search(r'Max budget: ([0-9.]+)', body)
            rows.append((ts, path.split("/models/")[1].split("/http.jsonl")[0], d.get("status"), etype.group(1) if etype else None,
                         alias.group(1)[-8:] if alias else None, cost.group(1)[:9] if cost else None, mx.group(1) if mx else None))
    rows.sort()
    print(f"== {run}: {len(rows)} hits")
    for r in rows[:4] + (rows[-4:] if len(rows) > 8 else rows[4:]):
        print("  " + time.strftime('%H:%M:%S', time.localtime(r[0])), *r[1:])
