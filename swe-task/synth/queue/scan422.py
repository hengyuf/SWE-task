#!/usr/bin/env python3
"""Count hub HTTP 422 / budget_exceeded responses in the tails of recently modified probe http.jsonl journals.
Prints counts and timestamps only (never bodies/headers). Usage: scan422.py MINUTES RUN [RUN...]"""
import glob, os, sys, time
S = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs"
mins = float(sys.argv[1]); runs = sys.argv[2:]
cut = time.time() - mins * 60
TAIL = 3 * 1024 * 1024
for run in runs:
    hits = []
    files = 0
    for path in glob.glob(f"{S}/{run}/build/*/*/round-*/probe-*/models/*/attempts/*/attempt-*/http.jsonl"):
        try:
            st = os.stat(path)
        except OSError:
            continue
        if st.st_mtime < cut:
            continue
        files += 1
        with open(path, "rb") as fh:
            fh.seek(max(0, st.st_size - TAIL))
            data = fh.read()
        for line in data.split(b"\n"):
            if b'"status":422' in line or b'budget_exceeded' in line:
                ts = None
                i = line.find(b'"timestamp":')
                if i >= 0:
                    try:
                        ts = float(line[i + 12:].split(b",")[0].split(b"}")[0])
                    except ValueError:
                        ts = None
                if ts is None or ts >= cut:
                    hits.append((ts, path.split("/models/")[1].split("/http.jsonl")[0]))
    last = max((h[0] for h in hits if h[0]), default=None)
    print(f"{run}: recent journals {files}, 422/budget hits in window {len(hits)}"
          + (f", last {time.strftime('%H:%M:%S', time.localtime(last))}, slots {sorted(set(h[1] for h in hits))[:6]}" if hits else ""))
