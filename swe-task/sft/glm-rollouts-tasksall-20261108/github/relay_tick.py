#!/usr/bin/env python3
"""One tick of the GitHub relay for glm-rollouts-tasksall (user 2026-11-09: progress every hour + decisions in an
issue thread of github.com/hengyuf/SWE-task; the user answers there). Run by the relay subagent:

    /usr/bin/python3 -I -B relay_tick.py [--wait 540]

1. posts every message file in outbox/ (oldest first) as an issue comment and moves it to outbox/sent/;
2. posts a progress note (progress.py + issue_progress.py) when the last one is >= PROGRESS_EVERY_S old;
3. waits up to --wait s (polling every 60 s) for NEW comments of the user (author USER, not a relay post: relay posts
   carry MARKER) and stops early when one arrives.
Prints one JSON line: {"replies": [{"id", "created_at", "url", "body"}], "posted": [...], "progress_posted": bool}.
State (issue number, last seen comment id, last progress time) is in state.json next to this file. Uses ~/.local/bin/gh
(the user's own login); never prints credentials.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
GR = HERE.parent
GH = str(Path.home() / ".local/bin/gh")
REPO = "hengyuf/SWE-task"
USER = "hengyuf"
MARKER = "<!-- swe-task-relay -->"
PROGRESS_EVERY_S = 3600
STATE = HERE / "state.json"


def gh(*args, timeout=60):
    result = subprocess.run([GH, *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"gh {args[0]} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def post(issue: int, body: str) -> str:
    tmp = HERE / ".post-body.md"
    tmp.write_text(body.rstrip() + "\n\n" + MARKER + "\n")
    return gh("issue", "comment", str(issue), "--repo", REPO, "--body-file", str(tmp)).strip()


def save(state):
    tmp = STATE.with_name(".state.json.tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(STATE)


def new_replies(issue: int, last_id: int):
    out = json.loads(gh("api", f"repos/{REPO}/issues/{issue}/comments?per_page=100&sort=created&direction=desc"))
    replies = [c for c in out if c["id"] > last_id and (c.get("user") or {}).get("login") == USER
               and MARKER not in (c.get("body") or "")]
    return sorted(replies, key=lambda c: c["id"]), max([c["id"] for c in out] + [last_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=float, default=540)
    args = parser.parse_args()
    state = json.loads(STATE.read_text())
    issue = int(state["issue"])
    posted, progress_posted = [], False
    for path in sorted((HERE / "outbox").glob("*.md")):
        url = post(issue, path.read_text())
        path.rename(HERE / "outbox" / "sent" / path.name)
        posted.append({"file": path.name, "url": url})
    if time.time() - state.get("last_progress", 0) >= PROGRESS_EVERY_S:
        subprocess.run(["/usr/bin/python3", "-I", "-B", str(GR / "progress.py")], capture_output=True, timeout=900)
        note = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "issue_progress.py")], capture_output=True,
                              text=True, timeout=120).stdout
        if note.strip():
            posted.append({"progress": post(issue, note)})
            state["last_progress"] = time.time()
            progress_posted = True
    save(state)
    deadline = time.time() + args.wait
    replies = []
    while True:
        try:
            replies, _ = new_replies(issue, int(state.get("last_seen", 0)))
        except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
            replies, state["last_error"] = [], str(error)[:300]
        if replies or time.time() >= deadline:
            break
        time.sleep(60)
    if replies:
        state["last_seen"] = replies[-1]["id"]
    save(state)
    print(json.dumps({"replies": [{"id": c["id"], "created_at": c["created_at"], "url": c["html_url"],
                                   "body": c["body"]} for c in replies],
                      "posted": posted, "progress_posted": progress_posted, "last_error": state.get("last_error")}))


if __name__ == "__main__":
    main()
