export const meta = {
  name: 'dashboard-host-keeper',
  description: 'Every 30 min: check the swe-task dashboard host and keep state/server.json pointing at a live, healthy server (restart with the same token before the 7-day limit or after a failure); alert main only on problems and actions',
  whenToUse: 'hengyuf swe-task: keep the dashboard web page host alive and its connection config current (args: {interval_min?: 30, checks_per_window?: 6, max_windows?: 400})',
  phases: [
    { title: 'Keep', detail: 'one agent per window of checks, 30 min apart; snapshots in dashboard/logs/host-watch.jsonl' },
  ],
}

// Saved at S/synth/workflows/dashboard_host_watch.js (main, 2026-10-20). hengyuf: "用一个subagent 每半小时检查网页host 状态",
// then "这个 dashboard/state/server.json 的配置是最新的吗, 让agent 不断更新这个config".
// Stop it: touch S/dashboard/state/host-watch.stop (the current window returns at its next check).

const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task'
const D = `${S}/dashboard`
const INTERVAL = (args && args.interval_min) || 30
const PER_WINDOW = (args && args.checks_per_window) || 6
const MAX_WINDOWS = (args && args.max_windows) || 400

const SNAPSHOT = {
  type: 'object',
  properties: {
    window: { type: 'integer' },
    checks: { type: 'integer' },
    ok_checks: { type: 'integer' },
    config_current: { type: 'boolean' },
    job: { type: 'string' },
    host_port: { type: 'string' },
    job_state: { type: 'string' },
    time_left: { type: 'string' },
    last_data_age_s: { type: 'number' },
    hub: { type: 'string' },
    actions: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
    alerts_sent: { type: 'array', items: { type: 'string' } },
    stop: { type: 'boolean' },
    stop_reason: { type: 'string' },
  },
  required: ['window', 'checks', 'ok_checks', 'config_current', 'actions', 'problems', 'alerts_sent', 'stop'],
}

function windowPrompt(w, previous) {
  return `You are the DASHBOARD HOST KEEPER for hengyuf's swe-task pipeline, window ${w + 1}. Run ${PER_WINDOW} checks, ${INTERVAL} min apart (the first one right away), then return your snapshot.
Your job is to keep ${D}/state/server.json CURRENT: it must always describe a live, healthy dashboard server, so hengyuf can connect with it. Message main only when a rule below says so.

S=${S}. The dashboard is ${D}; read its README.md, "Operations".
- A Slurm CPU job ("swe-task-dashboard", 7-day limit) runs server.py. It serves static/ and data/ with a token and runs the collector every 120 s.
- server.py WRITES state/server.json itself when it starts. Its keys are host, port, slurm_job_id, pid, started_at, url, local_url, token, ... and the url fields contain the token. So the file is updated only by (re)starting the server through launch.sh. NEVER edit state/server.json by hand: the token must match the running server's.
- hengyuf connects by forwarding <host>:<port> (VS Code Ports view or ssh -L) and opening http://localhost:<port>/?token=...
  - A restart with --reuse-token keeps the token and the browser sessions on the same port.
  - --nodelist <host> also keeps the node, so the forwarded address stays valid.

Previous window's snapshot (for continuity, may be null): ${previous ? JSON.stringify(previous) : 'null'}

## Secrets
- Never read .env files or S/secrets. Never print, log or pass the token or a URL containing it.
- Read state/server.json only inside a python3 -I process that keeps the token in memory, and print only non-secret fields.
- launch.sh start/status print a "Connect" block containing the token. Run launch.sh only as: (umask 077; bash ${D}/launch.sh <args> > /tmp/dash-keeper-$USER.out 2>&1). Then read that file only through: grep -v -i -e token -e 'X-Dashboard' /tmp/dash-keeper-$USER.out. Delete it afterwards.
- Never run launch.sh status or header.

## One check
1. Stop file: if ${D}/state/host-watch.stop exists, return stop=true with stop_reason "stop file".
2. Read the non-secret server.json fields (host, port, slurm_job_id, pid, started_at, stopped_at if present), plus ${D}/data/server_status.json server and collector.
3. Slurm, with a scrubbed env:
   env -i HOME=$HOME USER=$USER LANG=C.UTF-8 SLURM_CONF=/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf PATH=/cm/shared/apps/slurm/current/bin:/usr/bin:/bin squeue -h -u hengyuf -n swe-task-dashboard -o '%i %T %N %M %l %L'
   Use sacct for a job that squeue no longer lists.
4. HTTP from this host, no proxy, 15 s timeouts, in ONE python3 -I process that takes the token from server.json without printing it:
   - /healthz (no token, expect 200 and ok true);
   - / without the token (expect 403) and with header X-Dashboard-Token (expect 200);
   - /data/index.json, /data/fleet.json and /data/api.json with the header (expect 200 and generated_at younger than 300 s);
   - report run/task counts, the hub key NAME and remaining dollars.
5. CONFIG CURRENT means all of these hold:
   - server.json's slurm_job_id is RUNNING;
   - its host equals squeue's node;
   - its pid equals data/server_status.json's server.pid;
   - server_status's slurm_job_id equals server.json's;
   - server.json has no stopped_at;
   - /healthz answers at server.json host:port;
   - the token authenticates.
6. Once per window (first check), run the contract validator read-only: (cd ${D} && /usr/bin/python3 -I -B tests/validate_contract.py --sources), timeout 300 s.
   - KNOWN pre-existing issue, report only as "known": "api.json burn: eta_exhaust must be hub.sampled_at + remaining / rate ...".
   - Anything else is NEW.
7. Append ONE JSON line per check to ${D}/logs/host-watch.jsonl (mode 600; one write) with no token or URL. Fields: ts, window, job, job_state, host, port, pid, time_left, config_current, healthz, unauth_root, auth_root, data ages, runs, tasks, hub key/remaining, collector cycles/failures/consecutive_failures/last_ok, contract, actions, problems, ok.

## Keeping the config current (actions you MAY take, each followed at once by a message to main)
A. RENEWAL before the walltime. If the dashboard job has less than 45 min left (squeue %L):
   (umask 077; bash ${D}/launch.sh stop > /tmp/dash-keeper-$USER.out 2>&1)
   then, once squeue no longer lists a dashboard job (poll every 15 s, up to 5 min):
   (umask 077; bash ${D}/launch.sh start --reuse-token --nodelist <old host> > /tmp/dash-keeper-$USER.out 2>&1)
   If the new job is still PENDING after 10 min, run launch.sh stop again, then launch.sh start --reuse-token without --nodelist (any node; the address hengyuf forwards then changes, so say so).
B. FAILURE. Restart the same way (stop if a broken job still exists, then start --reuse-token --nodelist <old host>) when either:
   - no dashboard job is pending or running; or
   - the job is RUNNING but /healthz or the authenticated / fails 3 times within 5 min (retry every 2 min); or
   - all data files are older than 900 s while collector consecutive_failures >= 2.
C. After any start:
   - wait up to 10 min for state/server.json to show the NEW job id;
   - re-run the check;
   - message main with: old job, new job, host:port (same or changed), token kept (yes/no), config current (yes/no).
   If the new server is not healthy after 10 min, do not retry in a loop: message main and wait for the next check.
   At most 2 restarts per 24 h; after that, only alert.
D. Mismatch: if a healthy dashboard job runs but server.json does not describe it (other job id, host or pid), do NOT edit the file and do NOT restart. Message main with the details.

## Messages to main (SendMessage "main"; first line is a self-contained plain summary)
- Send on: every action (A/B/C), a mismatch (D), unauthenticated / not returning 403 (security regression), collector consecutive_failures >= 2, data older than 600 s, and a NEW contract problem.
- Do not repeat the same ongoing alert more than once every 2 h. Do send "recovered" when it clears.
- Never message about healthy checks.

## Other rules
- Do not edit dashboard code, data or state. Your writes are only logs/host-watch.jsonl and the temporary launch.sh output file.
- Return the snapshot after the window's ${PER_WINDOW} checks. problems are this window's problems, actions are what you did, alerts_sent is one line per message.
- stop=true only for the stop file.`
}

phase('Keep')
let previous = null
let failures = 0
for (let w = 0; w < MAX_WINDOWS; w++) {
  const snap = await agent(windowPrompt(w, previous), { label: `window ${w + 1}`, phase: 'Keep', schema: SNAPSHOT })
  if (!snap) {
    failures++
    log(`window ${w + 1}: no snapshot (${failures} in a row)`)
    if (failures >= 3) {
      log('3 windows in a row without a snapshot: stopping')
      return { windows: w + 1, stopped: 'no snapshots', last: previous }
    }
    continue
  }
  failures = 0
  previous = snap
  log(`window ${w + 1}: ${snap.ok_checks}/${snap.checks} ok; config ${snap.config_current ? 'current' : 'NOT current'}; job ${snap.job || '?'} ${snap.job_state || '?'} at ${snap.host_port || '?'} (left ${snap.time_left || '?'}); actions ${snap.actions.length}; alerts ${snap.alerts_sent.length}`)
  if (snap.stop) {
    return { windows: w + 1, stopped: snap.stop_reason || 'stop', last: snap }
  }
}
return { windows: MAX_WINDOWS, stopped: 'max windows', last: previous }
