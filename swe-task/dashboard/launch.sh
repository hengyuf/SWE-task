#!/usr/bin/env bash
# Run the swe-task dashboard (server.py + its collector thread) as a small CPU Slurm job, or here in the foreground.
#
#   launch.sh [start] [--time 7-00:00:00] [--partition P | --partitions a,b,c] [--nodelist NODE] [--port 8765] [--reuse-token]
#                     [--wait 600] [--dry-run]
#       Refuses when a swe-task-dashboard job of $USER is already pending or running. Partition: unless --partition
#       is given, tools/pick_partition.py test-submits this exact job (sbatch --test-only: nothing is submitted) to
#       the candidates whose MaxTime allows --time (cpu_interactive caps at 1 day, so the 7-day default leaves
#       cpu_datamover and cpu_dataprocessing) and takes the earliest estimated start (ties: more idle nodes, then
#       the listed order). Submits dashboard.sbatch with a scrubbed environment, waits up to --wait s for the
#       server's state/server.json, checks /healthz from this host, then prints how to connect. --reuse-token keeps
#       the previous token and, unless --port is given, the previous server's port: the session cookie is per port
#       (swe_dash_<port>, derived from the token and the port), so open browser sessions stay valid. --dry-run prints
#       the probe result and the sbatch command and submits nothing.
#   launch.sh stop [--dry-run]    scancel the swe-task-dashboard job(s) of $USER, by name
#   launch.sh status              the job, how to connect (URL with the token), the collector's last cycle
#   launch.sh local [--port 0] [--no-collect | --collect] [--reuse-token] [--interval 120]
#       server.py in the foreground on 127.0.0.1 of this host (VS Code forwards the port by itself), with its own
#       info file state/server-local.json. Serve-only by default while the Slurm job runs (that job collects).
#       --reuse-token keeps the local server's token and, unless --port is given, its port.
#   launch.sh header [--file ~/.dashboard-header] [--local]
#       Writes the line "X-Dashboard-Token: <token>" of the Slurm job's server (--local: of the local one) to a file
#       of mode 600, for tools: curl -H @$HOME/.dashboard-header http://localhost:<port>/data/index.json. Never put
#       the token itself on a command line (ps and the shell history keep it).
#
# The token is printed only here, on your terminal (and kept in state/server.json, mode 600), never into a log file.
# Test hook: DASHBOARD_SLURM_BIN replaces /cm/shared/apps/slurm/current/bin (tests/test_server.py: fake commands).
set -euo pipefail
S=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=swe-task-dashboard
SLURM_BIN=${DASHBOARD_SLURM_BIN:-/cm/shared/apps/slurm/current/bin}
ME=${USER:-$(id -un)}
PY=/usr/bin/python3
INFO=$D/state/server.json
# Scrubbed environment for everything this script starts: nothing from the calling shell (keys, session tokens,
# proxies) reaches Slurm, the job or the server. SLURM_CONF: login nodes run no slurmd (fleet.py does the same).
CLEAN=(env -i HOME="$HOME" USER="$ME" LOGNAME="$ME" LANG=C.UTF-8
       SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf}"
       PATH="$SLURM_BIN:/usr/local/bin:/usr/bin:/bin")

die() { echo "launch.sh: $*" >&2; exit 1; }
need() { (($# >= 2)) || die "$1 needs a value"; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

secs() {  # Slurm time limit -> seconds ("infinite"/"UNLIMITED" -> a huge number); status 1 on an unknown format
  local t=$1 d=0 h=0 m=0 s=0
  case "${t,,}" in infinite|unlimited) echo 999999999; return 0 ;; esac
  if [[ $t =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?(:([0-9]+))?$ ]]; then  # days-hours[:minutes[:seconds]]
    d=${BASH_REMATCH[1]} h=${BASH_REMATCH[2]} m=${BASH_REMATCH[4]:-0} s=${BASH_REMATCH[6]:-0}
  elif [[ $t =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then             # hours:minutes:seconds
    h=${BASH_REMATCH[1]} m=${BASH_REMATCH[2]} s=${BASH_REMATCH[3]}
  elif [[ $t =~ ^([0-9]+):([0-9]+)$ ]]; then                      # minutes:seconds
    m=${BASH_REMATCH[1]} s=${BASH_REMATCH[2]}
  elif [[ $t =~ ^[0-9]+$ ]]; then                                 # minutes
    m=$t
  else
    return 1
  fi
  echo $((10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

jobs_list() {  # "id|state|partition|node|elapsed|left|reason" lines: $USER's dashboard jobs still pending or running
  "${CLEAN[@]}" squeue -h -u "$ME" -n "$NAME" -t PENDING,RUNNING,CONFIGURING,SUSPENDED -o '%i|%T|%P|%N|%M|%L|%r'
}

show_jobs() {  # show_jobs <jobs_list output>
  local id state part node elapsed left reason
  while IFS='|' read -r id state part node elapsed left reason; do
    [[ -n "$id" ]] || continue
    printf '  job %s  %s  partition %s  node %s  elapsed %s  left %s%s\n' "$id" "$state" "$part" "${node:--}" \
      "$elapsed" "$left" "$([[ $state == PENDING ]] && echo "  reason $reason")"
  done <<<"$1"
}

field() {  # field <info file> <key>: one top-level value of a server.py info file (empty when missing)
  "$PY" -I -B -c 'import json, sys
try:
    value = json.load(open(sys.argv[1])).get(sys.argv[2])
except (OSError, ValueError, AttributeError):
    value = None
print("" if value is None else value)' "$1" "$2" </dev/null
}

healthz() {  # healthz <host> <port>: "ok" when /healthz answers from this host (no proxy, no token needed)
  "${CLEAN[@]}" "$PY" -I -B -c 'import json, sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open("http://%s:%s/healthz" % (sys.argv[1], sys.argv[2]), timeout=5) as response:
        print("ok" if json.load(response).get("ok") is True else "unexpected answer")
except Exception as error:
    print("%s: %s" % (type(error).__name__, error))' "$1" "$2" </dev/null
}

connect() {  # connect <node> <port> <token>: how to open the dashboard (the token appears on this terminal only)
  local node=$1 port=$2 token=$3 login
  login=$(hostname -f 2>/dev/null || hostname)
  echo "Connect (the token is the password: it is shown only here, on your terminal):"
  echo "  VS Code: Ports view -> \"Forward a Port\" -> enter $node:$port, then open"
  echo "             http://localhost:$port/?token=$token"
  echo "           (if VS Code forwards it to another local port, use that port in the URL)"
  echo "  ssh:     ssh -L $port:$node:$port $ME@$login     then open the same URL"
  echo "The first visit stores a session cookie and drops the token from the address bar; later visits need no token."
  echo "Use a regular browser: VS Code's built-in Simple Browser is a cross-site frame and gets no cookie."
  echo "Tools: '$D/launch.sh header' writes $HOME/.dashboard-header (mode 600), then"
  echo "  curl -sS -H @$HOME/.dashboard-header http://localhost:$port/data/index.json"
}

report() {  # report <job> <info file>: where the job's server listens, whether it answers from here, how to connect
  local job=$1 info=$2 node port token health
  node=$("${CLEAN[@]}" squeue -h -j "$job" -o %N 2>/dev/null </dev/null || true)
  node=${node:-$(field "$info" host)}
  port=$(field "$info" port)
  token=$(field "$info" token)
  [[ -n "$port" && -n "$token" ]] || { echo "$info of job $job is incomplete" >&2; return 1; }
  health=$(healthz "$node" "$port")
  echo "Dashboard job $job serves on $node:$port (/healthz from $(hostname -s): $health)."
  [[ $health == ok ]] || echo "  warning: not answering from this host; see $D/logs/$NAME-$job.out" >&2
  connect "$node" "$port" "$token"
}

collector_status() {  # one paragraph from data/server_status.json (written by the collecting server)
  "$PY" -I -B -c 'import json, sys, time
now = time.time()
def ago(ts):
    return "%d s ago" % (now - ts) if isinstance(ts, (int, float)) else "never"
try:
    doc = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    doc = None
try:
    index_age = ago(json.load(open(sys.argv[2])).get("generated_at"))
except (OSError, ValueError, AttributeError):
    index_age = "missing"
if not isinstance(doc, dict):
    print("collector: no data/server_status.json yet; data/index.json written %s" % index_age)
    sys.exit(0)
server, c = doc.get("server") or {}, doc.get("collector") or {}
print("collector (server on %s, job %s): %s; %s cycles, %s failed; last finished %s in %s s with %s errors; "
      "data/index.json written %s" % (server.get("host"), server.get("slurm_job_id") or "-", c.get("state"),
      c.get("cycles"), c.get("failures"), ago(c.get("last_finished_at")), c.get("last_seconds"),
      c.get("last_error_count"), index_age))
for key in ("import_error", "last_failure"):
    if c.get(key):
        print("  %s: %s" % (key, c[key]))
for error in (c.get("last_errors") or [])[:3]:
    print("  error: %s" % error)' "$D/data/server_status.json" "$D/data/index.json" </dev/null
}

# ------------------------------------------------------------------------------------------------------ commands

start() {
  local walltime=7-00:00:00 partition="" partitions=cpu_datamover,cpu_dataprocessing,cpu_interactive port=""
  local wait_s=600 dry=0 reuse=0 extra=() nodelist=""
  while (($#)); do
    case "$1" in
      --time) need "$@"; walltime=$2; shift 2 ;;
      --partition) need "$@"; partition=$2; shift 2 ;;
      --partitions) need "$@"; partitions=$2; shift 2 ;;
      --nodelist) need "$@"; nodelist=$2; shift 2 ;;  # e.g. keep the previous job's node, so the URL stays
      --port) need "$@"; port=$2; shift 2 ;;
      --wait) need "$@"; wait_s=$2; shift 2 ;;
      --reuse-token) reuse=1; extra+=(--reuse-token); shift ;;
      --dry-run) dry=1; shift ;;
      *) die "unknown argument $1 (see launch.sh --help)" ;;
    esac
  done
  # Only the token is kept: the session cookie derives from it and the port, so the browser sessions survive a
  # --reuse-token restart on the port the previous server had (8765 unless that was taken on its node).
  local previous=""
  if ((reuse)); then
    previous=$(field "$INFO" port)
    [[ $previous =~ ^[0-9]+$ ]] || previous=""
    if [[ -z "$port" && -n "$previous" ]]; then
      port=$previous
    elif [[ -n "$previous" && $port != "$previous" ]]; then
      echo "note: the previous server listened on port $previous; sessions are per port, so open the new URL" \
           "with the token once" >&2
    fi
  fi
  port=${port:-8765}
  [[ $port =~ ^[0-9]+$ ]] && ((port < 65536)) || die "bad --port $port"
  [[ $wait_s =~ ^[0-9]+$ ]] || die "bad --wait $wait_s"
  local want
  want=$(secs "$walltime") || die "bad --time $walltime"

  local running
  running=$(jobs_list) || die "squeue failed: cannot tell whether a dashboard job already runs"
  if [[ -n "$running" ]]; then
    echo "A $NAME job of $ME is already pending or running:" >&2
    show_jobs "$running" >&2
    ((dry)) || die "not submitting another one ('launch.sh status' shows its URL, 'launch.sh stop' cancels it)"
    echo "(dry run: a real start would refuse here)" >&2
  fi

  # Candidates whose MaxTime allows the walltime (sinfo %l: "infinite", "1-00:00:00", ...).
  local candidates=${partition:-$partitions} eligible=() p limit max
  for p in ${candidates//,/ }; do
    limit=$("${CLEAN[@]}" sinfo -h -p "$p" -o %l </dev/null) || die "sinfo failed"
    limit=${limit%%$'\n'*}
    if [[ -z "$limit" ]]; then
      echo "skip $p: no such partition" >&2
    elif max=$(secs "$limit") && ((max < want)); then
      echo "skip $p: MaxTime $limit < --time $walltime" >&2
    else
      eligible+=("$p")
    fi
  done
  ((${#eligible[@]})) || die "no partition among $candidates allows --time $walltime"

  local job_args=(--job-name="$NAME" --time="$walltime" ${nodelist:+--nodelist="$nodelist"} "$D/dashboard.sbatch"
                  --port "$port" "${extra[@]}")
  local pick chosen
  if [[ -n "$partition" ]]; then
    chosen=$partition
    pick="{\"partition\": \"$partition\", \"probes\": [], \"chosen_by\": \"--partition\"}"
  else
    pick=$(cd "$D" && "${CLEAN[@]}" "$PY" -I -B "$S/tools/pick_partition.py" \
           --partitions "$(IFS=,; echo "${eligible[*]}")" -- "${job_args[@]}" </dev/null) \
      || die "pick_partition.py failed"
    chosen=$(printf '%s' "$pick" | "$PY" -I -c 'import json, sys; print(json.load(sys.stdin)["partition"])') \
      || die "unexpected pick_partition.py output: $pick"
  fi
  local argv=(sbatch --parsable --partition="$chosen" "${job_args[@]}")
  if ((dry)); then
    echo "probe: $pick"
    echo "DRY: (cd $D && env -i <scrubbed> $(printf '%q ' "${argv[@]}"))"
    return 0
  fi

  mkdir -p "$D/logs"  # sbatch --output: Slurm cannot create the directory itself
  local out job
  out=$(cd "$D" && "${CLEAN[@]}" "${argv[@]}" </dev/null) || die "sbatch failed"
  job=${out%%;*}
  [[ $job =~ ^[0-9]+$ ]] || die "unexpected sbatch output: $out"
  local record='{"submitted_at": "%s", "job_id": "%s", "partition": "%s", "time": "%s", "port": %s, "log": "%s",'
  record+=' "partition_probe": %s}\n'  # launch history, like runs/<run>/launch.jsonl of tools/launch_build.sh
  # shellcheck disable=SC2059
  printf "$record" "$(date -Is)" "$job" "$chosen" "$walltime" "$port" "$D/logs/$NAME-$job.out" "$pick" \
    >>"$D/logs/launch.jsonl"
  echo "Submitted $NAME job $job to $chosen (time $walltime); log $D/logs/$NAME-$job.out"
  wait_for "$job" "$wait_s" || return 1
  local actual
  actual=$(field "$INFO" port)
  if ((reuse)) && [[ "$(field "$INFO" slurm_job_id)" == "$job" && -n "$actual" && $actual != "$port" ]]; then
    echo "note: port $port was taken on the node, so the server took $actual; sessions are per port: open the URL" \
         "with the token once" >&2
  fi
}

wait_for() {  # wait_for <job> <seconds>: until the job's server has written state/server.json, then report
  local job=$1 deadline=$((SECONDS + $2)) line last="" state ok
  echo "Waiting up to $2 s for the server of job $job..."
  while :; do
    line=$("${CLEAN[@]}" squeue -h -j "$job" -o '%T|%N|%r|%S' 2>&1 </dev/null) && ok=1 || ok=0
    line=${line%%$'\n'*}
    # Gone: no longer listed, purged ("Invalid job id"), or listed in a final state for a few minutes after the end.
    # (Any other squeue failure is transient: try again.)
    if [[ $line == *"Invalid job id"* ]] || ((ok)); then
      state=${line%%|*}
      case "$state" in
        ""|*"Invalid job id"*|COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED)
          echo "Job $job ended (${state:-no longer queued}) before its server came up;" \
               "the end of $D/logs/$NAME-$job.out:" >&2
          tail -n 20 "$D/logs/$NAME-$job.out" >&2 2>/dev/null || true
          return 1 ;;
      esac
      if [[ $state == RUNNING && "$(field "$INFO" slurm_job_id)" == "$job" && -z "$(field "$INFO" stopped_at)" ]]; then
        report "$job" "$INFO"
        return 0
      fi
      if [[ $line != "$last" ]]; then
        echo "  job $job: ${line//|/  }"
        last=$line
      fi
    fi
    if ((SECONDS >= deadline)); then
      echo "Job $job is not serving yet after $2 s; run '$D/launch.sh status' later for the URL." >&2
      return 0
    fi
    sleep 5
  done
}

stop() {
  local dry=0
  while (($#)); do
    case "$1" in
      --dry-run) dry=1; shift ;;
      *) die "unknown argument $1 (stop [--dry-run])" ;;
    esac
  done
  local running
  running=$(jobs_list) || die "squeue failed"
  if [[ -z "$running" ]]; then
    echo "No $NAME job of $ME is pending or running."
    return 0
  fi
  echo "Cancelling:"
  show_jobs "$running"
  local argv=(scancel --user="$ME" --name="$NAME")
  if ((dry)); then
    echo "DRY: ${argv[*]}"
    return 0
  fi
  "${CLEAN[@]}" "${argv[@]}" </dev/null
  echo "Done: the server stops on the SIGTERM and marks state/server.json stopped."
}

status() {
  (($# == 0)) || die "status takes no arguments"
  local running id state rest
  running=$(jobs_list) || die "squeue failed"
  if [[ -z "$running" ]]; then
    echo "No $NAME job of $ME is pending or running ('launch.sh start' submits one)."
  else
    echo "$NAME jobs of $ME:"
    show_jobs "$running"
    while IFS='|' read -r -u 3 id state rest; do
      if [[ $state == RUNNING && "$(field "$INFO" slurm_job_id)" == "$id" ]]; then
        report "$id" "$INFO" || true
      elif [[ $state == RUNNING ]]; then
        echo "Job $id runs but its server has not written state/server.json yet (see $D/logs/$NAME-$id.out)."
      fi
    done 3<<<"$running"
  fi
  local pid host
  pid=$(field "$D/state/server-local.json" pid)
  host=$(field "$D/state/server-local.json" host)
  if [[ -n "$pid" && $host == "$(hostname)" && -z "$(field "$D/state/server-local.json" stopped_at)" ]] \
     && kill -0 "$pid" 2>/dev/null; then
    echo "A local server (launch.sh local, pid $pid) serves on this host: $(field "$D/state/server-local.json" url)"
  fi
  collector_status
}

local_run() {
  local port="" collect=auto reuse=0 extra=() running previous
  while (($#)); do
    case "$1" in
      --port) need "$@"; port=$2; shift 2 ;;
      --interval) need "$@"; extra+=(--interval "$2"); shift 2 ;;
      --no-collect) collect=no; shift ;;
      --collect) collect=yes; shift ;;
      --reuse-token) reuse=1; extra+=(--reuse-token); shift ;;
      *) die "unknown argument $1 (local [--port N] [--no-collect | --collect] [--reuse-token] [--interval S])" ;;
    esac
  done
  if ((reuse)) && [[ -z "$port" ]]; then  # the session cookie is per port: the previous port keeps the sessions
    previous=$(field "$D/state/server-local.json" port)
    if [[ $previous =~ ^[0-9]+$ ]]; then
      port=$previous
    fi
  fi
  port=${port:-0}
  if [[ $collect == auto ]]; then
    collect=yes
    running=$(jobs_list 2>/dev/null || true)
    if grep -q '^[0-9_]*|RUNNING|' <<<"$running"; then
      collect=no
      echo "The $NAME Slurm job is running and collecting: serving its data only (--collect collects here too)."
    fi
  fi
  if [[ $collect == no ]]; then
    extra+=(--no-collect)
  fi
  echo "Serving on 127.0.0.1 of $(hostname -s); Ctrl-C stops it. VS Code forwards the port: open the URL below."
  cd "$D"
  exec "${CLEAN[@]}" "$PY" -I -B "$D/server.py" --host 127.0.0.1 --port "$port" \
    --info-file "$D/state/server-local.json" --print-url "${extra[@]}"
}

header() {
  local file="$HOME/.dashboard-header" info=$INFO
  while (($#)); do
    case "$1" in
      --file) need "$@"; file=$2; shift 2 ;;
      --local) info=$D/state/server-local.json; shift ;;
      *) die "unknown argument $1 (header [--file F] [--local])" ;;
    esac
  done
  # Python reads the token from the info file and writes it: it is never an argument of any command.
  "$PY" -I -B -c 'import json, os, re, sys
info, target = sys.argv[1], os.path.abspath(sys.argv[2])
try:
    token = json.load(open(info)).get("token")
except (OSError, ValueError, AttributeError):
    token = None
if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
    sys.exit("launch.sh: no token in %s: no server has started yet (launch.sh status)" % info)
temporary = "%s.%d.tmp" % (target, os.getpid())
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
try:
    with os.fdopen(fd, "w") as stream:
        stream.write("X-Dashboard-Token: %s\n" % token)
    os.replace(temporary, target)
except BaseException:
    os.unlink(temporary)
    raise' "$info" "$file" </dev/null || exit 1
  echo "Wrote $file (mode 600): the X-Dashboard-Token header of the server in $info. Use it as"
  echo "  curl -sS -H @$file http://localhost:$(field "$info" port)/data/index.json"
}

command=start
case "${1:-}" in
  start|stop|status|local|header) command=$1; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
case "$command" in
  start) start "$@" ;;
  stop) stop "$@" ;;
  status) status "$@" ;;
  local) local_run "$@" ;;
  header) header "$@" ;;
esac
