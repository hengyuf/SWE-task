## Remote repository sandbox
The repository lives at /app inside one remote OpenSandbox container owned by this job {{network_note}}. You are not
inside that container: reach it only through these wrappers in your working directory.
- `bin/osb-exec '<shell command>' [timeout_seconds]` runs the command with cwd /app (default timeout 1800 s), prints
  its stdout/stderr and exits with the remote exit code.
- `bin/osb-put <local_file> <remote_path>` and `bin/osb-get <remote_path> <local_file>` copy one file; tar a
  directory first to move it.
Keep remote scratch files under /tmp/work. Never read, print or copy credentials (.env, .api, auth.json, keys,
tokens). Do not touch any other sandbox, Slurm job, or any local path outside your working directory.
