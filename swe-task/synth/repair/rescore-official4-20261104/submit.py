#!/usr/bin/env python3
"""Submit the official4 re-score (tools/rescore_unscored.py score) as a CPU job through hengyuf's dual-account helper
(schdule_helper.sh babysit_submit: two held candidates of synth/run_babysit.sbatch on nemotron_agents_dev and
nemotron_rl_algo, then released; the first to start wins). run_babysit.sbatch reads the command from
$RESULTS_DIR/babysit/argv. The submitting environment is the login one scrubbed like queue4/run_queue.job_env():
no secret-like, session-bound or Slurm-job names (the helper submits with --export=ALL).
    /usr/bin/python3 -I -B submit.py --partition cpu_interactive [--dry-run]
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
HELPER = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh")
BATCH = S / "synth/run_babysit.sbatch"
EXP_NAME = "synth-rescore-official4"
SLURM_CONF = "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"
SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")
DROP_VARS = {"SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "TMUX_PANE", "DISPLAY",
             "XDG_RUNTIME_DIR", "XDG_SESSION_ID", "SYNTH_STUDENT_BASE_URL", "OLDPWD", "PWD", "SHLVL", "_", "HOSTNAME",
             "TMPDIR", "TMP", "TEMP", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}
DROP_PREFIXES = ("VSCODE_", "CLAUDE", "ANTHROPIC", "TERM_PROGRAM", "BABYSIT_", "SLURM_", "SBATCH_", "SRUN_", "SALLOC_",
                 "PMI_", "PMIX_", "OMPI_", "GIT_")


def job_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (SECRET_NAME.search(k.upper()) or k in DROP_VARS or k.upper().startswith(DROP_PREFIXES))}
    parts = [p for p in (env.get("PATH") or "").split(":") if p]
    for extra in (SLURM_BIN, "/usr/local/bin", "/usr/bin", "/bin"):
        if extra not in parts:
            parts.append(extra)
    user = os.environ.get("USER") or "hengyuf"
    env.update(PATH=":".join(parts), SLURM_CONF=SLURM_CONF, USER=user, LOGNAME=user, RESULTS_DIR=str(HERE),
               BATCH_SCRIPT=str(BATCH), EXP_NAME=EXP_NAME)
    env.setdefault("HOME", f"/home/{user}")
    env.setdefault("LANG", "C.UTF-8")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runs = [entry["run"] for entry in json.loads((S / "synth/queue4/queue.json").read_text())["runs"]]
    argv = ["tools/rescore_unscored.py", "score", "--runs", *runs, "--concurrency", "3", "--passes", "6",
            "--pause-s", "900", "--max-wait-s", "14400"]
    sbatch_args = [f"--partition={args.partition}", f"--time={args.time}", "--cpus-per-task=8", "--mem=32G"]
    print(json.dumps({"argv": argv, "sbatch_args": sbatch_args, "results_dir": str(HERE), "exp_name": EXP_NAME}))
    if args.dry_run:
        return 0
    (HERE / "babysit").mkdir(exist_ok=True)
    temporary = HERE / "babysit/.argv.tmp"
    temporary.write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    temporary.replace(HERE / "babysit/argv")
    (HERE / "babysit/sbatch_args.json").write_text(json.dumps(sbatch_args) + "\n")
    result = subprocess.run(["bash", "-c", 'source "$0" && babysit_submit "$@"', str(HELPER), *sbatch_args],
                            env=job_env(), capture_output=True, text=True, timeout=900, cwd=str(HERE))
    print(result.stdout.strip())
    print(result.stderr.strip()[-2000:])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
