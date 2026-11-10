"""glm-rollouts-tasksall: run ATP's job-local deepswe_stage.py with a verified local copy of its two GitHub release
downloads (2026-11-09; GitHub release assets answered HTTP 504 on and off from 2026-11-08 23:22, so legs died in ATP
staging). Nothing in the shared ATP app or in the staged scripts is edited.

    python3 -B stage_wrapper.py <scratch>/scripts/deepswe_stage.py --app <app> --scratch <scratch>

driver.py makes rollouts.probe() run this instead of `python3 -B <scratch>/scripts/deepswe_stage.py ...` (same
arguments, same environment). It imports that deepswe_stage.py as a module and rebinds two of its globals:
* urlopen: for exactly the OpenCode 1.17.11 baseline and ripgrep 15.1.0 musl release URLs it returns the cached file
  (release-cache/release-cache.json: sha256 re-checked on every use; byte-identical to what legs downloaded from GitHub
  on 2026-11-08, cross-checked against job 21548031's staged copies); a missing or mismatching cache file, or any other
  URL, goes to the real urlopen;
* run: the inner staging run (the job's CPython 3.13 re-running deepswe_stage.py) is started through this wrapper too.
Then it does what deepswe_stage.py's __main__ does (stage(), then the "DeepSWE runtime ready" line).
Works under the system python3 (3.10) and the staged CPython 3.13.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CACHE = json.loads((HERE / "release-cache" / "release-cache.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith("deepswe_stage.py"):
        raise SystemExit("usage: stage_wrapper.py <scratch>/scripts/deepswe_stage.py --app APP --scratch SCRATCH")
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    import argparse
    import deepswe_stage

    real_urlopen = deepswe_stage.urlopen
    real_run = deepswe_stage.run

    def urlopen(url, *args, **kwargs):
        entry = CACHE["assets"].get(url) if isinstance(url, str) else None
        if entry:
            path = HERE / "release-cache" / entry["file"]
            if path.is_file() and sha256(path) == entry["sha256"]:
                print(f"glm-rollouts-tasksall stage_wrapper: {url} from the verified cache {path}", flush=True)
                return open(path, "rb")
            print(f"glm-rollouts-tasksall stage_wrapper: cache for {url} missing or mismatching; downloading",
                  flush=True)
        return real_urlopen(url, *args, **kwargs)

    def run(command, cwd, env):
        command = list(command)
        if len(command) >= 2 and str(command[1]).endswith("scripts/deepswe_stage.py"):
            command = [command[0], str(Path(__file__).resolve()), *command[1:]]
        return real_run(command, cwd, env)

    deepswe_stage.urlopen = urlopen
    deepswe_stage.run = run
    parser = argparse.ArgumentParser(description=deepswe_stage.__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    arguments = parser.parse_args()
    result = deepswe_stage.stage(arguments.app, arguments.scratch)
    print(f"DeepSWE runtime ready: {result['scratch'] / 'staged.json'}")


if __name__ == "__main__":
    main()
