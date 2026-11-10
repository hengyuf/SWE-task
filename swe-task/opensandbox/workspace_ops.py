"""Workspace archives and OCI build contexts for the OpenSandbox helpers."""

from __future__ import annotations

import json
import fnmatch
import posixpath
import re
import shlex
import shutil
import tarfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


EXCLUDED_DIRS = (".codex", ".ssh", ".aws", ".cache", "__pycache__", ".venv", "venv", "node_modules")
EXCLUDED_FILES = (".env", ".env.*", ".git-credentials", ".netrc", ".npmrc", ".pypirc", ".git/config")
EXCLUDES = EXCLUDED_DIRS + EXCLUDED_FILES
_SYSTEM_TREES = ("/bin", "/sbin", "/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", "/run", "/var")
_HELPER_ROOT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/opensandbox").resolve()
_LOCAL_ROOTS = (_HELPER_ROOT, Path("/tmp").resolve(), Path("/raid/scratch").resolve())


def validate_workdir(value: str) -> str:
    """Require a specific repository directory, never a system directory."""
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or any(ord(char) < 32 for char in value)):
        raise ValueError("Repository directory must be an absolute path")
    path = PurePosixPath(value)
    if ".." in path.parts or posixpath.normpath(value) != value:
        raise ValueError("Repository directory must be normalized, without '..' or a trailing slash")
    if value in ("/", "/tmp", "/home", "/root", "/opt", "/raid", "/raid/scratch"):
        raise ValueError("Refusing a root or shared parent directory")
    if any(value == root or value.startswith(root + "/") for root in _SYSTEM_TREES):
        raise ValueError("Refusing a system directory")
    return value


def _local_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not any(path == root or root in path.parents for root in _LOCAL_ROOTS):
        raise ValueError("Local artifacts must be under this helper directory, /tmp, or /raid/scratch")
    return path


def _new_directory(value: str | Path) -> Path:
    path = _local_path(value)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    return path


def _archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return str(path) if str(path) != "." else ""


def excluded_path(relative: str) -> bool:
    """Report the default path exclusions used by backup (not a secret scanner)."""
    parts = PurePosixPath(relative).parts
    return (any(part in EXCLUDED_DIRS for part in parts)
            or any(fnmatch.fnmatchcase(part, pattern) for part in parts
                   for pattern in EXCLUDED_FILES if "/" not in pattern)
            or any(parts[i:i + 2] == (".git", "config") for i in range(len(parts) - 1)))


def validate_archive(archive_path: str | Path) -> None:
    """Reject escaping links, duplicate paths, and devices before remote extraction."""
    with tarfile.open(archive_path, "r:gz") as archive:
        entries = {}
        for member in archive:
            name = _archive_name(member.name)
            if name in entries:
                raise ValueError(f"Duplicate archive path: {name!r}")
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError(f"Unsupported archive entry: {member.name!r}")
            if not name and not member.isdir():
                raise ValueError("Archive root must be a directory")
            entries[name] = member
        links = {name for name, member in entries.items() if member.issym() or member.islnk()}

        def resolve(parts, seen=frozenset()):
            resolved = []
            for part in parts:
                if part in ("", "."):
                    continue
                if part == "..":
                    if not resolved:
                        raise ValueError("Archive symlink escapes the workspace")
                    resolved.pop()
                    continue
                resolved.append(part)
                key = "/".join(resolved)
                entry = entries.get(key)
                if entry is not None and entry.issym():
                    target = entry.linkname
                    if not target or posixpath.isabs(target) or "\x00" in target or key in seen:
                        raise ValueError(f"Unsafe or cyclic archive symlink: {key!r}")
                    resolved = resolve(resolved[:-1] + list(PurePosixPath(target).parts), seen | {key})
            return resolved

        for name, member in entries.items():
            if any(str(parent) in links for parent in PurePosixPath(name).parents):
                raise ValueError(f"Archive entry traverses a link: {name!r}")
            if member.issym():
                resolve(PurePosixPath(name).parts)
            elif member.islnk():
                target = _archive_name(member.linkname)
                if target not in entries or not entries[target].isfile():
                    raise ValueError(f"Hardlink must target an archived regular file: {name!r}")


async def _checked(client, sandbox_id, command, *, timeout=300):
    result = await client.exec(sandbox_id, command, timeout=timeout)
    if result.exit_code != 0:
        raise RuntimeError(f"Workspace operation failed (exit {result.exit_code}): {result.stderr or result.stdout}")
    return result


async def _remove_remote(client, sandbox_id, remote):
    try:
        await _checked(client, sandbox_id, f"rm -f -- {shlex.quote(remote)}", timeout=30)
    except Exception as error:
        warnings.warn(f"Temporary archive cleanup failed: {type(error).__name__}; {remote}")


async def _capture(client, sandbox_id, command, workdir, timeout=60):
    try:
        result = await client.exec(sandbox_id, command, cwd=workdir, timeout=timeout)
        return {"exit_code": result.exit_code, "available": result.exit_code == 0}, result.stdout
    except Exception as error:
        return {"exit_code": None, "available": False, "error_type": type(error).__name__}, ""


async def backup(client, sandbox_id: str, workdir: str, output_dir: str | Path) -> dict:
    """Save a repository archive, binary Git patch, and best-effort inventories."""
    workdir = validate_workdir(workdir)
    output = _new_directory(output_dir)
    quoted = shlex.quote(workdir)
    await _checked(client, sandbox_id,
                   f"test -d {quoted} && test \"$(readlink -f -- {quoted})\" = {quoted}")
    remote = f"/tmp/swe-task-backup-{uuid.uuid4().hex}.tar.gz"
    archive = output / "workspace.tar.gz"
    excludes = " ".join(shlex.quote("--exclude=" + pattern) for pattern in EXCLUDES)
    try:
        await _checked(client, sandbox_id,
                       f"tar {excludes} -czf {shlex.quote(remote)} -C {quoted} .", timeout=900)
        await client.download(sandbox_id, remote, archive.with_suffix(".part"))
    finally:
        await _remove_remote(client, sandbox_id, remote)
    validate_archive(archive.with_suffix(".part"))
    archive.with_suffix(".part").rename(archive)
    commit, commit_output = await _capture(client, sandbox_id, "git rev-parse HEAD", workdir)
    pathspecs = [f":(exclude,glob)**/{pattern}" for pattern in EXCLUDES]
    pathspecs += [f":(exclude,glob)**/{pattern}/**" for pattern in EXCLUDED_DIRS]
    patch, patch_output = await _capture(client, sandbox_id, "git diff --binary HEAD -- . " +
                                         " ".join(shlex.quote(item) for item in pathspecs), workdir, 120)
    if patch["available"]:
        (output / "git.patch").write_text(patch_output, encoding="utf-8")
    inventories = {}
    for filename, command in (
        ("packages-python.json", "python3 -m pip list --format=json"),
        ("packages-os.txt", "dpkg-query -W -f='${Package}\\t${Version}\\n'"),
    ):
        status, contents = await _capture(client, sandbox_id, command, workdir)
        inventories[filename] = status
        if status["available"]:
            (output / filename).write_text(contents, encoding="utf-8")
    manifest = {
        "sandbox_id": sandbox_id, "workdir": workdir,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive.name, "excludes": list(EXCLUDES),
        "git_commit": commit_output.strip() if commit["available"] else None,
        "git_commit_status": commit, "git_patch_status": patch,
        "inventories": inventories,
        "limitations": ["Workspace files only; packages outside this directory require a setup recipe.",
                        "Path exclusions do not remove secrets from Git history.",
                        "Stop workspace writers and checkpoint databases before taking an archive."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


async def restore(client, sandbox_id: str, backup_dir: str | Path, destination: str,
                  replace: bool = False) -> dict:
    """Restore into an empty directory; explicit replacement also preserves deletions."""
    destination = validate_workdir(destination)
    source = _local_path(backup_dir)
    archive = source / "workspace.tar.gz"
    validate_archive(archive)
    remote = f"/tmp/swe-task-restore-{uuid.uuid4().hex}.tar.gz"
    suffix = uuid.uuid4().hex
    parent = posixpath.dirname(destination)
    script = f"""set -eu
destination={shlex.quote(destination)}
parent={shlex.quote(parent)}
staging={shlex.quote(destination + '.restore-' + suffix)}
previous={shlex.quote(destination + '.previous-' + suffix)}
test -d "$parent"
test "$(readlink -f -- "$parent")" = "$parent"
test ! -L "$destination"
if test -e "$destination"; then
    test -d "$destination"
    {'true' if replace else 'test -z "$(find "$destination" -mindepth 1 -maxdepth 1 -printf x -quit)"'}
fi
mkdir -- "$staging"
cleanup() {{
    status=$?
    trap - EXIT INT TERM
    if test -e "$previous" && test ! -e "$destination" && test ! -L "$destination"; then
        if ! mv -- "$previous" "$destination"; then
            printf '%s\\n' "Restore rollback failed; original directory retained at $previous" >&2
            if test "$status" -eq 0; then status=1; fi
        fi
    fi
    if ! rm -rf -- "$staging"; then
        if test "$status" -eq 0; then status=1; fi
    fi
    exit "$status"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
tar --no-same-owner -xzf {shlex.quote(remote)} -C "$staging"
if test -e "$destination"; then mv -- "$destination" "$previous"; fi
if mv -- "$staging" "$destination"; then
    rm -rf -- "$previous"
else
    exit 1
fi
"""
    try:
        await client.upload(sandbox_id, archive, remote)
        await _checked(client, sandbox_id, script, timeout=900)
    finally:
        await _remove_remote(client, sandbox_id, remote)
    return {"sandbox_id": sandbox_id, "destination": destination, "replaced": replace}


async def export_build_context(client, sandbox_id: str, workdir: str, output_dir: str | Path,
                               base_image: str, setup_script: str | Path | None = None) -> dict:
    """Export an explicit base-image recipe; this does not export the container rootfs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", base_image):
        raise ValueError("Base image must be a literal OCI image reference")
    workdir = validate_workdir(workdir)
    if setup_script is not None and not Path(setup_script).is_file():
        raise FileNotFoundError(setup_script)
    manifest = await backup(client, sandbox_id, workdir, output_dir)
    output = _local_path(output_dir)
    clean = f"rm -rf -- {shlex.quote(workdir)} && mkdir -p -- {shlex.quote(workdir)}"
    lines = [f"FROM {base_image}", "USER 0", "RUN " + json.dumps(["/bin/sh", "-c", clean]),
             "ADD " + json.dumps(["workspace.tar.gz", workdir + "/"]),
             "WORKDIR " + json.dumps(workdir)]
    if setup_script is not None:
        shutil.copyfile(setup_script, output / "setup.sh")
        lines += ['COPY ["setup.sh", "/tmp/swe-task-setup.sh"]',
                  'RUN ["/bin/sh", "-c", "/bin/sh /tmp/swe-task-setup.sh && rm /tmp/swe-task-setup.sh"]']
    lines += ['ENTRYPOINT []', 'CMD ["tail", "-f", "/dev/null"]']
    (output / "Dockerfile").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / ".dockerignore").write_text("*\n!Dockerfile\n!workspace.tar.gz\n!setup.sh\n", encoding="utf-8")
    manifest["build"] = {"base_image": base_image, "platform": "linux/amd64",
                          "setup_script": "setup.sh" if setup_script is not None else None,
                          "note": "Only the workspace is exported; reproduce all other changes in setup.sh."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
