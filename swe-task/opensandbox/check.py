#!/usr/bin/env python3
"""Small offline checks and an opt-in, owned-resource Cell-3 round trip.

Run on CPU-node scratch. The live mode creates and deletes only its own IDs.
"""
import argparse
import asyncio
import io
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from unittest.mock import patch

from sandbox_helpers import Client, Config, DEFAULT_ENV_FILE
from opensandbox.models.sandboxes import SandboxFilter, SnapshotFilter
import workspace_ops

IMAGE = "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh75679ajj3b8dtd7se3h7z0a1833y6r-v1.1"


def make_archive(path, name, *, link=None):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if link is not None:
            member.type = tarfile.SYMTYPE
            member.linkname = link
            archive.addfile(member)
        else:
            payload = b"fixture\n"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def offline(root, record):
    """Validate actual archive payloads, including traversal and escaping links."""
    safe = root / "safe.tar.gz"
    make_archive(safe, "sub/file.txt")
    workspace_ops.validate_archive(safe)
    for index, (name, link) in enumerate([
        ("../../outside", None), ("/etc/escape", None),
        ("link", "/etc/passwd"), ("sub/link", "../../outside"),
    ]):
        unsafe = root / f"unsafe-{index}.tar.gz"
        make_archive(unsafe, name, link=link)
        try:
            workspace_ops.validate_archive(unsafe)
        except (ValueError, RuntimeError, tarfile.TarError):
            pass
        else:
            raise AssertionError(f"Unsafe archive accepted: case {index}")
    record("archive_traversal_and_link_validation")
    for path in (".env", "sub/.env.local", ".codex/auth.json", "sub/.git/config", ".venv/bin/python"):
        assert workspace_ops.excluded_path(path), path
    for path in ("src/main.py", "uv.lock", "Dockerfile", ".git/HEAD"):
        assert not workspace_ops.excluded_path(path), path
    for path in ("/", "/tmp", "/etc/project", "/app/../etc"):
        try:
            workspace_ops.validate_workdir(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe workspace root accepted")
    record("workspace_roots_and_exclusions")

    env_file = root / "test.env"
    env_file.write_text("export OPENSANDBOX_DOMAIN='https://example.invalid/'\n"
                        "OPENSANDBOX_API_KEY='$(touch must-not-execute)'\n"
                        "NVIDIA_API_KEY=not-an-opensandbox-setting\n")
    with patch.dict(os.environ, {}, clear=True):
        config = Config.load(env_file)
        assert config.domain == "example.invalid" and config.tls_verify
        assert config.api_key == "$(touch must-not-execute)"
        assert config.api_key not in repr(config)
    with patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "override", "OPENSANDBOX_TLS_VERIFY": "false"}):
        config = Config.load(env_file)
        assert config.api_key == "override" and not config.tls_verify
    assert not Path("must-not-execute").exists()
    record("literal_credentials_and_environment_precedence")

    # Exercise the real shell interface using a fake runner, without publishing anything.
    context = root / "build context with spaces"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    runner = root / "bin"
    runner.mkdir()
    fake = runner / "buildctl-daemonless.sh"
    fake.write_text("#!/bin/sh\nset -eu\n"
                    "printf '%s\\n' \"$@\" > \"$FAKE_ARGS\"\n"
                    "while test $# -gt 0; do\n"
                    "  if test \"$1\" = --metadata-file; then shift; metadata=$1; fi\n"
                    "  shift\ndone\n"
                    "if test \"${FAKE_INVALID:-0}\" = 1; then echo '{}' > \"$metadata\"; else\n"
                    "  printf '{\\n  \"containerimage.digest\": \"sha256:"
                    + "a" * 64 + "\"\\n}\\n' > \"$metadata\"\nfi\n")
    fake.chmod(0o700)
    docker_config = root / "docker-config"
    docker_config.mkdir()
    (docker_config / "config.json").write_text("{}\n")
    environment = dict(os.environ, PATH=str(runner) + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
                       DOCKER_CONFIG=str(docker_config), FAKE_ARGS=str(root / "build-args.txt"))
    script = Path(__file__).with_name("build_image.sh")
    output = root / "metadata.json"
    result = subprocess.run(["bash", str(script), str(context), "example.invalid/team/task:check", str(output)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "example.invalid/team/task@sha256:" + "a" * 64
    arguments = (root / "build-args.txt").read_text().splitlines()
    assert "context=" + str(context) in arguments
    assert "type=image,name=example.invalid/team/task:check,push=true" in arguments
    result = subprocess.run(["bash", str(script), str(context), "example.invalid/team/task:check", str(root / "bad-metadata.json")],
                            env=dict(environment, FAKE_INVALID="1"), capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout
    record("build_script_arguments_and_missing_digest_failure")


async def live(args, root, report, record):
    token = "helper-check-" + uuid.uuid4().hex[:12]
    sandbox_ids, snapshot_ids = [], []
    cleanup_errors = []
    metadata = {"purpose": "swe-task-opensandbox-check", "run": token}
    report["run"] = token
    config = Config.load(args.env_file)
    async with Client(config, timeout=60) as client:
        def cli(*arguments, expected=0):
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("sandbox_helpers.py")),
                 "--env-file", str(args.env_file), *map(str, arguments)],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == expected, result.stderr
            return result

        async def command(sid, text, expected=0, **kwargs):
            result = await client.exec(sid, text, timeout=60, **kwargs)
            if result.exit_code != expected:
                raise AssertionError(
                    f"Remote fixture command returned {result.exit_code}, expected {expected}: "
                    f"{result.stderr[-1500:]}"
                )
            return result

        async def new(*, via_cli=False, **kwargs):
            if via_cli:
                data = json.loads(cli("create", "--image", kwargs["image"], "--ttl", "1200",
                                      "--metadata", "purpose=" + metadata["purpose"],
                                      "--metadata", "run=" + token).stdout)
            else:
                data = await client.create(ttl=1200, metadata=metadata, **kwargs)
            sandbox_ids.append(data["id"])
            report.setdefault("created_sandbox_ids", []).append(data["id"])
            return data["id"]

        try:
            sid = await new(image=args.image, via_cli=True)
            report["image"] = args.image
            record("create")
            result = await command(sid, "printf 'hello'; printf 'problem' >&2; exit 7", expected=7)
            assert result.stdout == "hello" and result.stderr == "problem", result
            record("separate_streams_and_exit_7")
            cli_result = cli("exec", sid, "printf 'cli-out'; printf 'cli-err' >&2; exit 7", expected=7)
            assert cli_result.stdout == "cli-out" and cli_result.stderr == "cli-err"
            cli_result = cli("exec", sid, "exit 7", "--json", expected=7)
            assert json.loads(cli_result.stdout)["exit_code"] == 7
            record("cli_create_and_exit_status")
            found = json.loads(cli("find", "--metadata", "run=" + token).stdout)
            assert sid in json.dumps(found)
            record("find_by_ownership_metadata")
            await client.info(sid)
            await client.renew(sid, 1200)
            # An entirely separate client must attach after the first client disconnects.
            async with Client(config, timeout=60) as other:
                attached = await other.exec(sid, "printf attached", timeout=30)
                assert attached.exit_code == 0 and attached.stdout == "attached"
            await command(sid, "printf still-alive")
            record("persistent_id_and_renew")

            result = await client.exec(sid, "printf started; sleep 30", timeout=2)
            assert result.timed_out and result.exit_code == 124, result
            report["timeout"] = {
                "execution_id": result.execution_id,
                "interrupt_attempted": result.interrupt_attempted,
                "interrupted": result.interrupted,
                "remote_running": result.remote_running,
            }
            record("bounded_command_timeout")

            original = root / "binary.bin"
            original.write_bytes(bytes(range(256)) * 4096)
            await client.upload(sid, original, "/tmp/osb-binary.bin")
            downloaded = root / "downloaded.bin"
            await client.download(sid, "/tmp/osb-binary.bin", downloaded)
            assert downloaded.read_bytes() == original.read_bytes()
            cli_destination = root / "new-parent" / "downloaded.bin"
            cli("download", sid, "/tmp/osb-binary.bin", cli_destination)
            assert cli_destination.read_bytes() == original.read_bytes()
            record("binary_file_transfer")

            fixture = f"/tmp/{token}"
            await command(sid, f"""set -eu
mkdir -p {shlex.quote(fixture)}
cd {shlex.quote(fixture)}
git init -q
git config user.name 'OpenSandbox check'
git config user.email 'check@example.invalid'
printf 'keep\\n' > tracked.txt
printf 'delete me\\n' > deleted.txt
git add tracked.txt deleted.txt
git commit -qm baseline
rm deleted.txt
printf 'modified\\n' >> tracked.txt
printf 'untracked\\n' > untracked.txt
printf '#!/bin/sh\\nprintf fixture\\n' > executable.sh
chmod 755 executable.sh
ln -s untracked.txt relative-link
printf 'DUMMY_SECRET=excluded\\n' > .env
mkdir .codex .venv .cache
printf '{{"dummy":true}}' > .codex/auth.json
printf 'excluded' > .venv/marker
printf 'excluded' > .cache/marker
""")
            await client.upload(sid, original, fixture + "/binary.bin")
            backup_dir = root / "backup"
            await workspace_ops.backup(client, sid, fixture, backup_dir)
            archives = list(backup_dir.glob("*.tar.gz"))
            assert len(archives) == 1, archives
            with tarfile.open(archives[0]) as archive:
                names = {member.name.removeprefix("./") for member in archive}
            assert "untracked.txt" in names and "binary.bin" in names
            assert not any(n == ".env" or n.startswith((".codex/", ".venv/", ".cache/")) for n in names)
            record("backup_excludes_credentials_and_caches")

            second = await new(image=args.image)
            destination = fixture + "-restored"
            await workspace_ops.restore(client, second, backup_dir, destination)
            verification = """set -eu
test -d .git
git rev-parse --verify HEAD
test ! -e deleted.txt
test ! -e .env
test ! -e .codex/auth.json
test ! -e .venv/marker
test -x executable.sh
test -L relative-link
test "$(cat relative-link)" = untracked
test "$(wc -c < binary.bin)" = 1048576
git diff --binary
"""
            await command(second, verification, cwd=destination)
            await command(second, "touch extra-file", cwd=destination)
            try:
                await workspace_ops.restore(client, second, backup_dir, destination)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Restore overwrote a nonempty destination without replace=True")
            class InterruptedRestore:
                """Inject SIGTERM exactly between renames in this owned fixture."""
                async def upload(self, *items, **options):
                    return await client.upload(*items, **options)

                async def exec(self, sandbox_id, text, **options):
                    boundary = 'if mv -- "$staging" "$destination"; then'
                    if boundary in text:
                        text = text.replace(boundary, 'kill -TERM $$\n' + boundary, 1)
                    return await client.exec(sandbox_id, text, **options)

            try:
                await workspace_ops.restore(InterruptedRestore(), second, backup_dir, destination, replace=True)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Injected restore interruption did not fail")
            await command(second, "test -e extra-file && test -f tracked.txt", cwd=destination)
            record("interrupted_restore_rolls_back")
            await workspace_ops.restore(client, second, backup_dir, destination, replace=True)
            await command(second, "test ! -e extra-file", cwd=destination)
            record("restore_history_modes_links_and_deletions")
            await client.delete(second)
            try:
                await client.info(second)
            except Exception as error:
                assert getattr(error, "status_code", None) == 404
            else:
                raise AssertionError("Restore target still exists after delete")
            sandbox_ids.remove(second)

            context = root / "build-context"
            await workspace_ops.export_build_context(
                client, sid, fixture, context, base_image=args.image,
            )
            assert (context / "Dockerfile").is_file()
            assert args.image in (context / "Dockerfile").read_text()
            record("export_build_context")

            try:
                snap = await client.snapshot(sid, name=token, wait=True, timeout=180)
            except Exception as error:
                if getattr(error, "resource_id", None):
                    snapshot_ids.append(error.resource_id)
                raise
            snapshot_ids.append(snap["id"])
            report["created_snapshot_ids"] = list(snapshot_ids)
            restored = await new(snapshot_id=snap["id"])
            await command(restored, "test -f tracked.txt && test ! -e deleted.txt && test -L relative-link", cwd=fixture)
            record("snapshot_ready_and_restore")
        finally:
            # A timed-out POST may have succeeded remotely before its response was lost.
            # Recover only this check's exact unique metadata/name; never replay creation.
            try:
                async with asyncio.timeout(30):
                    page = await client.manager.list_sandbox_infos(SandboxFilter(metadata=metadata, page=1, page_size=100))
                    assert not page.pagination.has_next_page, "Unexpected owned-resource count"
                    for info in page.sandbox_infos:
                        assert all((info.metadata or {}).get(k) == v for k, v in metadata.items())
                        if info.id not in sandbox_ids:
                            sandbox_ids.append(info.id)
                            report.setdefault("cleanup_recovered_sandbox_ids", []).append(info.id)
                    page = await client.manager.list_snapshots(SnapshotFilter(name=token, page=1, page_size=100))
                    assert not page.pagination.has_next_page, "Unexpected owned-snapshot count"
                    for info in page.snapshot_infos:
                        assert info.name == token and info.sandbox_id in sandbox_ids
                        if info.id not in snapshot_ids:
                            snapshot_ids.append(info.id)
                            report.setdefault("cleanup_recovered_snapshot_ids", []).append(info.id)
            except Exception as error:
                cleanup_errors.append({"kind": "ownership_audit", "error_type": type(error).__name__})
            for sid in reversed(sandbox_ids):
                try:
                    await client.delete(sid)
                    try:
                        await client.info(sid)
                    except Exception as error:
                        # An explicit not-found response establishes deletion, not any exception.
                        status = getattr(error, "status_code", None)
                        code = getattr(error, "code", None)
                        if status != 404 and "notfound" not in type(error).__name__.lower() and "not_found" not in str(code).lower():
                            raise
                    else:
                        raise AssertionError("Sandbox still exists after delete")
                except Exception as error:
                    cleanup_errors.append({"kind": "sandbox", "id": sid, "error_type": type(error).__name__})
            for snapshot_id in reversed(snapshot_ids):
                try:
                    await client.delete_snapshot(snapshot_id)
                    try:
                        await client.snapshot_info(snapshot_id)
                    except Exception as error:
                        status = getattr(error, "status_code", None)
                        if status != 404:
                            raise
                    else:
                        raise AssertionError("Snapshot still exists after delete")
                except Exception as error:
                    cleanup_errors.append({"kind": "snapshot", "id": snapshot_id, "error_type": type(error).__name__})
            report["cleanup_errors"] = cleanup_errors
            if not cleanup_errors:
                try:
                    async with asyncio.timeout(30):
                        remaining = await client.manager.list_sandbox_infos(SandboxFilter(metadata=metadata, page=1, page_size=100))
                        snapshots = await client.manager.list_snapshots(SnapshotFilter(name=token, page=1, page_size=100))
                        assert remaining.pagination.total_items == 0 and snapshots.pagination.total_items == 0
                        report["cleanup_audit"] = "Exact run filters returned zero sandboxes and snapshots"
                except Exception as error:
                    cleanup_errors.append({"kind": "final_ownership_audit", "error_type": type(error).__name__})
            report["cleanup_complete"] = not cleanup_errors
            if cleanup_errors:
                raise RuntimeError("Owned-resource cleanup needs attention; see report IDs")


def main():
    logging.getLogger("opensandbox").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Create disposable Cell-3 resources")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"checks": [], "live": args.live, "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def record(name):
        report["checks"].append(name)
        print("PASS", name, flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="osb-check-") as scratch:
            root = Path(scratch)
            offline(root, record)
            if args.live:
                asyncio.run(live(args, root, report, record))
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
