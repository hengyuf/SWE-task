"""Freeze and verify the small executable source set for a DeepSWE campaign."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def verify_source(source: Path) -> Path:
    source = Path(source).resolve()
    record = json.loads((source / "source-manifest.json").read_text())
    expected = record["files"]
    actual = {str(path.relative_to(source)) for path in source.rglob("*")
              if path.is_file() and path != source / "source-manifest.json"}
    if actual != set(expected):
        raise ValueError("Campaign source snapshot has missing or unexpected files")
    for relative, digest in expected.items():
        path = source / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Campaign source snapshot changed: {relative}")
    return source


def _revision_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise ValueError("Source revision must be a simple name of 1 to 64 letters, digits, dots, underscores, or hyphens")
    return value


def select_source(campaign: Path, source: Path | None = None) -> Path:
    """Select only a verified original snapshot or one named campaign revision."""
    campaign = Path(campaign).resolve()
    original = campaign / "source"
    chosen = original if source is None else Path(source)
    if not chosen.is_absolute():
        chosen = campaign / chosen
    if chosen.is_symlink() or (campaign / "source-revisions").is_symlink() or original.is_symlink():
        raise ValueError("Campaign source paths cannot be symlinks")
    chosen = chosen.resolve()
    if chosen != original:
        if chosen.parent != campaign / "source-revisions":
            raise ValueError("Source must be the campaign original or a direct child of source-revisions")
        _revision_name(chosen.name)
    verify_source(original)
    settings = json.loads((campaign / "manifest.json").read_text())["settings"]
    if json.loads((original / "configs/deepswe.json").read_text()) != settings:
        raise ValueError("Campaign settings changed since its original source snapshot")
    verified = original if chosen == original else verify_source(chosen)
    if json.loads((verified / "configs/deepswe.json").read_text()) != settings:
        raise ValueError("Source revision settings differ from the immutable campaign settings")
    return verified


def freeze_source(app: Path, campaign: Path, revision: str | None = None) -> Path:
    from deepswe_stage import DATASET_REVISION, GYM_COMMIT, PATCHES, RG_VERSION

    app, campaign = Path(app).resolve(), Path(campaign).resolve()
    if revision is not None:
        _revision_name(revision)
    source = campaign / "source" if revision is None else campaign / "source-revisions" / revision
    with (campaign / "source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if revision is not None:
            if not (campaign / "source").is_dir():
                raise ValueError("An original source snapshot is required before freezing a revision")
            select_source(campaign)
        if source.exists():
            return select_source(campaign, source)
        if revision is None and any((campaign / "models").glob("*/attempts/*/attempt-*")):
            raise ValueError("Cannot freeze previously executed work retroactively; this campaign has no source snapshot")
        settings = json.loads((campaign / "manifest.json").read_text())["settings"]
        if settings["gym_commit"] != GYM_COMMIT or settings["dataset_revision"] != DATASET_REVISION:
            raise ValueError("Campaign settings do not match the supported Gym and dataset pins")
        if tuple(int(part) for part in settings["python_version"].split(".")[:2]) < (3, 13):
            raise ValueError("The pinned Gym stack requires Python 3.13 or newer")
        files = {"deepswe.sbatch", "configs/deepswe.json",
                 *("overlays/" + name for name in PATCHES),
                 *(str(path.relative_to(app)) for path in (app / "scripts").glob("*deepswe*.py"))}
        approval = "configs/deepswe_historical_recovery.json"
        if ((app / approval).exists()
                and json.loads((app / approval).read_text()).get("campaign") == str(campaign)):
            files.add(approval)
        required = {"scripts/deepswe_stage.py", "scripts/deepswe_source.py", "scripts/run_deepswe.py"}
        if not required <= files:
            raise ValueError("DeepSWE source is missing its bootstrap, runner, or snapshot helper")
        source.parent.mkdir(exist_ok=True)
        temporary = source.parent / (source.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.mkdir()
        record = {"files": {}, "pins": {key: settings[key] for key in
                  ("gym_commit", "dataset_revision", "opencode_version", "python_version")}}
        record["pins"]["ripgrep_version"] = RG_VERSION
        try:
            for relative in sorted(files):
                original, destination = app / relative, temporary / relative
                if original.is_symlink():
                    raise ValueError(f"Source snapshot requires regular files: {relative}")
                data = ((json.dumps(settings, indent=2, sort_keys=True) + "\n").encode()
                        if relative == "configs/deepswe.json" else original.read_bytes())
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                record["files"][relative] = hashlib.sha256(data).hexdigest()
            with (temporary / "source-manifest.json").open("x") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for directory in [temporary, *(path for path in temporary.rglob("*") if path.is_dir())]:
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            temporary.rename(source)
            for parent in {source.parent, campaign}:
                descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return select_source(campaign, source)
