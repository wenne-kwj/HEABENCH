#!/usr/bin/env python3
"""Download and unpack the versioned HEABench data bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "registry.json"


def datasets() -> dict[str, dict]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {entry["task"]: entry for entry in registry["datasets"]}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise RuntimeError(f"archive member escapes extraction root: {member}")
    return target


def unpack(archive: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    safe_target(staging, member.filename)
                bundle.extractall(staging)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    safe_target(staging, member.name)
                    if member.issym() or member.islnk():
                        raise RuntimeError("symbolic links are not allowed in data bundles")
                bundle.extractall(staging)
        else:
            raise RuntimeError(f"unsupported archive format: {archive}")

        children = list(staging.iterdir())
        source = children[0] if len(children) == 1 and children[0].is_dir() else staging
        if destination.exists():
            raise RuntimeError(f"destination already exists: {destination}")
        shutil.move(str(source), destination)


def install(entry: dict, destination_root: Path, archive_override: Path | None) -> None:
    destination = destination_root / entry["directory"]
    if archive_override is None and not entry.get("public_url"):
        raise RuntimeError(
            f"{entry['task']} has not been uploaded yet; "
            "set public_url in data/registry.json or pass --archive"
        )

    if archive_override is not None:
        archive = archive_override.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        remove_after = False
    else:
        suffix = Path(str(entry["public_url"])).name or f"{entry['task']}.bundle"
        handle = tempfile.NamedTemporaryFile(prefix="heabench_", suffix=f"_{suffix}", delete=False)
        handle.close()
        archive = Path(handle.name)
        print(f"Downloading {entry['public_url']}")
        urllib.request.urlretrieve(entry["public_url"], archive)
        remove_after = True

    try:
        expected = entry.get("archive_sha256")
        if expected and digest(archive) != expected:
            raise RuntimeError(f"checksum mismatch for {entry['task']}")
        unpack(archive, destination)
    finally:
        if remove_after:
            archive.unlink(missing_ok=True)
    print(f"Installed {entry['task']} at {destination}")


def main() -> None:
    available = datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*available, "all"], default="all")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--archive",
        type=Path,
        help="install one task from a local archive instead of its public URL",
    )
    parser.add_argument("--list", action="store_true", help="show registered bundles and exit")
    args = parser.parse_args()

    if args.list:
        for task, entry in available.items():
            print(f"{task}\t{entry['version']}\t{entry['status']}\t{entry.get('public_url') or '-'}")
        return
    if args.archive is not None and args.task == "all":
        parser.error("--archive requires one explicit --task")

    selected = available.values() if args.task == "all" else [available[args.task]]
    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        install(entry, args.destination, args.archive)


if __name__ == "__main__":
    main()
