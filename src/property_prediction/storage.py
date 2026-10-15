from __future__ import annotations

from pathlib import Path

from .spec import BenchmarkSpec


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_heavy_artifact_path(
    candidate: str | Path,
    spec: BenchmarkSpec,
) -> Path:
    """Validate a path without creating, deleting, or modifying it."""

    path = Path(candidate).resolve()
    if not _is_within(path, spec.nas_write_root) or path == spec.nas_write_root:
        raise ValueError(
            f"heavy artifact path must be a child of {spec.nas_write_root}: {path}"
        )
    if _is_within(path, spec.raw_proxy_root):
        raise ValueError("raw proxy is read-only")
    if _is_within(path, spec.benchmark_root):
        raise ValueError("canonical benchmark dataset is read-only")
    return path


def validate_data_input_path(
    candidate: str | Path,
    spec: BenchmarkSpec,
) -> Path:
    """Validate an existing, read-only benchmark input path."""

    path = Path(candidate).resolve()
    allowed = (
        _is_within(path, spec.benchmark_root)
        or _is_within(path, spec.nas_write_root)
    )
    if not allowed:
        raise ValueError(
            "benchmark input must be under the data or artifact root: "
            f"{path}"
        )
    if _is_within(path, spec.raw_proxy_root):
        raise ValueError("raw proxy is not a model-ready benchmark input")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_local_metadata_path(
    candidate: str | Path,
    spec: BenchmarkSpec,
) -> Path:
    path = Path(candidate).resolve()
    if not _is_within(path, spec.local_project_root):
        raise ValueError(
            f"local code/metadata path must be under {spec.local_project_root}: {path}"
        )
    return path
