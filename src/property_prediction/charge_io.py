from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, TextIO

import numpy as np


@dataclass(frozen=True)
class ChargeSourceRef:
    dataset_entry_id: str
    proxy_path: Path
    manifest_path: Path
    stage_location: str


@dataclass(frozen=True)
class ChgcarMetadata:
    path: Path
    cell: np.ndarray
    atom_count: int
    grid_shape: tuple[int, int, int]
    volume_A3: float


def _required_line(handle: TextIO, context: str) -> str:
    line = handle.readline()
    if line == "":
        raise ValueError(f"unexpected EOF while reading {context}")
    return line


def _integer_tokens(line: str) -> list[int] | None:
    parts = line.split()
    if not parts:
        return None
    try:
        return [int(value) for value in parts]
    except ValueError:
        return None


def resolve_static_charge_source(
    row: Mapping[str, str], proxy_root: str | Path
) -> ChargeSourceRef:
    """Resolve the selected static CHGCAR through the read-only proxy.

    The sample-root ``CHGCAR`` is deliberately never considered.  Some source
    trees contain an empty placeholder there; the canonical manifest's
    ``static_stage_location`` is the only allowed stage selector.
    """

    entry_id = row["dataset_entry_id"]
    stage = row["static_stage_location"].strip()
    relative_stage = PurePosixPath(stage)
    if not stage or relative_stage.is_absolute() or ".." in relative_stage.parts:
        raise ValueError(f"unsafe static stage for {entry_id}: {stage!r}")
    current = Path(proxy_root) / "current"
    proxy_path = current / "samples" / entry_id / Path(*relative_stage.parts) / "CHGCAR"
    manifest_path = Path(row["static_chgcar_path"])
    if not proxy_path.is_file() or proxy_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty proxy static CHGCAR: {proxy_path}")
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty manifest static CHGCAR: {manifest_path}")
    if not proxy_path.samefile(manifest_path):
        raise ValueError(
            f"proxy/manifest CHGCAR mismatch for {entry_id}: "
            f"{proxy_path} != {manifest_path}"
        )
    return ChargeSourceRef(entry_id, proxy_path, manifest_path, stage)


def _read_header(handle: TextIO, path: Path) -> ChgcarMetadata:
    _required_line(handle, "CHGCAR comment")
    scale = float(_required_line(handle, "CHGCAR scale").split()[0])
    cell = np.asarray(
        [
            [float(value) for value in _required_line(handle, f"cell row {i}").split()[:3]]
            for i in range(3)
        ],
        dtype=np.float64,
    )
    cell *= scale

    line = _required_line(handle, "species or atom counts")
    counts = _integer_tokens(line)
    if counts is None:
        counts = _integer_tokens(_required_line(handle, "atom counts"))
    if counts is None:
        raise ValueError(f"could not parse atom counts in {path}")
    atom_count = int(sum(counts))

    coordinate_mode = _required_line(handle, "coordinate mode")
    if coordinate_mode.strip().lower().startswith("s"):
        coordinate_mode = _required_line(handle, "coordinate mode after Selective dynamics")
    if not coordinate_mode.strip():
        raise ValueError(f"missing coordinate mode in {path}")
    for _ in range(atom_count):
        _required_line(handle, "atomic positions")

    grid_line = _required_line(handle, "charge grid shape")
    while not grid_line.strip():
        grid_line = _required_line(handle, "charge grid shape")
    grid_values = _integer_tokens(grid_line)
    if grid_values is None or len(grid_values) < 3:
        raise ValueError(f"could not parse grid shape in {path}: {grid_line!r}")
    grid_shape = tuple(int(value) for value in grid_values[:3])
    volume = float(abs(np.linalg.det(cell)))
    if volume <= 0.0 or min(grid_shape) <= 0:
        raise ValueError(f"invalid cell/grid metadata in {path}")
    return ChgcarMetadata(path, cell, atom_count, grid_shape, volume)


def read_chgcar_metadata(path: str | Path) -> ChgcarMetadata:
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        return _read_header(handle, source)


def fourier_resample_periodic_grid(
    grid: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """Single-step periodic FFT crop/pad with the sample mean preserved."""

    source = np.asarray(grid)
    if source.ndim != 3:
        raise ValueError(f"expected a 3-D grid, got {source.shape}")
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 3 or min(target_shape) <= 0:
        raise ValueError(f"invalid target shape {target_shape}")
    if source.shape == target_shape:
        return source.astype(np.float32, copy=True)

    spectrum = np.fft.fftshift(np.fft.fftn(source))
    resized = np.zeros(target_shape, dtype=np.complex128)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for source_dim, target_dim in zip(source.shape, target_shape):
        keep = min(source_dim, target_dim)
        source_start = (source_dim - keep) // 2
        target_start = (target_dim - keep) // 2
        source_slices.append(slice(source_start, source_start + keep))
        target_slices.append(slice(target_start, target_start + keep))
    resized[tuple(target_slices)] = spectrum[tuple(source_slices)]
    scale = float(np.prod(target_shape, dtype=np.float64) / source.size)
    result = np.fft.ifftn(np.fft.ifftshift(resized)).real * scale
    # Numerical roundoff or even/odd FFT boundary handling must not alter the
    # zero-frequency component used for the electron-integral audit.
    result += float(source.mean(dtype=np.float64) - result.mean(dtype=np.float64))
    return result.astype(np.float32, copy=False)


def read_total_charge_density(
    path: str | Path, *, target_shape: tuple[int, int, int] | None = None
) -> tuple[np.ndarray, ChgcarMetadata]:
    """Read only the first (total-charge) CHGCAR block as e/angstrom^3."""

    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        metadata = _read_header(handle, source)
        total = int(np.prod(metadata.grid_shape, dtype=np.int64))
        values = np.empty(total, dtype=np.float32)
        cursor = 0
        for line in handle:
            parsed = np.fromstring(line, sep=" ", dtype=np.float32)
            if parsed.size == 0:
                continue
            take = min(int(parsed.size), total - cursor)
            values[cursor : cursor + take] = parsed[:take]
            cursor += take
            if cursor == total:
                break
        if cursor != total:
            raise ValueError(f"expected {total} charge values in {source}, found {cursor}")

    density = values.reshape(metadata.grid_shape, order="F") / metadata.volume_A3
    if target_shape is not None:
        density = fourier_resample_periodic_grid(density, target_shape)
    density = np.asarray(density, dtype=np.float32)
    if not np.isfinite(density).all():
        raise ValueError(f"non-finite charge values in {source}")
    return density, metadata


def integrated_electron_count(density: np.ndarray, volume_A3: float) -> float:
    return float(np.asarray(density, dtype=np.float64).mean() * volume_A3)
