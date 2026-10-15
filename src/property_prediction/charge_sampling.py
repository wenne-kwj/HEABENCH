from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def uniform_fractional_positions(
    sample_shape: tuple[int, int, int] = (8, 16, 16),
) -> np.ndarray:
    """Deterministic periodic point lattice at fractional voxel centers."""

    if len(sample_shape) != 3 or min(sample_shape) <= 0:
        raise ValueError(f"invalid sample shape {sample_shape}")
    axes = [
        (np.arange(count, dtype=np.float64) + 0.5) / count
        for count in sample_shape
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=-1).astype(np.float32)


def sample_native_periodic_grid(
    grid: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Sample unsmoothed native FFT-grid values at periodic fractional points."""

    density = np.asarray(grid)
    points = np.asarray(positions, dtype=np.float64)
    if density.ndim != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected grid [D,H,W] and positions [N,3]")
    shape = np.asarray(density.shape, dtype=np.int64)
    # VASP FFT samples lie on i/N. Select the nearest native sample to each
    # deterministic fractional point; modulo makes the convention periodic.
    index = np.rint(points * shape[None, :]).astype(np.int64) % shape[None, :]
    return np.asarray(density[index[:, 0], index[:, 1], index[:, 2]], dtype=np.float32)


def fourier_resample_multiple(
    grid: np.ndarray,
    target_shapes: Iterable[tuple[int, int, int]],
) -> dict[tuple[int, int, int], np.ndarray]:
    """Create several direct low-pass grids from one native FFT."""

    source = np.asarray(grid)
    if source.ndim != 3:
        raise ValueError(f"expected a 3-D source grid, got {source.shape}")
    shapes = [tuple(int(value) for value in shape) for shape in target_shapes]
    if any(len(shape) != 3 or min(shape) <= 0 for shape in shapes):
        raise ValueError("all target shapes must have three positive dimensions")
    spectrum = np.fft.fftshift(np.fft.fftn(source))
    source_size = float(source.size)
    source_mean = float(source.mean(dtype=np.float64))
    results: dict[tuple[int, int, int], np.ndarray] = {}
    for target in shapes:
        resized = np.zeros(target, dtype=np.complex128)
        source_slices: list[slice] = []
        target_slices: list[slice] = []
        for source_dim, target_dim in zip(source.shape, target):
            keep = min(source_dim, target_dim)
            source_start = (source_dim - keep) // 2
            target_start = (target_dim - keep) // 2
            source_slices.append(slice(source_start, source_start + keep))
            target_slices.append(slice(target_start, target_start + keep))
        resized[tuple(target_slices)] = spectrum[tuple(source_slices)]
        result = np.fft.ifftn(np.fft.ifftshift(resized)).real
        result *= float(np.prod(target, dtype=np.float64)) / source_size
        result += source_mean - float(result.mean(dtype=np.float64))
        results[target] = result.astype(np.float32, copy=False)
    return results
