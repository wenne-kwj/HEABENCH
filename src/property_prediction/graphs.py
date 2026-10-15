from __future__ import annotations

from itertools import product

import numpy as np


def lattice_heights(cell: np.ndarray) -> np.ndarray:
    cell = np.asarray(cell, dtype=np.float64)
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 1.0e-10:
        raise ValueError("cell volume is non-positive")
    return np.asarray([
        volume / np.linalg.norm(np.cross(cell[1], cell[2])),
        volume / np.linalg.norm(np.cross(cell[2], cell[0])),
        volume / np.linalg.norm(np.cross(cell[0], cell[1])),
    ])


def periodic_neighbor_slots(
    fractional_positions: np.ndarray,
    cell: np.ndarray,
    *,
    cutoff: float,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted CGCNN neighbor slots for one row-vector periodic cell."""

    frac = np.asarray(fractional_positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError("fractional_positions must have shape [atoms, 3]")
    if max_neighbors <= 0 or cutoff <= 0:
        raise ValueError("cutoff and max_neighbors must be positive")
    num_atoms = frac.shape[0]
    positions = frac @ cell
    limits = np.ceil(cutoff / lattice_heights(cell)).astype(int)
    shifts = np.asarray(
        list(product(*(range(-int(n), int(n) + 1) for n in limits))),
        dtype=np.int64,
    )
    translations = shifts @ cell
    vectors = (
        positions[None, None, :, :]
        + translations[:, None, None, :]
        - positions[None, :, None, :]
    )
    distances = np.linalg.norm(vectors, axis=-1)
    same_atom = np.eye(num_atoms, dtype=bool)[None, :, :]
    zero_shift = np.all(shifts == 0, axis=1)[:, None, None]
    valid = (distances <= cutoff) & (distances > 1.0e-8)
    valid &= ~(zero_shift & same_atom)

    neighbor_index = np.zeros((num_atoms, max_neighbors), dtype=np.int64)
    neighbor_distance = np.full(
        (num_atoms, max_neighbors), cutoff + 1.0, dtype=np.float32
    )
    neighbor_count = np.zeros(num_atoms, dtype=np.int16)
    for source in range(num_atoms):
        shift_index, destination = np.nonzero(valid[:, source, :])
        if destination.size == 0:
            continue
        local_distance = distances[shift_index, source, destination]
        order = np.argsort(local_distance, kind="stable")[:max_neighbors]
        count = len(order)
        neighbor_index[source, :count] = destination[order]
        neighbor_distance[source, :count] = local_distance[order].astype(np.float32)
        neighbor_count[source] = count
    return neighbor_index, neighbor_distance, neighbor_count


def gaussian_distance(distance: np.ndarray, *, cutoff: float, step: float) -> np.ndarray:
    centers = np.arange(0.0, cutoff + step, step, dtype=np.float32)
    values = np.asarray(distance, dtype=np.float32)[..., None]
    return np.exp(-((values - centers) ** 2) / (step**2)).astype(np.float32)
