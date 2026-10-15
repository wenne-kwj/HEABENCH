#!/usr/bin/env python3
"""Consume the released Task 3 structure-derived Gaussian field view."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


from .paths import DEFAULT_TASK3_DATA_ROOT

TASK3_ROOT = DEFAULT_TASK3_DATA_ROOT


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_periodic_trilinear(field: np.ndarray, fractional: np.ndarray) -> np.ndarray:
    """Sample a periodic [nx,ny,nz] field at fractional coordinates."""

    grid = np.asarray(field)
    points = np.mod(np.asarray(fractional, dtype=np.float64), 1.0)
    if grid.ndim != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected a 3-D field and [N,3] fractional coordinates")
    shape = np.asarray(grid.shape, dtype=np.int64)
    scaled = points * shape[None, :]
    lower = np.floor(scaled).astype(np.int64)
    fraction = scaled - lower
    result = np.zeros(len(points), dtype=np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.asarray((dx, dy, dz), dtype=np.int64)
                index = (lower + offset[None, :]) % shape[None, :]
                weight = np.prod(
                    np.where(offset[None, :] == 1, fraction, 1.0 - fraction), axis=1
                )
                result += (
                    np.asarray(grid[index[:, 0], index[:, 1], index[:, 2]], dtype=np.float32)
                    * weight.astype(np.float32)
                )
    return result


class _LazyGaussianParent:
    def __init__(
        self,
        field: np.ndarray,
        fractional_positions: np.ndarray,
        cell: np.ndarray,
        charge_offsets: np.ndarray,
    ) -> None:
        self.field = field
        self.fractional_positions = np.asarray(fractional_positions, dtype=np.float64)
        self.cell = np.asarray(cell, dtype=np.float64)
        self.charge_offsets = np.asarray(charge_offsets, dtype=np.float64)
        self.fractional_offsets = self.charge_offsets @ np.linalg.inv(self.cell)
        self.shape = (len(self.fractional_positions), len(self.charge_offsets))

    def __getitem__(self, center_atom: int) -> np.ndarray:
        atom = int(center_atom)
        if atom < 0 or atom >= self.shape[0]:
            raise IndexError(atom)
        query = self.fractional_positions[atom][None, :] + self.fractional_offsets
        return sample_periodic_trilinear(self.field, query)


class _GaussianChargeView:
    def __init__(self, task3_root: Path, view_root: Path, capacity: int) -> None:
        contract = json.loads((view_root / "view_contract.json").read_text(encoding="utf-8"))
        gaussian_root = Path(contract["gaussian_root"])
        if not gaussian_root.is_absolute():
            gaussian_root = view_root / gaussian_root
        self.gaussian_root = gaussian_root.expanduser().resolve()
        if not (self.gaussian_root / "density_e_A3.npy").is_file():
            raise FileNotFoundError(
                "Gaussian field cache is missing. Install the Global Multimodal "
                f"Probing bundle next to the magnetic bundle: {self.gaussian_root}"
            )
        rows = _read_csv(view_root / "nsp_gaussian_mapping.csv")
        self.index = {row["parent_id"]: int(row["gaussian_index"]) for row in rows}
        self.density = np.load(self.gaussian_root / "density_e_A3.npy", mmap_mode="r")
        parents = np.load(task3_root / "nsp_parent_structures.npz", allow_pickle=False)
        self.parent_ids = np.asarray(parents["parent_ids"]).astype(str)
        self.z = np.asarray(parents["z"])
        self.cell = np.asarray(parents["cell"], dtype=np.float64)
        self.frac = np.asarray(parents["frac"], dtype=np.float64)
        self.parent_position = {value: index for index, value in enumerate(self.parent_ids.tolist())}
        self.charge_offsets = np.load(task3_root / "charge_offsets.npy", allow_pickle=False)
        self.capacity = max(0, int(capacity))
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, Any]:
        parent_id = Path(relative_path).stem
        if parent_id in self.cache:
            value = self.cache.pop(parent_id)
            self.cache[parent_id] = value
            return value
        parent_position = self.parent_position[parent_id]
        value = {
            "rho_total": _LazyGaussianParent(
                self.density[self.index[parent_id]],
                self.frac[parent_position],
                self.cell[parent_position],
                self.charge_offsets,
            )
        }
        if self.capacity > 0:
            self.cache[parent_id] = value
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return value


def make_gaussian_dataset_class(base_class, view_root: Path):
    class GaussianTask3NSPSiteDataset(base_class):
        def __init__(self, root=TASK3_ROOT, *, cache_size=2, **kwargs):
            super().__init__(root, cache_size=cache_size, **kwargs)
            self.gaussian_view_root = Path(view_root).expanduser().resolve()
            if not (self.gaussian_view_root / "_SUCCESS").is_file():
                raise RuntimeError(f"incomplete Gaussian view: {self.gaussian_view_root}")
            self._charges = _GaussianChargeView(self.root, self.gaussian_view_root, cache_size)

    return GaussianTask3NSPSiteDataset
