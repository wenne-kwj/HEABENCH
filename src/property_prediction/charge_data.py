from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BenchmarkIndex
from .spec import TargetFamily


@dataclass(frozen=True)
class ChargeNormalizer:
    scale: float
    mean: float
    std: float
    sampled_values: int

    def transform(self, values: Tensor) -> Tensor:
        scaled = torch.sign(values) * torch.log1p(values.abs() / self.scale)
        return (scaled - self.mean) / self.std

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "transform": "signed_log1p_then_standardize",
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
            "sampled_train_values": self.sampled_values,
        }


def fit_charge_normalizer(
    array: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int = 31,
    points_per_sample: int = 2048,
) -> ChargeNormalizer:
    """Fit a deterministic, train-only transform without materializing the cache."""

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    row_size = int(np.prod(array.shape[1:]))
    count = min(points_per_sample, row_size)
    for index in train_indices:
        flat = np.asarray(array[int(index)]).reshape(-1)
        if count == row_size:
            chosen = flat
        else:
            chosen = flat[rng.choice(row_size, size=count, replace=False)]
        pieces.append(np.asarray(chosen, dtype=np.float32))
    sample = np.concatenate(pieces).astype(np.float64, copy=False)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError("charge cache contains no finite training values")
    scale = max(float(np.mean(np.abs(finite))), 1.0e-8)
    transformed = np.sign(finite) * np.log1p(np.abs(finite) / scale)
    mean = float(transformed.mean())
    std = max(float(transformed.std()), 1.0e-8)
    return ChargeNormalizer(scale, mean, std, int(finite.size))


class ChargeConditionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        index: BenchmarkIndex,
        split: str,
        family: TargetFamily,
        cache_dir: Path,
    ):
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            raise ValueError("charge cache is not complete")
        self.index = index
        self.family = family
        self.cache_dir = cache_dir
        self.ids = np.load(cache_dir / "dataset_entry_id.npy", allow_pickle=False).astype(str)
        self.density = np.load(cache_dir / "density_e_A3.npy", mmap_mode="r")
        canonical = np.asarray(index.ids)
        if len(self.ids) != index.spec.population or not np.array_equal(self.ids, canonical):
            raise ValueError("charge cache ID/order mismatch")
        self.row_indices = index.indices(split)
        self.cache_index = {value: position for position, value in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.row_indices[item])
        record = self.index.record(index)
        target, target_mask = self.index.family_target(index, self.family)
        cache_position = self.cache_index[record.dataset_entry_id]
        with np.load(record.structure_path, allow_pickle=False) as payload:
            return {
                "dataset_entry_id": record.dataset_entry_id,
                "composition_z": torch.from_numpy(payload["composition_z"].astype(np.int64)),
                "composition_fraction": torch.from_numpy(payload["composition_fraction"].astype(np.float32)),
                "charge_density": torch.from_numpy(
                    np.asarray(self.density[cache_position], dtype=np.float32).copy()
                ),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
            }
