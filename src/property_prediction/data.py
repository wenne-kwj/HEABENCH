from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .spec import BenchmarkSpec, TargetFamily


@dataclass(frozen=True)
class RecordRef:
    index: int
    dataset_entry_id: str
    split: str
    structure_path: Path


class BenchmarkIndex:
    """Validated index over labels and structure shards; charge is not read."""

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec
        manifest_path = spec.benchmark_root / "manifest.csv"
        labels_path = spec.benchmark_root / "labels.npz"
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != spec.population:
            raise ValueError("manifest population does not match the benchmark spec")
        self.ids = tuple(row["dataset_entry_id"] for row in self.rows)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("dataset_entry_id values must be unique")
        with np.load(labels_path, allow_pickle=False) as payload:
            label_ids = tuple(payload["dataset_entry_id"].astype(str).tolist())
            if label_ids != self.ids:
                raise ValueError("manifest and labels.npz ID/order mismatch")
            self.splits = payload["split"].astype(str)
            required_targets = {
                target for family in spec.target_families.values() for target in family.targets
            }
            self.labels = {
                target: np.asarray(payload[target], dtype=np.float32)
                for target in required_targets
            }
        for row in self.rows:
            required = spec.required_encut_eV
            if (float(row["static_encut_eV"]) != required
                    or float(row["band_encut_eV"]) != required):
                raise ValueError(f"ENCUT violation for {row['dataset_entry_id']}")

    def indices(self, split: str) -> np.ndarray:
        if split not in self.spec.split_counts:
            raise ValueError(f"unknown split {split!r}")
        indices = np.flatnonzero(self.splits == split)
        expected = self.spec.split_counts[split]
        if len(indices) != expected:
            raise ValueError(f"{split} contains {len(indices)} rows, expected {expected}")
        return indices

    def record(self, index: int) -> RecordRef:
        row = self.rows[index]
        return RecordRef(
            index=index,
            dataset_entry_id=row["dataset_entry_id"],
            split=str(self.splits[index]),
            structure_path=self.spec.benchmark_root / row["structure_npz"],
        )

    def family_target(self, index: int, family: TargetFamily) -> tuple[np.ndarray, np.ndarray]:
        if len(family.targets) == 1:
            value = np.asarray(self.labels[family.targets[0]][index], dtype=np.float32)
            target = value.reshape(family.output_shape)
        else:
            target = np.asarray([self.labels[name][index] for name in family.targets], dtype=np.float32)
        mask = np.isfinite(target)
        return np.where(mask, target, 0.0).astype(np.float32), mask


class CompositionStructureDataset(Dataset[dict[str, Any]]):
    """Composition records with optional fixed-size periodic structure arrays."""

    def __init__(self, index: BenchmarkIndex, split: str, family: TargetFamily,
                 *, include_structure: bool):
        self.index = index
        self.family = family
        self.include_structure = include_structure
        self.row_indices = index.indices(split)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.row_indices[item])
        record = self.index.record(index)
        target, target_mask = self.index.family_target(index, self.family)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            sample: dict[str, Any] = {
                "dataset_entry_id": record.dataset_entry_id,
                "composition_z": torch.from_numpy(payload["composition_z"].astype(np.int64)),
                "composition_fraction": torch.from_numpy(payload["composition_fraction"].astype(np.float32)),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
            }
            if self.include_structure:
                sample.update({
                    "atomic_numbers": torch.from_numpy(payload["z"].astype(np.int64)),
                    "cell": torch.from_numpy(payload["cell"].astype(np.float32)),
                    "cartesian_positions": torch.from_numpy(payload["r"].astype(np.float32)),
                    "fractional_positions": torch.from_numpy(payload["frac"].astype(np.float32)),
                })
        return sample
