from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BenchmarkIndex
from .external import upstream_path
from .graphs import gaussian_distance
from .spec import TargetFamily


ATOM_INIT = upstream_path("HEABENCH_CGCNN_ROOT", "third_party/CGCNN") / (
    "data/sample-regression/atom_init.json"
)


def load_atom_feature_table(path: Path = ATOM_INIT) -> Tensor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_dim = len(next(iter(payload.values())))
    table = np.zeros((119, feature_dim), dtype=np.float32)
    for key, value in payload.items():
        table[int(key)] = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(table)


class CGCNNConditionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        index: BenchmarkIndex,
        split: str,
        family: TargetFamily,
        cache_dir: Path,
        *,
        cutoff: float = 6.0,
        step: float = 0.2,
    ):
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete" or metadata["population"] != index.spec.population:
            raise ValueError("a complete full-population structure cache is required")
        if float(metadata["cutoff_A"]) != cutoff:
            raise ValueError("structure cache cutoff mismatch")
        self.index = index
        self.family = family
        self.row_indices = index.indices(split)
        self.cutoff = cutoff
        self.step = step
        self.neighbor_index = np.load(cache_dir / "neighbor_index.npy", mmap_mode="r")
        self.neighbor_distance = np.load(cache_dir / "neighbor_distance.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.row_indices[item])
        record = self.index.record(index)
        target, target_mask = self.index.family_target(index, self.family)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            result = {
                "dataset_entry_id": record.dataset_entry_id,
                "composition_z": torch.from_numpy(payload["composition_z"].astype(np.int64)),
                "composition_fraction": torch.from_numpy(payload["composition_fraction"].astype(np.float32)),
                "atomic_numbers": torch.from_numpy(payload["z"].astype(np.int64)),
                "neighbor_index": torch.from_numpy(np.asarray(self.neighbor_index[index]).astype(np.int64)),
                "neighbor_distance": torch.from_numpy(np.asarray(self.neighbor_distance[index]).copy()),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
            }
        return result


def collate_cgcnn(samples: Sequence[dict[str, Any]], *, cutoff: float = 6.0,
                  step: float = 0.2, atom_table: Tensor | None = None) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    atom_table = load_atom_feature_table() if atom_table is None else atom_table
    atom_parts, neighbor_parts, index_parts, crystal_parts = [], [], [], []
    offset = 0
    for sample in samples:
        atomic_numbers = sample["atomic_numbers"]
        if int(atomic_numbers.max()) >= atom_table.shape[0]:
            raise ValueError("atomic number is outside the feature table")
        atom_parts.append(atom_table[atomic_numbers])
        distance = sample["neighbor_distance"].numpy()
        neighbor_parts.append(torch.from_numpy(gaussian_distance(distance, cutoff=cutoff, step=step)))
        index_parts.append(sample["neighbor_index"] + offset)
        count = len(atomic_numbers)
        crystal_parts.append(torch.arange(count, dtype=torch.long) + offset)
        offset += count
    return {
        "dataset_entry_id": [sample["dataset_entry_id"] for sample in samples],
        "composition_z": torch.stack([sample["composition_z"] for sample in samples]),
        "composition_fraction": torch.stack([sample["composition_fraction"] for sample in samples]),
        "atom_features": torch.cat(atom_parts),
        "neighbor_features": torch.cat(neighbor_parts),
        "neighbor_indices": torch.cat(index_parts),
        "crystal_atom_indices": crystal_parts,
        "target": torch.stack([sample["target"] for sample in samples]),
        "target_mask": torch.stack([sample["target_mask"] for sample in samples]),
    }
