from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .data import BenchmarkIndex
from .graphs import lattice_heights
from .spec import TargetFamily
from .structure_data import load_atom_feature_table


def periodic_neighbor_vector_slots(
    fractional_positions: np.ndarray,
    cell: np.ndarray,
    *,
    cutoff: float,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted neighbor slots including the exact selected periodic edge vector."""

    frac = np.asarray(fractional_positions, dtype=np.float64)
    cell64 = np.asarray(cell, dtype=np.float64)
    positions = frac @ cell64
    limits = np.ceil(cutoff / lattice_heights(cell64)).astype(int)
    shifts = np.asarray(
        list(product(*(range(-int(n), int(n) + 1) for n in limits))), dtype=np.int64
    )
    vectors = (
        positions[None, None, :, :]
        + (shifts @ cell64)[:, None, None, :]
        - positions[None, :, None, :]
    )
    distances = np.linalg.norm(vectors, axis=-1)
    atom_count = len(frac)
    same_atom = np.eye(atom_count, dtype=bool)[None, :, :]
    zero_shift = np.all(shifts == 0, axis=1)[:, None, None]
    valid = (distances <= cutoff) & (distances > 1.0e-8)
    valid &= ~(zero_shift & same_atom)

    neighbor_index = np.zeros((atom_count, max_neighbors), dtype=np.uint16)
    neighbor_distance = np.full((atom_count, max_neighbors), cutoff + 1.0, dtype=np.float32)
    neighbor_vector = np.zeros((atom_count, max_neighbors, 3), dtype=np.float32)
    neighbor_count = np.zeros(atom_count, dtype=np.uint8)
    for source in range(atom_count):
        shift_index, destination = np.nonzero(valid[:, source, :])
        local_distance = distances[shift_index, source, destination]
        order = np.argsort(local_distance, kind="stable")[:max_neighbors]
        count = len(order)
        neighbor_index[source, :count] = destination[order]
        neighbor_distance[source, :count] = local_distance[order].astype(np.float32)
        neighbor_vector[source, :count] = vectors[
            shift_index[order], source, destination[order]
        ].astype(np.float32)
        neighbor_count[source] = count
    return neighbor_index, neighbor_distance, neighbor_vector, neighbor_count


class ComFormerConditionDataset(Dataset[dict[str, Any]]):
    def __init__(self, index: BenchmarkIndex, split: str, family: TargetFamily,
                 cache_dir: Path, *, cutoff: float = 6.0):
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete" or metadata["population"] != index.spec.population:
            raise ValueError("a complete full-population ComFormer cache is required")
        if float(metadata["cutoff_A"]) != cutoff:
            raise ValueError("ComFormer cache cutoff mismatch")
        self.index = index
        self.family = family
        self.row_indices = index.indices(split)
        self.neighbor_index = np.load(cache_dir / "neighbor_index.npy", mmap_mode="r")
        self.neighbor_vector = np.load(cache_dir / "neighbor_vector.npy", mmap_mode="r")
        self.neighbor_count = np.load(cache_dir / "neighbor_count.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.row_indices[item])
        record = self.index.record(index)
        target, target_mask = self.index.family_target(index, self.family)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            return {
                "dataset_entry_id": record.dataset_entry_id,
                "composition_z": torch.from_numpy(payload["composition_z"].astype(np.int64)),
                "composition_fraction": torch.from_numpy(payload["composition_fraction"].astype(np.float32)),
                "atomic_numbers": torch.from_numpy(payload["z"].astype(np.int64)),
                "neighbor_index": torch.from_numpy(np.asarray(self.neighbor_index[index]).astype(np.int64)),
                "neighbor_vector": torch.from_numpy(np.asarray(self.neighbor_vector[index]).copy()),
                "neighbor_count": torch.from_numpy(np.asarray(self.neighbor_count[index]).astype(np.int64)),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
            }


def collate_comformer(samples: Sequence[dict[str, Any]],
                      *, atom_table: Tensor | None = None) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    atom_table = load_atom_feature_table() if atom_table is None else atom_table
    graphs: list[Data] = []
    for sample in samples:
        atomic_numbers = sample["atomic_numbers"]
        sources, destinations, vectors = [], [], []
        for source, count in enumerate(sample["neighbor_count"].tolist()):
            if count <= 0:
                continue
            sources.append(torch.full((count,), source, dtype=torch.long))
            destinations.append(sample["neighbor_index"][source, :count])
            vectors.append(sample["neighbor_vector"][source, :count])
        graph = Data(
            x=atom_table[atomic_numbers],
            edge_index=torch.stack([torch.cat(sources), torch.cat(destinations)]),
            edge_attr=torch.cat(vectors),
        )
        graphs.append(graph)
    return {
        "dataset_entry_id": [sample["dataset_entry_id"] for sample in samples],
        "composition_z": torch.stack([sample["composition_z"] for sample in samples]),
        "composition_fraction": torch.stack([sample["composition_fraction"] for sample in samples]),
        "structure_graph": Batch.from_data_list(graphs),
        "target": torch.stack([sample["target"] for sample in samples]),
        "target_mask": torch.stack([sample["target_mask"] for sample in samples]),
    }
