from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import dgl
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BenchmarkIndex
from .external import upstream_path
from .spec import TargetFamily
from .structure_data import load_atom_feature_table


ALIGNN_ROOT = upstream_path("HEABENCH_ALIGNN_ROOT", "third_party/ALIGNN")
if str(ALIGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(ALIGNN_ROOT))

from alignn.graphs import compute_bond_cosines  # noqa: E402


class ALIGNNConditionDataset(Dataset[dict[str, Any]]):
    """ALIGNN view over the frozen 6 A / 32-neighbor periodic cache."""

    def __init__(
        self,
        index: BenchmarkIndex,
        split: str,
        family: TargetFamily,
        cache_dir: Path,
        *,
        cutoff: float = 6.0,
        max_neighbors: int = 32,
    ):
        metadata = json.loads(
            (cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            metadata["status"] != "complete"
            or metadata["population"] != index.spec.population
        ):
            raise ValueError("a complete full-population structure cache is required")
        if float(metadata["cutoff_A"]) != cutoff:
            raise ValueError("ALIGNN structure-cache cutoff mismatch")
        if int(metadata["max_neighbors"]) != max_neighbors:
            raise ValueError("ALIGNN structure-cache neighbor limit mismatch")
        self.index = index
        self.family = family
        self.row_indices = index.indices(split)
        self.neighbor_index = np.load(
            cache_dir / "neighbor_index.npy", mmap_mode="r"
        )
        self.neighbor_vector = np.load(
            cache_dir / "neighbor_vector.npy", mmap_mode="r"
        )
        self.neighbor_count = np.load(
            cache_dir / "neighbor_count.npy", mmap_mode="r"
        )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.row_indices[item])
        record = self.index.record(index)
        target, target_mask = self.index.family_target(index, self.family)
        with np.load(record.structure_path, allow_pickle=False) as payload:
            return {
                "dataset_entry_id": record.dataset_entry_id,
                "composition_z": torch.from_numpy(
                    payload["composition_z"].astype(np.int64)
                ),
                "composition_fraction": torch.from_numpy(
                    payload["composition_fraction"].astype(np.float32)
                ),
                "atomic_numbers": torch.from_numpy(payload["z"].astype(np.int64)),
                "neighbor_index": torch.from_numpy(
                    np.asarray(self.neighbor_index[index]).astype(np.int64)
                ),
                "neighbor_vector": torch.from_numpy(
                    np.asarray(self.neighbor_vector[index]).copy()
                ),
                "neighbor_count": torch.from_numpy(
                    np.asarray(self.neighbor_count[index]).astype(np.int64)
                ),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
            }


def _sample_graph(sample: dict[str, Any], atom_table: Tensor) -> tuple[Any, Any]:
    atomic_numbers = sample["atomic_numbers"]
    if int(atomic_numbers.max()) >= atom_table.shape[0]:
        raise ValueError("atomic number is outside the feature table")

    sources: list[Tensor] = []
    destinations: list[Tensor] = []
    vectors: list[Tensor] = []
    for source, count in enumerate(sample["neighbor_count"].tolist()):
        if count <= 0:
            continue
        sources.append(torch.full((count,), source, dtype=torch.long))
        destinations.append(sample["neighbor_index"][source, :count])
        vectors.append(sample["neighbor_vector"][source, :count])
    if not sources:
        raise ValueError("ALIGNN graph contains no periodic edges")

    source = torch.cat(sources)
    destination = torch.cat(destinations)
    graph = dgl.graph(
        (source, destination), num_nodes=int(atomic_numbers.numel())
    )
    graph.ndata["atom_features"] = atom_table[atomic_numbers]
    graph.edata["r"] = torch.cat(vectors).to(torch.float32)

    # This exactly follows upstream ALIGNN: directed bonds become line-graph
    # nodes, and adjacent directed bonds are connected with their angle cosine.
    line_graph = graph.line_graph(shared=True)
    line_graph.apply_edges(compute_bond_cosines)
    return graph, line_graph


def collate_alignn(
    samples: Sequence[dict[str, Any]], *, atom_table: Tensor | None = None
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    atom_table = load_atom_feature_table() if atom_table is None else atom_table
    pairs = [_sample_graph(sample, atom_table) for sample in samples]
    return {
        "dataset_entry_id": [sample["dataset_entry_id"] for sample in samples],
        "composition_z": torch.stack([sample["composition_z"] for sample in samples]),
        "composition_fraction": torch.stack(
            [sample["composition_fraction"] for sample in samples]
        ),
        "structure_graph": (
            dgl.batch([pair[0] for pair in pairs]),
            dgl.batch([pair[1] for pair in pairs]),
        ),
        "target": torch.stack([sample["target"] for sample in samples]),
        "target_mask": torch.stack([sample["target_mask"] for sample in samples]),
    }


def make_alignn_collate(atom_table: Tensor | None = None):
    table = load_atom_feature_table() if atom_table is None else atom_table
    return partial(collate_alignn, atom_table=table)
