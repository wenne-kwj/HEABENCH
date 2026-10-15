from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import BenchmarkIndex
from .spec import TargetFamily


DENSE_FIELD_SHAPE = (56, 40, 40)
STRUCTURE_ENCODERS = ("cgcnn", "ecomformer", "alignn")


def build_structure_dataset(
    encoder: str,
    index: BenchmarkIndex,
    split: str,
    family: TargetFamily,
    cache_dir: Path,
) -> Dataset:
    if encoder == "cgcnn":
        from .structure_data import CGCNNConditionDataset

        return CGCNNConditionDataset(index, split, family, cache_dir)
    if encoder == "ecomformer":
        from .comformer_data import ComFormerConditionDataset

        return ComFormerConditionDataset(index, split, family, cache_dir)
    if encoder == "alignn":
        from .alignn_data import ALIGNNConditionDataset

        return ALIGNNConditionDataset(index, split, family, cache_dir)
    raise ValueError(f"unknown structure encoder {encoder!r}")


class DenseFieldFusionDataset(Dataset[dict[str, Any]]):
    """Attach a complete dense DFT or Gaussian field to a structure dataset."""

    def __init__(
        self,
        structure_dataset: Dataset,
        index: BenchmarkIndex,
        field_cache_dir: Path,
    ) -> None:
        metadata = json.loads(
            (field_cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError("dense field cache is not complete")
        if int(metadata.get("population", -1)) != index.spec.population:
            raise ValueError("dense field cache population mismatch")
        if tuple(metadata.get("shape", ())) != DENSE_FIELD_SHAPE:
            raise ValueError("fusion requires a 56x40x40 dense field")
        ids = np.load(
            field_cache_dir / "dataset_entry_id.npy", allow_pickle=False
        ).astype(str)
        if not np.array_equal(ids, np.asarray(index.ids)):
            raise ValueError("dense field cache ID/order mismatch")
        self.structure_dataset = structure_dataset
        self.field_cache_dir = field_cache_dir
        self.metadata = metadata
        self.cache_index = {value: position for position, value in enumerate(ids)}
        self.density = np.load(
            field_cache_dir / "density_e_A3.npy", mmap_mode="r"
        )

    def __len__(self) -> int:
        return len(self.structure_dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = dict(self.structure_dataset[item])
        position = self.cache_index[sample["dataset_entry_id"]]
        sample["charge_density"] = torch.from_numpy(
            np.asarray(self.density[position], dtype=np.float32).copy()
        )
        return sample


def _attach_dense_field(
    batch: dict[str, Any], samples: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    batch["charge_density"] = torch.stack(
        [sample["charge_density"] for sample in samples]
    )
    return batch


def collate_fusion_cgcnn(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from .structure_data import collate_cgcnn

    return _attach_dense_field(collate_cgcnn(samples), samples)


def collate_fusion_ecomformer(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from .comformer_data import collate_comformer

    return _attach_dense_field(collate_comformer(samples), samples)


def collate_fusion_alignn(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from .alignn_data import collate_alignn

    return _attach_dense_field(collate_alignn(samples), samples)


def fusion_collate(encoder: str) -> Callable[[Sequence[dict[str, Any]]], dict[str, Any]]:
    functions = {
        "cgcnn": collate_fusion_cgcnn,
        "ecomformer": collate_fusion_ecomformer,
        "alignn": collate_fusion_alignn,
    }
    if encoder not in functions:
        raise ValueError(f"unknown structure encoder {encoder!r}")
    return functions[encoder]
