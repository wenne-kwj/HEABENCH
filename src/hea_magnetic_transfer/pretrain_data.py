"""Task 3 NSP pre-training data loader.

The immutable dataset stores one structure and charge shard per 100-atom
parent.  This module exposes atom-centred sites to the existing HEA
EquiformerV2 + PointVector towers without copying the extracted data.
"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from hea_retrieval.data import DEFAULT_UNIFORM_CHARGE_POINTS, random_rotation_matrix


from .paths import DEFAULT_TASK3_DATA_ROOT
EXPECTED_ATOMS = 100
EXPECTED_CHARGE_POINTS = 4096


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class _ShardCache:
    """Small per-process LRU cache for atom-parent NPZ shards."""

    def __init__(self, root: Path, capacity: int = 2) -> None:
        self.root = root
        self.capacity = max(0, int(capacity))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        key = str(relative_path)
        if self.capacity > 0 and key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        path = self.root / key
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            value = {name: np.asarray(payload[name]) for name in payload.files}
        if self.capacity > 0:
            self._cache[key] = value
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return value


class Task3NSPSiteDataset(Dataset[dict[str, Any]]):
    """Random access to all 199,300 atom-centred NSP positive pairs."""

    def __init__(
        self,
        root: str | Path = DEFAULT_TASK3_DATA_ROOT,
        *,
        cache_size: int = 2,
        use_center_indicator: bool = True,
        require_success: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Task 3 data root does not exist: {self.root}")
        if require_success and not (self.root / "_SUCCESS").is_file():
            raise RuntimeError(f"Task 3 data root has no _SUCCESS marker: {self.root}")
        rows = _read_csv(self.root / "nsp_manifest.csv")
        rows.sort(key=lambda row: int(row["parent_index"]))
        if len(rows) != 1993:
            raise ValueError(f"expected 1993 NSP parents, found {len(rows)}")
        expected_indices = list(range(len(rows)))
        actual_indices = [int(row["parent_index"]) for row in rows]
        if actual_indices != expected_indices:
            raise ValueError("NSP parent_index is not contiguous and ordered")
        parent_ids = [row["parent_id"] for row in rows]
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("duplicate NSP parent IDs")
        if any(int(row["n_atoms"]) != EXPECTED_ATOMS for row in rows):
            raise ValueError("Task 3 NSP contains a parent that is not 100 atoms")
        self.parents = rows
        self.parent_ids = parent_ids
        self.parent_index_by_id = {parent_id: index for index, parent_id in enumerate(parent_ids)}
        self.use_center_indicator = bool(use_center_indicator)
        self._structures = _ShardCache(self.root, cache_size)
        self._charges = _ShardCache(self.root, cache_size)
        charge_offsets = np.load(self.root / "charge_offsets.npy", allow_pickle=False)
        self.charge_xyz = np.asarray(charge_offsets, dtype=np.float32)
        if self.charge_xyz.shape != (EXPECTED_CHARGE_POINTS, 3):
            raise ValueError(f"unexpected charge offset shape {self.charge_xyz.shape}")
        if not np.isfinite(self.charge_xyz).all():
            raise ValueError("charge offsets contain NaN/Inf")

    def __len__(self) -> int:
        return len(self.parents) * EXPECTED_ATOMS

    def index_for(self, parent_id: str, center_atom: int) -> int:
        if parent_id not in self.parent_index_by_id:
            raise KeyError(parent_id)
        atom = int(center_atom)
        if atom < 0 or atom >= EXPECTED_ATOMS:
            raise IndexError(atom)
        return self.parent_index_by_id[parent_id] * EXPECTED_ATOMS + atom

    def parent_split(self, parent_id: str) -> str:
        return self.parents[self.parent_index_by_id[parent_id]]["split"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(index)
        parent_index, center_atom = divmod(position, EXPECTED_ATOMS)
        parent = self.parents[parent_index]
        parent_id = parent["parent_id"]
        structure_rel = f"nsp/structures/{parent_id}.npz"
        charge_rel = f"nsp/charges/{parent_id}.npz"
        structure = self._structures.get(structure_rel)
        charge = self._charges.get(charge_rel)
        offsets = np.asarray(structure["offsets"], dtype=np.int64)
        if offsets.shape != (EXPECTED_ATOMS + 1,):
            raise ValueError(f"malformed structure offsets for {parent_id}")
        lo, hi = int(offsets[center_atom]), int(offsets[center_atom + 1])
        z = np.asarray(structure["z"][lo:hi], dtype=np.int64)
        xyz = np.asarray(structure["xyz"][lo:hi], dtype=np.float32)
        distance = np.asarray(structure["distance"][lo:hi], dtype=np.float32)
        center_mask = np.asarray(structure["is_center"][lo:hi], dtype=np.bool_)
        atom_index = np.asarray(structure["atom_index"][lo:hi], dtype=np.int64)
        if z.ndim != 1 or xyz.shape != (z.shape[0], 3):
            raise ValueError(f"malformed local structure for {parent_id} site {center_atom}")
        if distance.shape != z.shape or center_mask.shape != z.shape:
            raise ValueError(f"malformed local auxiliaries for {parent_id} site {center_atom}")
        if int(center_mask.sum()) != 1:
            raise ValueError(f"site {parent_id}:{center_atom} does not have one centre")
        center_position = int(np.flatnonzero(center_mask)[0])
        if int(atom_index[center_position]) != center_atom or abs(float(distance[center_position])) > 1e-5:
            raise ValueError(f"site {parent_id}:{center_atom} centre mapping is inconsistent")
        rho_total = np.asarray(charge["rho_total"][center_atom], dtype=np.float32)
        if rho_total.shape != (EXPECTED_CHARGE_POINTS,):
            raise ValueError(f"malformed charge patch for {parent_id} site {center_atom}")
        if not np.isfinite(xyz).all() or not np.isfinite(rho_total).all():
            raise ValueError(f"non-finite Task 3 sample {parent_id} site {center_atom}")
        sample: dict[str, Any] = {
            "site_id": f"{parent_id}_site_{center_atom:03d}",
            "patch_id": f"{parent_id}_site_{center_atom:03d}",
            "parent_id": parent_id,
            "parent_index": parent_index,
            "center_atom": center_atom,
            "split": parent["split"],
            "source": parent["source"],
            "structure_z": z,
            "structure_xyz": xyz,
            "structure_distance": distance,
            "structure_center_mask": center_mask,
            "charge_xyz": self.charge_xyz,
            "rho_total": rho_total,
        }
        if self.use_center_indicator:
            # The existing Equiformer adapter accepts role values 0/1/2.  Task 3
            # uses only 0=neighbour and 1=the unique atom-centre; there is no
            # pair endpoint-B semantics in this dataset.
            sample["structure_role"] = center_mask.astype(np.int64)
        return sample


class ScheduledSiteBatchSampler(Sampler[list[int]]):
    """Batch the fixed 50-epoch, 20-sites-per-parent training schedule.

    Parents are shuffled per epoch, then their selected sites are concatenated
    before chunking.  A batch therefore touches at most two parents in the
    common case, preserving shard locality while avoiding one short batch per
    parent when 20 sites are trained with batch size 16.
    """

    def __init__(
        self,
        dataset: Task3NSPSiteDataset,
        schedule_path: str | Path,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError("in-batch InfoNCE requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        with np.load(Path(schedule_path), allow_pickle=False) as payload:
            parent_ids = np.asarray(payload["parent_ids"]).astype(str)
            site_indices = np.asarray(payload["site_indices"], dtype=np.int16)
        if site_indices.ndim != 3 or site_indices.shape[1] != len(parent_ids):
            raise ValueError(f"malformed site schedule shape {site_indices.shape}")
        if np.any((site_indices < 0) | (site_indices >= EXPECTED_ATOMS)):
            raise ValueError("site schedule contains an out-of-range atom")
        for row in site_indices.reshape(-1, site_indices.shape[-1]):
            if len(np.unique(row)) != len(row):
                raise ValueError("site schedule repeats an atom within one parent epoch")
        positions: list[int] = []
        for parent_id in parent_ids.tolist():
            if dataset.parent_split(parent_id) != "train":
                raise ValueError(f"scheduled non-train parent {parent_id}")
            positions.append(dataset.parent_index_by_id[parent_id])
        self.parent_ids = parent_ids
        self.parent_positions = np.asarray(positions, dtype=np.int64)
        self.site_indices = site_indices
        self.epoch = 0

    @property
    def epochs(self) -> int:
        return int(self.site_indices.shape[0])

    @property
    def samples_per_epoch(self) -> int:
        return int(self.site_indices.shape[1] * self.site_indices.shape[2])

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        if value < 0 or value >= self.epochs:
            raise IndexError(f"epoch {value} outside schedule [0,{self.epochs})")
        self.epoch = value

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        parent_order = np.arange(len(self.parent_positions), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(parent_order)
        batch: list[int] = []
        for schedule_position in parent_order.tolist():
            parent_position = int(self.parent_positions[schedule_position])
            atoms = np.array(self.site_indices[self.epoch, schedule_position], copy=True)
            if self.shuffle:
                rng.shuffle(atoms)
            for atom in atoms.tolist():
                batch.append(parent_position * EXPECTED_ATOMS + int(atom))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.samples_per_epoch / self.batch_size)


def validation_indices(
    dataset: Task3NSPSiteDataset,
    *,
    max_sites: int | None = None,
) -> list[int]:
    rows = _read_csv(dataset.root / "nsp_validation_subset.csv")
    result: list[int] = []
    for row in rows:
        if dataset.parent_split(row["parent_id"]) != "validation":
            raise ValueError(f"validation subset contains non-validation parent {row['parent_id']}")
        result.append(dataset.index_for(row["parent_id"], int(row["center_atom"])))
    if len(result) != 3980 or len(set(result)) != len(result):
        raise ValueError(f"expected 3980 unique validation sites, found {len(result)}")
    if max_sites is not None and int(max_sites) > 0:
        result = result[: int(max_sites)]
    return result


def split_indices(dataset: Task3NSPSiteDataset, split: str) -> list[int]:
    result: list[int] = []
    for parent_index, parent in enumerate(dataset.parents):
        if parent["split"] == split:
            start = parent_index * EXPECTED_ATOMS
            result.extend(range(start, start + EXPECTED_ATOMS))
    return result


def load_density_stats(root: str | Path) -> dict[str, float | int | str]:
    path = Path(root) / "density_stats.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mean = float(value["rho_mean"])
    std = float(value["rho_std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"invalid density statistics in {path}")
    return value


def collate_task3_sites(
    samples: Sequence[dict[str, Any]],
    *,
    rho_mean: float,
    rho_std: float,
    rotate: bool = False,
    rotation: Tensor | None = None,
    charge_points: int | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty Task 3 batch")
    if not math.isfinite(float(rho_mean)) or not math.isfinite(float(rho_std)) or rho_std <= 0:
        raise ValueError("density normalization must be finite with positive std")
    batch_size = len(samples)
    max_atoms = max(len(np.asarray(sample["structure_z"])) for sample in samples)
    z = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    xyz = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    distance = torch.zeros((batch_size, max_atoms), dtype=torch.float32)
    center_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    structure_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    has_role = ["structure_role" in sample for sample in samples]
    if any(has_role) and not all(has_role):
        raise ValueError("centre indicator must be present for every sample or none")
    role = torch.zeros((batch_size, max_atoms), dtype=torch.long) if all(has_role) else None
    charge_xyz_rows: list[Tensor] = []
    rho_rows: list[Tensor] = []
    rho_raw_rows: list[Tensor] = []
    volume_mask_rows: list[Tensor] = []
    for row_index, sample in enumerate(samples):
        count = len(np.asarray(sample["structure_z"]))
        z[row_index, :count] = torch.as_tensor(sample["structure_z"], dtype=torch.long)
        xyz[row_index, :count] = torch.as_tensor(sample["structure_xyz"], dtype=torch.float32)
        distance[row_index, :count] = torch.as_tensor(
            sample["structure_distance"], dtype=torch.float32
        )
        center_mask[row_index, :count] = torch.as_tensor(
            sample["structure_center_mask"], dtype=torch.bool
        )
        structure_mask[row_index, :count] = True
        if role is not None:
            role[row_index, :count] = torch.as_tensor(sample["structure_role"], dtype=torch.long)
        charge_xyz = torch.as_tensor(sample["charge_xyz"], dtype=torch.float32)
        rho_raw = torch.as_tensor(sample["rho_total"], dtype=torch.float32)
        source_indices = torch.arange(charge_xyz.shape[0], dtype=torch.long)
        if charge_points is not None:
            point_count = int(charge_points)
            if point_count < 8 or point_count > charge_xyz.shape[0]:
                raise ValueError(f"invalid charge_points={point_count}")
            selected = torch.linspace(0, charge_xyz.shape[0] - 1, point_count).round().long()
            charge_xyz = charge_xyz[selected]
            rho_raw = rho_raw[selected]
            source_indices = source_indices[selected]
        charge_xyz_rows.append(charge_xyz)
        rho_raw_rows.append(rho_raw)
        rho_rows.append((rho_raw - float(rho_mean)) / float(rho_std))
        volume_mask_rows.append(source_indices.lt(DEFAULT_UNIFORM_CHARGE_POINTS))
    charge_xyz = torch.stack(charge_xyz_rows)
    rho_total_raw = torch.stack(rho_raw_rows)
    rho_total = torch.stack(rho_rows)
    if rotate:
        if rotation is None:
            rotation = random_rotation_matrix()
        rotation = rotation.to(dtype=xyz.dtype, device=xyz.device)
        xyz = xyz @ rotation.T
        charge_xyz = charge_xyz @ rotation.T
    result: dict[str, Any] = {
        "structure_z": z,
        "structure_xyz": xyz,
        "structure_distance": distance,
        "structure_center_mask": center_mask,
        "structure_mask": structure_mask,
        "charge_xyz": charge_xyz,
        "rho_total": rho_total,
        "rho_total_raw": rho_total_raw,
        "rho_magnetization": torch.zeros_like(rho_total),
        "charge_mask": torch.ones_like(rho_total, dtype=torch.bool),
        "charge_volume_mask": torch.stack(volume_mask_rows),
        "patch_id": [sample["site_id"] for sample in samples],
        "site_id": [sample["site_id"] for sample in samples],
        "parent_id": [sample["parent_id"] for sample in samples],
        "pair_id": [sample["site_id"] for sample in samples],
        "center_atom": [int(sample["center_atom"]) for sample in samples],
        "split": [sample["split"] for sample in samples],
    }
    if role is not None:
        result["structure_role"] = role
    if rotation is not None:
        result["rotation"] = rotation
    return result


def make_task3_collate_fn(
    *,
    rho_mean: float,
    rho_std: float,
    rotate: bool,
    charge_points: int | None = None,
):
    def _collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collate_task3_sites(
            samples,
            rho_mean=rho_mean,
            rho_std=rho_std,
            rotate=rotate,
            charge_points=charge_points,
        )

    return _collate


def validate_pretraining_contract(root: str | Path = DEFAULT_TASK3_DATA_ROOT) -> dict[str, Any]:
    data_root = Path(root).expanduser().resolve()
    required = (
        "_SUCCESS",
        "dataset_contract.json",
        "validation_report.json",
        "density_stats.json",
        "nsp_manifest.csv",
        "nsp_train_site_schedule.npz",
        "nsp_validation_subset.csv",
        "charge_offsets.npy",
    )
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Task 3 data contract is incomplete: {missing}")
    validation = json.loads((data_root / "validation_report.json").read_text(encoding="utf-8"))
    if not validation.get("valid"):
        raise ValueError("Task 3 validation_report.json is not valid")
    contract = json.loads((data_root / "dataset_contract.json").read_text(encoding="utf-8"))
    if contract.get("pretraining_objective") != "symmetric in-batch InfoNCE only":
        raise ValueError("Task 3 objective contract changed unexpectedly")
    dataset = Task3NSPSiteDataset(data_root, cache_size=0)
    sampler = ScheduledSiteBatchSampler(
        dataset,
        data_root / "nsp_train_site_schedule.npz",
        batch_size=16,
        seed=int(contract["schedule_seed"]),
    )
    val_indices = validation_indices(dataset)
    return {
        "valid": True,
        "data_root": str(data_root),
        "nsp_parents": len(dataset.parents),
        "nsp_sites": len(dataset),
        "train_parents": len(sampler.parent_ids),
        "train_sites_per_epoch": sampler.samples_per_epoch,
        "schedule_epochs": sampler.epochs,
        "validation_sites": len(val_indices),
        "charge_points": int(dataset.charge_xyz.shape[0]),
        "use_center_indicator_supported": True,
    }


__all__ = [
    "DEFAULT_TASK3_DATA_ROOT",
    "EXPECTED_ATOMS",
    "EXPECTED_CHARGE_POINTS",
    "Task3NSPSiteDataset",
    "ScheduledSiteBatchSampler",
    "validation_indices",
    "split_indices",
    "load_density_stats",
    "collate_task3_sites",
    "make_task3_collate_fn",
    "validate_pretraining_contract",
]
