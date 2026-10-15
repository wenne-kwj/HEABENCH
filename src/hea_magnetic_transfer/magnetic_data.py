"""Parent-level SP data access for Task 3 magnetic transfer."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hea_retrieval.data import random_rotation_matrix

from .pretrain_data import DEFAULT_TASK3_DATA_ROOT, EXPECTED_ATOMS


# Frozen label budgets reported in the paper. The 300-parent membership is
# part of the released bundle; older internal pilot budgets are not public
# benchmark conditions.
SP_BUDGETS = (50, 300, 620)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_sp_magnetic_contract(
    root: str | Path = DEFAULT_TASK3_DATA_ROOT,
) -> dict[str, Any]:
    data_root = Path(root).expanduser().resolve()
    required = (
        "_SUCCESS",
        "_SP_EXTRACT_SUCCESS",
        "dataset_contract.json",
        "validation_report.json",
        "sp_manifest.csv",
        "sp_site_manifest.csv",
        "sp_labels.npz",
        "sp_budgets.json",
        "sp_budget_membership.csv",
    )
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SP magnetic contract is incomplete: {missing}")
    validation = json.loads((data_root / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True:
        raise ValueError("validation_report.json is not valid")
    parents = _read_csv(data_root / "sp_manifest.csv")
    parents.sort(key=lambda row: int(row["parent_index"]))
    if len(parents) != 886:
        raise ValueError(f"expected 886 SP parents, found {len(parents)}")
    if [int(row["parent_index"]) for row in parents] != list(range(886)):
        raise ValueError("SP parent_index is not contiguous")
    if len({row["parent_id"] for row in parents}) != len(parents):
        raise ValueError("duplicate SP parent IDs")
    split_counts = {
        split: sum(row["split"] == split for row in parents)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 620, "validation": 133, "test": 133}:
        raise ValueError(f"unexpected SP split counts: {split_counts}")
    with np.load(data_root / "sp_labels.npz", allow_pickle=False) as payload:
        label_ids = np.asarray(payload["parent_ids"]).astype(str)
        site_moments = np.asarray(payload["site_moments"], dtype=np.float32)
        cell_moments = np.asarray(payload["cell_moments"], dtype=np.float32)
        active_mask = np.asarray(payload["active_mask"], dtype=np.bool_)
    if label_ids.tolist() != [row["parent_id"] for row in parents]:
        raise ValueError("SP labels are not aligned with parent_index")
    if site_moments.shape != (886, EXPECTED_ATOMS):
        raise ValueError(f"unexpected site label shape: {site_moments.shape}")
    if cell_moments.shape != (886,) or active_mask.shape != site_moments.shape:
        raise ValueError("unexpected cell label or active-mask shape")
    if not np.isfinite(site_moments).all() or not np.isfinite(cell_moments).all():
        raise ValueError("SP magnetic labels contain NaN/Inf")
    if not np.array_equal(active_mask, np.abs(site_moments) > np.float32(0.1)):
        raise ValueError("stored active mask does not follow the float32 >0.1 contract")
    budgets_payload = json.loads((data_root / "sp_budgets.json").read_text(encoding="utf-8"))
    budgets = budgets_payload.get("budgets", {})
    train_ids = {row["parent_id"] for row in parents if row["split"] == "train"}
    previous: set[str] = set()
    for size in SP_BUDGETS:
        ids = list(budgets.get(str(size), []))
        current = set(ids)
        if len(ids) != size or len(current) != size:
            raise ValueError(f"SP budget {size} has wrong size or duplicates")
        if not previous.issubset(current) or not current.issubset(train_ids):
            raise ValueError(f"SP budget {size} is not nested or train-only")
        previous = current
    shard_root = data_root / "sp" / "structures"
    missing_shards = [row["parent_id"] for row in parents if not (shard_root / f"{row['parent_id']}.npz").is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing {len(missing_shards)} SP structure shards")
    overlap = validation.get("nsp_sp_overlap_counts", {})
    if any(int(value) != 0 for value in overlap.values()):
        raise ValueError(f"NSP/SP structure overlap is nonzero: {overlap}")
    train_positions = np.asarray(
        [index for index, row in enumerate(parents) if row["split"] == "train"],
        dtype=np.int64,
    )
    train_site = site_moments[train_positions]
    return {
        "valid": True,
        "data_root": str(data_root),
        "parents": len(parents),
        "sites": len(parents) * EXPECTED_ATOMS,
        "split_counts": split_counts,
        "budget_counts": {str(size): len(budgets[str(size)]) for size in SP_BUDGETS},
        "budgets_nested_train_only": True,
        "site_label_shape": list(site_moments.shape),
        "cell_label_shape": list(cell_moments.shape),
        "train_site_mean_muB": float(train_site.mean()),
        "train_site_std_muB": float(train_site.std()),
        "train_active_fraction": float(active_mask[train_positions].mean()),
        "nsp_sp_overlap_counts": overlap,
    }


def build_sp_parent_edge_cache(
    sample: dict[str, Any],
    *,
    cutoff: float = 5.0,
    max_neighbors: int = 64,
) -> dict[str, np.ndarray]:
    """Build one reusable, rotation-invariant graph catalog for an SP parent."""

    from hea_retrieval.models.structure.equiformer import build_radius_edges

    if cutoff <= 0 or max_neighbors <= 0:
        raise ValueError("edge cutoff and max_neighbors must be positive")
    node_offsets = np.asarray(sample["structure_offsets"], dtype=np.int64)
    if node_offsets.shape != (EXPECTED_ATOMS + 1,):
        raise ValueError("SP parent must contain exactly 100 patch offsets")
    z = np.asarray(sample["structure_z"], dtype=np.int64)
    xyz = np.asarray(sample["structure_xyz"], dtype=np.float32)
    edge_parts: list[np.ndarray] = []
    edge_offsets = [0]
    for site in range(EXPECTED_ATOMS):
        lo, hi = int(node_offsets[site]), int(node_offsets[site + 1])
        local_z = torch.as_tensor(z[lo:hi], dtype=torch.long).unsqueeze(0)
        local_xyz = torch.as_tensor(xyz[lo:hi], dtype=torch.float32).unsqueeze(0)
        local_mask = torch.ones_like(local_z, dtype=torch.bool)
        with torch.no_grad():
            local_edges, _ = build_radius_edges(
                local_z,
                local_xyz,
                local_mask,
                cutoff=float(cutoff),
                max_neighbors=int(max_neighbors),
            )
        local_numpy = local_edges.cpu().numpy().astype(np.int64, copy=False)
        edge_parts.append(local_numpy + lo)
        edge_offsets.append(edge_offsets[-1] + int(local_numpy.shape[1]))
    edge_index = np.concatenate(edge_parts, axis=1)
    return {
        "structure_edge_index": edge_index,
        "structure_node_offsets": np.array(node_offsets, copy=True),
        "structure_edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "edge_cutoff": np.asarray(float(cutoff), dtype=np.float32),
        "edge_max_neighbors": np.asarray(int(max_neighbors), dtype=np.int64),
    }


def _collate_precomputed_edges(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Tensor]:
    presence = ["structure_edge_index" in sample for sample in samples]
    if not any(presence):
        return {}
    if not all(presence):
        raise ValueError("a parent batch cannot mix cached and dynamic graphs")
    edge_parts: list[Tensor] = []
    node_offsets = [0]
    edge_offsets = [0]
    node_base = 0
    edge_base = 0
    for sample in samples:
        local_nodes = np.asarray(sample["structure_node_offsets"], dtype=np.int64)
        structure_offsets = np.asarray(sample["structure_offsets"], dtype=np.int64)
        local_edges = np.asarray(sample["structure_edge_index"], dtype=np.int64)
        local_edge_offsets = np.asarray(
            sample["structure_edge_offsets"], dtype=np.int64
        )
        if not np.array_equal(local_nodes, structure_offsets):
            raise ValueError("cached node offsets disagree with the structure shard")
        if (
            local_edge_offsets.shape != (EXPECTED_ATOMS + 1,)
            or int(local_edge_offsets[0]) != 0
            or int(local_edge_offsets[-1]) != local_edges.shape[1]
            or np.any(np.diff(local_edge_offsets) < 0)
        ):
            raise ValueError("cached edge offsets are malformed")
        if local_edges.ndim != 2 or local_edges.shape[0] != 2:
            raise ValueError("cached edge index must have shape [2,E]")
        if local_edges.size and (
            int(local_edges.min()) < 0
            or int(local_edges.max()) >= int(local_nodes[-1])
        ):
            raise ValueError("cached edge index is outside the packed parent nodes")
        edge_parts.append(torch.as_tensor(local_edges, dtype=torch.long) + node_base)
        node_offsets.extend((local_nodes[1:] + node_base).tolist())
        edge_offsets.extend((local_edge_offsets[1:] + edge_base).tolist())
        node_base += int(local_nodes[-1])
        edge_base += int(local_edges.shape[1])
    return {
        "structure_edge_index": torch.cat(edge_parts, dim=1),
        "structure_node_offsets": torch.tensor(node_offsets, dtype=torch.long),
        "structure_edge_offsets": torch.tensor(edge_offsets, dtype=torch.long),
    }


class Task3SPParentDataset(Dataset[dict[str, Any]]):
    """Return all 100 atom-centred structure patches and labels for one SP parent."""

    def __init__(
        self,
        root: str | Path = DEFAULT_TASK3_DATA_ROOT,
        *,
        split: str,
        budget: int | None = None,
        require_success: bool = True,
        edge_cache_root: str | Path | None = None,
        require_edge_cache: bool = False,
        edge_cutoff: float = 5.0,
        edge_max_neighbors: int = 64,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.edge_cache_root = (
            Path(edge_cache_root).expanduser().resolve()
            if edge_cache_root is not None
            else None
        )
        self.require_edge_cache = bool(require_edge_cache)
        self.edge_cutoff = float(edge_cutoff)
        self.edge_max_neighbors = int(edge_max_neighbors)
        if self.edge_cutoff <= 0 or self.edge_max_neighbors <= 0:
            raise ValueError("edge cutoff and max_neighbors must be positive")
        if self.require_edge_cache and self.edge_cache_root is None:
            raise ValueError("require_edge_cache needs edge_cache_root")
        if require_success and not (self.root / "_SUCCESS").is_file():
            raise RuntimeError(f"Task 3 data root has no _SUCCESS marker: {self.root}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown SP split: {split}")
        if budget is not None and split != "train":
            raise ValueError("SP label budgets are defined only for the train split")
        if budget is not None and int(budget) not in SP_BUDGETS:
            raise ValueError(f"budget must be one of {SP_BUDGETS}")
        rows = _read_csv(self.root / "sp_manifest.csv")
        rows.sort(key=lambda row: int(row["parent_index"]))
        self.all_parents = rows
        self.parent_by_id = {row["parent_id"]: row for row in rows}
        with np.load(self.root / "sp_labels.npz", allow_pickle=False) as payload:
            label_ids = np.asarray(payload["parent_ids"]).astype(str)
            self.site_moments = np.asarray(payload["site_moments"], dtype=np.float32)
            self.cell_moments = np.asarray(payload["cell_moments"], dtype=np.float32)
            self.active_mask = np.asarray(payload["active_mask"], dtype=np.bool_)
        if label_ids.tolist() != [row["parent_id"] for row in rows]:
            raise ValueError("SP labels and parent manifest are misaligned")
        if budget is None:
            parent_ids = [row["parent_id"] for row in rows if row["split"] == split]
        else:
            payload = json.loads((self.root / "sp_budgets.json").read_text(encoding="utf-8"))
            parent_ids = list(payload["budgets"][str(int(budget))])
        for parent_id in parent_ids:
            if parent_id not in self.parent_by_id:
                raise KeyError(f"unknown SP budget parent: {parent_id}")
            if self.parent_by_id[parent_id]["split"] != split:
                raise ValueError(f"parent {parent_id} is outside requested split {split}")
        self.parent_ids = parent_ids
        self.split = split
        self.budget = int(budget) if budget is not None else None

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(index)
        parent_id = self.parent_ids[position]
        row = self.parent_by_id[parent_id]
        parent_index = int(row["parent_index"])
        shard_path = self.root / "sp" / "structures" / f"{parent_id}.npz"
        with np.load(shard_path, allow_pickle=False) as payload:
            shard = {name: np.asarray(payload[name]) for name in payload.files}
        offsets = np.asarray(shard["offsets"], dtype=np.int64)
        z = np.asarray(shard["z"], dtype=np.int64)
        xyz = np.asarray(shard["xyz"], dtype=np.float32)
        distance = np.asarray(shard["distance"], dtype=np.float32)
        is_center = np.asarray(shard["is_center"], dtype=np.bool_)
        atom_index = np.asarray(shard["atom_index"], dtype=np.int64)
        if offsets.shape != (EXPECTED_ATOMS + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != len(z):
            raise ValueError(f"malformed offsets for {parent_id}")
        if xyz.shape != (len(z), 3) or distance.shape != z.shape:
            raise ValueError(f"malformed structure arrays for {parent_id}")
        if is_center.shape != z.shape or atom_index.shape != z.shape:
            raise ValueError(f"malformed centre arrays for {parent_id}")
        if not np.isfinite(xyz).all() or not np.isfinite(distance).all():
            raise ValueError(f"non-finite SP structure shard {parent_id}")
        for site in range(EXPECTED_ATOMS):
            lo, hi = int(offsets[site]), int(offsets[site + 1])
            centres = np.flatnonzero(is_center[lo:hi])
            if len(centres) != 1:
                raise ValueError(f"{parent_id}:{site} does not have one centre")
            centre = lo + int(centres[0])
            if int(atom_index[centre]) != site or abs(float(distance[centre])) > 1e-5:
                raise ValueError(f"{parent_id}:{site} centre mapping is inconsistent")
        sample: dict[str, Any] = {
            "parent_id": parent_id,
            "parent_index": parent_index,
            "split": row["split"],
            "structure_z": z,
            "structure_xyz": xyz,
            "structure_distance": distance,
            "structure_center_mask": is_center,
            "structure_atom_index": atom_index,
            "structure_offsets": offsets,
            "site_moments": np.array(self.site_moments[parent_index], copy=True),
            "cell_moment": float(self.cell_moments[parent_index]),
            "active_mask": np.array(self.active_mask[parent_index], copy=True),
        }


        if self.edge_cache_root is not None:
            edge_path = self.edge_cache_root / f"{parent_id}.npz"
            if not edge_path.is_file():
                if self.require_edge_cache:
                    raise FileNotFoundError(f"missing cached graph: {edge_path}")
            else:
                with np.load(edge_path, allow_pickle=False) as payload:
                    cached = {
                        name: np.array(payload[name], copy=True)
                        for name in payload.files
                    }
                required = {
                    "structure_edge_index",
                    "structure_node_offsets",
                    "structure_edge_offsets",
                    "edge_cutoff",
                    "edge_max_neighbors",
                }
                missing = sorted(required.difference(cached))
                if missing:
                    raise ValueError(f"cached graph {edge_path} is missing {missing}")
                cached_nodes = np.asarray(
                    cached["structure_node_offsets"], dtype=np.int64
                )
                if not np.array_equal(cached_nodes, offsets):
                    raise ValueError(
                        f"cached graph node offsets disagree for {parent_id}"
                    )
                if not math.isclose(
                    float(cached["edge_cutoff"]),
                    self.edge_cutoff,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(f"cached graph cutoff disagrees for {parent_id}")
                if int(cached["edge_max_neighbors"]) != self.edge_max_neighbors:
                    raise ValueError(
                        f"cached graph max_neighbors disagrees for {parent_id}"
                    )
                sample.update(
                    {
                        "structure_edge_index": np.asarray(
                            cached["structure_edge_index"], dtype=np.int64
                        ),
                        "structure_node_offsets": cached_nodes,
                        "structure_edge_offsets": np.asarray(
                            cached["structure_edge_offsets"], dtype=np.int64
                        ),
                    }
                )
        return sample


def collate_sp_parents(
    samples: Sequence[dict[str, Any]],
    *,
    rotate: bool = False,
    rotations: Tensor | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty SP parent batch")
    parent_count = len(samples)
    if rotations is not None and tuple(rotations.shape) != (parent_count, 3, 3):
        raise ValueError("rotations must have shape [B_parent,3,3]")
    if rotations is None and rotate:
        rotations = torch.stack([random_rotation_matrix() for _ in samples])
    patch_sizes: list[int] = []
    for sample in samples:
        offsets = np.asarray(sample["structure_offsets"], dtype=np.int64)
        if offsets.shape != (EXPECTED_ATOMS + 1,):
            raise ValueError("every SP parent must contain exactly 100 site patches")
        patch_sizes.extend(np.diff(offsets).astype(int).tolist())
    max_atoms = max(patch_sizes)
    patch_count = parent_count * EXPECTED_ATOMS
    z = torch.zeros((patch_count, max_atoms), dtype=torch.long)
    xyz = torch.zeros((patch_count, max_atoms, 3), dtype=torch.float32)
    distance = torch.zeros((patch_count, max_atoms), dtype=torch.float32)
    center_mask = torch.zeros((patch_count, max_atoms), dtype=torch.bool)
    structure_mask = torch.zeros((patch_count, max_atoms), dtype=torch.bool)
    role = torch.zeros((patch_count, max_atoms), dtype=torch.long)
    patch_parent = torch.empty(patch_count, dtype=torch.long)
    patch_site = torch.empty(patch_count, dtype=torch.long)
    site_ids: list[str] = []
    for parent_position, sample in enumerate(samples):
        offsets = np.asarray(sample["structure_offsets"], dtype=np.int64)
        local_rotation = rotations[parent_position] if rotations is not None else None
        for site in range(EXPECTED_ATOMS):
            flat = parent_position * EXPECTED_ATOMS + site
            lo, hi = int(offsets[site]), int(offsets[site + 1])
            count = hi - lo
            local_z = torch.as_tensor(sample["structure_z"][lo:hi], dtype=torch.long)
            local_xyz = torch.as_tensor(sample["structure_xyz"][lo:hi], dtype=torch.float32)
            local_distance = torch.as_tensor(
                sample["structure_distance"][lo:hi], dtype=torch.float32
            )
            local_center = torch.as_tensor(
                sample["structure_center_mask"][lo:hi], dtype=torch.bool
            )
            if int(local_center.sum()) != 1:
                raise ValueError(f"{sample['parent_id']}:{site} has an invalid centre")
            if local_rotation is not None:
                local_xyz = local_xyz @ local_rotation.to(dtype=local_xyz.dtype).T
            z[flat, :count] = local_z
            xyz[flat, :count] = local_xyz
            distance[flat, :count] = local_distance
            center_mask[flat, :count] = local_center
            structure_mask[flat, :count] = True
            role[flat, :count] = local_center.long()
            patch_parent[flat] = parent_position
            patch_site[flat] = site
            site_ids.append(f"{sample['parent_id']}_site_{site:03d}")
    site_targets = torch.stack(
        [torch.as_tensor(sample["site_moments"], dtype=torch.float32) for sample in samples]
    )
    cell_targets = torch.tensor(
        [float(sample["cell_moment"]) for sample in samples], dtype=torch.float32
    )
    active_mask = torch.stack(
        [torch.as_tensor(sample["active_mask"], dtype=torch.bool) for sample in samples]
    )
    if site_targets.shape != (parent_count, EXPECTED_ATOMS):
        raise ValueError("unexpected SP site-label shape after collation")
    if not torch.isfinite(site_targets).all() or not torch.isfinite(cell_targets).all():
        raise ValueError("non-finite SP targets after collation")
    result: dict[str, Any] = {
        "structure_z": z,
        "structure_xyz": xyz,
        "structure_distance": distance,
        "structure_center_mask": center_mask,
        "structure_role": role,
        "structure_mask": structure_mask,
        "patch_parent": patch_parent,
        "patch_site": patch_site,
        "site_targets": site_targets,
        "cell_targets": cell_targets,
        "active_mask": active_mask,
        "parent_id": [sample["parent_id"] for sample in samples],
        "parent_index": torch.tensor(
            [int(sample["parent_index"]) for sample in samples], dtype=torch.long
        ),
        "split": [sample["split"] for sample in samples],
        "site_id": site_ids,
        "sites_per_parent": EXPECTED_ATOMS,
    }
    result.update(_collate_precomputed_edges(samples))
    if rotations is not None:
        result["rotation"] = rotations.to(dtype=torch.float32)
    return result


def make_sp_parent_collate_fn(*, rotate: bool):
    def _collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collate_sp_parents(samples, rotate=rotate)

    return _collate


__all__ = [
    "SP_BUDGETS",
    "Task3SPParentDataset",
    "build_sp_parent_edge_cache",
    "collate_sp_parents",
    "make_sp_parent_collate_fn",
    "validate_sp_magnetic_contract",
]
