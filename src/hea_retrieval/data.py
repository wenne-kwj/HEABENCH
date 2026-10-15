"""Loader and collation for ``hea-local-retrieval-v1``.

The extraction pipeline stores variable-length structure patches as flat arrays
plus an ``offsets`` index and stores fixed-size charge patches in separate NPZ
shards.  This module is intentionally independent of PyG/OpenPoints so it can
also be used for manifest audits and CPU-side tests.
"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


from .paths import DEFAULT_DATA_ROOT

# Every stored charge patch contains 3,968 shared quasi-uniform volume points
# followed by 128 pair-focused points.  The latter improve near-endpoint
# resolution but are not volume quadrature samples.  Keep this contract
# explicit so physically integrated features can use only the unbiased prefix
# while learned point-cloud descriptors may still consume all 4,096 points.
DEFAULT_UNIFORM_CHARGE_POINTS = 3968


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class PatchMeta:
    patch_id: str
    parent_id: str
    split: str
    pair_id: str
    view_id: str
    structure_shard: str
    structure_row: int
    charge_shard: str
    charge_row: int
    n_structure_atoms: int
    n_charge_points: int


class _ShardCache:
    """Small LRU cache for compressed NPZ shards.

    A complete charge dataset is hundreds of GB when uncompressed, so loading
    every shard at process start is prohibited.  The default cache of two
    shards keeps nearby samples cheap while bounding worker memory.
    """

    def __init__(self, root: Path, kind: str, capacity: int = 2) -> None:
        self.root = root / kind
        self.capacity = max(0, int(capacity))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, name: str) -> dict[str, np.ndarray]:
        name = str(name)
        if self.capacity > 0 and name in self._cache:
            value = self._cache.pop(name)
            self._cache[name] = value
            return value
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            value = {key: np.asarray(data[key]) for key in data.files}
        if self.capacity > 0:
            self._cache[name] = value
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return value


class PatchStore:
    """Random access to one row of the structure and charge shard stores."""

    def __init__(self, root: str | Path = DEFAULT_DATA_ROOT, cache_size: int = 2) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"retrieval data root does not exist: {self.root}")
        manifest_path = self.root / "patch_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        rows = _read_csv(manifest_path)
        self.rows = rows
        self.by_id: dict[str, dict[str, str]] = {}
        for row in rows:
            patch_id = row.get("patch_id", "")
            if not patch_id or patch_id in self.by_id:
                raise ValueError(f"duplicate or empty patch_id in {manifest_path}: {patch_id!r}")
            self.by_id[patch_id] = row
        self._structures = _ShardCache(self.root, "structures", cache_size)
        self._charges = _ShardCache(self.root, "charges", cache_size)

    def __len__(self) -> int:
        return len(self.rows)

    def metadata(self, patch_id: str) -> PatchMeta:
        row = self.by_id[patch_id]
        return PatchMeta(
            patch_id=patch_id,
            parent_id=row["parent_id"],
            split=row["split"],
            pair_id=row["pair_id"],
            view_id=row["view_id"],
            structure_shard=row["structure_shard"],
            structure_row=int(row["structure_row"]),
            charge_shard=row["charge_shard"],
            charge_row=int(row["charge_row"]),
            n_structure_atoms=int(row["n_structure_atoms"]),
            n_charge_points=int(row.get("n_charge_points", 4096)),
        )

    def get(self, patch_id: str) -> dict[str, Any]:
        """Return a single patch as NumPy arrays plus immutable metadata."""

        meta = self.metadata(patch_id)
        s = self._structures.get(meta.structure_shard)
        c = self._charges.get(meta.charge_shard)
        row = meta.structure_row
        offsets = s["offsets"]
        if row < 0 or row + 1 >= len(offsets):
            raise IndexError(f"structure row {row} out of range for {meta.structure_shard}")
        lo, hi = int(offsets[row]), int(offsets[row + 1])
        charge_row = meta.charge_row
        z = np.asarray(s["z"][lo:hi], dtype=np.int64)
        xyz = np.asarray(s["xyz"][lo:hi], dtype=np.float32)
        role = np.asarray(s["role"][lo:hi], dtype=np.int64)
        distance = np.asarray(s["distance"][lo:hi], dtype=np.float32)
        image_shift = np.asarray(s["image_shift"][lo:hi], dtype=np.int16)
        charge_xyz = np.asarray(c["xyz"][charge_row], dtype=np.float32)
        rho_total = np.asarray(c["rho_total"][charge_row], dtype=np.float32)
        if "rho_magnetization" in c:
            rho_mag = np.asarray(c["rho_magnetization"][charge_row], dtype=np.float32)
        else:
            rho_mag = np.zeros_like(rho_total, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or z.shape[0] != xyz.shape[0]:
            raise ValueError(f"malformed structure patch {patch_id}")
        if z.shape[0] != meta.n_structure_atoms:
            raise ValueError(
                f"structure atom count mismatch for {patch_id}: "
                f"manifest={meta.n_structure_atoms}, shard={z.shape[0]}"
            )
        if distance.shape != (z.shape[0],) or role.shape != (z.shape[0],):
            raise ValueError(f"malformed structure auxiliary fields {patch_id}")
        if image_shift.shape != (z.shape[0], 3):
            raise ValueError(f"malformed image_shift field {patch_id}")
        if charge_xyz.shape != (rho_total.shape[0], 3):
            raise ValueError(f"malformed charge patch {patch_id}")
        if rho_total.shape != (meta.n_charge_points,) or charge_xyz.shape[0] != meta.n_charge_points:
            raise ValueError(
                f"charge point count mismatch for {patch_id}: "
                f"manifest={meta.n_charge_points}, shard={charge_xyz.shape[0]}"
            )
        if role.shape[0] != z.shape[0] or np.any((role < 0) | (role > 2)):
            raise ValueError(f"invalid role values in {patch_id}")
        return {
            "patch_id": meta.patch_id,
            "parent_id": meta.parent_id,
            "split": meta.split,
            "pair_id": meta.pair_id,
            "view_id": meta.view_id,
            "structure_z": z,
            "structure_xyz": xyz,
            "structure_role": role,
            "structure_distance": distance,
            "structure_image_shift": image_shift,
            "charge_xyz": charge_xyz,
            "rho_total": rho_total,
            "rho_magnetization": rho_mag,
        }


class RetrievalPatchDataset(Dataset[dict[str, Any]]):
    """Dataset of canonical anchor patches (one structure + one charge)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_ROOT,
        split: str | None = "train",
        manifest: str = "anchor_manifest.csv",
        cache_size: int = 2,
        patch_ids: Sequence[str] | None = None,
    ) -> None:
        self.store = PatchStore(root, cache_size=cache_size)
        manifest_rows = _read_csv(self.store.root / manifest)
        available = self.store.by_id
        selected: list[str] = []
        for row in manifest_rows:
            patch_id = row.get("patch_id", "")
            if patch_id not in available:
                raise KeyError(f"{patch_id} from {manifest} is absent in patch_manifest.csv")
            if split is None or row.get("split") == split:
                selected.append(patch_id)
        if patch_ids is not None:
            requested = set(patch_ids)
            selected = [x for x in selected if x in requested]
        self.patch_ids = selected
        self._meta_rows = {r["patch_id"]: r for r in manifest_rows if r.get("patch_id") in set(selected)}

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.store.get(self.patch_ids[index])


def random_rotation_matrix(
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Draw a uniform-ish proper 3-D rotation using QR decomposition."""

    q, r = torch.linalg.qr(torch.randn((3, 3), generator=generator, device=device, dtype=dtype))
    diag = torch.sign(torch.diag(r))
    signs = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q @ torch.diag(signs)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _apply_rotation(x: Tensor, rotation: Tensor) -> Tensor:
    return x @ rotation.transpose(0, 1)


def collate_patches(
    samples: Sequence[dict[str, Any]],
    *,
    rho_mean: float = 0.0,
    rho_std: float = 1.0,
    rho_mag_mean: float = 0.0,
    rho_mag_std: float = 1.0,
    rotate: bool = False,
    rotation: Tensor | None = None,
    charge_points: int | None = None,
) -> dict[str, Any]:
    """Pad structures and stack charge clouds into model-ready tensors.

    If ``rotate`` is enabled, exactly one matrix is applied to both modalities
    in this batch.  This is important: independently rotating structure and
    density would destroy the physical correspondence.
    """

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for name, value in (
        ("rho_mean", rho_mean),
        ("rho_std", rho_std),
        ("rho_mag_mean", rho_mag_mean),
        ("rho_mag_std", rho_mag_std),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(rho_std) < 0 or float(rho_mag_std) < 0:
        raise ValueError("density standard deviations must be non-negative")
    device = torch.device("cpu")
    bsz = len(samples)
    max_atoms = max(int(np.asarray(s["structure_z"]).shape[0]) for s in samples)
    z = torch.zeros((bsz, max_atoms), dtype=torch.long, device=device)
    xyz = torch.zeros((bsz, max_atoms, 3), dtype=torch.float32, device=device)
    role = torch.zeros((bsz, max_atoms), dtype=torch.long, device=device)
    distance = torch.zeros((bsz, max_atoms), dtype=torch.float32, device=device)
    mask = torch.zeros((bsz, max_atoms), dtype=torch.bool, device=device)

    charge_xyz_list: list[Tensor] = []
    rho_list: list[Tensor] = []
    rho_raw_list: list[Tensor] = []
    mag_list: list[Tensor] = []
    charge_mask_list: list[Tensor] = []
    charge_volume_mask_list: list[Tensor] = []
    for i, sample in enumerate(samples):
        n = int(np.asarray(sample["structure_z"]).shape[0])
        z[i, :n] = torch.as_tensor(sample["structure_z"], dtype=torch.long)
        xyz[i, :n] = torch.as_tensor(sample["structure_xyz"], dtype=torch.float32)
        role[i, :n] = torch.as_tensor(sample["structure_role"], dtype=torch.long)
        distance[i, :n] = torch.as_tensor(sample["structure_distance"], dtype=torch.float32)
        mask[i, :n] = True
        cx = torch.as_tensor(sample["charge_xyz"], dtype=torch.float32)
        rt = torch.as_tensor(sample["rho_total"], dtype=torch.float32)
        rm = torch.as_tensor(sample["rho_magnetization"], dtype=torch.float32)
        source_indices = torch.arange(cx.shape[0], dtype=torch.long)
        if charge_points is not None:
            if charge_points < 8:
                raise ValueError("charge_points must be at least 8")
            if charge_points > cx.shape[0]:
                raise ValueError(f"charge_points={charge_points} exceeds stored {cx.shape[0]}")
            # Deterministic, evenly spread subsampling for bounded diagnostics. Formal
            # training uses all 4096 points by leaving this argument unset.
            idx = torch.linspace(0, cx.shape[0] - 1, charge_points).round().long()
            cx, rt, rm = cx[idx], rt[idx], rm[idx]
            source_indices = source_indices[idx]
        charge_xyz_list.append(cx)
        rho_raw_list.append(rt)
        rho_list.append((rt - float(rho_mean)) / max(float(rho_std), 1e-8))
        mag_list.append((rm - float(rho_mag_mean)) / max(float(rho_mag_std), 1e-8))
        charge_mask_list.append(torch.ones_like(rt, dtype=torch.bool))
        charge_volume_mask_list.append(
            source_indices.lt(
                min(
                    DEFAULT_UNIFORM_CHARGE_POINTS,
                    int(np.asarray(sample["rho_total"]).shape[0]),
                )
            )
        )

    charge_xyz = torch.stack(charge_xyz_list, dim=0)
    rho_total = torch.stack(rho_list, dim=0)
    rho_total_raw = torch.stack(rho_raw_list, dim=0)
    rho_mag = torch.stack(mag_list, dim=0)
    charge_mask = torch.stack(charge_mask_list, dim=0)
    charge_volume_mask = torch.stack(charge_volume_mask_list, dim=0)
    if rotate:
        if rotation is None:
            rotation = random_rotation_matrix(device=device)
        rotation = rotation.to(device=device, dtype=xyz.dtype)
        xyz = _apply_rotation(xyz, rotation)
        charge_xyz = _apply_rotation(charge_xyz, rotation)

    result: dict[str, Any] = {
        "structure_z": z,
        "structure_xyz": xyz,
        "structure_role": role,
        "structure_distance": distance,
        "structure_mask": mask,
        "charge_xyz": charge_xyz,
        "rho_total": rho_total,
        "rho_total_raw": rho_total_raw,
        "rho_magnetization": rho_mag,
        "charge_mask": charge_mask,
        "charge_volume_mask": charge_volume_mask,
        "patch_id": [s["patch_id"] for s in samples],
        "parent_id": [s["parent_id"] for s in samples],
        "pair_id": [s["pair_id"] for s in samples],
        "split": [s["split"] for s in samples],
    }
    if rotation is not None:
        result["rotation"] = rotation
    return result


def make_collate_fn(
    *,
    rho_mean: float = 0.0,
    rho_std: float = 1.0,
    rho_mag_mean: float = 0.0,
    rho_mag_std: float = 1.0,
    rotate: bool = False,
    charge_points: int | None = None,
):
    def _fn(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collate_patches(
            samples,
            rho_mean=rho_mean,
            rho_std=rho_std,
            rho_mag_mean=rho_mag_mean,
            rho_mag_std=rho_mag_std,
            rotate=rotate,
            charge_points=charge_points,
        )

    return _fn


class ShardBatchSampler(Sampler[list[int]]):
    """Batch indices grouped by charge shard to reduce compressed-NPZ churn."""

    def __init__(self, dataset: RetrievalPatchDataset, batch_size: int, shuffle: bool = True, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        groups: dict[str, list[int]] = {}
        for idx, patch_id in enumerate(dataset.patch_ids):
            shard = dataset.store.metadata(patch_id).charge_shard
            groups.setdefault(shard, []).append(idx)
        self.groups = list(groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        groups = [list(g) for g in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
        for group in groups:
            if self.shuffle:
                rng.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                # Keep the final short batch.  The loss supports arbitrary
                # batch sizes and dropping it would bias shard boundaries.
                yield batch

    def __len__(self) -> int:
        return sum(math.ceil(len(g) / self.batch_size) for g in self.groups)


def compute_density_stats(
    dataset: RetrievalPatchDataset,
    *,
    max_patches: int = 2048,
    point_stride: int = 16,
) -> dict[str, float | int]:
    """Compute train-only density statistics with bounded I/O/memory."""

    if max_patches <= 0 or point_stride <= 0:
        raise ValueError("max_patches and point_stride must be positive")
    n = min(len(dataset), max_patches)
    total_n = mag_n = 0
    total_mean = total_m2 = mag_mean = mag_m2 = 0.0
    for i in range(n):
        sample = dataset[i]
        for key, state in (("rho_total", "total"), ("rho_magnetization", "mag")):
            values = np.asarray(sample[key], dtype=np.float64)[::point_stride].ravel()
            mean, m2, count = (total_mean, total_m2, total_n) if state == "total" else (mag_mean, mag_m2, mag_n)
            for value in values:
                count += 1
                delta = float(value) - mean
                mean += delta / count
                m2 += delta * (float(value) - mean)
            if state == "total":
                total_mean, total_m2, total_n = mean, m2, count
            else:
                mag_mean, mag_m2, mag_n = mean, m2, count
    return {
        "rho_mean": total_mean,
        "rho_std": math.sqrt(total_m2 / max(total_n - 1, 1)),
        "rho_mag_mean": mag_mean,
        "rho_mag_std": math.sqrt(mag_m2 / max(mag_n - 1, 1)),
        "patches": n,
        "point_stride": point_stride,
    }


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_UNIFORM_CHARGE_POINTS",
    "PatchStore",
    "RetrievalPatchDataset",
    "collate_patches",
    "make_collate_fn",
    "ShardBatchSampler",
    "compute_density_stats",
    "random_rotation_matrix",
]
