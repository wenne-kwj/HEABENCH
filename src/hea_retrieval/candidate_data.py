"""Candidate-set loading and collation for local-matching evaluation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .data import (
    DEFAULT_DATA_ROOT,
    PatchStore,
    _read_csv,
    collate_patches,
    random_rotation_matrix,
)


def validate_candidate_manifest(root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    """Stream-audit the 1-positive/10-negative candidate contract."""

    root = Path(root)
    candidate_path = root / "candidate_manifest.csv"
    patch_path = root / "patch_manifest.csv"
    patch_rows = {r["patch_id"]: r for r in _read_csv(patch_path)}
    errors: list[str] = []
    groups = 0
    rows_count = 0
    same_pair_negative = 0
    cross_split = 0
    direction_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    positive_position_counts: dict[str, dict[str, int]] = {}
    seen_keys: set[tuple[str, str]] = set()
    current_key: tuple[str, str] | None = None
    current: list[dict[str, str]] = []

    def error(message: str) -> None:
        if len(errors) < 50:
            errors.append(message)

    def audit(rows: list[dict[str, str]], key: tuple[str, str] | None) -> None:
        nonlocal groups, same_pair_negative, cross_split
        if not rows or key is None:
            return
        groups += 1
        query_id, direction = key
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        if direction not in {"structure_to_charge", "charge_to_structure"}:
            error(f"invalid direction {direction!r} for {query_id}")
        if len(rows) != 11:
            error(f"{query_id}/{direction} has {len(rows)} candidates")
        try:
            labels = [int(row.get("label", "-1")) for row in rows]
        except ValueError:
            labels = []
        if len(labels) != len(rows) or any(label not in (0, 1) for label in labels):
            error(f"{query_id}/{direction} has invalid labels")
            labels = [-1] * len(rows)
        if sum(label == 1 for label in labels) != 1:
            error(f"{query_id}/{direction} does not have exactly one positive")
        elif labels:
            position = str(labels.index(1))
            by_position = positive_position_counts.setdefault(direction, {})
            by_position[position] = by_position.get(position, 0) + 1
        qrow = patch_rows.get(query_id)
        if qrow is None:
            error(f"unknown query patch {query_id}")
            return
        split = qrow["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        candidate_ids: set[str] = set()
        for row, label in zip(rows, labels):
            candidate_id = row.get("candidate_patch_id", "")
            if candidate_id in candidate_ids:
                error(f"duplicate candidate {candidate_id} in {query_id}/{direction}")
            candidate_ids.add(candidate_id)
            crow = patch_rows.get(candidate_id)
            if crow is None:
                error(f"unknown candidate patch {candidate_id}")
                continue
            if crow["split"] != split or row.get("same_split", "").lower() != "true":
                cross_split += 1
                error(f"cross-split candidate {query_id}->{candidate_id}")
            if label == 0 and (
                row.get("same_pair_id", "").lower() == "true"
                or crow["pair_id"] == qrow["pair_id"]
            ):
                same_pair_negative += 1
                error(f"same-pair negative {query_id}->{candidate_id}")

    if not candidate_path.exists():
        return {
            "rows": 0,
            "groups": 0,
            "expected_groups": 2 * len(patch_rows),
            "error_count": 1,
            "errors": [f"missing {candidate_path}"],
            "valid": False,
        }
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_patch_id", "candidate_patch_id", "direction", "label"}
        if not required.issubset(reader.fieldnames or set()):
            error(f"missing candidate columns: {sorted(required)}")
        for row in reader:
            rows_count += 1
            key = (row.get("query_patch_id", ""), row.get("direction", ""))
            if current_key is None:
                current_key = key
            elif key != current_key:
                audit(current, current_key)
                seen_keys.add(current_key)
                if key in seen_keys:
                    error(f"non-contiguous candidate group {key}")
                current = []
                current_key = key
            current.append(row)
        audit(current, current_key)

    expected_groups = 2 * len(patch_rows)
    return {
        "rows": rows_count,
        "groups": groups,
        "expected_groups": expected_groups,
        "direction_counts": direction_counts,
        "split_group_counts": split_counts,
        "positive_position_counts": positive_position_counts,
        "same_pair_id_as_negative": same_pair_negative > 0,
        "same_pair_negative_count": same_pair_negative,
        "cross_split_count": cross_split,
        "same_split_only": cross_split == 0,
        "error_count": len(errors),
        "errors": errors,
        "valid": (
            rows_count == expected_groups * 11
            and groups == expected_groups
            and not errors
            and same_pair_negative == 0
            and cross_split == 0
        ),
    }


@dataclass(frozen=True)
class CandidateGroupMeta:
    """Compact description of one explicit 11-way retrieval query."""

    query_patch_id: str
    direction: str
    candidate_patch_ids: tuple[str, ...]
    target: int
    split: str


class CandidateGroupDataset(Dataset[dict[str, Any]]):
    """Dataset backed by candidate_manifest.csv.

    Each item is one query and its explicit candidate set.  The manifest is
    parsed as a stream (rather than materialising 860k dictionaries), and all
    split/pair exclusions are checked before items become visible to a loader.
    """

    VALID_DIRECTIONS = frozenset({"structure_to_charge", "charge_to_structure"})

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_ROOT,
        split: str | None = "train",
        manifest: str = "candidate_manifest.csv",
        cache_size: int = 2,
        direction: str | None = None,
        max_groups: int | None = None,
        strict: bool = True,
    ) -> None:
        self.store = PatchStore(root, cache_size=cache_size)
        if direction is not None and direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"unknown candidate direction: {direction}")
        manifest_path = self.store.root / manifest
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.groups = self._read_groups(manifest_path, strict=strict)
        if direction is not None:
            self.groups = [g for g in self.groups if g.direction == direction]
        if split is not None:
            self.groups = [g for g in self.groups if g.split == split]
        if max_groups is not None:
            if max_groups <= 0:
                raise ValueError("max_groups must be positive when supplied")
            limit = int(max_groups)
            # The manifest is ordered by direction. Keep both directions
            # represented when a bounded diagnostic subset is requested.
            if direction is None:
                buckets: dict[str, list[CandidateGroupMeta]] = {}
                for group in self.groups:
                    buckets.setdefault(group.direction, []).append(group)
                selected: list[CandidateGroupMeta] = []
                while len(selected) < limit and buckets:
                    progressed = False
                    for name in sorted(tuple(buckets)):
                        bucket = buckets[name]
                        if bucket:
                            selected.append(bucket.pop(0))
                            progressed = True
                            if len(selected) >= limit:
                                break
                        if not bucket:
                            buckets.pop(name, None)
                    if not progressed:
                        break
                self.groups = selected
            else:
                self.groups = self.groups[:limit]

    def _read_groups(self, path: Path, *, strict: bool) -> list[CandidateGroupMeta]:
        patch_rows = self.store.by_id
        groups: list[CandidateGroupMeta] = []
        seen_keys: set[tuple[str, str]] = set()
        all_keys: set[tuple[str, str]] = set()
        current_key: tuple[str, str] | None = None
        current: list[dict[str, str]] = []

        def finish(rows: list[dict[str, str]], key: tuple[str, str] | None) -> None:
            if not rows or key is None:
                return
            query_id, direction = key
            if direction not in self.VALID_DIRECTIONS:
                raise ValueError(f"invalid candidate direction {direction!r}")
            qrow = patch_rows.get(query_id)
            if qrow is None:
                raise KeyError(f"candidate query patch is absent: {query_id}")
            if len(rows) != 11:
                raise ValueError(f"{query_id}/{direction} has {len(rows)} candidates; expected 11")
            labels = [int(r.get("label", "-1")) for r in rows]
            if any(label not in (0, 1) for label in labels) or sum(labels) != 1:
                raise ValueError(f"{query_id}/{direction} must contain exactly one positive")
            target = labels.index(1)
            query_pair = qrow["pair_id"]
            query_split = qrow["split"]
            candidate_ids: list[str] = []
            for row, label in zip(rows, labels):
                candidate_id = row.get("candidate_patch_id", "")
                crow = patch_rows.get(candidate_id)
                if crow is None:
                    raise KeyError(f"candidate patch is absent: {candidate_id}")
                if candidate_id in candidate_ids:
                    raise ValueError(f"duplicate candidate {candidate_id} in {query_id}/{direction}")
                candidate_ids.append(candidate_id)
                if crow["split"] != query_split or row.get("same_split", "").lower() != "true":
                    raise ValueError(f"cross-split candidate in {query_id}/{direction}: {candidate_id}")
                if label == 0 and (
                    row.get("same_pair_id", "").lower() == "true"
                    or crow["pair_id"] == query_pair
                ):
                    raise ValueError(f"reverse/same pair used as negative: {query_id}->{candidate_id}")
            groups.append(
                CandidateGroupMeta(
                    query_patch_id=query_id,
                    direction=direction,
                    candidate_patch_ids=tuple(candidate_ids),
                    target=target,
                    split=query_split,
                )
            )

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"query_patch_id", "candidate_patch_id", "direction", "label"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"{path} is missing candidate columns: {sorted(required)}")
            for row in reader:
                key = (row["query_patch_id"], row["direction"])
                all_keys.add(key)
                if current_key is None:
                    current_key = key
                elif key != current_key:
                    finish(current, current_key)
                    seen_keys.add(current_key)
                    if key in seen_keys:
                        raise ValueError(f"candidate groups are not contiguous: {key}")
                    current = []
                    current_key = key
                current.append(row)
            finish(current, current_key)

        if strict:
            expected = 2 * len(patch_rows)
            expected_keys = {
                (patch_id, direction)
                for patch_id in patch_rows
                for direction in self.VALID_DIRECTIONS
            }
            if len(groups) != expected or all_keys != expected_keys:
                missing = sorted(expected_keys - all_keys)[:5]
                extra = sorted(all_keys - expected_keys)[:5]
                raise ValueError(
                    f"candidate manifest has {len(groups)} groups; expected {expected}; "
                    f"missing={missing}, extra={extra}"
                )
        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.groups[index]
        return {
            "meta": meta,
            "query": self.store.get(meta.query_patch_id),
            "candidates": [self.store.get(pid) for pid in meta.candidate_patch_ids],
        }


class CandidateGroupBatchSampler(Sampler[list[int]]):
    """Batch explicit groups while keeping a single direction per batch."""

    def __init__(
        self,
        dataset: CandidateGroupDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        max_groups: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be positive when supplied")
        self.max_groups = None if max_groups is None else int(max_groups)
        buckets: dict[str, list[int]] = {}
        for index, meta in enumerate(dataset.groups):
            buckets.setdefault(meta.direction, []).append(index)
        self.buckets = list(buckets.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        buckets = [list(bucket) for bucket in self.buckets]
        for indices in buckets:
            if self.shuffle:
                rng.shuffle(indices)
        if self.max_groups is not None:
            limit = min(self.max_groups, sum(len(bucket) for bucket in buckets))
            selected: list[list[int]] = [[] for _ in buckets]
            # Round-robin selection keeps both directions represented and
            # changes the subset across epochs through the seeded shuffle.
            while sum(len(item) for item in selected) < limit:
                progressed = False
                for bucket_index, indices in enumerate(buckets):
                    if indices and sum(len(item) for item in selected) < limit:
                        selected[bucket_index].append(indices.pop())
                        progressed = True
                if not progressed:
                    break
            buckets = selected
        for bucket in buckets:
            batches.extend(
                bucket[start : start + self.batch_size]
                for start in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        lengths = [len(bucket) for bucket in self.buckets]
        if self.max_groups is not None:
            limit = min(self.max_groups, sum(lengths))
            quotas = [0 for _ in lengths]
            assigned = 0
            while assigned < limit:
                progressed = False
                for index, length in enumerate(lengths):
                    if quotas[index] < length and assigned < limit:
                        quotas[index] += 1
                        assigned += 1
                        progressed = True
                if not progressed:
                    break
            lengths = quotas
        return sum(math.ceil(length / self.batch_size) for length in lengths)


def _reshape_candidate_tensor(value: Tensor, batch_size: int, candidates: int) -> Tensor:
    if value.shape[0] != batch_size * candidates:
        raise ValueError(
            f"candidate tensor first dimension {value.shape[0]} != {batch_size}*{candidates}"
        )
    return value.reshape(batch_size, candidates, *value.shape[1:])


def collate_candidate_groups(
    groups: Sequence[dict[str, Any]],
    *,
    rho_mean: float = 0.0,
    rho_std: float = 1.0,
    rho_mag_mean: float = 0.0,
    rho_mag_std: float = 1.0,
    rotate: bool = False,
    charge_points: int | None = None,
) -> dict[str, Any]:
    """Collate explicit candidate groups for either retrieval direction.

    The same random rotation is applied to the query and every candidate
    patch.  Thus the positive pair remains physically aligned while negatives
    receive the same augmentation convention.
    """

    if not groups:
        raise ValueError("cannot collate empty candidate groups")
    metas = [item["meta"] for item in groups]
    directions = {meta.direction for meta in metas}
    if len(directions) != 1:
        raise ValueError("a candidate batch must contain one retrieval direction")
    candidate_counts = {len(meta.candidate_patch_ids) for meta in metas}
    if candidate_counts != {11}:
        raise ValueError(f"candidate groups must have 11 candidates, got {candidate_counts}")
    batch_size = len(groups)
    candidates = 11
    rotation = random_rotation_matrix() if rotate else None
    kwargs = dict(
        rho_mean=rho_mean,
        rho_std=rho_std,
        rho_mag_mean=rho_mag_mean,
        rho_mag_std=rho_mag_std,
        rotate=False,
        rotation=rotation,
        charge_points=charge_points,
    )
    query_batch = collate_patches([item["query"] for item in groups], **kwargs)
    flat_candidates = [
        candidate
        for item in groups
        for candidate in item["candidates"]
    ]
    candidate_batch = collate_patches(flat_candidates, **kwargs)
    result: dict[str, Any] = {
        "direction": metas[0].direction,
        "target": torch.tensor([meta.target for meta in metas], dtype=torch.long),
        "query_patch_id": [meta.query_patch_id for meta in metas],
        "candidate_patch_ids": [list(meta.candidate_patch_ids) for meta in metas],
        "split": [meta.split for meta in metas],
    }
    for field in (
        "structure_z",
        "structure_xyz",
        "structure_role",
        "structure_distance",
        "structure_mask",
        "charge_xyz",
        "rho_total",
        "rho_total_raw",
        "rho_magnetization",
        "charge_mask",
        "charge_volume_mask",
    ):
        result[f"query_{field}"] = query_batch[field]
        result[f"candidate_{field}"] = _reshape_candidate_tensor(
            candidate_batch[field], batch_size, candidates
        )
    if rotation is not None:
        result["rotation"] = rotation
    return result


def make_candidate_collate_fn(
    *,
    rho_mean: float = 0.0,
    rho_std: float = 1.0,
    rho_mag_mean: float = 0.0,
    rho_mag_std: float = 1.0,
    rotate: bool = False,
    charge_points: int | None = None,
):
    def _fn(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collate_candidate_groups(
            groups,
            rho_mean=rho_mean,
            rho_std=rho_std,
            rho_mag_mean=rho_mag_mean,
            rho_mag_std=rho_mag_std,
            rotate=rotate,
            charge_points=charge_points,
        )

    return _fn


__all__ = [
    "validate_candidate_manifest",
    "CandidateGroupMeta",
    "CandidateGroupDataset",
    "CandidateGroupBatchSampler",
    "collate_candidate_groups",
    "make_candidate_collate_fn",
]
