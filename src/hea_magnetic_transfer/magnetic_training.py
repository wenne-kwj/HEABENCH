"""Data-loading, edge-cache and validation helpers for magnetic training."""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .conditions import normalize_condition
from .magnetic_data import (
    Task3SPParentDataset,
    build_sp_parent_edge_cache,
    collate_sp_parents,
)
from .magnetic_objective import MagneticMetricAccumulator, move_magnetic_batch


def _utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _write_log(path: Path, message: str) -> None:
    line = f"[{_utc()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def prepare_edge_cache(
    datasets: dict[str, Task3SPParentDataset],
    cache_root: Path,
    *,
    cutoff: float,
    max_neighbors: int,
    log_path: Path,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=False)
    running = cache_root / "_RUNNING"
    running.write_text(f"started {_utc()}\n", encoding="utf-8")
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    total_edges = 0
    total_bytes = 0
    try:
        for split, dataset in datasets.items():
            split_root = cache_root / split
            split_root.mkdir()
            split_started = time.perf_counter()
            for position in range(len(dataset)):
                sample = dataset[position]
                graph = build_sp_parent_edge_cache(
                    sample, cutoff=float(cutoff), max_neighbors=int(max_neighbors)
                )
                path = split_root / f"{sample['parent_id']}.npz"
                _atomic_npz(path, graph)
                edges = int(graph["structure_edge_index"].shape[1])
                size = int(path.stat().st_size)
                total_edges += edges
                total_bytes += size
                records.append(
                    {
                        "split": split,
                        "position": position,
                        "parent_id": sample["parent_id"],
                        "edges": edges,
                        "bytes": size,
                    }
                )
                completed = position + 1
                if completed == 1 or completed % 50 == 0 or completed == len(dataset):
                    _write_log(log_path, f"edge cache split={split} {completed}/{len(dataset)}")
            _write_log(
                log_path,
                f"edge cache split={split} completed parents={len(dataset)} "
                f"seconds={time.perf_counter() - split_started:.1f}",
            )
        result = {
            "cache_root": str(cache_root),
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "parents": len(records),
            "edges": total_edges,
            "bytes": total_bytes,
            "seconds": time.perf_counter() - started,
            "records": records,
        }
        _atomic_json(cache_root / "manifest.json", result)
        (cache_root / "_SUCCESS").write_text(
            f"edge cache completed {_utc()} parents={len(records)}\n",
            encoding="utf-8",
        )
        return result
    finally:
        if running.exists():
            running.unlink()


def _parent_ids(dataset: Task3SPParentDataset | Subset) -> list[str]:
    if isinstance(dataset, Subset):
        source = dataset.dataset
        if not isinstance(source, Task3SPParentDataset):
            raise TypeError("edge-cache subsets must wrap Task3SPParentDataset")
        return [source.parent_ids[int(index)] for index in dataset.indices]
    return list(dataset.parent_ids)


def validate_edge_cache(
    datasets: dict[str, Task3SPParentDataset | Subset],
    cache_root: Path,
    *,
    cutoff: float,
    max_neighbors: int,
) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    if not (cache_root / "_SUCCESS").is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete edge cache: {cache_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not math.isclose(
        float(manifest.get("cutoff", float("nan"))),
        float(cutoff),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("edge-cache cutoff disagrees with the model")
    if int(manifest.get("max_neighbors", -1)) != int(max_neighbors):
        raise ValueError("edge-cache max_neighbors disagrees with the model")
    required = 0
    for split, dataset in datasets.items():
        parent_ids = _parent_ids(dataset)
        missing = [
            parent_id
            for parent_id in parent_ids
            if not (cache_root / split / f"{parent_id}.npz").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"edge cache is missing {len(missing)} {split} parents"
            )
        required += len(parent_ids)
    return {
        "cache_root": str(cache_root),
        "cutoff": float(cutoff),
        "max_neighbors": int(max_neighbors),
        "parents": required,
        "source_manifest_parents": int(manifest.get("parents", -1)),
        "edges": int(manifest.get("edges", -1)),
        "bytes": int(manifest.get("bytes", -1)),
        "seconds": 0.0,
        "external_reuse": True,
    }


def make_parent_loader(
    dataset: Task3SPParentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    rotate: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": partial(collate_sp_parents, rotate=bool(rotate)),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": bool(persistent_workers),
                "prefetch_factor": int(prefetch_factor),
                "multiprocessing_context": "spawn",
            }
        )
    return DataLoader(**kwargs)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    precision: str,
    delta: float,
    cell_weight: float,
    log_path: Path,
) -> dict[str, float]:
    model.eval()
    accumulator = MagneticMetricAccumulator(delta=delta, cell_weight=cell_weight)
    started = time.perf_counter()
    for position, batch in enumerate(loader, start=1):
        moved = move_magnetic_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            accumulator.update(model(moved), moved)
        if position % 50 == 0:
            _write_log(log_path, f"validation progress parents={position}")
    metrics = accumulator.compute()
    metrics["seconds"] = time.perf_counter() - started
    metrics["seconds_per_parent"] = metrics["seconds"] / metrics["parents"]
    return metrics


def validate_training_args(args: argparse.Namespace) -> None:
    normalize_condition(args.mode)
    if args.budget not in {50, 300, 620}:
        raise ValueError("formal budget must be one of 50, 300, 620")
    if args.parent_batch_size != 1:
        raise ValueError("magnetic training requires parent_batch_size=1")
    positive = {
        "epochs": args.epochs,
        "encoder_site_batch_size": args.encoder_site_batch_size,
        "gradient_accumulation_parents": args.gradient_accumulation_parents,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "grad_clip": args.grad_clip,
        "huber_delta_muB": args.huber_delta_muB,
        "edge_cutoff": args.edge_cutoff,
        "edge_max_neighbors": args.edge_max_neighbors,
        "log_every": args.log_every,
    }
    bad = [name for name, value in positive.items() if float(value) <= 0]
    if bad:
        raise ValueError(f"these settings must be positive: {bad}")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("invalid DataLoader worker settings")
    if args.weight_decay < 0 or args.cell_weight < 0:
        raise ValueError("weight_decay and cell_weight must be non-negative")
    if args.early_stopping_patience < 0 or args.warmup_epochs < 0:
        raise ValueError("patience and warmup_epochs must be non-negative")
    if args.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if not args.precompute_edges:
        raise ValueError("magnetic training requires precompute_edges=true")
    if args.maximum_allocated_gib <= 0:
        raise ValueError("maximum_allocated_gib must be positive")
    if args.maximum_reserved_gib < 0 or args.empty_cache_every_optimizer_steps < 0:
        raise ValueError("memory cache settings must be non-negative")
