"""Formal held-out test evaluation for Task 3 magnetic transfer models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from hea_retrieval.data import random_rotation_matrix

from .conditions import normalize_condition
from .magnetic_data import (
    DEFAULT_TASK3_DATA_ROOT,
    Task3SPParentDataset,
    build_sp_parent_edge_cache,
    collate_sp_parents,
    validate_sp_magnetic_contract,
)
from .magnetic_model import build_magnetic_model
from .magnetic_objective import (
    MagneticMetricAccumulator,
    align_predictions,
    move_magnetic_batch,
)
from .pretrain_data import EXPECTED_ATOMS


from .paths import DEFAULT_EVALUATION_ROOT

# Index is atomic number.  Task 3 currently uses transition-metal HEAs, but the
# complete table avoids silently mislabelling a future valid structure.
CHEMICAL_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
)


def _utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _write_log(path: Path, message: str) -> None:
    line = f"[{_utc()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            default=float,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and undefined floating values to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _new_run_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = root / f"{stamp}_{name}"
    candidate = base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stamp}_{name}_{suffix:02d}"
    candidate.mkdir()
    return candidate


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid magnetic checkpoint: {path}")
    if not isinstance(payload.get("args"), dict):
        raise ValueError(f"checkpoint has no resolved training args: {path}")
    return payload


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["parent_id"]: row for row in rows}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def _safe_correlation(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if prediction.size < 2 or float(prediction.std()) == 0.0 or float(target.std()) == 0.0:
        return None
    return float(np.corrcoef(prediction, target)[0, 1])


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must be non-empty and equally shaped")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("prediction or target contains NaN/Inf")
    error = prediction - target
    absolute = np.abs(error)
    squared = np.square(error)
    target_ss = float(np.square(target - target.mean()).sum())
    r2 = None if target_ss == 0.0 else 1.0 - float(squared.sum()) / target_ss
    return {
        "count": float(prediction.size),
        "mae_muB": float(absolute.mean()),
        "rmse_muB": float(np.sqrt(squared.mean())),
        "median_absolute_error_muB": float(np.median(absolute)),
        "p90_absolute_error_muB": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_muB": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error_muB": float(np.quantile(absolute, 0.99)),
        "max_absolute_error_muB": float(absolute.max()),
        "bias_muB": float(error.mean()),
        "r2": r2,
        "pearson_r": _safe_correlation(prediction, target),
    }


def _bootstrap_parent_ci(
    site_absolute_error: np.ndarray,
    active_mask: np.ndarray,
    cell_absolute_error: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    parent_count = int(site_absolute_error.shape[0])
    if site_absolute_error.shape != active_mask.shape:
        raise ValueError("site errors and active mask disagree")
    if cell_absolute_error.shape != (parent_count,):
        raise ValueError("cell errors and parent count disagree")
    if replicates <= 0:
        return {}
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, parent_count, size=(replicates, parent_count))
    site_by_parent = site_absolute_error.mean(axis=1)
    active_sum = (site_absolute_error * active_mask).sum(axis=1)
    active_count = active_mask.sum(axis=1)
    site_samples = site_by_parent[indices].mean(axis=1)
    active_samples = active_sum[indices].sum(axis=1) / active_count[indices].sum(axis=1)
    cell_samples = cell_absolute_error[indices].mean(axis=1)

    def describe(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "estimate": float(point),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_standard_error": float(values.std(ddof=1)),
            "replicates": float(replicates),
        }

    return {
        "site_mae_muB": describe(site_samples, float(site_absolute_error.mean())),
        "active_site_mae_muB": describe(
            active_samples,
            float(site_absolute_error[active_mask].mean()),
        ),
        "cell_mae_muB": describe(cell_samples, float(cell_absolute_error.mean())),
    }


def _atomic_symbol(atomic_number: int) -> str:
    if atomic_number <= 0 or atomic_number >= len(CHEMICAL_SYMBOLS):
        return f"Z{atomic_number}"
    return CHEMICAL_SYMBOLS[atomic_number]


def _deterministic_rotation(parent_id: str, seed: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return random_rotation_matrix(generator=generator)


@dataclass(frozen=True)
class RotationCollator:
    seed: int

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        rotations = torch.stack(
            [_deterministic_rotation(sample["parent_id"], self.seed) for sample in samples]
        )
        return collate_sp_parents(samples, rotations=rotations)


def _make_loader(
    dataset: Task3SPParentDataset,
    *,
    num_workers: int,
    rotated: bool,
    rotation_seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": RotationCollator(rotation_seed) if rotated else collate_sp_parents,
    }
    if num_workers > 0:
        kwargs.update({"prefetch_factor": 2, "persistent_workers": False})
    return DataLoader(**kwargs)


def _prepare_test_edge_cache(
    source_dataset: Task3SPParentDataset,
    root: Path,
    *,
    cutoff: float,
    max_neighbors: int,
    log_path: Path,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    (root / "_RUNNING").write_text(f"started {_utc()}\n", encoding="utf-8")
    started = time.perf_counter()
    total_edges = 0
    total_bytes = 0
    records: list[dict[str, Any]] = []
    try:
        for position in range(len(source_dataset)):
            sample = source_dataset[position]
            graph = build_sp_parent_edge_cache(
                sample,
                cutoff=float(cutoff),
                max_neighbors=int(max_neighbors),
            )
            path = root / f"{sample['parent_id']}.npz"
            _atomic_npz(path, graph)
            edges = int(graph["structure_edge_index"].shape[1])
            size = int(path.stat().st_size)
            total_edges += edges
            total_bytes += size
            records.append(
                {
                    "position": position,
                    "parent_id": sample["parent_id"],
                    "edges": edges,
                    "bytes": size,
                }
            )
            completed = position + 1
            if completed == 1 or completed % 25 == 0 or completed == len(source_dataset):
                _write_log(log_path, f"edge cache test {completed}/{len(source_dataset)}")
        result = {
            "parents": len(source_dataset),
            "edges": total_edges,
            "bytes": total_bytes,
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "seconds": time.perf_counter() - started,
            "records": records,
        }
        _atomic_json(root / "manifest.json", result)
        (root / "_RUNNING").unlink()
        (root / "_SUCCESS").write_text(
            f"edge cache completed {_utc()} parents={len(source_dataset)}\n",
            encoding="utf-8",
        )
        return result
    except BaseException:
        (root / "_FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str,
    delta: float,
    cell_weight: float,
    log_path: Path,
    label: str,
    collect_metrics: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    model.eval()
    accumulator = MagneticMetricAccumulator(delta=delta, cell_weight=cell_weight)
    parent_ids: list[str] = []
    parent_indices: list[np.ndarray] = []
    atomic_numbers: list[np.ndarray] = []
    site_targets: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    raw_sites: list[np.ndarray] = []
    aligned_sites: list[np.ndarray] = []
    cell_targets: list[np.ndarray] = []
    raw_cells: list[np.ndarray] = []
    aligned_cells: list[np.ndarray] = []
    chosen_signs: list[np.ndarray] = []
    started = time.perf_counter()
    autocast_enabled = precision == "bf16" and device.type == "cuda"
    for position, batch in enumerate(loader, start=1):
        center_z = batch["structure_z"][batch["structure_center_mask"]]
        center_z = center_z.reshape(len(batch["parent_id"]), EXPECTED_ATOMS)
        moved = move_magnetic_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(moved)
        prediction = {
            "site_moments": output["site_moments"].float(),
            "cell_moment_per_atom": output["cell_moment_per_atom"].float(),
        }
        if collect_metrics:
            accumulator.update(prediction, moved)
        aligned_site, aligned_cell, sign = align_predictions(
            prediction["site_moments"],
            prediction["cell_moment_per_atom"],
            moved["site_targets"],
            moved["cell_targets"],
            delta=delta,
            cell_weight=cell_weight,
        )
        parent_ids.extend(batch["parent_id"])
        parent_indices.append(batch["parent_index"].cpu().numpy())
        atomic_numbers.append(center_z.cpu().numpy())
        site_targets.append(batch["site_targets"].cpu().numpy())
        active_masks.append(batch["active_mask"].cpu().numpy())
        raw_sites.append(prediction["site_moments"].cpu().numpy())
        aligned_sites.append(aligned_site.cpu().numpy())
        cell_targets.append(batch["cell_targets"].cpu().numpy())
        raw_cells.append(
            (prediction["cell_moment_per_atom"] * float(EXPECTED_ATOMS)).cpu().numpy()
        )
        aligned_cells.append(aligned_cell.cpu().numpy())
        chosen_signs.append(sign.cpu().numpy())
        if position == 1 or position % 25 == 0 or position == len(loader):
            _write_log(log_path, f"{label} inference {position}/{len(loader)}")
    elapsed = time.perf_counter() - started
    arrays = {
        "parent_ids": np.asarray(parent_ids),
        "parent_indices": np.concatenate(parent_indices).astype(np.int64),
        "atomic_numbers": np.concatenate(atomic_numbers).astype(np.int64),
        "site_targets": np.concatenate(site_targets).astype(np.float32),
        "active_mask": np.concatenate(active_masks).astype(np.bool_),
        "raw_site_predictions": np.concatenate(raw_sites).astype(np.float32),
        "aligned_site_predictions": np.concatenate(aligned_sites).astype(np.float32),
        "cell_targets": np.concatenate(cell_targets).astype(np.float32),
        "raw_cell_predictions": np.concatenate(raw_cells).astype(np.float32),
        "aligned_cell_predictions": np.concatenate(aligned_cells).astype(np.float32),
        "chosen_sign": np.concatenate(chosen_signs).astype(np.float32),
    }
    metrics = accumulator.compute() if collect_metrics else {}
    return arrays, metrics, elapsed


def _group_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    active_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in sorted(np.unique(groups).tolist()):
        selected = groups == group
        active = selected & active_mask
        metrics = regression_metrics(prediction[selected], target[selected])
        row: dict[str, Any] = {
            "atomic_number": int(group),
            "element": _atomic_symbol(int(group)),
            "site_count": int(selected.sum()),
            "active_site_count": int(active.sum()),
            "active_fraction": float(active.sum() / selected.sum()),
            **metrics,
        }
        if active.any():
            active_metrics = regression_metrics(prediction[active], target[active])
            row.update(
                {
                    "active_mae_muB": active_metrics["mae_muB"],
                    "active_rmse_muB": active_metrics["rmse_muB"],
                    "active_bias_muB": active_metrics["bias_muB"],
                    "active_r2": active_metrics["r2"],
                }
            )
        else:
            row.update(
                {
                    "active_mae_muB": None,
                    "active_rmse_muB": None,
                    "active_bias_muB": None,
                    "active_r2": None,
                }
            )
        rows.append(row)
    return rows


def _magnitude_bin_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    absolute_target = np.abs(target)
    bins = (
        ("[0,0.1]", 0.0, 0.1, True),
        ("(0.1,0.5]", 0.1, 0.5, False),
        ("(0.5,1.0]", 0.5, 1.0, False),
        ("(1.0,2.0]", 1.0, 2.0, False),
        ("(2.0,+inf)", 2.0, float("inf"), False),
    )
    rows: list[dict[str, Any]] = []
    for label, lower, upper, include_lower in bins:
        lower_mask = absolute_target >= lower if include_lower else absolute_target > lower
        selected = lower_mask & (absolute_target <= upper)
        if not selected.any():
            continue
        rows.append(
            {
                "target_abs_bin_muB": label,
                **regression_metrics(prediction[selected], target[selected]),
            }
        )
    return rows


def _render_report(summary: dict[str, Any], element_rows: list[dict[str, Any]]) -> str:
    official = summary["official_test_metrics"]
    detailed = summary["detailed_test_metrics"]
    ci = summary["bootstrap_95ci"]
    rotation = summary["rotation_consistency"]
    zero = summary["zero_baseline"]
    checkpoint = summary["checkpoint"]
    model_mode = str(checkpoint.get("mode", "unknown")).upper()

    def optional_metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f}"

    lines = [
        f"# {model_mode} 磁矩预测独立测试集完整评估",
        "",
        f"- 评估状态：完成",
        f"- 数据划分：独立 test split（{int(official['parents'])} 个父结构，{int(official['sites'])} 个原子位点）",
        f"- 模型：{model_mode}，seed={checkpoint['seed']}，训练预算={checkpoint['budget']}，best checkpoint 来自第 {checkpoint['human_epoch']} 个 epoch",
        "- 自旋约定：每个父结构使用一个共享的 ±1 符号，按训练时同一 Huber 目标选择；不存在逐原子独立翻转。",
        "",
        "## 核心结果",
        "",
        f"| 指标 | {model_mode} test | 95% bootstrap CI | 零预测基线 |",
        "|---|---:|---:|---:|",
        f"| 全部位点 MAE | {official['site_mae_muB']:.6f} μB | [{ci['site_mae_muB']['ci95_low']:.6f}, {ci['site_mae_muB']['ci95_high']:.6f}] | {zero['site']['mae_muB']:.6f} μB |",
        f"| 活性位点 MAE | {official['active_site_mae_muB']:.6f} μB | [{ci['active_site_mae_muB']['ci95_low']:.6f}, {ci['active_site_mae_muB']['ci95_high']:.6f}] | {zero['active_site']['mae_muB']:.6f} μB |",
        f"| 晶胞总磁矩 MAE | {official['cell_mae_muB']:.6f} μB | [{ci['cell_mae_muB']['ci95_low']:.6f}, {ci['cell_mae_muB']['ci95_high']:.6f}] | {zero['cell']['mae_muB']:.6f} μB |",
        f"| 全部位点 RMSE | {detailed['site']['rmse_muB']:.6f} μB | — | {zero['site']['rmse_muB']:.6f} μB |",
        f"| 活性位点 RMSE | {detailed['active_site']['rmse_muB']:.6f} μB | — | {zero['active_site']['rmse_muB']:.6f} μB |",
        f"| 晶胞总磁矩 RMSE | {detailed['cell']['rmse_muB']:.6f} μB | — | {zero['cell']['rmse_muB']:.6f} μB |",
        "",
        "## 补充诊断",
        "",
        f"- site R²：{detailed['site']['r2']:.6f}；active-site R²：{detailed['active_site']['r2']:.6f}；cell R²：{detailed['cell']['r2']:.6f}。",
        f"- 采用负方向对齐的父结构比例：{official['minus_orientation_fraction']:.4%}。",
        f"- 随机旋转前后 site 预测平均绝对差：{rotation['site_mean_absolute_difference_muB']:.3e} μB；最大差：{rotation['site_max_absolute_difference_muB']:.3e} μB。",
        f"- test 推理耗时：{summary['runtime']['base_inference_seconds']:.2f} 秒，即 {summary['runtime']['seconds_per_parent']:.4f} 秒/父结构。",
        "",
        "## 按元素结果",
        "",
        "| 元素 | 位点数 | 活性位点数 | MAE (μB) | 活性位点 MAE (μB) | RMSE (μB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in element_rows:
        lines.append(
            f"| {row['element']} | {row['site_count']} | {row['active_site_count']} | "
            f"{row['mae_muB']:.6f} | {optional_metric(row['active_mae_muB'])} | "
            f"{row['rmse_muB']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 结果解释边界",
            "",
            f"本报告只使用一次冻结的 validation 最优 checkpoint 对独立 test split 评估。test 标签没有参与 checkpoint 选择或超参数调整。这里的 {model_mode} 结果是单次 seed={checkpoint['seed']}、budget={checkpoint['budget']} 的独立测试结果；要估计随机种子方差，仍需完成预先设计的其余 seeds 后再汇总。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if args.num_workers < 0 or args.bootstrap_replicates < 0:
        raise ValueError("worker and bootstrap counts must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_dir = _new_run_dir(evaluation_root, args.name)
    log_path = run_dir / "run.log"
    running_path = run_dir / "_RUNNING"
    running_path.write_text(f"started {_utc()}\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        _write_log(log_path, f"formal magnetic held-out test evaluation started run_dir={run_dir}")
        contract = validate_sp_magnetic_contract(data_root)
        payload = _load_checkpoint(checkpoint_path)
        training_args = dict(payload["args"])
        mode = normalize_condition(str(training_args.get("mode", "")))
        training_args["mode"] = mode
        if Path(training_args["data_root"]).resolve() != data_root:
            raise ValueError("checkpoint and requested evaluation data roots disagree")
        if int(training_args.get("best_epoch", payload.get("best_epoch", -1))) not in {
            -1,
            int(payload.get("epoch", -1)),
        }:
            raise ValueError("checkpoint epoch is not the recorded best epoch")

        model = build_magnetic_model(
            output_dim=int(training_args["output_dim"]),
            structure_kwargs=dict(training_args["structure_kwargs"]),
            head_hidden_dim=int(training_args["head_hidden_dim"]),
            head_dropout=float(training_args["head_dropout"]),
            encoder_site_batch_size=int(
                args.encoder_site_batch_size
                if args.encoder_site_batch_size is not None
                else training_args["encoder_site_batch_size"]
            ),
            encoder_gradient_checkpointing=False,
        )
        incompatible = model.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict model-state load unexpectedly returned incompatibilities")
        model.to(device)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        source_test = Task3SPParentDataset(data_root, split="test")
        edge_cache_root = run_dir / "edge_cache" / "test"
        edge_result = _prepare_test_edge_cache(
            source_test,
            edge_cache_root,
            cutoff=float(training_args["edge_cutoff"]),
            max_neighbors=int(training_args["edge_max_neighbors"]),
            log_path=log_path,
        )
        test_dataset = Task3SPParentDataset(
            data_root,
            split="test",
            edge_cache_root=edge_cache_root,
            require_edge_cache=True,
            edge_cutoff=float(training_args["edge_cutoff"]),
            edge_max_neighbors=int(training_args["edge_max_neighbors"]),
        )
        if len(test_dataset) != 133:
            raise ValueError(f"formal test split must have 133 parents, got {len(test_dataset)}")

        loader = _make_loader(
            test_dataset,
            num_workers=int(args.num_workers),
            rotated=False,
            rotation_seed=int(args.rotation_seed),
        )
        arrays, official, base_seconds = _predict(
            model,
            loader,
            device,
            precision=args.precision,
            delta=float(training_args["huber_delta_muB"]),
            cell_weight=float(training_args["cell_weight"]),
            log_path=log_path,
            label="base test",
            collect_metrics=True,
        )
        if arrays["site_targets"].shape != (133, EXPECTED_ATOMS):
            raise RuntimeError("formal test predictions have an unexpected shape")
        if len(set(arrays["parent_ids"].tolist())) != 133:
            raise RuntimeError("formal test predictions contain duplicate parent IDs")

        rotation: dict[str, Any]
        rotation_seconds = 0.0
        if args.rotation_check:
            rotated_loader = _make_loader(
                test_dataset,
                num_workers=int(args.num_workers),
                rotated=True,
                rotation_seed=int(args.rotation_seed),
            )
            rotated, _, rotation_seconds = _predict(
                model,
                rotated_loader,
                device,
                precision=args.precision,
                delta=float(training_args["huber_delta_muB"]),
                cell_weight=float(training_args["cell_weight"]),
                log_path=log_path,
                label="rotated test",
                collect_metrics=False,
            )
            if not np.array_equal(arrays["parent_ids"], rotated["parent_ids"]):
                raise RuntimeError("base and rotated parent orders disagree")
            site_difference = (
                rotated["raw_site_predictions"] - arrays["raw_site_predictions"]
            )
            cell_difference = (
                rotated["raw_cell_predictions"] - arrays["raw_cell_predictions"]
            )
            rotation = {
                "enabled": True,
                "seed": int(args.rotation_seed),
                "site_mean_absolute_difference_muB": float(np.abs(site_difference).mean()),
                "site_rmse_difference_muB": float(np.sqrt(np.square(site_difference).mean())),
                "site_max_absolute_difference_muB": float(np.abs(site_difference).max()),
                "site_p95_absolute_difference_muB": float(
                    np.quantile(np.abs(site_difference), 0.95)
                ),
                "site_p99_absolute_difference_muB": float(
                    np.quantile(np.abs(site_difference), 0.99)
                ),
                "site_count_absolute_difference_gt_0p01_muB": int(
                    (np.abs(site_difference) > 0.01).sum()
                ),
                "site_count_absolute_difference_gt_0p05_muB": int(
                    (np.abs(site_difference) > 0.05).sum()
                ),
                "site_count_absolute_difference_gt_0p1_muB": int(
                    (np.abs(site_difference) > 0.1).sum()
                ),
                "cell_mean_absolute_difference_muB": float(np.abs(cell_difference).mean()),
                "cell_rmse_difference_muB": float(np.sqrt(np.square(cell_difference).mean())),
                "cell_max_absolute_difference_muB": float(np.abs(cell_difference).max()),
            }
        else:
            rotation = {"enabled": False}

        aligned_site = arrays["aligned_site_predictions"].astype(np.float64)
        target_site = arrays["site_targets"].astype(np.float64)
        active = arrays["active_mask"]
        aligned_cell = arrays["aligned_cell_predictions"].astype(np.float64)
        target_cell = arrays["cell_targets"].astype(np.float64)
        site_error = aligned_site - target_site
        cell_error = aligned_cell - target_cell
        detailed = {
            "site": regression_metrics(aligned_site, target_site),
            "active_site": regression_metrics(aligned_site[active], target_site[active]),
            "inactive_site": regression_metrics(aligned_site[~active], target_site[~active]),
            "cell": regression_metrics(aligned_cell, target_cell),
        }
        if not math.isclose(
            detailed["site"]["mae_muB"], official["site_mae_muB"], rel_tol=0.0, abs_tol=2e-6
        ):
            raise RuntimeError("saved predictions disagree with official site MAE")
        if not math.isclose(
            detailed["active_site"]["mae_muB"],
            official["active_site_mae_muB"],
            rel_tol=0.0,
            abs_tol=2e-6,
        ):
            raise RuntimeError("saved predictions disagree with official active-site MAE")

        bootstrap = _bootstrap_parent_ci(
            np.abs(site_error),
            active,
            np.abs(cell_error),
            replicates=int(args.bootstrap_replicates),
            seed=int(args.bootstrap_seed),
        )
        zero = {
            "site": regression_metrics(np.zeros_like(target_site), target_site),
            "active_site": regression_metrics(
                np.zeros_like(target_site[active]), target_site[active]
            ),
            "inactive_site": regression_metrics(
                np.zeros_like(target_site[~active]), target_site[~active]
            ),
            "cell": regression_metrics(np.zeros_like(target_cell), target_cell),
        }
        flat_prediction = aligned_site.reshape(-1)
        flat_target = target_site.reshape(-1)
        flat_active = active.reshape(-1)
        flat_z = arrays["atomic_numbers"].reshape(-1)
        element_rows = _group_metrics(
            flat_prediction,
            flat_target,
            flat_z,
            flat_active,
        )
        magnitude_rows = _magnitude_bin_metrics(flat_prediction, flat_target)

        manifest = _read_manifest(data_root / "sp_manifest.csv")
        parent_rows: list[dict[str, Any]] = []
        site_rows: list[dict[str, Any]] = []
        for parent_position, parent_id_value in enumerate(arrays["parent_ids"]):
            parent_id = str(parent_id_value)
            metadata = manifest[parent_id]
            parent_active = active[parent_position]
            parent_site_abs = np.abs(site_error[parent_position])
            parent_rows.append(
                {
                    "parent_id": parent_id,
                    "parent_index": int(arrays["parent_indices"][parent_position]),
                    "sample_name": metadata["sample_name"],
                    "composition": metadata["composition"],
                    "chosen_sign": float(arrays["chosen_sign"][parent_position]),
                    "site_mae_muB": float(parent_site_abs.mean()),
                    "site_rmse_muB": float(np.sqrt(np.square(site_error[parent_position]).mean())),
                    "active_site_count": int(parent_active.sum()),
                    "active_site_mae_muB": (
                        float(parent_site_abs[parent_active].mean())
                        if parent_active.any()
                        else None
                    ),
                    "cell_target_muB": float(target_cell[parent_position]),
                    "cell_prediction_raw_muB": float(arrays["raw_cell_predictions"][parent_position]),
                    "cell_prediction_aligned_muB": float(aligned_cell[parent_position]),
                    "cell_absolute_error_muB": float(abs(cell_error[parent_position])),
                }
            )
            for site in range(EXPECTED_ATOMS):
                atomic_number = int(arrays["atomic_numbers"][parent_position, site])
                site_rows.append(
                    {
                        "parent_id": parent_id,
                        "parent_index": int(arrays["parent_indices"][parent_position]),
                        "sample_name": metadata["sample_name"],
                        "composition": metadata["composition"],
                        "site_index": site,
                        "atomic_number": atomic_number,
                        "element": _atomic_symbol(atomic_number),
                        "target_muB": float(target_site[parent_position, site]),
                        "prediction_raw_muB": float(arrays["raw_site_predictions"][parent_position, site]),
                        "prediction_aligned_muB": float(aligned_site[parent_position, site]),
                        "error_muB": float(site_error[parent_position, site]),
                        "absolute_error_muB": float(parent_site_abs[site]),
                        "active": bool(active[parent_position, site]),
                        "chosen_sign": float(arrays["chosen_sign"][parent_position]),
                    }
                )

        parent_rows.sort(key=lambda row: row["site_mae_muB"], reverse=True)
        worst = {
            "by_site_mae": parent_rows[:10],
            "by_cell_absolute_error": sorted(
                parent_rows,
                key=lambda row: row["cell_absolute_error_muB"],
                reverse=True,
            )[:10],
        }
        peak_allocated = (
            float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            float(torch.cuda.max_memory_reserved(device) / (1024**3))
            if device.type == "cuda"
            else 0.0
        )
        summary: dict[str, Any] = {
            "status": "complete",
            "completed_at": _utc(),
            "evaluation_protocol": {
                "split": "test",
                "checkpoint_selection_split": "validation",
                "test_used_for_selection": False,
                "global_spin_flip": "one_shared_sign_per_parent",
                "active_site_threshold_muB": 0.1,
                "precision": args.precision,
                "rotation_check": bool(args.rotation_check),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "mode": training_args["mode"],
                "budget": int(training_args["budget"]),
                "seed": int(training_args["seed"]),
                "epoch": int(payload["epoch"]),
                "human_epoch": int(payload["epoch"]) + 1,
                "validation_best_site_mae_muB": float(payload["best_metric"]),
                "strict_state_load": True,
                "parameters": parameter_count,
            },
            "data_contract": contract,
            "official_test_metrics": official,
            "detailed_test_metrics": detailed,
            "bootstrap_95ci": bootstrap,
            "zero_baseline": zero,
            "rotation_consistency": rotation,
            "per_element_metrics": element_rows,
            "target_magnitude_bin_metrics": magnitude_rows,
            "worst_parents": worst,
            "runtime": {
                "edge_cache_seconds": float(edge_result["seconds"]),
                "base_inference_seconds": base_seconds,
                "rotation_inference_seconds": rotation_seconds,
                "seconds_per_parent": base_seconds / len(test_dataset),
                "peak_memory_allocated_gib": peak_allocated,
                "peak_memory_reserved_gib": peak_reserved,
            },
            "environment": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(device),
            },
        }

        _atomic_npz(run_dir / "predictions.npz", arrays)
        parent_fields = list(parent_rows[0])
        site_fields = list(site_rows[0])
        element_fields = list(element_rows[0])
        magnitude_fields = list(magnitude_rows[0])
        _write_csv(run_dir / "parent_predictions.csv", parent_rows, parent_fields)
        _write_csv(run_dir / "site_predictions.csv", site_rows, site_fields)
        _write_csv(run_dir / "per_element_metrics.csv", element_rows, element_fields)
        _write_csv(run_dir / "magnitude_bin_metrics.csv", magnitude_rows, magnitude_fields)
        _atomic_json(run_dir / "test_metrics.json", summary)
        report = _render_report(summary, element_rows)
        (run_dir / "report_zh.md").write_text(report, encoding="utf-8")

        artifact_names = (
            "predictions.npz",
            "parent_predictions.csv",
            "site_predictions.csv",
            "per_element_metrics.csv",
            "magnitude_bin_metrics.csv",
            "test_metrics.json",
            "report_zh.md",
        )
        elapsed = time.perf_counter() - started
        _atomic_json(
            run_dir / "summary.json",
            {
                "status": "complete",
                "completed_at": _utc(),
                "run_dir": str(run_dir),
                "parents": int(official["parents"]),
                "sites": int(official["sites"]),
                "site_mae_muB": float(official["site_mae_muB"]),
                "active_site_mae_muB": float(official["active_site_mae_muB"]),
                "cell_mae_muB": float(official["cell_mae_muB"]),
                "elapsed_seconds": elapsed,
                "artifacts": list(artifact_names),
            },
        )
        running_path.unlink()
        (run_dir / "_SUCCESS").write_text(
            f"formal test evaluation completed {_utc()} parents={len(test_dataset)}\n",
            encoding="utf-8",
        )
        _write_log(
            log_path,
            f"evaluation complete site_mae={official['site_mae_muB']:.6f} "
            f"active_site_mae={official['active_site_mae_muB']:.6f} "
            f"cell_mae={official['cell_mae_muB']:.6f} elapsed_seconds={elapsed:.1f}",
        )
        return run_dir
    except BaseException as exc:
        failure = {
            "status": "failed",
            "failed_at": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(run_dir / "_FAILED", failure)
        _write_log(log_path, f"evaluation failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen magnetic-transfer checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default=str(DEFAULT_TASK3_DATA_ROOT))
    parser.add_argument("--output-root", dest="evaluation_root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(
        name="heldout_test", precision="bf16", num_workers=2,
        encoder_site_batch_size=None, rotation_check=True,
        rotation_seed=20260920, bootstrap_replicates=5000,
        bootstrap_seed=20260920,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_dir = evaluate(args)
    print(run_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "RotationCollator",
    "build_arg_parser",
    "evaluate",
    "regression_metrics",
]
