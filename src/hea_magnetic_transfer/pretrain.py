"""Train Task 3 EquiformerV2 + PointVector from scratch on NSP sites.

Only symmetric in-batch InfoNCE is implemented here.  Candidate loss and the
Task 2 pair manifests are intentionally outside this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from hea_retrieval.losses import symmetric_infonce
from hea_retrieval.models.factory import build_model

from .pretrain_data import (
    DEFAULT_TASK3_DATA_ROOT,
    ScheduledSiteBatchSampler,
    Task3NSPSiteDataset,
    load_density_stats,
    make_task3_collate_fn,
    validate_pretraining_contract,
    validation_indices,
)


from .paths import CONFIG_PATH, DEFAULT_PRETRAIN_RUN_ROOT as DEFAULT_RUN_ROOT


def _utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_log(path: Path, message: str) -> None:
    line = f"[{_utc()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _read_metrics(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or "model" not in value:
        raise ValueError(f"invalid checkpoint: {path}")
    return value


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    structure_rows: list[torch.Tensor] = []
    charge_rows: list[torch.Tensor] = []
    ids: list[str] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        structure, charge = model(moved)
        if not torch.isfinite(structure).all() or not torch.isfinite(charge).all():
            raise FloatingPointError("validation embedding contains NaN/Inf")
        structure_rows.append(structure.cpu())
        charge_rows.append(charge.cpu())
        ids.extend(str(value) for value in batch["site_id"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("validation site IDs are empty or non-unique")
    structure = torch.cat(structure_rows)
    charge = torch.cat(charge_rows)
    n = len(ids)
    forward_hits = {1: 0, 5: 0, 10: 0}
    reverse_hits = {1: 0, 5: 0, 10: 0}
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        targets = torch.arange(start, stop).unsqueeze(1)
        forward = (structure[start:stop] @ charge.T).topk(min(10, n), dim=1).indices
        reverse = (charge[start:stop] @ structure.T).topk(min(10, n), dim=1).indices
        for k in forward_hits:
            width = min(k, n)
            forward_hits[k] += int((forward[:, :width] == targets).any(dim=1).sum())
            reverse_hits[k] += int((reverse[:, :width] == targets).any(dim=1).sum())
    return {
        **{f"s2c_recall@{k}": forward_hits[k] / n for k in forward_hits},
        **{f"c2s_recall@{k}": reverse_hits[k] / n for k in reverse_hits},
        "validation_sites": float(n),
    }


def _score(metrics: dict[str, Any]) -> float:
    return 0.5 * (float(metrics["s2c_recall@1"]) + float(metrics["c2s_recall@1"]))


def _normalise_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("model kwargs must be a JSON object")
    return dict(value)


def _install_signal_handlers() -> None:
    def _raise(signum, _frame):
        raise RuntimeError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)


def train(args: argparse.Namespace) -> Path:
    if args.objective != "inbatch":
        raise ValueError("Task 3 pre-training supports only inbatch InfoNCE")
    if args.structure.lower() not in {"equiformer", "equiformerv2", "e"}:
        raise ValueError("Task 3 formal pre-training requires EquiformerV2")
    if args.charge.lower() not in {"pointvector", "p"}:
        raise ValueError("Task 3 formal pre-training requires PointVector")
    if args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("invalid batch size")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.grad_clip <= 0 or args.temperature <= 0:
        raise ValueError("gradient clipping and temperature must be positive")
    if args.num_workers < 0 or args.cache_size < 0:
        raise ValueError("num_workers and cache_size must be non-negative")
    _install_signal_handlers()
    _seed_everything(int(args.seed))
    data_root = Path(args.data_root).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _new_run_dir(Path(args.run_root), str(args.name))
    log_path = run_dir / "run.log"
    running_marker = run_dir / "_RUNNING"
    running_marker.write_text(f"started {_utc()}\n", encoding="utf-8")
    _write_log(log_path, f"starting Task 3 NSP pre-training device={device} objective=inbatch")

    try:
        contract_audit = validate_pretraining_contract(data_root)
        _atomic_json(run_dir / "data_contract_audit.json", contract_audit)
        stats = load_density_stats(data_root)
        dataset = Task3NSPSiteDataset(
            data_root,
            cache_size=args.cache_size,
            use_center_indicator=args.use_center_indicator,
        )
        train_sampler = ScheduledSiteBatchSampler(
            dataset,
            data_root / "nsp_train_site_schedule.npz",
            args.batch_size,
            seed=args.seed,
            shuffle=True,
        )
        if args.epochs <= 0 or args.epochs > train_sampler.epochs:
            raise ValueError(
                f"epochs must be within [1,{train_sampler.epochs}], got {args.epochs}"
            )
        val_indices = validation_indices(
            dataset,
            max_sites=args.validation_max_sites if args.validation_max_sites > 0 else None,
        )
        train_collate = make_task3_collate_fn(
            rho_mean=float(stats["rho_mean"]),
            rho_std=float(stats["rho_std"]),
            rotate=bool(args.rotate),
            charge_points=args.charge_points,
        )
        validation_collate = make_task3_collate_fn(
            rho_mean=float(stats["rho_mean"]),
            rho_std=float(stats["rho_std"]),
            rotate=False,
            charge_points=args.charge_points,
        )
        common_loader = {
            "num_workers": int(args.num_workers),
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collate,
            **common_loader,
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=validation_collate,
            **common_loader,
        )
        structure_kwargs = _normalise_kwargs(args.structure_kwargs)
        charge_kwargs = _normalise_kwargs(args.charge_kwargs)
        model = build_model(
            args.structure,
            args.charge,
            output_dim=args.output_dim,
            structure_kwargs=structure_kwargs,
            charge_kwargs=charge_kwargs,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1)
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        tower_counts = {
            "structure": sum(parameter.numel() for parameter in model.structure_encoder.parameters()),
            "charge": sum(parameter.numel() for parameter in model.charge_encoder.parameters()),
        }

        resume_checkpoint = (
            Path(args.resume_checkpoint).expanduser().resolve()
            if args.resume_checkpoint
            else None
        )
        resume_metrics_path = (
            Path(args.resume_metrics).expanduser().resolve()
            if args.resume_metrics
            else (resume_checkpoint.parent / "metrics.jsonl" if resume_checkpoint else None)
        )
        history = _read_metrics(resume_metrics_path)
        start_epoch = 0
        global_step = 0
        best_score = -float("inf")
        if resume_checkpoint is not None:
            payload = _load_checkpoint(resume_checkpoint)
            model.load_state_dict(payload["model"], strict=True)
            if payload.get("optimizer") is not None:
                optimizer.load_state_dict(payload["optimizer"])
                _move_optimizer_state(optimizer, device)
            if payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload.get("epoch", -1)) + 1
            global_step = int(payload.get("global_step", 0))
            best_score = float(payload.get("best_score", -float("inf")))
            if start_epoch >= args.epochs:
                raise ValueError("resume checkpoint already reaches requested epochs")
            _write_log(log_path, f"resuming {resume_checkpoint} at epoch={start_epoch}")

        resolved_config = vars(args).copy()
        resolved_config.update(
            {
                "run_dir": str(run_dir),
                "data_root": str(data_root),
                "device": str(device),
                "objective": "inbatch",
                "from_scratch": resume_checkpoint is None,
                "pretrained_weights": None,
                "density_stats": stats,
                "parameter_count": parameter_count,
                "tower_parameter_counts": tower_counts,
                "train_sites_per_epoch": train_sampler.samples_per_epoch,
                "train_batches_per_epoch": len(train_sampler),
                "validation_sites": len(val_indices),
                "structure_kwargs": structure_kwargs,
                "charge_kwargs": charge_kwargs,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
        _atomic_json(run_dir / "config.json", resolved_config)
        _write_log(
            log_path,
            f"data train_sites={train_sampler.samples_per_epoch} "
            f"train_batches={len(train_sampler)} validation_sites={len(val_indices)}",
        )
        _write_log(
            log_path,
            f"model parameters={parameter_count} structure={tower_counts['structure']} "
            f"charge={tower_counts['charge']} from_scratch={resume_checkpoint is None}",
        )

        metrics_path = run_dir / "metrics.jsonl"
        last_epoch = start_epoch - 1

        def checkpoint_payload(epoch: int) -> dict[str, Any]:
            return {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_score": float(best_score),
                "args": resolved_config,
            }

        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for record in history:
                metrics_handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
            metrics_handle.flush()
            stop_requested = False
            for epoch in range(start_epoch, args.epochs):
                last_epoch = epoch
                train_sampler.set_epoch(epoch)
                model.train()
                losses: list[float] = []
                epoch_start = time.perf_counter()
                for step_in_epoch, batch in enumerate(train_loader, start=1):
                    moved = _move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    structure, charge = model(moved)
                    loss = symmetric_infonce(
                        structure, charge, temperature=float(args.temperature)
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss epoch={epoch} global_step={global_step}"
                        )
                    loss.backward()
                    grad_norm = clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        raise FloatingPointError(
                            f"non-finite gradient norm epoch={epoch} global_step={global_step}"
                        )
                    optimizer.step()
                    global_step += 1
                    loss_value = float(loss.detach().cpu())
                    losses.append(loss_value)
                    if args.log_every > 0 and global_step % args.log_every == 0:
                        _write_log(
                            log_path,
                            f"epoch={epoch} step={global_step} step_in_epoch={step_in_epoch} "
                            f"loss={loss_value:.6f} grad_norm={float(grad_norm):.6f}",
                        )
                        _atomic_json(
                            run_dir / "progress.json",
                            {
                                "status": "running",
                                "updated_at": _utc(),
                                "epoch": epoch,
                                "global_step": global_step,
                                "step_in_epoch": step_in_epoch,
                                "train_batches_per_epoch": len(train_sampler),
                                "recent_loss": loss_value,
                                "gradient_norm": float(grad_norm),
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                        )
                    if args.max_steps > 0 and global_step >= args.max_steps:
                        stop_requested = True
                        break
                if not losses:
                    raise RuntimeError(f"epoch {epoch} produced no optimizer steps")
                scheduler.step()
                elapsed = time.perf_counter() - epoch_start
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(np.mean(losses)),
                    "train_loss_last": losses[-1],
                    "lr": optimizer.param_groups[0]["lr"],
                    "steps_in_epoch": len(losses),
                    "seconds": elapsed,
                }
                should_eval = (
                    (epoch + 1) % max(int(args.eval_every), 1) == 0 or stop_requested
                )
                if should_eval:
                    evaluation = evaluate_retrieval(model, val_loader, device)
                    record.update(evaluation)
                    score = _score(evaluation)
                    if score > best_score:
                        best_score = score
                        _atomic_torch_save(checkpoint_payload(epoch), run_dir / "best.pt")
                metrics_handle.write(
                    json.dumps(record, ensure_ascii=False, default=float) + "\n"
                )
                metrics_handle.flush()
                if (epoch + 1) % max(int(args.save_every), 1) == 0 or stop_requested:
                    _atomic_torch_save(checkpoint_payload(epoch), run_dir / "last.pt")
                _atomic_json(
                    run_dir / "progress.json",
                    {
                        "status": "running",
                        "updated_at": _utc(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": record["train_loss"],
                        "best_score": best_score,
                        "lr": record["lr"],
                        "last_validation": {
                            key: value
                            for key, value in record.items()
                            if "recall" in key or key == "validation_sites"
                        },
                    },
                )
                _write_log(
                    log_path,
                    f"epoch={epoch} completed steps={len(losses)} mean_loss={record['train_loss']:.6f} "
                    f"best_score={best_score:.6f} seconds={elapsed:.1f}",
                )
                if stop_requested:
                    break
        if last_epoch < 0:
            raise RuntimeError("no training epoch was executed")
        # Always make last.pt represent the terminal state, including when
        # save_every does not divide the requested final epoch.
        _atomic_torch_save(checkpoint_payload(last_epoch), run_dir / "last.pt")
        _atomic_json(
            run_dir / "progress.json",
            {
                "status": "complete",
                "updated_at": _utc(),
                "epoch": last_epoch,
                "global_step": global_step,
                "best_score": best_score,
            },
        )
        (run_dir / "_SUCCESS").write_text(
            f"training completed {_utc()} epoch={last_epoch} step={global_step}\n",
            encoding="utf-8",
        )
        _write_log(log_path, f"training completed epoch={last_epoch} step={global_step}")
        return run_dir
    except BaseException as exc:
        (run_dir / "_FAILED").write_text(
            f"{_utc()} {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        _atomic_json(
            run_dir / "progress.json",
            {
                "status": "failed",
                "updated_at": _utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        _write_log(log_path, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        if running_marker.exists():
            running_marker.unlink()
        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["pretraining"]
    parser = argparse.ArgumentParser(description="Pretrain the fixed structure-DFT pair model.")
    parser.add_argument("--data-root", default=str(DEFAULT_TASK3_DATA_ROOT))
    parser.add_argument("--output-dir", dest="run_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", dest="resume_checkpoint", default=None)
    parser.set_defaults(
        config=str(CONFIG_PATH), run_root=str(DEFAULT_RUN_ROOT),
        name="structure_dft_pretraining", structure="equiformer",
        charge="pointvector", objective="inbatch",
        batch_size=int(protocol["batch_size"]), eval_batch_size=32,
        epochs=int(protocol["epochs"]), max_steps=-1, eval_every=1, save_every=1,
        lr=float(protocol["learning_rate"]), weight_decay=1e-4, grad_clip=1.0,
        temperature=float(protocol["temperature"]),
        output_dim=int(protocol["embedding_dim"]), cache_size=2, num_workers=0,
        charge_points=None, validation_max_sites=0, rotate=True,
        use_center_indicator=True, log_every=20, seed=int(protocol["seed"]),
        resume_metrics=None, structure_kwargs={}, charge_kwargs={},
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    args.structure_kwargs = _normalise_kwargs(args.structure_kwargs)
    args.charge_kwargs = _normalise_kwargs(args.charge_kwargs)
    return args


if __name__ == "__main__":
    train(parse_args())
