"""Formal S0, S-DFT, and S-Gauss magnetic-transfer training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from .magnetic_data import (
    DEFAULT_TASK3_DATA_ROOT,
    Task3SPParentDataset,
    validate_sp_magnetic_contract,
)
from .conditions import normalize_condition, uses_pretrained_encoder
from .magnetic_model import (
    DEFAULT_PRETRAINED_CHECKPOINT,
    initialize_magnetic_condition,
)
from .magnetic_objective import (
    global_spin_flip_huber,
    move_magnetic_batch,
)
from .magnetic_training import (
    evaluate_model as _evaluate,
    make_parent_loader as _make_loader,
    prepare_edge_cache as _prepare_edge_cache,
    validate_edge_cache as _validate_existing_edge_cache,
    validate_training_args as _validate_args,
)


from .paths import CONFIG_PATH, DEFAULT_DOWNSTREAM_RUN_ROOT as DEFAULT_RUN_ROOT


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


def _install_signal_handlers() -> None:
    def _raise(signum, _frame):
        raise RuntimeError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)


def _normalise_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("structure_kwargs must be a JSON object")
    return dict(value)


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
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid magnetic checkpoint: {path}")
    return payload


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _lr_scale(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if total_epochs <= 0 or epoch < 0 or epoch >= total_epochs:
        raise ValueError("epoch must be within the configured training range")
    warmup = min(max(int(warmup_epochs), 0), int(total_epochs))
    if warmup and epoch < warmup:
        return float(epoch + 1) / float(warmup)
    decay_epochs = total_epochs - warmup
    if decay_epochs <= 1:
        return 1.0
    progress = float(epoch - warmup) / float(decay_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> tuple[float, list[float]]:
    scale = _lr_scale(epoch, total_epochs, warmup_epochs)
    values: list[float] = []
    for group in optimizer.param_groups:
        initial = float(group["initial_lr"])
        group["lr"] = initial * scale
        values.append(float(group["lr"]))
    return scale, values


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    )


def train(args: argparse.Namespace) -> Path:
    args.mode = normalize_condition(args.mode)
    if args.mode == "s_gauss" and int(args.budget) != 50:
        raise ValueError("the published S-Gauss control is defined only at budget 50")
    _validate_args(args)
    _install_signal_handlers()
    _seed_everything(int(args.seed))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal magnetic training requires CUDA")
    torch.cuda.set_device(device)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not report BF16 support")

    data_root = Path(args.data_root).expanduser().resolve()
    pretrained_checkpoint = Path(args.pretrained_checkpoint).expanduser().resolve()
    if uses_pretrained_encoder(args.mode) and not pretrained_checkpoint.is_file():
        raise FileNotFoundError(pretrained_checkpoint)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty run directory: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _new_run_dir(Path(args.run_root), str(args.name))

    log_path = run_dir / "run.log"
    running_marker = run_dir / "_RUNNING"
    running_marker.write_text(f"started {_utc()}\n", encoding="utf-8")
    _write_log(
        log_path,
        f"starting magnetic transfer mode={args.mode} budget={args.budget} "
        f"seed={args.seed} device={device} precision={args.precision}",
    )
    model: torch.nn.Module | None = None
    train_loader: DataLoader | None = None
    validation_loader: DataLoader | None = None
    started = time.perf_counter()
    success = False
    try:
        contract = validate_sp_magnetic_contract(data_root)
        _atomic_json(run_dir / "data_contract_audit.json", contract)
        base_train = Task3SPParentDataset(
            data_root,
            split="train",
            budget=int(args.budget),
        )
        base_validation = Task3SPParentDataset(
            data_root,
            split="validation",
        )
        if args.validation_max_parents > 0:
            base_validation = Subset(
                base_validation,
                range(min(args.validation_max_parents, len(base_validation))),
            )

        cache_datasets = {"train": base_train, "validation": base_validation}
        if args.edge_cache_root:
            edge_cache_root = Path(args.edge_cache_root).expanduser().resolve()
            edge_result = _validate_existing_edge_cache(
                cache_datasets,
                edge_cache_root,
                cutoff=float(args.edge_cutoff),
                max_neighbors=int(args.edge_max_neighbors),
            )
            _write_log(
                log_path,
                f"reusing external edge cache={edge_cache_root} "
                f"required_parents={edge_result['parents']}",
            )
        else:
            edge_cache_root = run_dir / "edge_cache"
            edge_result = _prepare_edge_cache(
                cache_datasets,
                edge_cache_root,
                cutoff=float(args.edge_cutoff),
                max_neighbors=int(args.edge_max_neighbors),
                log_path=log_path,
            )
            _write_log(
                log_path,
                f"edge precompute completed parents={edge_result['parents']} "
                f"seconds={edge_result['seconds']:.1f}",
            )
        train_dataset = Task3SPParentDataset(
            data_root,
            split="train",
            budget=int(args.budget),
            edge_cache_root=edge_cache_root / "train",
            require_edge_cache=True,
            edge_cutoff=float(args.edge_cutoff),
            edge_max_neighbors=int(args.edge_max_neighbors),
        )
        validation_dataset: Task3SPParentDataset | Subset
        full_validation_dataset = Task3SPParentDataset(
            data_root,
            split="validation",
            edge_cache_root=edge_cache_root / "validation",
            require_edge_cache=True,
            edge_cutoff=float(args.edge_cutoff),
            edge_max_neighbors=int(args.edge_max_neighbors),
        )
        if args.validation_max_parents > 0:
            validation_dataset = Subset(
                full_validation_dataset,
                range(min(args.validation_max_parents, len(full_validation_dataset))),
            )
        else:
            validation_dataset = full_validation_dataset

        train_generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
        train_loader = _make_loader(
            train_dataset,
            batch_size=int(args.parent_batch_size),
            shuffle=True,
            rotate=bool(args.rotate_train),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
            generator=train_generator,
            device=device,
        )
        validation_loader = _make_loader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            rotate=False,
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
            generator=None,
            device=device,
        )

        model, initialization = initialize_magnetic_condition(
            args.mode,
            seed=int(args.seed),
            checkpoint=pretrained_checkpoint,
            output_dim=int(args.output_dim),
            structure_kwargs=_normalise_mapping(args.structure_kwargs),
            head_hidden_dim=int(args.head_hidden_dim),
            head_dropout=float(args.head_dropout),
            encoder_site_batch_size=int(args.encoder_site_batch_size),
            encoder_gradient_checkpointing=bool(
                args.encoder_gradient_checkpointing
            ),
        )
        model.to(device)
        encoder_parameters = list(model.structure_encoder.parameters())
        head_parameters = list(model.site_head.parameters()) + list(
            model.cell_head.parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_parameters,
                    "lr": float(args.encoder_lr),
                    "initial_lr": float(args.encoder_lr),
                    "name": "structure_encoder",
                },
                {
                    "params": head_parameters,
                    "lr": float(args.head_lr),
                    "initial_lr": float(args.head_lr),
                    "name": "magnetic_heads",
                },
            ],
            weight_decay=float(args.weight_decay),
        )

        resume_checkpoint = (
            Path(args.resume_checkpoint).expanduser().resolve()
            if args.resume_checkpoint
            else None
        )
        resume_metrics = (
            Path(args.resume_metrics).expanduser().resolve()
            if args.resume_metrics
            else (resume_checkpoint.parent / "metrics.jsonl" if resume_checkpoint else None)
        )
        history = _read_metrics(resume_metrics)
        start_epoch = 0
        global_parent_step = 0
        optimizer_step = 0
        best_metric = float("inf")
        best_epoch = -1
        bad_validation_count = 0
        if resume_checkpoint is not None:
            payload = _load_checkpoint(resume_checkpoint)
            model.load_state_dict(payload["model"], strict=True)
            if payload.get("optimizer") is not None:
                optimizer.load_state_dict(payload["optimizer"])
                for group, initial in zip(
                    optimizer.param_groups,
                    (float(args.encoder_lr), float(args.head_lr)),
                ):
                    group["initial_lr"] = initial
                _move_optimizer_state(optimizer, device)
            start_epoch = int(payload.get("epoch", -1)) + 1
            global_parent_step = int(payload.get("global_parent_step", 0))
            optimizer_step = int(payload.get("optimizer_step", 0))
            best_metric = float(payload.get("best_metric", float("inf")))
            best_epoch = int(payload.get("best_epoch", -1))
            bad_validation_count = int(payload.get("bad_validation_count", 0))
            if start_epoch >= args.epochs:
                raise ValueError("resume checkpoint already reaches requested epochs")
            _write_log(
                log_path,
                f"resuming checkpoint={resume_checkpoint} epoch={start_epoch}",
            )

        parameter_counts = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "structure_encoder": sum(
                parameter.numel() for parameter in encoder_parameters
            ),
            "heads": sum(parameter.numel() for parameter in head_parameters),
        }
        resolved_config = vars(args).copy()
        resolved_config.update(
            {
                "run_dir": str(run_dir),
                "data_root": str(data_root),
                "pretrained_checkpoint": str(pretrained_checkpoint),
                "device": str(device),
                "initialization": initialization,
                "parameter_counts": parameter_counts,
                "train_parents": len(train_dataset),
                "validation_parents": len(validation_dataset),
                "steps_per_epoch": len(train_loader),
                "optimizer_steps_per_epoch": math.ceil(
                    len(train_loader) / args.gradient_accumulation_parents
                ),
                "edge_cache": {
                    key: value for key, value in edge_result.items() if key != "records"
                },
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
        _atomic_json(run_dir / "config.json", resolved_config)
        _write_log(
            log_path,
            f"data train_parents={len(train_dataset)} "
            f"validation_parents={len(validation_dataset)} workers={args.num_workers}",
        )
        _write_log(
            log_path,
            f"model parameters={parameter_counts['total']} "
            f"encoder_microbatch={args.encoder_site_batch_size} "
            f"gradient_accumulation={args.gradient_accumulation_parents}",
        )

        metrics_path = run_dir / "metrics.jsonl"
        last_epoch = start_epoch - 1
        last_validation: dict[str, float] = {}
        stopped_early = False
        stop_requested = False

        def checkpoint_payload(epoch: int) -> dict[str, Any]:
            return {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": int(epoch),
                "global_parent_step": int(global_parent_step),
                "optimizer_step": int(optimizer_step),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "bad_validation_count": int(bad_validation_count),
                "args": resolved_config,
            }

        torch.cuda.reset_peak_memory_stats(device)
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for old_record in history:
                metrics_handle.write(
                    json.dumps(old_record, ensure_ascii=False, default=float) + "\n"
                )
            metrics_handle.flush()
            for epoch in range(start_epoch, int(args.epochs)):
                last_epoch = epoch
                lr_scale, learning_rates = _set_learning_rates(
                    optimizer,
                    epoch=epoch,
                    total_epochs=int(args.epochs),
                    warmup_epochs=int(args.warmup_epochs),
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_start = time.perf_counter()
                losses: list[float] = []
                site_losses: list[float] = []
                cell_losses: list[float] = []
                gradient_norms: list[float] = []
                minus_count = 0
                parent_count = 0
                recent_gradient_norm = float("nan")
                loader_steps = len(train_loader)
                for step_in_epoch, batch in enumerate(train_loader, start=1):
                    if (
                        args.max_parent_steps > 0
                        and global_parent_step >= args.max_parent_steps
                    ):
                        stop_requested = True
                        break
                    group_position = (step_in_epoch - 1) % int(
                        args.gradient_accumulation_parents
                    )
                    group_start = step_in_epoch - group_position
                    remaining = loader_steps - group_start + 1
                    group_target = min(
                        int(args.gradient_accumulation_parents), remaining
                    )
                    moved = move_magnetic_batch(batch, device)
                    with _autocast(device, args.precision):
                        predictions = model(moved)
                        selected = global_spin_flip_huber(
                            predictions["site_moments"],
                            predictions["cell_moment_per_atom"],
                            moved["site_targets"],
                            moved["cell_targets"],
                            delta=float(args.huber_delta_muB),
                            cell_weight=float(args.cell_weight),
                        )
                        loss = selected["loss"]
                        scaled_loss = loss / float(group_target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss epoch={epoch} parent_step={global_parent_step}"
                        )
                    scaled_loss.backward()
                    batch_parents = len(batch["parent_id"])
                    global_parent_step += batch_parents
                    parent_count += batch_parents
                    loss_value = float(loss.detach().float().cpu())
                    losses.append(loss_value)
                    site_losses.append(
                        float(selected["site_loss"].detach().float().cpu())
                    )
                    cell_losses.append(
                        float(selected["cell_loss"].detach().float().cpu())
                    )
                    minus_count += int(selected["minus_selected"].sum().cpu())
                    end_of_group = (
                        group_position + 1 == group_target
                        or step_in_epoch == loader_steps
                    )
                    if end_of_group:
                        gradient_norm = clip_grad_norm_(
                            model.parameters(), float(args.grad_clip)
                        )
                        if not torch.isfinite(torch.as_tensor(gradient_norm)):
                            raise FloatingPointError(
                                f"non-finite gradient epoch={epoch} "
                                f"parent_step={global_parent_step}"
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        recent_gradient_norm = float(
                            torch.as_tensor(gradient_norm).cpu()
                        )
                        gradient_norms.append(recent_gradient_norm)
                        peak_allocated_gib = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        )
                        if peak_allocated_gib > float(args.maximum_allocated_gib):
                            raise MemoryError(
                                f"allocated CUDA memory {peak_allocated_gib:.3f} GiB "
                                f"exceeds configured limit "
                                f"{args.maximum_allocated_gib:.3f} GiB"
                            )
                        if (
                            args.empty_cache_every_optimizer_steps > 0
                            and optimizer_step
                            % int(args.empty_cache_every_optimizer_steps)
                            == 0
                        ):
                            torch.cuda.empty_cache()
                        current_reserved_gib = (
                            torch.cuda.memory_reserved(device) / 2**30
                        )
                        if (
                            args.maximum_reserved_gib > 0
                            and current_reserved_gib
                            > float(args.maximum_reserved_gib)
                        ):
                            raise MemoryError(
                                f"current reserved CUDA memory "
                                f"{current_reserved_gib:.3f} GiB exceeds configured "
                                f"limit {args.maximum_reserved_gib:.3f} GiB"
                            )
                    if (
                        step_in_epoch == 1
                        or step_in_epoch % int(args.log_every) == 0
                        or step_in_epoch == loader_steps
                    ):
                        elapsed = time.perf_counter() - epoch_start
                        rate = parent_count / max(elapsed, 1e-12)
                        peak_allocated = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        )
                        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
                        _write_log(
                            log_path,
                            f"epoch={epoch} parent={step_in_epoch}/{loader_steps} "
                            f"loss={loss_value:.6f} grad_norm={recent_gradient_norm:.6f} "
                            f"parents_per_second={rate:.4f} "
                            f"reserved_gib={peak_reserved:.3f}",
                        )
                        _atomic_json(
                            run_dir / "progress.json",
                            {
                                "status": "running",
                                "updated_at": _utc(),
                                "epoch": epoch,
                                "human_epoch": epoch + 1,
                                "step_in_epoch": step_in_epoch,
                                "steps_per_epoch": loader_steps,
                                "global_parent_step": global_parent_step,
                                "optimizer_step": optimizer_step,
                                "recent_loss": loss_value,
                                "recent_gradient_norm": recent_gradient_norm,
                                "parents_per_second": rate,
                                "encoder_lr": learning_rates[0],
                                "head_lr": learning_rates[1],
                                "peak_memory_allocated_gib": peak_allocated,
                                "peak_memory_reserved_gib": peak_reserved,
                                "best_validation_site_mae_muB": best_metric,
                                "best_epoch": best_epoch,
                            },
                        )
                    del moved, predictions, selected, loss, scaled_loss
                    if (
                        args.max_parent_steps > 0
                        and global_parent_step >= args.max_parent_steps
                    ):
                        stop_requested = True
                        break
                if not losses:
                    raise RuntimeError(f"epoch {epoch} produced no training parents")

                train_seconds = time.perf_counter() - epoch_start
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "human_epoch": epoch + 1,
                    "global_parent_step": global_parent_step,
                    "optimizer_step": optimizer_step,
                    "train_loss": float(np.mean(losses)),
                    "train_site_huber": float(np.mean(site_losses)),
                    "train_cell_huber_per_atom": float(np.mean(cell_losses)),
                    "train_minus_orientation_fraction": minus_count / parent_count,
                    "train_parents": parent_count,
                    "optimizer_steps": len(gradient_norms),
                    "mean_gradient_norm": (
                        float(np.mean(gradient_norms))
                        if gradient_norms
                        else float("nan")
                    ),
                    "lr_scale": lr_scale,
                    "encoder_lr": learning_rates[0],
                    "head_lr": learning_rates[1],
                    "train_seconds": train_seconds,
                    "train_seconds_per_parent": train_seconds / parent_count,
                    "precision": args.precision,
                }
                should_validate = (
                    (epoch + 1) % int(args.eval_every) == 0
                    or epoch + 1 == int(args.epochs)
                    or stop_requested
                )
                if should_validate:
                    _write_log(log_path, f"epoch={epoch} validation started")
                    validation = _evaluate(
                        model,
                        validation_loader,
                        device,
                        precision=args.precision,
                        delta=float(args.huber_delta_muB),
                        cell_weight=float(args.cell_weight),
                        log_path=log_path,
                    )
                    last_validation = validation
                    record["validation"] = validation
                    metric = float(validation[args.primary_validation_metric])
                    improved = metric < (
                        best_metric - float(args.minimum_improvement)
                    )
                    if improved:
                        best_metric = metric
                        best_epoch = epoch
                        bad_validation_count = 0
                        _write_log(
                            log_path,
                            f"new best {args.primary_validation_metric}={metric:.6f} "
                            f"epoch={epoch}; saving best.pt",
                        )
                        _atomic_torch_save(
                            checkpoint_payload(epoch), run_dir / "best.pt"
                        )
                    else:
                        bad_validation_count += 1
                    record.update(
                        {
                            "validation_metric": metric,
                            "validation_improved": improved,
                            "best_metric": best_metric,
                            "best_epoch": best_epoch,
                            "bad_validation_count": bad_validation_count,
                        }
                    )
                    if (
                        args.early_stopping_patience > 0
                        and bad_validation_count
                        >= int(args.early_stopping_patience)
                    ):
                        stopped_early = True
                metrics_handle.write(
                    json.dumps(record, ensure_ascii=False, default=float) + "\n"
                )
                metrics_handle.flush()
                if (
                    (epoch + 1) % int(args.save_every) == 0
                    or stop_requested
                    or stopped_early
                ):
                    _write_log(log_path, f"epoch={epoch} saving last.pt")
                    _atomic_torch_save(
                        checkpoint_payload(epoch), run_dir / "last.pt"
                    )
                peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
                peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
                _atomic_json(
                    run_dir / "progress.json",
                    {
                        "status": "running",
                        "updated_at": _utc(),
                        "epoch": epoch,
                        "human_epoch": epoch + 1,
                        "steps_per_epoch": loader_steps,
                        "global_parent_step": global_parent_step,
                        "optimizer_step": optimizer_step,
                        "train_loss": record["train_loss"],
                        "train_seconds": train_seconds,
                        "best_validation_site_mae_muB": best_metric,
                        "best_epoch": best_epoch,
                        "bad_validation_count": bad_validation_count,
                        "last_validation": last_validation,
                        "peak_memory_allocated_gib": peak_allocated,
                        "peak_memory_reserved_gib": peak_reserved,
                    },
                )
                _write_log(
                    log_path,
                    f"epoch={epoch} completed train_loss={record['train_loss']:.6f} "
                    f"train_seconds={train_seconds:.1f} best_metric={best_metric:.6f}",
                )
                if stop_requested or stopped_early:
                    break

        if last_epoch < 0:
            raise RuntimeError("no training epoch was executed")
        _atomic_torch_save(checkpoint_payload(last_epoch), run_dir / "last.pt")
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        summary = {
            "status": "complete",
            "completed_at": _utc(),
            "mode": args.mode,
            "budget": int(args.budget),
            "seed": int(args.seed),
            "last_epoch": int(last_epoch),
            "human_epochs_completed": int(last_epoch + 1),
            "global_parent_step": int(global_parent_step),
            "optimizer_step": int(optimizer_step),
            "best_validation_site_mae_muB": float(best_metric),
            "best_epoch": int(best_epoch),
            "last_validation": last_validation,
            "stopped_early": bool(stopped_early),
            "max_parent_steps_reached": bool(stop_requested),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_gib": peak_allocated,
            "peak_memory_reserved_gib": peak_reserved,
        }
        _atomic_json(run_dir / "summary.json", summary)
        _atomic_json(run_dir / "progress.json", summary)
        (run_dir / "_SUCCESS").write_text(
            f"training completed {_utc()} epoch={last_epoch} "
            f"parent_step={global_parent_step}\n",
            encoding="utf-8",
        )
        _write_log(
            log_path,
            f"training completed epoch={last_epoch} "
            f"best_site_mae={best_metric:.6f}",
        )
        success = True
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
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        _write_log(log_path, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        if running_marker.exists():
            running_marker.unlink()
        train_loader = None
        validation_loader = None
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
        if not success and not (run_dir / "_FAILED").exists():
            (run_dir / "_FAILED").write_text(
                f"{_utc()} training ended without success\n",
                encoding="utf-8",
            )


def build_arg_parser() -> argparse.ArgumentParser:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["transfer"]
    parser = argparse.ArgumentParser(description="Train a Low-Label Magnetic Transfer model.")
    parser.add_argument("mode", choices=("s0", "s_dft", "s_gauss"))
    parser.add_argument("budget", type=int, choices=(50, 300, 620))
    parser.add_argument("--data-root", default=str(DEFAULT_TASK3_DATA_ROOT))
    parser.add_argument("--checkpoint", dest="pretrained_checkpoint", default=str(DEFAULT_PRETRAINED_CHECKPOINT))
    parser.add_argument("--output-dir", dest="run_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", dest="resume_checkpoint", default=None)
    parser.set_defaults(
        config=str(CONFIG_PATH), run_root=str(DEFAULT_RUN_ROOT), name=None,
        seed=int(protocol["seed"]), epochs=int(protocol["epochs"]),
        early_stopping_patience=20, minimum_improvement=0.0, output_dim=256,
        head_hidden_dim=128, head_dropout=0.1, encoder_site_batch_size=32,
        encoder_gradient_checkpointing=True, parent_batch_size=1,
        gradient_accumulation_parents=4, num_workers=2, prefetch_factor=2,
        persistent_workers=True, precompute_edges=True, edge_cutoff=5.0,
        edge_max_neighbors=64, edge_cache_root=None, precision="bf16",
        eval_every=2, save_every=1,
        encoder_lr=float(protocol["encoder_learning_rate"]),
        head_lr=float(protocol["head_learning_rate"]), weight_decay=1e-4,
        grad_clip=1.0, warmup_epochs=5, scheduler="warmup_cosine",
        rotate_train=True, huber_delta_muB=0.5, cell_weight=1.0,
        primary_validation_metric="site_mae_muB", log_every=10,
        maximum_allocated_gib=76.0, maximum_reserved_gib=0.0,
        empty_cache_every_optimizer_steps=1, validation_max_parents=0,
        max_parent_steps=-1, resume_metrics=None, structure_kwargs={},
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "s_gauss" and args.budget != 50:
        parser.error("the published S-Gauss control is defined only at budget 50")
    args.name = args.name or f"{args.mode}_budget{args.budget}"
    args.structure_kwargs = _normalise_mapping(args.structure_kwargs)
    return args


if __name__ == "__main__":
    output = train(parse_args())
    print(output, flush=True)
