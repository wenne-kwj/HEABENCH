
"""Train HEABench structure--charge towers with symmetric in-batch InfoNCE.

Explicit candidate sets are reserved for evaluation and never contribute
gradients in the reported experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import (
    DEFAULT_DATA_ROOT,
    RetrievalPatchDataset,
    ShardBatchSampler,
    compute_density_stats,
    make_collate_fn,
)
from .losses import symmetric_infonce
from .models.factory import build_model


from .paths import CONFIG_PATH, DEFAULT_RUN_ROOT


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_run_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = root / f"{stamp}_{name}"
    path = base
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"{stamp}_{name}_{suffix:02d}"
    path.mkdir()
    return path


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _write_log(path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp} UTC] {message}\n")
        handle.flush()


def _mark_failed(run_dir: Path, log_path: Path, exc: BaseException) -> None:
    """Leave a durable failure marker without overwriting a completed run."""

    if (run_dir / "_SUCCESS").exists() or (run_dir / "_FAILED").exists():
        return
    (run_dir / "_FAILED").write_text(
        f"training failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
    )
    _write_log(log_path, f"FAILED {type(exc).__name__}: {exc}")


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    """Load a training checkpoint across supported PyTorch versions."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint does not contain a model state: {path}")
    return payload


def _read_metric_history(path: Path) -> list[dict[str, Any]]:
    """Read valid JSONL records, tolerating a truncated final line."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _retrieval_score(record: dict[str, Any]) -> float | None:
    """Return the validation score used for best-checkpoint selection."""

    if "structure_to_charge_recall@1" in record:
        left = record.get("structure_to_charge_recall@1")
        right = record.get("charge_to_structure_recall@1")
    else:
        left = record.get("s2c_recall@1")
        right = record.get("c2s_recall@1")
    if left is None or right is None:
        return None
    return 0.5 * (float(left) + float(right))


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors after loading a CPU checkpoint."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so an interrupted write is recoverable."""

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def encode_dataset(
    model,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Encode canonical anchors, retaining structure/charge side by side."""

    model.eval()
    vectors: list[torch.Tensor] = []
    ids: list[str] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        structure, charge = model(moved)
        vectors.append(torch.cat([structure, charge], dim=1).cpu())
        ids.extend(batch["patch_id"])
    if not vectors:
        return torch.empty((0, 0)), ids
    return torch.cat(vectors, dim=0), ids


@torch.no_grad()
def evaluate_retrieval(
    model,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate global canonical-anchor retrieval in both directions."""

    model.eval()
    structures: list[torch.Tensor] = []
    charges: list[torch.Tensor] = []
    ids: list[str] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        structure, charge = model(moved)
        structures.append(structure.cpu())
        charges.append(charge.cpu())
        ids.extend(batch["patch_id"])
    if not ids:
        return {
            "s2c_recall@1": 0.0,
            "s2c_recall@5": 0.0,
            "s2c_recall@10": 0.0,
            "c2s_recall@1": 0.0,
            "c2s_recall@5": 0.0,
            "c2s_recall@10": 0.0,
            "n": 0.0,
        }
    structure = torch.cat(structures, dim=0)
    charge = torch.cat(charges, dim=0)
    hits = {1: 0, 5: 0, 10: 0}
    reverse_hits = {1: 0, 5: 0, 10: 0}
    n = len(ids)
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        logits = structure[start:stop] @ charge.T
        ranking = logits.topk(min(10, n), dim=1).indices
        reverse_logits = charge[start:stop] @ structure.T
        reverse_ranking = reverse_logits.topk(min(10, n), dim=1).indices
        target = torch.arange(start, stop).unsqueeze(1)
        for k in hits:
            kk = min(k, n)
            hits[k] += int((ranking[:, :kk] == target).any(dim=1).sum())
            reverse_hits[k] += int((reverse_ranking[:, :kk] == target).any(dim=1).sum())
    return {
        **{f"s2c_recall@{k}": hits[k] / n for k in hits},
        **{f"c2s_recall@{k}": reverse_hits[k] / n for k in reverse_hits},
        "n": float(n),
    }


def _make_anchor_loader(
    dataset: RetrievalPatchDataset,
    *,
    batch_size: int,
    collate,
    device: torch.device,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
) -> tuple[DataLoader, ShardBatchSampler | None]:
    if shuffle:
        sampler = ShardBatchSampler(dataset, batch_size, shuffle=True, seed=seed)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        return loader, sampler
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, None


def _normalise_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("model kwargs must be a JSON object")
    return dict(value)


def _train_impl(args: argparse.Namespace, failure_state: dict[str, Any] | None = None) -> Path:
    _seed_everything(args.seed)
    data_root = Path(args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    if args.objective != "inbatch":
        raise ValueError("formal Task 2 training supports only inbatch InfoNCE")
    if args.candidate_eval:
        raise ValueError(
            "candidate sets are evaluated by hea_retrieval.evaluate, not during training"
        )
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    resume_checkpoint = (
        Path(args.resume_checkpoint).expanduser().resolve()
        if args.resume_checkpoint
        else None
    )
    if resume_checkpoint is not None and not resume_checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
    run_name = args.name or f"{args.structure}_{args.charge}"
    if resume_checkpoint is not None:
        run_name += "_resume"
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else _new_run_dir(DEFAULT_RUN_ROOT, run_name)
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    if failure_state is not None:
        failure_state["run_dir"] = run_dir
        failure_state["log_path"] = log_path
    _write_log(log_path, f"starting objective=inbatch device={device}")

    resume_metrics = (
        Path(args.resume_metrics).expanduser().resolve()
        if args.resume_metrics
        else (resume_checkpoint.parent / "metrics.jsonl" if resume_checkpoint else None)
    )
    resume_history = _read_metric_history(resume_metrics) if resume_metrics else []
    if resume_checkpoint is not None:
        _write_log(
            log_path,
            f"resuming checkpoint={resume_checkpoint} "
            f"history={resume_metrics} records={len(resume_history)}",
        )

    train_ds = RetrievalPatchDataset(data_root, split="train", cache_size=args.cache_size)
    val_ds = RetrievalPatchDataset(data_root, split="validation", cache_size=args.cache_size)
    stats = compute_density_stats(
        train_ds, max_patches=args.stats_patches, point_stride=args.stats_stride
    )
    (run_dir / "density_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    collate = make_collate_fn(
        rho_mean=float(stats["rho_mean"]),
        rho_std=float(stats["rho_std"]),
        rho_mag_mean=float(stats["rho_mag_mean"]),
        rho_mag_std=float(stats["rho_mag_std"]),
        rotate=args.rotate,
        charge_points=args.charge_points,
    )
    train_anchor_loader, train_anchor_sampler = _make_anchor_loader(
        train_ds,
        batch_size=args.batch_size,
        collate=collate,
        device=device,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_anchor_loader, _ = _make_anchor_loader(
        val_ds,
        batch_size=args.eval_batch_size,
        collate=collate,
        device=device,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
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
    start_epoch = 0
    global_step = 0
    best = -float("inf")
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_payload = _load_torch_checkpoint(resume_checkpoint)
        model.load_state_dict(resume_payload["model"], strict=True)
        if resume_payload.get("optimizer") is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            _move_optimizer_state(optimizer, device)
        if resume_payload.get("scheduler") is not None:
            scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        if resume_history:
            global_step = int(resume_history[-1].get("step", 0))
            scores = [
                score for score in (_retrieval_score(record) for record in resume_history)
                if score is not None
            ]
            if scores:
                best = max(scores)
        if "best_score" in resume_payload:
            best = max(best, float(resume_payload["best_score"]))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}; "
                f"target epochs={args.epochs}"
            )
        initial_name = "best.pt" if resume_checkpoint.name == "best.pt" else "last.pt"
        shutil.copy2(resume_checkpoint, run_dir / initial_name)
        (run_dir / "resume.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(resume_checkpoint),
                    "source_metrics": str(resume_metrics) if resume_metrics else None,
                    "source_epoch": int(resume_payload.get("epoch", -1)),
                    "start_epoch": start_epoch,
                    "source_global_step": global_step,
                    "source_best_score": best,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    config = vars(args).copy()
    config.update(
        {
            "run_dir": str(run_dir),
            "density_stats": stats,
            "device": str(device),
            "structure_kwargs": structure_kwargs,
            "charge_kwargs": charge_kwargs,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "resume_source_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "resume_source_metrics": str(resume_metrics) if resume_metrics else None,
            "resume_start_epoch": start_epoch,
            "resume_source_global_step": global_step,
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    last_epoch = start_epoch - 1
    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best,
            "args": config,
        }

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics:
            for previous in resume_history:
                metrics.write(json.dumps(previous, ensure_ascii=False, default=float) + "\n")
            metrics.flush()
            for epoch in range(start_epoch, args.epochs):
                last_epoch = epoch
                model.train()
                if train_anchor_sampler is not None:
                    train_anchor_sampler.seed = args.seed + epoch
                losses: list[float] = []
                step_start = time.perf_counter()
                for batch in train_anchor_loader:
                    moved = _move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    structure, charge = model(moved)
                    loss = symmetric_infonce(
                        structure, charge, temperature=args.temperature
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at step {global_step}"
                        )
                    loss.backward()
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    global_step += 1
                    if args.log_every > 0 and global_step % args.log_every == 0:
                        _write_log(log_path, f"epoch={epoch} step={global_step} loss={losses[-1]:.6f}")
                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break
                scheduler.step()
                elapsed = time.perf_counter() - step_start
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "lr": optimizer.param_groups[0]["lr"],
                    "steps_in_epoch": len(losses),
                    "seconds": elapsed,
                    "inbatch_loss": float(np.mean(losses)) if losses else float("nan"),
                }
                should_eval = (
                    (epoch + 1) % max(args.eval_every, 1) == 0
                    or (args.max_steps > 0 and global_step >= args.max_steps)
                )
                if should_eval:
                    if args.global_eval:
                        record.update(evaluate_retrieval(model, val_anchor_loader, device))
                        score = float(
                            0.5
                            * (
                                record.get("s2c_recall@1", 0.0)
                                + record.get("c2s_recall@1", 0.0)
                            )
                        )
                        if score > best:
                            best = score
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "epoch": epoch,
                                    "args": config,
                                },
                                run_dir / "best.pt",
                            )
                metrics.write(
                    json.dumps(record, ensure_ascii=False, default=float) + "\n"
                )
                metrics.flush()
                # Keep a recoverable checkpoint at every epoch boundary.  The
                # atomic write prevents a killed process from corrupting it.
                _atomic_torch_save(
                    checkpoint_payload(epoch),
                    run_dir / "last.pt",
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if last_epoch < 0:
            raise RuntimeError("no training epochs were executed")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": last_epoch,
                "args": config,
            },
            run_dir / "last.pt",
        )
        (run_dir / "_SUCCESS").write_text("training completed\n", encoding="utf-8")
        _write_log(log_path, f"completed steps={global_step} epochs={last_epoch + 1}")
    except Exception as exc:
        _mark_failed(run_dir, log_path, exc)
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return run_dir


def train(args: argparse.Namespace) -> Path:
    """Run training and mark failures occurring during setup as well as updates."""

    failure_state: dict[str, Any] = {}
    try:
        return _train_impl(args, failure_state)
    except Exception as exc:
        run_dir = failure_state.get("run_dir")
        log_path = failure_state.get("log_path")
        if isinstance(run_dir, Path) and isinstance(log_path, Path):
            _mark_failed(run_dir, log_path, exc)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description="Train a Local Structure-Charge Matching model.")
    p.add_argument("structure", choices=("geoformer", "equiformer", "mace", "chgnet"))
    p.add_argument("charge", choices=("pointvector", "x3d", "ptv3", "lshce"))
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir", dest="run_dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", dest="resume_checkpoint", default=None)
    p.set_defaults(
        config=str(CONFIG_PATH), name=None,
        batch_size=int(values["batch_size"]), eval_batch_size=32,
        epochs=int(values["epochs"]), max_steps=-1, eval_every=1,
        lr=float(values["learning_rate"]), weight_decay=1e-4, grad_clip=1.0,
        temperature=float(values["temperature"]),
        output_dim=int(values["embedding_dim"]), stats_patches=2048,
        stats_stride=16, cache_size=2, num_workers=0, charge_points=None,
        rotate=True, objective="inbatch", candidate_eval=False, global_eval=True,
        log_every=20, seed=int(values["seed"]), resume_metrics=None,
        structure_kwargs={}, charge_kwargs={},
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.charge == "lshce":
        stats_path = CONFIG_PATH.with_name("lshce_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        args.charge_kwargs = {
            "rho_ref": stats["rho_ref"],
            "physical_mean": stats["physical_mean"],
            "physical_std": stats["physical_std"],
        }
    args.structure_kwargs = _normalise_kwargs(args.structure_kwargs)
    args.charge_kwargs = _normalise_kwargs(args.charge_kwargs)
    return args


if __name__ == "__main__":
    train(parse_args())
