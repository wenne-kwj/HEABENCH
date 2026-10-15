#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from property_prediction.data import BenchmarkIndex, CompositionStructureDataset
from property_prediction.metrics import regression_metrics
from property_prediction.models.conditions import CompositionConditionModel
from property_prediction.objectives import TargetNormalizer, masked_family_smooth_l1
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import validate_heavy_artifact_path


DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "config.json"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the composition baseline.")
    parser.add_argument("family")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", dest="training_seed", type=int, default=None)
    parser.set_defaults(
        spec=DEFAULT_SPEC, epochs=160, batch_size=64, eval_batch_size=128,
        num_workers=4, lr=1.0e-3, weight_decay=1.0e-5, patience=30,
    )
    args = parser.parse_args()
    args.run_dir = args.output_dir / "run"
    args.checkpoint_dir = args.output_dir / "checkpoints"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_targets(index: BenchmarkIndex, family, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [index.family_target(int(i), family) for i in index.indices(split)]
    return (
        torch.from_numpy(np.stack([pair[0] for pair in pairs])),
        torch.from_numpy(np.stack([pair[1] for pair in pairs])),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, normalizer, family, device):
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        predictions.append(normalizer.inverse(output).cpu())
        targets.append(batch["target"].cpu())
        masks.append(batch["target_mask"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    normalized_target = normalizer.transform(target, mask)
    normalized_prediction = normalizer.transform(prediction, mask)
    loss = masked_family_smooth_l1(
        normalized_prediction, normalized_target, mask, family=family.name
    )
    return float(loss), regression_metrics(prediction, target, mask, family)


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    spec = load_benchmark_spec(args.spec)
    training_seed = spec.seed if args.training_seed is None else args.training_seed
    if args.family not in spec.target_families:
        raise ValueError(f"unknown family {args.family!r}")
    family = spec.target_families[args.family]
    run_dir = validate_heavy_artifact_path(args.run_dir, spec)
    checkpoint_dir = validate_heavy_artifact_path(args.checkpoint_dir, spec)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty run directory: {run_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty checkpoint directory: {checkpoint_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_seed)
    device = torch.device(args.device)
    index = BenchmarkIndex(spec)
    train_target, train_mask = gather_targets(index, family, "train")
    normalizer = TargetNormalizer.fit(train_target, train_mask)
    datasets = {
        split: CompositionStructureDataset(
            index, split, family, include_structure=False
        )
        for split in ("train", "validation")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        ),
    }
    model = CompositionConditionModel(
        family.output_shape,
        latent_dim=spec.latent_dim,
        hidden_dim=spec.head_hidden_dim,
        dropout=spec.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config = {
        "status": "validation_only_training",
        "information_condition": "C",
        "family": family.name,
        "targets": list(family.targets),
        "output_shape": list(family.output_shape),
        "dataset": str(spec.benchmark_root),
        "split_counts": {key: len(value) for key, value in datasets.items()},
        "split_seed": spec.seed,
        "training_seed": training_seed,
        "model": "ElementFractionSetEncoder + common gated interface + common family head",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "trainable_parameters": trainable,
        "test_evaluated": False,
    }
    json_dump(run_dir / "config.json", config)
    torch.save({"mean": normalizer.mean, "std": normalizer.std}, checkpoint_dir / "normalizer.pt")

    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            target = normalizer.transform(batch["target"], batch["target_mask"])
            prediction = model(batch)
            loss = masked_family_smooth_l1(
                prediction, target, batch["target_mask"], family=family.name
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        scheduler.step()
        validation_loss, validation_metrics = evaluate(
            model, loaders["validation"], normalizer, family, device
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - started,
        }
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std": normalizer.std,
                    "config": config,
                },
                checkpoint_dir / "best.pt",
            )
            json_dump(run_dir / "best_validation_metrics.json", validation_metrics)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    json_dump(
        run_dir / "status.json",
        {
            "status": "validation_complete",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "test_evaluated": False,
        },
    )
    print(json.dumps({"status": "validation_complete", "family": family.name,
                      "best_epoch": best_epoch, "best_validation_loss": best_loss}), flush=True)


if __name__ == "__main__":
    main()
