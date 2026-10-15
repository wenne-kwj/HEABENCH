#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from property_prediction.charge_data import ChargeNormalizer, fit_charge_normalizer
from property_prediction.data import BenchmarkIndex
from property_prediction.fusion_data import (
    DenseFieldFusionDataset,
    build_structure_dataset,
    fusion_collate,
)
from property_prediction.metrics import regression_metrics
from property_prediction.models.fusion_conditions import (
    CompositionStructureChargeModel,
)
from property_prediction.objectives import TargetNormalizer, masked_family_smooth_l1
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import (
    validate_data_input_path,
    validate_heavy_artifact_path,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
DEFAULT_SPEC = TASK_ROOT / "config.json"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "HEABENCH_T1_ARTIFACT_ROOT",
        REPOSITORY_ROOT / "artifacts/t1_property_prediction",
    )
)
FIELD_ROOT = Path(
    os.environ.get(
        "HEABENCH_T1_FIELD_ROOT",
        REPOSITORY_ROOT / "data/t1_property_prediction/fields",
    )
)
FIELD_CACHES = {
    "dft": FIELD_ROOT / "charge_encoder_inputs_v1/mcdc_56x40x40",
    "gaussian": FIELD_ROOT / "gaussian_zval_56x40x40_sigma0p35_v1",
}
STRUCTURE_CACHES = {
    "cgcnn": ARTIFACT_ROOT / "cache/structure_cgcnn_r6_n32_v1",
    "ecomformer": ARTIFACT_ROOT / "cache/structure_ecomformer_r6_n32_v1",
    "alignn": ARTIFACT_ROOT / "cache/structure_ecomformer_r6_n32_v1",
}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fixed ALIGNN + field fusion model.")
    parser.add_argument("field", choices=("dft", "gaussian"))
    parser.add_argument("family")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", dest="training_seed", type=int, default=None)
    parser.set_defaults(
        spec=DEFAULT_SPEC, structure_encoder="alignn", structure_cache_dir=None,
        field_cache_dir=None, epochs=140, batch_size=8, eval_batch_size=8,
        num_workers=4, lr=1.0e-3, weight_decay=1.0e-5, patience=25,
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


def dump(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def gather_targets(
    index: BenchmarkIndex, family, split: str
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [index.family_target(int(i), family) for i in index.indices(split)]
    return (
        torch.from_numpy(np.stack([item[0] for item in pairs])),
        torch.from_numpy(np.stack([item[1] for item in pairs])),
    )


def field_normalizer(
    field: str, cache_dir: Path, index: BenchmarkIndex
) -> ChargeNormalizer:
    existing = cache_dir / "train_normalization_seed31.json"
    if field == "dft" and existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        return ChargeNormalizer(
            float(payload["scale"]),
            float(payload["mean"]),
            float(payload["std"]),
            int(payload["sampled_train_values"]),
        )
    array = np.load(cache_dir / "density_e_A3.npy", mmap_mode="r")
    return fit_charge_normalizer(
        array, index.indices("train"), seed=index.spec.seed
    )


def periodic_translate_dense(density: torch.Tensor) -> torch.Tensor:
    shifted = []
    for sample in density:
        shifts = tuple(
            int(torch.randint(0, size, ()).item()) for size in sample.shape
        )
        shifted.append(torch.roll(sample, shifts=shifts, dims=(0, 1, 2)))
    return torch.stack(shifted)


def move_batch(
    batch: dict, device: torch.device, normalizer: ChargeNormalizer, *, augment: bool
) -> dict:
    result = {}
    for key, value in batch.items():
        if key == "structure_graph" and isinstance(value, (tuple, list)):
            result[key] = tuple(
                graph.to(device, non_blocking=True) for graph in value
            )
        elif key == "structure_graph":
            result[key] = value.to(device, non_blocking=True)
        elif key == "crystal_atom_indices":
            result[key] = [item.to(device, non_blocking=True) for item in value]
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    density = normalizer.transform(result["charge_density"])
    result["charge_density"] = (
        periodic_translate_dense(density) if augment else density
    )
    return result


@torch.no_grad()
def evaluate(model, loader, target_normalizer, input_normalizer, family, device):
    model.eval()
    predictions, targets, masks = [], [], []
    gate_values = {"structure": [], "charge_density": []}
    contribution_ratios = {"structure": [], "charge_density": []}
    for batch in loader:
        batch = move_batch(
            batch, device, input_normalizer, augment=False
        )
        normalized_prediction, diagnostics = model(
            batch, return_diagnostics=True
        )
        predictions.append(target_normalizer.inverse(normalized_prediction).cpu())
        targets.append(batch["target"].cpu())
        masks.append(batch["target_mask"].cpu())
        for name in gate_values:
            gate_values[name].append(diagnostics["gates"][name].cpu())
            contribution_ratios[name].append(
                diagnostics["contribution_ratio_to_composition"][name].cpu()
            )
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    normalized_target = target_normalizer.transform(target, mask)
    normalized_prediction = target_normalizer.transform(prediction, mask)
    loss = masked_family_smooth_l1(
        normalized_prediction,
        normalized_target,
        mask,
        family=family.name,
    )
    diagnostics = {}
    for name in gate_values:
        gate = torch.cat(gate_values[name]).flatten()
        ratio = torch.cat(contribution_ratios[name]).flatten()
        diagnostics[name] = {
            "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()),
            "gate_q05": float(torch.quantile(gate, 0.05)),
            "gate_q50": float(torch.quantile(gate, 0.50)),
            "gate_q95": float(torch.quantile(gate, 0.95)),
            "gate_fraction_below_0p05": float((gate < 0.05).float().mean()),
            "gate_fraction_above_0p95": float((gate > 0.95).float().mean()),
            "contribution_ratio_mean": float(ratio.mean()),
            "contribution_ratio_std": float(ratio.std()),
        }
    return (
        float(loss),
        regression_metrics(prediction, target, mask, family),
        diagnostics,
    )


def parameter_counts(model: CompositionStructureChargeModel) -> dict[str, int]:
    names = (
        "composition_encoder",
        "structure_encoder",
        "charge_encoder",
        "composition_adapter",
        "structure_adapter",
        "charge_adapter",
        "fusion",
        "head",
    )
    counts = {
        name: sum(
            parameter.numel()
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        )
        for name in names
    }
    counts["total"] = sum(counts.values())
    return counts


def main() -> None:
    args = arguments()
    spec = load_benchmark_spec(args.spec)
    if args.family not in spec.target_families:
        raise ValueError(args.family)
    family = spec.target_families[args.family]
    run_dir = validate_heavy_artifact_path(args.run_dir, spec)
    checkpoint_dir = validate_heavy_artifact_path(args.checkpoint_dir, spec)
    structure_cache = validate_heavy_artifact_path(
        args.structure_cache_dir or STRUCTURE_CACHES[args.structure_encoder], spec
    )
    field_cache = validate_data_input_path(
        args.field_cache_dir or FIELD_CACHES[args.field], spec
    )
    for path in (run_dir, checkpoint_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(path)
        path.mkdir(parents=True, exist_ok=True)

    set_seed(spec.seed if args.training_seed is None else args.training_seed)
    device = torch.device(args.device)
    index = BenchmarkIndex(spec)
    train_target, train_mask = gather_targets(index, family, "train")
    target_normalizer = TargetNormalizer.fit(train_target, train_mask)
    input_normalizer = field_normalizer(args.field, field_cache, index)
    datasets = {}
    for split in ("train", "validation"):
        structure_dataset = build_structure_dataset(
            args.structure_encoder, index, split, family, structure_cache
        )
        datasets[split] = DenseFieldFusionDataset(
            structure_dataset, index, field_cache
        )
    collate = fusion_collate(args.structure_encoder)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        ),
    }
    model = CompositionStructureChargeModel(
        family.output_shape,
        structure_encoder_name=args.structure_encoder,
        charge_encoder_name="mcdc",
        latent_dim=spec.latent_dim,
        hidden_dim=spec.head_hidden_dim,
        dropout=spec.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    config = {
        "status": "validation_only_training",
        "information_condition": "C+S+rho" if args.field == "dft" else "C+S+G",
        "family": family.name,
        "targets": list(family.targets),
        "output_shape": list(family.output_shape),
        "dataset": str(spec.benchmark_root),
        "seed": spec.seed,
        "structure_encoder": args.structure_encoder,
        "structure_cache": str(structure_cache),
        "charge_encoder": "MCDC_3DCNN_dense_56x40x40",
        "field": args.field,
        "field_cache": str(field_cache),
        "field_contract": datasets["train"].metadata,
        "input_normalization": input_normalizer.to_dict(),
        "periodic_translation_augmentation": True,
        "fusion": "common_composition_anchor_gated_residual",
        "head": "common_family_head",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "parameter_counts": parameter_counts(model),
        "test_evaluated": False,
    }
    dump(run_dir / "config.json", config)
    torch.save(
        {
            "target_mean": target_normalizer.mean,
            "target_std": target_normalizer.std,
            "input_normalizer": input_normalizer.to_dict(),
        },
        checkpoint_dir / "normalizers.pt",
    )

    best = math.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = move_batch(
                batch, device, input_normalizer, augment=True
            )
            target = target_normalizer.transform(
                batch["target"], batch["target_mask"]
            )
            prediction = model(batch)
            loss = masked_family_smooth_l1(
                prediction,
                target,
                batch["target_mask"],
                family=family.name,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_loss, validation_metrics, diagnostics = evaluate(
            model,
            loaders["validation"],
            target_normalizer,
            input_normalizer,
            family,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        if validation_loss < best:
            best = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "config": config,
                },
                checkpoint_dir / "best.pt",
            )
            dump(run_dir / "best_validation_metrics.json", validation_metrics)
            dump(run_dir / "best_fusion_diagnostics.json", diagnostics)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    dump(
        run_dir / "status.json",
        {
            "status": "validation_complete",
            "best_epoch": best_epoch,
            "best_validation_loss": best,
            "test_evaluated": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "validation_complete",
                "structure_encoder": args.structure_encoder,
                "field": args.field,
                "family": family.name,
                "best_epoch": best_epoch,
                "best_validation_loss": best,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
