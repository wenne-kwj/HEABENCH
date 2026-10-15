#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from property_prediction.charge_data import ChargeConditionDataset, ChargeNormalizer
from property_prediction.data import (
    BenchmarkIndex,
    CompositionStructureDataset,
)
from property_prediction.metrics import regression_metrics
from property_prediction.models.charge_conditions import CompositionChargeModel
from property_prediction.models.conditions import CompositionConditionModel
from property_prediction.objectives import TargetNormalizer, masked_family_smooth_l1
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import validate_heavy_artifact_path


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
SPEC = TASK_ROOT / "config.json"
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
STRUCTURE_CACHE = ARTIFACT_ROOT / "cache/structure_ecomformer_r6_n32_v1"
CGCNN_CACHE = ARTIFACT_ROOT / "cache/structure_cgcnn_r6_n32_v1"
DFT_CACHE = FIELD_ROOT / "charge_encoder_inputs_v1/mcdc_56x40x40"
POINTVECTOR_CACHE = (
    FIELD_ROOT / "charge_encoder_inputs_v1/pointvector_native_2048"
)
GAUSSIAN_CACHE = FIELD_ROOT / "gaussian_zval_56x40x40_sigma0p35_v1"
CONDITIONS = (
    "C",
    "C_S_cgcnn",
    "C_S_ecomformer",
    "C_S_alignn",
    "C_rho_pointvector",
    "C_rho_mcdc",
    "C_S_gaussian",
    "C_S_rho",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen probing checkpoint on test.")
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("family")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", dest="training_seed", type=int, default=None)
    parser.set_defaults(spec=SPEC, batch_size=None, num_workers=4)
    return parser.parse_args()


def dump(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def target_normalizer(index: BenchmarkIndex, family) -> TargetNormalizer:
    pairs = [
        index.family_target(int(position), family)
        for position in index.indices("train")
    ]
    target = torch.from_numpy(np.stack([item[0] for item in pairs]))
    mask = torch.from_numpy(np.stack([item[1] for item in pairs]))
    return TargetNormalizer.fit(target, mask)


def charge_normalizer(payload: dict[str, Any]) -> ChargeNormalizer:
    return ChargeNormalizer(
        float(payload["scale"]),
        float(payload["mean"]),
        float(payload["std"]),
        int(payload["sampled_train_values"]),
    )


def field_normalizer(cache_dir: Path) -> ChargeNormalizer:
    payload = json.loads(
        (cache_dir / "train_normalization_seed31.json").read_text(
            encoding="utf-8"
        )
    )
    return charge_normalizer(payload)


def dft_normalizer() -> ChargeNormalizer:
    return field_normalizer(DFT_CACHE)


def checkpoint_path(
    condition: str, family: str, training_seed: int, split_seed: int
) -> Path:
    checkpoint_root = ARTIFACT_ROOT / "checkpoints"
    if training_seed != split_seed:
        checkpoint_root = checkpoint_root / "seeds" / f"seed{training_seed}"
    relative_roots = {
        "C": Path("C"),
        "C_S_cgcnn": Path("C_S/cgcnn"),
        "C_S_ecomformer": Path("C_S/ecomformer"),
        "C_S_alignn": Path("C_S/alignn"),
        "C_rho_pointvector": Path("C_rho/pointvector"),
        "C_rho_mcdc": Path("C_rho/mcdc"),
        "C_S_gaussian": Path("C_S_gaussian/alignn_mcdc"),
        "C_S_rho": Path("C_S_rho/alignn_mcdc"),
    }
    return checkpoint_root / relative_roots[condition] / family / "best.pt"


def build_condition(
    condition: str, index: BenchmarkIndex, family, split: str = "test"
):
    input_normalizer = None
    if condition == "C":
        dataset = CompositionStructureDataset(
            index, split, family, include_structure=False
        )
        model = CompositionConditionModel(family.output_shape)
        collate = None
        default_batch = 128
    elif condition == "C_S_alignn":
        from property_prediction.alignn_data import (
            ALIGNNConditionDataset,
            make_alignn_collate,
        )
        from property_prediction.models.alignn_conditions import (
            CompositionALIGNNModel,
        )

        dataset = ALIGNNConditionDataset(
            index, split, family, STRUCTURE_CACHE
        )
        model = CompositionALIGNNModel(family.output_shape)
        collate = make_alignn_collate()
        default_batch = 8
    elif condition == "C_S_cgcnn":
        from property_prediction.models.structure_conditions import (
            CompositionCGCNNModel,
        )
        from property_prediction.structure_data import (
            CGCNNConditionDataset,
            collate_cgcnn,
            load_atom_feature_table,
        )

        dataset = CGCNNConditionDataset(
            index, split, family, CGCNN_CACHE
        )
        model = CompositionCGCNNModel(family.output_shape)
        collate = partial(
            collate_cgcnn,
            cutoff=6.0,
            step=0.2,
            atom_table=load_atom_feature_table(),
        )
        default_batch = 64
    elif condition == "C_S_ecomformer":
        from property_prediction.comformer_data import (
            ComFormerConditionDataset,
            collate_comformer,
        )
        from property_prediction.models.comformer_conditions import (
            CompositionComFormerModel,
        )
        from property_prediction.structure_data import load_atom_feature_table

        dataset = ComFormerConditionDataset(
            index, split, family, STRUCTURE_CACHE
        )
        model = CompositionComFormerModel(family.output_shape)
        collate = partial(
            collate_comformer, atom_table=load_atom_feature_table()
        )
        default_batch = 16
    elif condition == "C_rho_pointvector":
        dataset = ChargeConditionDataset(
            index, split, family, POINTVECTOR_CACHE
        )
        model = CompositionChargeModel(
            family.output_shape, encoder_name="pointvector"
        )
        collate = None
        input_normalizer = field_normalizer(POINTVECTOR_CACHE)
        default_batch = 32
    elif condition == "C_rho_mcdc":
        dataset = ChargeConditionDataset(index, split, family, DFT_CACHE)
        model = CompositionChargeModel(
            family.output_shape, encoder_name="mcdc"
        )
        collate = None
        input_normalizer = dft_normalizer()
        default_batch = 16
    else:
        from property_prediction.fusion_data import (
            DenseFieldFusionDataset,
            build_structure_dataset,
            fusion_collate,
        )
        from property_prediction.models.fusion_conditions import (
            CompositionStructureChargeModel,
        )

        field_cache = DFT_CACHE if condition == "C_S_rho" else GAUSSIAN_CACHE
        structure_dataset = build_structure_dataset(
            "alignn", index, split, family, STRUCTURE_CACHE
        )
        dataset = DenseFieldFusionDataset(
            structure_dataset, index, field_cache
        )
        model = CompositionStructureChargeModel(
            family.output_shape,
            structure_encoder_name="alignn",
            charge_encoder_name="mcdc",
        )
        collate = fusion_collate("alignn")
        default_batch = 8
    return dataset, model, collate, input_normalizer, default_batch


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
    input_normalizer: ChargeNormalizer | None,
) -> dict[str, Any]:
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
    if input_normalizer is not None:
        result["charge_density"] = input_normalizer.transform(
            result["charge_density"]
        )
    return result


def main() -> None:
    args = arguments()
    spec = load_benchmark_spec(args.spec)
    training_seed = spec.seed if args.training_seed is None else args.training_seed
    if args.family not in spec.target_families:
        raise ValueError(args.family)
    family = spec.target_families[args.family]
    index = BenchmarkIndex(spec)
    checkpoint = checkpoint_path(
        args.condition, args.family, training_seed, spec.seed
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    test_run_root = ARTIFACT_ROOT / "runs/final_test"
    test_prediction_root = ARTIFACT_ROOT / "predictions/final_test"
    if training_seed != spec.seed:
        seed_path = Path("seeds") / f"seed{training_seed}"
        test_run_root = test_run_root / seed_path
        test_prediction_root = test_prediction_root / seed_path
    output_dir = validate_heavy_artifact_path(
        test_run_root / args.condition / args.family, spec
    )
    prediction_dir = validate_heavy_artifact_path(
        test_prediction_root / args.condition, spec
    )
    prediction_path = prediction_dir / f"{args.family}.npz"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"test already evaluated: {output_dir}")
    if prediction_path.exists():
        raise FileExistsError(f"test prediction already exists: {prediction_path}")

    dataset, model, collate, input_normalizer, default_batch = build_condition(
        args.condition, index, family
    )
    saved = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    if args.condition == "C_S_gaussian":
        input_normalizer = charge_normalizer(
            saved["config"]["input_normalization"]
        )
    elif args.condition == "C_S_rho":
        input_normalizer = dft_normalizer()

    device = torch.device(args.device)
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or default_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    normalizer = target_normalizer(index, family)
    identifiers, predictions, targets, masks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            identifiers.extend(batch["dataset_entry_id"])
            batch = move_batch(batch, device, input_normalizer)
            output = model(batch)
            predictions.append(normalizer.inverse(output).cpu())
            targets.append(batch["target"].cpu())
            masks.append(batch["target_mask"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    expected_ids = [index.ids[int(position)] for position in index.indices("test")]
    if identifiers != expected_ids:
        raise ValueError("test prediction order does not match the frozen split")
    normalized_prediction = normalizer.transform(prediction, mask)
    normalized_target = normalizer.transform(target, mask)
    loss = masked_family_smooth_l1(
        normalized_prediction,
        normalized_target,
        mask,
        family=family.name,
    )
    metrics = regression_metrics(prediction, target, mask, family)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        dataset_entry_id=np.asarray(identifiers),
        prediction=prediction.numpy(),
        target=target.numpy(),
        mask=mask.numpy(),
    )
    dump(
        output_dir / "metrics.json",
        {
            "status": "test_complete",
            "condition": args.condition,
            "family": family.name,
            "split_seed": spec.seed,
            "training_seed": training_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(saved["epoch"]),
            "validation_loss": float(saved["validation_loss"]),
            "test_loss": float(loss),
            "metrics": metrics,
            "sample_count": len(identifiers),
            "test_evaluated": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "test_complete",
                "condition": args.condition,
                "family": family.name,
                "training_seed": training_seed,
                "test_loss": float(loss),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
