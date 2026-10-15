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

from property_prediction.charge_data import (
    ChargeConditionDataset,
    ChargeNormalizer,
    fit_charge_normalizer,
)
from property_prediction.data import BenchmarkIndex
from property_prediction.metrics import regression_metrics
from property_prediction.models.charge_conditions import CompositionChargeModel
from property_prediction.objectives import TargetNormalizer, masked_family_smooth_l1
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import (
    validate_data_input_path,
    validate_heavy_artifact_path,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
DEFAULT_SPEC = TASK_ROOT / "config.json"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "HEABENCH_T1_FIELD_ROOT",
        REPOSITORY_ROOT / "data/t1_property_prediction/fields/charge_encoder_inputs_v1",
    )
)
CACHE_SUBDIR = {
    "mcdc": "mcdc_56x40x40",
    "pointvector": "pointvector_native_2048",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a charge-field baseline.")
    parser.add_argument("encoder", choices=("mcdc", "pointvector"))
    parser.add_argument("family")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", dest="training_seed", type=int, default=None)
    parser.set_defaults(
        spec=DEFAULT_SPEC, cache_root=DEFAULT_CACHE_ROOT, epochs=140,
        batch_size=None, eval_batch_size=None, num_workers=4, lr=1.0e-3,
        weight_decay=1.0e-5, patience=25,
    )
    args = parser.parse_args()
    args.run_dir = args.output_dir / "run"
    args.checkpoint_dir = args.output_dir / "checkpoints"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dump(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gather_targets(index: BenchmarkIndex, family, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [index.family_target(int(i), family) for i in index.indices(split)]
    return torch.from_numpy(np.stack([x[0] for x in pairs])), torch.from_numpy(np.stack([x[1] for x in pairs]))


def load_or_fit_charge_normalizer(
    cache_dir: Path, index: BenchmarkIndex
) -> ChargeNormalizer:
    path = cache_dir / "train_normalization_seed31.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        return ChargeNormalizer(
            float(payload["scale"]), float(payload["mean"]), float(payload["std"]),
            int(payload["sampled_train_values"]),
        )
    ids = np.load(cache_dir / "dataset_entry_id.npy", allow_pickle=False).astype(str)
    if not np.array_equal(ids, np.asarray(index.ids)):
        raise ValueError("normalizer requires full canonical cache order")
    array = np.load(cache_dir / "density_e_A3.npy", mmap_mode="r")
    return fit_charge_normalizer(
        array, index.indices("train"), seed=index.spec.seed
    )


def periodic_translate(density: torch.Tensor, encoder: str) -> torch.Tensor:
    shape = (8, 16, 16) if encoder == "pointvector" else tuple(density.shape[-3:])
    grid = density.reshape(density.shape[0], *shape)
    shifted = []
    for sample in grid:
        shifts = tuple(int(torch.randint(0, size, ()).item()) for size in shape)
        shifted.append(torch.roll(sample, shifts=shifts, dims=(0, 1, 2)))
    result = torch.stack(shifted)
    return result.reshape_as(density)


def move_batch(batch: dict, device: torch.device, charge_normalizer: ChargeNormalizer,
               *, encoder: str, augment: bool) -> dict:
    result = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    density = charge_normalizer.transform(result["charge_density"])
    result["charge_density"] = periodic_translate(density, encoder) if augment else density
    return result


@torch.no_grad()
def evaluate(model, loader, target_normalizer, charge_normalizer, family, device, encoder):
    model.eval(); predictions=[]; targets=[]; masks=[]
    for batch in loader:
        batch=move_batch(batch,device,charge_normalizer,encoder=encoder,augment=False)
        predictions.append(target_normalizer.inverse(model(batch)).cpu())
        targets.append(batch["target"].cpu()); masks.append(batch["target_mask"].cpu())
    prediction=torch.cat(predictions); target=torch.cat(targets); mask=torch.cat(masks)
    normalized_target=target_normalizer.transform(target,mask)
    normalized_prediction=target_normalizer.transform(prediction,mask)
    loss=masked_family_smooth_l1(normalized_prediction,normalized_target,mask,family=family.name)
    return float(loss), regression_metrics(prediction,target,mask,family)


def component_parameters(model: CompositionChargeModel) -> dict[str, int]:
    names=("composition_encoder","charge_encoder","composition_adapter","charge_adapter","fusion","head")
    counts={name:sum(p.numel() for p in getattr(model,name).parameters() if p.requires_grad) for name in names}
    counts["total"]=sum(counts.values())
    return counts


def main() -> None:
    args=arguments(); spec=load_benchmark_spec(args.spec)
    training_seed=spec.seed if args.training_seed is None else args.training_seed
    if args.family not in spec.target_families: raise ValueError(args.family)
    family=spec.target_families[args.family]
    run_dir=validate_heavy_artifact_path(args.run_dir,spec)
    checkpoint_dir=validate_heavy_artifact_path(args.checkpoint_dir,spec)
    cache_dir=validate_data_input_path(
        Path(args.cache_root)/CACHE_SUBDIR[args.encoder], spec
    )
    for path in (run_dir,checkpoint_dir):
        if path.exists() and any(path.iterdir()): raise FileExistsError(path)
        path.mkdir(parents=True,exist_ok=True)
    set_seed(training_seed); device=torch.device(args.device); index=BenchmarkIndex(spec)
    target,mask=gather_targets(index,family,"train")
    target_normalizer=TargetNormalizer.fit(target,mask)
    charge_normalizer=load_or_fit_charge_normalizer(cache_dir,index)
    datasets={split:ChargeConditionDataset(index,split,family,cache_dir) for split in ("train","validation")}
    default_batch=16
    batch_size=args.batch_size or default_batch
    eval_batch_size=args.eval_batch_size or default_batch*2
    loaders={
        "train":DataLoader(datasets["train"],batch_size=batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0),
        "validation":DataLoader(datasets["validation"],batch_size=eval_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0),
    }
    model=CompositionChargeModel(family.output_shape,encoder_name=args.encoder,latent_dim=spec.latent_dim,hidden_dim=spec.head_hidden_dim,dropout=spec.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    config={
        "status":"validation_only_training","information_condition":"C+rho",
        "family":family.name,"targets":list(family.targets),"output_shape":list(family.output_shape),
        "dataset":str(spec.benchmark_root),"cache":str(cache_dir),
        "split_seed":spec.seed,"training_seed":training_seed,
        "composition_encoder":"ElementFractionSetEncoder_v1","charge_encoder":args.encoder,
        "input_contract":json.loads((cache_dir/"metadata.json").read_text()),
        "charge_normalization":charge_normalizer.to_dict(),
        "periodic_translation_augmentation":True,
        "fusion":"common_composition_anchor_gated_residual","head":"common_family_head",
        "optimizer":"AdamW","lr":args.lr,"weight_decay":args.weight_decay,
        "epochs":args.epochs,"patience":args.patience,"batch_size":batch_size,
        "parameter_counts":component_parameters(model),"test_evaluated":False,
    }
    dump(run_dir/"config.json",config)
    torch.save({"target_mean":target_normalizer.mean,"target_std":target_normalizer.std,
                "charge_normalizer":charge_normalizer.to_dict()},checkpoint_dir/"normalizers.pt")
    best=math.inf; best_epoch=0; stale=0; history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loaders["train"]:
            batch=move_batch(batch,device,charge_normalizer,encoder=args.encoder,augment=True)
            normalized=target_normalizer.transform(batch["target"],batch["target_mask"])
            prediction=model(batch)
            loss=masked_family_smooth_l1(prediction,normalized,batch["target_mask"],family=family.name)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        val_loss,val_metrics=evaluate(model,loaders["validation"],target_normalizer,charge_normalizer,family,device,args.encoder)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"lr":scheduler.get_last_lr()[0],"elapsed_sec":time.time()-started}
        history.append(row)
        if val_loss<best:
            best=val_loss; best_epoch=epoch; stale=0
            torch.save({"model":model.state_dict(),"epoch":epoch,"validation_loss":val_loss,"config":config},checkpoint_dir/"best.pt")
            dump(run_dir/"best_validation_metrics.json",val_metrics)
        else: stale+=1
        if epoch==1 or epoch%10==0: print(json.dumps(row,sort_keys=True),flush=True)
        if stale>=args.patience: break
    with (run_dir/"history.jsonl").open("w",encoding="utf-8") as handle:
        for row in history: handle.write(json.dumps(row,sort_keys=True)+"\n")
    dump(run_dir/"status.json",{"status":"validation_complete","best_epoch":best_epoch,"best_validation_loss":best,"test_evaluated":False})
    print(json.dumps({"status":"validation_complete","encoder":args.encoder,"family":family.name,"best_epoch":best_epoch,"best_validation_loss":best}),flush=True)


if __name__=="__main__": main()
