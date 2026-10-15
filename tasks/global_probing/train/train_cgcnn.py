#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from property_prediction.data import BenchmarkIndex
from property_prediction.metrics import regression_metrics
from property_prediction.models.structure_conditions import CompositionCGCNNModel
from property_prediction.objectives import TargetNormalizer, masked_family_smooth_l1
from property_prediction.spec import load_benchmark_spec
from property_prediction.storage import validate_heavy_artifact_path
from property_prediction.structure_data import (
    CGCNNConditionDataset,
    collate_cgcnn,
    load_atom_feature_table,
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
DEFAULT_CACHE = ARTIFACT_ROOT / "cache/structure_cgcnn_r6_n32_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CGCNN structure baseline.")
    parser.add_argument("family")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", dest="training_seed", type=int, default=None)
    parser.set_defaults(
        spec=DEFAULT_SPEC, cache_dir=DEFAULT_CACHE, epochs=140, batch_size=32,
        eval_batch_size=64, num_workers=4, lr=1.0e-3,
        weight_decay=1.0e-5, patience=25,
    )
    args = parser.parse_args()
    args.run_dir = args.output_dir / "run"
    args.checkpoint_dir = args.output_dir / "checkpoints"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_targets(index: BenchmarkIndex, family, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [index.family_target(int(i), family) for i in index.indices(split)]
    return torch.from_numpy(np.stack([x[0] for x in pairs])), torch.from_numpy(np.stack([x[1] for x in pairs]))


def move_batch(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        elif key == "crystal_atom_indices":
            result[key] = [item.to(device, non_blocking=True) for item in value]
        else:
            result[key] = value
    return result


@torch.no_grad()
def evaluate(model, loader, normalizer, family, device):
    model.eval(); predictions=[]; targets=[]; masks=[]
    for batch in loader:
        batch = move_batch(batch, device)
        predictions.append(normalizer.inverse(model(batch)).cpu())
        targets.append(batch["target"].cpu()); masks.append(batch["target_mask"].cpu())
    prediction=torch.cat(predictions); target=torch.cat(targets); mask=torch.cat(masks)
    normalized_target=normalizer.transform(target,mask)
    normalized_prediction=normalizer.transform(prediction,mask)
    loss=masked_family_smooth_l1(normalized_prediction,normalized_target,mask,family=family.name)
    return float(loss), regression_metrics(prediction,target,mask,family)


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    args=arguments(); spec=load_benchmark_spec(args.spec)
    training_seed=spec.seed if args.training_seed is None else args.training_seed
    if args.family not in spec.target_families: raise ValueError(args.family)
    family=spec.target_families[args.family]
    run_dir=validate_heavy_artifact_path(args.run_dir,spec)
    checkpoint_dir=validate_heavy_artifact_path(args.checkpoint_dir,spec)
    cache_dir=validate_heavy_artifact_path(args.cache_dir,spec)
    for path in (run_dir,checkpoint_dir):
        if path.exists() and any(path.iterdir()): raise FileExistsError(path)
        path.mkdir(parents=True,exist_ok=True)
    set_seed(training_seed); device=torch.device(args.device); index=BenchmarkIndex(spec)
    train_target,train_mask=gather_targets(index,family,"train")
    normalizer=TargetNormalizer.fit(train_target,train_mask)
    datasets={split:CGCNNConditionDataset(index,split,family,cache_dir) for split in ("train","validation")}
    collate=partial(collate_cgcnn,cutoff=6.0,step=0.2,atom_table=load_atom_feature_table())
    loaders={
        "train":DataLoader(datasets["train"],batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,collate_fn=collate),
        "validation":DataLoader(datasets["validation"],batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,collate_fn=collate),
    }
    model=CompositionCGCNNModel(family.output_shape,latent_dim=spec.latent_dim,hidden_dim=spec.head_hidden_dim,dropout=spec.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    config={
        "status":"validation_only_training","information_condition":"C+S","family":family.name,
        "targets":list(family.targets),"output_shape":list(family.output_shape),
        "dataset":str(spec.benchmark_root),"cache":str(cache_dir),
        "split_seed":spec.seed,"training_seed":training_seed,
        "composition_encoder":"ElementFractionSetEncoder_v1","structure_encoder":"upstream_CGCNN_trunk",
        "fusion":"common_composition_anchor_gated_residual","head":"common_family_head",
        "cutoff_A":6.0,"max_neighbors":32,"gaussian_step_A":0.2,
        "optimizer":"AdamW","lr":args.lr,"weight_decay":args.weight_decay,
        "epochs":args.epochs,"patience":args.patience,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_evaluated":False,
    }
    dump(run_dir/"config.json",config)
    torch.save({"mean":normalizer.mean,"std":normalizer.std},checkpoint_dir/"normalizer.pt")
    best=math.inf; best_epoch=0; stale=0; history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); batch_losses=[]
        for batch in loaders["train"]:
            batch=move_batch(batch,device)
            target=normalizer.transform(batch["target"],batch["target_mask"])
            prediction=model(batch)
            loss=masked_family_smooth_l1(prediction,target,batch["target_mask"],family=family.name)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            batch_losses.append(float(loss.detach()))
        scheduler.step()
        val_loss,val_metrics=evaluate(model,loaders["validation"],normalizer,family,device)
        record={"epoch":epoch,"train_loss":float(np.mean(batch_losses)),"validation_loss":val_loss,"lr":scheduler.get_last_lr()[0],"elapsed_sec":time.time()-started}
        history.append(record)
        if val_loss<best:
            best=val_loss; best_epoch=epoch; stale=0
            torch.save({"model":model.state_dict(),"epoch":epoch,"validation_loss":val_loss,"normalizer_mean":normalizer.mean,"normalizer_std":normalizer.std,"config":config},checkpoint_dir/"best.pt")
            dump(run_dir/"best_validation_metrics.json",val_metrics)
        else: stale+=1
        if epoch==1 or epoch%10==0: print(json.dumps(record,sort_keys=True),flush=True)
        if stale>=args.patience: break
    with (run_dir/"history.jsonl").open("w",encoding="utf-8") as handle:
        for row in history: handle.write(json.dumps(row,sort_keys=True)+"\n")
    dump(run_dir/"status.json",{"status":"validation_complete","best_epoch":best_epoch,"best_validation_loss":best,"test_evaluated":False})
    print(json.dumps({"status":"validation_complete","family":family.name,"best_epoch":best_epoch,"best_validation_loss":best}),flush=True)


if __name__=="__main__": main()
