"""Standalone test evaluation for the HEA local structure--charge retrievers.

The training entry point evaluates only the validation split while fitting.  This
module loads a frozen ``best.pt`` checkpoint and evaluates a held-out split
without touching the optimizer or the source data.  The primary result is
global canonical-anchor retrieval; the explicit 11-way candidate task and
fixed-rotation robustness are reported as secondary diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .candidate_data import CandidateGroupDataset
from .data import DEFAULT_DATA_ROOT, RetrievalPatchDataset, collate_patches
from .models.factory import build_model


from .paths import DEFAULT_EVAL_ROOT


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _log(path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{stamp} UTC] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )


def _fixed_rotations() -> list[tuple[str, torch.Tensor | None]]:
    """Return deterministic proper rotations used for robustness checks.

    The first entry is the unrotated canonical test result.  Subsequent
    matrices are simple axis rotations/permutations; the same matrix is
    applied to structure and charge coordinates by ``collate_patches``.
    """

    def t(values: list[list[float]]) -> torch.Tensor:
        matrix = torch.tensor(values, dtype=torch.float32)
        if not torch.allclose(matrix @ matrix.T, torch.eye(3), atol=1e-6):
            raise ValueError("fixed rotation is not orthogonal")
        if float(torch.linalg.det(matrix)) <= 0:
            raise ValueError("fixed rotation must preserve orientation")
        return matrix

    return [
        ("identity", None),
        ("z90", t([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        ("z180", t([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),
        ("x90", t([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
        ("x180", t([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
        ("y90", t([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
        ("y180", t([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])),
        ("cyclic120", t([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
    ]


@torch.inference_mode()
def _encode_split(
    model: torch.nn.Module,
    dataset: RetrievalPatchDataset,
    *,
    stats: dict[str, float],
    batch_size: int,
    device: torch.device,
    rotation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    """Encode one split with a fixed, shared structure/charge rotation."""

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_patches(
            samples,
            rho_mean=float(stats["rho_mean"]),
            rho_std=float(stats["rho_std"]),
            rho_mag_mean=float(stats.get("rho_mag_mean", 0.0)),
            rho_mag_std=float(stats.get("rho_mag_std", 1.0)),
            rotate=False,
            rotation=rotation,
            charge_points=None,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    structure_chunks: list[torch.Tensor] = []
    charge_chunks: list[torch.Tensor] = []
    patch_ids: list[str] = []
    parent_ids: list[str] = []
    pair_ids: list[str] = []
    model.eval()
    for batch in loader:
        moved = _move_batch(batch, device)
        structure, charge = model(moved)
        if not torch.isfinite(structure).all() or not torch.isfinite(charge).all():
            raise FloatingPointError("non-finite embedding emitted during evaluation")
        # Projection heads already normalize, but normalize once more so that
        # the similarity definition is explicit and robust to future adapters.
        structure_chunks.append(F.normalize(structure, dim=-1).cpu())
        charge_chunks.append(F.normalize(charge, dim=-1).cpu())
        patch_ids.extend(str(x) for x in batch["patch_id"])
        parent_ids.extend(str(x) for x in batch["parent_id"])
        pair_ids.extend(str(x) for x in batch["pair_id"])
    if not patch_ids:
        raise ValueError("evaluation split is empty")
    return (
        torch.cat(structure_chunks, dim=0),
        torch.cat(charge_chunks, dim=0),
        patch_ids,
        parent_ids,
        pair_ids,
    )


def _one_direction_metrics(
    query: torch.Tensor,
    candidates: torch.Tensor,
    target_indices: list[int],
    *,
    prefix: str,
    chunk_size: int = 512,
) -> tuple[dict[str, float], np.ndarray]:
    if query.ndim != 2 or candidates.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if query.shape[1] != candidates.shape[1] or len(target_indices) != query.shape[0]:
        raise ValueError("embedding dimensions or target count disagree")
    n = int(query.shape[0])
    ranks = torch.empty(n, dtype=torch.int64)
    hit = {1: 0, 5: 0, 10: 0}
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        scores = query[start:stop] @ candidates.T
        local_target = torch.as_tensor(target_indices[start:stop], dtype=torch.long)
        target_scores = scores[torch.arange(stop - start), local_target]
        # Rank is one plus the number of strictly larger similarities.  Exact
        # ties are exceptionally unlikely for learned float embeddings; this
        # convention is deterministic and mildly conservative for ties.
        local_ranks = 1 + (scores > target_scores.unsqueeze(1)).sum(dim=1)
        ranks[start:stop] = local_ranks
        for k in hit:
            hit[k] += int((local_ranks <= k).sum())
    rank_float = ranks.to(torch.float64)
    values = {
        f"{prefix}_recall@{k}": hit[k] / n for k in hit
    }
    values.update(
        {
            f"{prefix}_mrr": float((1.0 / rank_float).mean()),
            f"{prefix}_mean_rank": float(rank_float.mean()),
            f"{prefix}_median_rank": float(rank_float.median()),
            f"{prefix}_n": float(n),
        }
    )
    return values, ranks.numpy()


def global_metrics(
    structure: torch.Tensor,
    charge: torch.Tensor,
    structure_ids: list[str],
    charge_ids: list[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute all-test canonical retrieval in both directions."""

    if len(structure_ids) != len(charge_ids):
        raise ValueError("structure and charge embedding counts disagree")
    if set(structure_ids) != set(charge_ids):
        raise ValueError("structure and charge patch-id sets disagree")
    charge_index = {patch_id: i for i, patch_id in enumerate(charge_ids)}
    structure_index = {patch_id: i for i, patch_id in enumerate(structure_ids)}
    s_targets = [charge_index[patch_id] for patch_id in structure_ids]
    c_targets = [structure_index[patch_id] for patch_id in charge_ids]
    s_values, s_ranks = _one_direction_metrics(
        structure, charge, s_targets, prefix="s2c"
    )
    c_values, c_ranks = _one_direction_metrics(
        charge, structure, c_targets, prefix="c2s"
    )
    result = {**s_values, **c_values, "n": float(len(structure_ids))}
    for k in (1, 5, 10):
        result[f"mean_recall@{k}"] = 0.5 * (
            result[f"s2c_recall@{k}"] + result[f"c2s_recall@{k}"]
        )
    result["mean_mrr"] = 0.5 * (result["s2c_mrr"] + result["c2s_mrr"])
    result["mean_rank"] = 0.5 * (result["s2c_mean_rank"] + result["c2s_mean_rank"])
    result["median_rank"] = 0.5 * (
        result["s2c_median_rank"] + result["c2s_median_rank"]
    )
    return result, s_ranks, c_ranks


def _candidate_metrics(
    structure: torch.Tensor,
    charge: torch.Tensor,
    patch_index: dict[str, int],
    groups: Iterable[Any],
    *,
    temperature: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate explicit 11-way groups directly from cached embeddings."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hit: dict[str, dict[int, int]] = {
        "all": {1: 0, 5: 0, 10: 0},
        "structure_to_charge": {1: 0, 5: 0, 10: 0},
        "charge_to_structure": {1: 0, 5: 0, 10: 0},
    }
    counts = {"all": 0, "structure_to_charge": 0, "charge_to_structure": 0}
    loss_sum = 0.0
    rank_rows: list[dict[str, Any]] = []
    for meta in groups:
        q_idx = patch_index.get(meta.query_patch_id)
        if q_idx is None:
            raise KeyError(f"candidate query is absent from encoded split: {meta.query_patch_id}")
        candidate_indices = []
        for patch_id in meta.candidate_patch_ids:
            idx = patch_index.get(patch_id)
            if idx is None:
                raise KeyError(f"candidate is absent from encoded split: {patch_id}")
            candidate_indices.append(idx)
        if meta.direction == "structure_to_charge":
            query = structure[q_idx]
            candidates = charge[candidate_indices]
        elif meta.direction == "charge_to_structure":
            query = charge[q_idx]
            candidates = structure[candidate_indices]
        else:
            raise ValueError(f"unknown candidate direction {meta.direction!r}")
        logits = query @ candidates.T
        target = int(meta.target)
        target_score = logits[target]
        rank = 1 + int((logits > target_score).sum())
        loss_sum += float(torch.logsumexp(logits / temperature, dim=0) - logits[target] / temperature)
        counts["all"] += 1
        counts[meta.direction] += 1
        for k in (1, 5, 10):
            if rank <= k:
                hit["all"][k] += 1
                hit[meta.direction][k] += 1
        rank_rows.append(
            {
                "query_patch_id": meta.query_patch_id,
                "direction": meta.direction,
                "rank": rank,
            }
        )
    if counts["all"] == 0:
        raise ValueError("test candidate split is empty")
    result: dict[str, float] = {
        "candidate_groups": float(counts["all"]),
        "candidate_loss": loss_sum / counts["all"],
    }
    for direction, n in counts.items():
        prefix = "candidate" if direction == "all" else direction
        for k in (1, 5, 10):
            result[f"{prefix}_recall@{k}"] = hit[direction][k] / max(n, 1)
        if direction != "all":
            result[f"{direction}_groups"] = float(n)
    return result, rank_rows


def _load_checkpoint(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any], dict[str, float]]:
    run_dir = checkpoint.parent
    config_path = run_dir / "config.json"
    stats_path = run_dir / "density_stats.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    model = build_model(
        config["structure"],
        config["charge"],
        output_dim=int(config.get("output_dim", 256)),
        structure_kwargs=config.get("structure_kwargs", {}),
        charge_kwargs=config.get("charge_kwargs", {}),
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch versions do not expose weights_only.
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain a state dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model, config, {str(k): float(v) for k, v in stats.items()}


def _expected_test_counts(data_root: Path, split: str) -> tuple[int, int]:
    """Return anchor and candidate counts from the manifests for audit logs."""

    anchor_ds = RetrievalPatchDataset(data_root, split=split, cache_size=1)
    candidate_ds = CandidateGroupDataset(data_root, split=split, cache_size=1, strict=True)
    return len(anchor_ds), len(candidate_ds)


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_run = checkpoint.parent
    model, train_config, stats = _load_checkpoint(checkpoint)
    data_root = Path(args.data_root or train_config.get("data_root", DEFAULT_DATA_ROOT))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    model.to(device)
    model.eval()
    if args.run_dir is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = DEFAULT_EVAL_ROOT / f"{stamp}_{train_config['structure']}_{train_config['charge']}"
    else:
        run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evaluation directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    try:
        _log(log_path, f"starting checkpoint={checkpoint} split={args.split} device={device}")
        _log(log_path, f"source_run={source_run} objective={train_config.get('objective')}")
        if not data_root.exists():
            raise FileNotFoundError(data_root)
        anchor_count, candidate_count = _expected_test_counts(data_root, args.split)
        _log(log_path, f"audit anchors={anchor_count} candidate_groups={candidate_count}")
        if args.split == "test" and anchor_count != 3904:
            raise ValueError(f"expected 3904 test anchors, found {anchor_count}")
        if args.split == "test" and candidate_count != 7808:
            raise ValueError(f"expected 7808 test candidate groups, found {candidate_count}")

        dataset = RetrievalPatchDataset(data_root, split=args.split, cache_size=args.cache_size)
        candidate_dataset = CandidateGroupDataset(
            data_root, split=args.split, cache_size=args.cache_size, strict=True
        )
        rotations = _fixed_rotations()
        if args.rotations > 0:
            rotations = rotations[: max(1, min(args.rotations, len(rotations)))]
        temperature = float(train_config.get("temperature", 0.07))
        rotation_metrics: dict[str, Any] = {}
        identity_payload: tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[str]] | None = None
        identity_global: dict[str, float] | None = None
        identity_s_ranks: np.ndarray | None = None
        identity_c_ranks: np.ndarray | None = None
        identity_candidate: dict[str, float] | None = None
        identity_candidate_rows: list[dict[str, Any]] | None = None

        for rotation_name, rotation in rotations:
            _log(log_path, f"encoding rotation={rotation_name}")
            encoded = _encode_split(
                model,
                dataset,
                stats=stats,
                batch_size=args.eval_batch_size,
                device=device,
                rotation=rotation,
            )
            structure, charge, patch_ids, parent_ids, pair_ids = encoded
            global_result, s_ranks, c_ranks = global_metrics(
                structure, charge, patch_ids, patch_ids
            )
            patch_index = {patch_id: i for i, patch_id in enumerate(patch_ids)}
            candidate_result, candidate_rows = _candidate_metrics(
                structure,
                charge,
                patch_index,
                candidate_dataset.groups,
                temperature=temperature,
            )
            rotation_metrics[rotation_name] = {
                "global": global_result,
                "candidate": candidate_result,
            }
            _log(
                log_path,
                f"rotation={rotation_name} global_mean_R1={global_result['mean_recall@1']:.6f} "
                f"global_mean_R5={global_result['mean_recall@5']:.6f} "
                f"candidate_R1={candidate_result['candidate_recall@1']:.6f}",
            )
            if rotation_name == "identity":
                identity_payload = encoded
                identity_global = global_result
                identity_s_ranks = s_ranks
                identity_c_ranks = c_ranks
                identity_candidate = candidate_result
                identity_candidate_rows = candidate_rows

        if identity_payload is None or identity_global is None:
            raise RuntimeError("identity rotation was not evaluated")
        structure, charge, patch_ids, parent_ids, pair_ids = identity_payload
        np.savez_compressed(
            run_dir / "embeddings.npz",
            patch_id=np.asarray(patch_ids),
            parent_id=np.asarray(parent_ids),
            pair_id=np.asarray(pair_ids),
            structure=structure.numpy().astype(np.float32),
            charge=charge.numpy().astype(np.float32),
        )
        if identity_s_ranks is None or identity_c_ranks is None:
            raise RuntimeError("identity ranks were not produced")
        np.savez_compressed(
            run_dir / "global_ranks.npz",
            patch_id=np.asarray(patch_ids),
            parent_id=np.asarray(parent_ids),
            pair_id=np.asarray(pair_ids),
            s2c_rank=identity_s_ranks,
            c2s_rank=identity_c_ranks,
        )
        if identity_candidate_rows is None:
            raise RuntimeError("identity candidate ranks were not produced")
        np.savez_compressed(
            run_dir / "candidate_ranks.npz",
            query_patch_id=np.asarray([r["query_patch_id"] for r in identity_candidate_rows]),
            direction=np.asarray([r["direction"] for r in identity_candidate_rows]),
            rank=np.asarray([r["rank"] for r in identity_candidate_rows], dtype=np.int64),
        )

        report = {
            "checkpoint": str(checkpoint),
            "source_training_run": str(source_run),
            "data_root": str(data_root),
            "split": args.split,
            "anchor_count": anchor_count,
            "candidate_group_count": candidate_count,
            "candidate_groups_by_direction": {
                "structure_to_charge": sum(
                    1 for g in candidate_dataset.groups if g.direction == "structure_to_charge"
                ),
                "charge_to_structure": sum(
                    1 for g in candidate_dataset.groups if g.direction == "charge_to_structure"
                ),
            },
            "temperature": temperature,
            "eval_batch_size": args.eval_batch_size,
            "rotations": [name for name, _ in rotations],
            "identity_global": identity_global,
            "identity_candidate": identity_candidate,
            "rotation_metrics": rotation_metrics,
        }
        _write_json(run_dir / "metrics.json", report)
        _write_json(
            run_dir / "eval_config.json",
            {
                "checkpoint": str(checkpoint),
                "source_training_config": train_config,
                "data_root": str(data_root),
                "split": args.split,
                "device": str(device),
                "eval_batch_size": args.eval_batch_size,
                "cache_size": args.cache_size,
                "rotations": [name for name, _ in rotations],
                "seed": args.seed,
            },
        )
        (run_dir / "_SUCCESS").write_text("evaluation completed\n", encoding="utf-8")
        _log(log_path, "completed evaluation")
        return run_dir
    except Exception as exc:
        if not (run_dir / "_SUCCESS").exists():
            (run_dir / "_FAILED").write_text(
                f"evaluation failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            _log(log_path, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen local-matching checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", dest="run_dir", default=None)
    parser.set_defaults(
        split="test", eval_batch_size=32, cache_size=2,
        rotations=8, seed=20260905,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.cache_size < 0:
        raise ValueError("--cache-size must be non-negative")
    if args.rotations <= 0:
        raise ValueError("--rotations must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    _seed_everything(parsed.seed)
    evaluate(parsed)
