"""Global-spin-invariant loss and unified metrics for Task 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .pretrain_data import EXPECTED_ATOMS


def _validate_shapes(
    site_predictions: Tensor,
    cell_predictions_per_atom: Tensor,
    site_targets: Tensor,
    cell_targets: Tensor,
) -> None:
    if site_predictions.ndim != 2 or site_predictions.shape[1] != EXPECTED_ATOMS:
        raise ValueError("site predictions must have shape [B_parent,100]")
    if site_targets.shape != site_predictions.shape:
        raise ValueError("site prediction and target shapes disagree")
    if cell_predictions_per_atom.shape != (site_predictions.shape[0],):
        raise ValueError("cell predictions must have shape [B_parent]")
    if cell_targets.shape != cell_predictions_per_atom.shape:
        raise ValueError("cell prediction and target shapes disagree")
    values = (site_predictions, cell_predictions_per_atom, site_targets, cell_targets)
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("magnetic predictions or targets contain NaN/Inf")


def global_spin_flip_huber(
    site_predictions: Tensor,
    cell_predictions_per_atom: Tensor,
    site_targets: Tensor,
    cell_targets: Tensor,
    *,
    delta: float = 0.5,
    cell_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Apply one shared +1/-1 orientation choice to each full configuration."""

    _validate_shapes(
        site_predictions,
        cell_predictions_per_atom,
        site_targets,
        cell_targets,
    )
    if delta <= 0 or cell_weight < 0:
        raise ValueError("Huber delta must be positive and cell_weight non-negative")
    cell_targets_per_atom = cell_targets / float(EXPECTED_ATOMS)

    plus_site = F.huber_loss(
        site_predictions, site_targets, reduction="none", delta=float(delta)
    ).mean(dim=1)
    minus_site = F.huber_loss(
        -site_predictions, site_targets, reduction="none", delta=float(delta)
    ).mean(dim=1)
    plus_cell = F.huber_loss(
        cell_predictions_per_atom,
        cell_targets_per_atom,
        reduction="none",
        delta=float(delta),
    )
    minus_cell = F.huber_loss(
        -cell_predictions_per_atom,
        cell_targets_per_atom,
        reduction="none",
        delta=float(delta),
    )
    plus_total = plus_site + float(cell_weight) * plus_cell
    minus_total = minus_site + float(cell_weight) * minus_cell
    use_minus = minus_total < plus_total
    chosen_sign = torch.where(
        use_minus,
        -torch.ones_like(plus_total),
        torch.ones_like(plus_total),
    )
    chosen_site = torch.where(use_minus, minus_site, plus_site)
    chosen_cell = torch.where(use_minus, minus_cell, plus_cell)
    per_parent = torch.where(use_minus, minus_total, plus_total)
    loss = per_parent.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("global-spin-invariant magnetic loss is non-finite")
    return {
        "loss": loss,
        "per_parent_loss": per_parent,
        "site_loss": chosen_site.mean(),
        "cell_loss": chosen_cell.mean(),
        "chosen_sign": chosen_sign,
        "minus_selected": use_minus,
        "plus_total": plus_total,
        "minus_total": minus_total,
    }


def align_predictions(
    site_predictions: Tensor,
    cell_predictions_per_atom: Tensor,
    site_targets: Tensor,
    cell_targets: Tensor,
    *,
    delta: float = 0.5,
    cell_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    selection = global_spin_flip_huber(
        site_predictions,
        cell_predictions_per_atom,
        site_targets,
        cell_targets,
        delta=delta,
        cell_weight=cell_weight,
    )
    sign = selection["chosen_sign"]
    aligned_sites = site_predictions * sign[:, None]
    aligned_cells = cell_predictions_per_atom * sign * float(EXPECTED_ATOMS)
    return aligned_sites, aligned_cells, sign


@dataclass
class MagneticMetricAccumulator:
    delta: float = 0.5
    cell_weight: float = 1.0
    parent_count: int = 0
    site_count: int = 0
    active_count: int = 0
    minus_count: int = 0
    site_abs_sum: float = 0.0
    active_abs_sum: float = 0.0
    cell_abs_sum: float = 0.0
    loss_sum: float = 0.0
    site_loss_sum: float = 0.0
    cell_loss_sum: float = 0.0

    @torch.no_grad()
    def update(
        self,
        predictions: dict[str, Tensor],
        batch: dict[str, Any],
    ) -> None:
        site_predictions = predictions["site_moments"]
        cell_predictions = predictions["cell_moment_per_atom"]
        site_targets = batch["site_targets"]
        cell_targets = batch["cell_targets"]
        active_mask = batch["active_mask"].bool()
        if active_mask.shape != site_targets.shape:
            raise ValueError("active-mask shape disagrees with site targets")
        selection = global_spin_flip_huber(
            site_predictions,
            cell_predictions,
            site_targets,
            cell_targets,
            delta=self.delta,
            cell_weight=self.cell_weight,
        )
        sign = selection["chosen_sign"]
        aligned_sites = site_predictions * sign[:, None]
        aligned_cells = cell_predictions * sign * float(EXPECTED_ATOMS)
        absolute_site = (aligned_sites - site_targets).abs()
        absolute_cell = (aligned_cells - cell_targets).abs()
        parents = int(site_targets.shape[0])
        self.parent_count += parents
        self.site_count += int(site_targets.numel())
        active = int(active_mask.sum())
        self.active_count += active
        self.minus_count += int(selection["minus_selected"].sum())
        self.site_abs_sum += float(absolute_site.sum().cpu())
        self.active_abs_sum += float(absolute_site[active_mask].sum().cpu())
        self.cell_abs_sum += float(absolute_cell.sum().cpu())
        self.loss_sum += float(selection["per_parent_loss"].sum().cpu())
        self.site_loss_sum += float(selection["site_loss"].cpu()) * parents
        self.cell_loss_sum += float(selection["cell_loss"].cpu()) * parents

    def compute(self) -> dict[str, float]:
        if self.parent_count <= 0 or self.site_count <= 0:
            raise ValueError("no magnetic predictions were accumulated")
        if self.active_count <= 0:
            raise ValueError("evaluation set contains no active magnetic sites")
        return {
            "loss": self.loss_sum / self.parent_count,
            "site_huber": self.site_loss_sum / self.parent_count,
            "cell_huber_per_atom": self.cell_loss_sum / self.parent_count,
            "site_mae_muB": self.site_abs_sum / self.site_count,
            "active_site_mae_muB": self.active_abs_sum / self.active_count,
            "cell_mae_muB": self.cell_abs_sum / self.parent_count,
            "minus_orientation_fraction": self.minus_count / self.parent_count,
            "parents": float(self.parent_count),
            "sites": float(self.site_count),
            "active_sites": float(self.active_count),
        }


def move_magnetic_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cpu_index_keys = {"structure_node_offsets", "structure_edge_offsets"}
    return {
        key: (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value) and key not in cpu_index_keys
            else value
        )
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_magnetic(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    delta: float = 0.5,
    cell_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    accumulator = MagneticMetricAccumulator(delta=delta, cell_weight=cell_weight)
    for batch in loader:
        moved = move_magnetic_batch(batch, device)
        predictions = model(moved)
        accumulator.update(predictions, moved)
    return accumulator.compute()


__all__ = [
    "MagneticMetricAccumulator",
    "align_predictions",
    "evaluate_magnetic",
    "global_spin_flip_huber",
    "move_magnetic_batch",
]
