from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetNormalizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, target: Tensor, mask: Tensor, eps: float = 1.0e-8) -> "TargetNormalizer":
        if target.shape != mask.shape:
            raise ValueError("target and mask shapes must match")
        if target.ndim < 2:
            raise ValueError("target must include batch and output axes")
        count = mask.sum(dim=0).clamp_min(1)
        masked = torch.where(mask, target, torch.zeros_like(target))
        mean = masked.sum(dim=0) / count
        centered = torch.where(mask, target - mean, torch.zeros_like(target))
        variance = centered.square().sum(dim=0) / count
        std = variance.sqrt()
        std = torch.where(std > eps, std, torch.ones_like(std))
        return cls(mean=mean, std=std)

    def transform(self, target: Tensor, mask: Tensor) -> Tensor:
        normalized = (target - self.mean.to(target)) / self.std.to(target)
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def inverse(self, normalized: Tensor) -> Tensor:
        return normalized * self.std.to(normalized) + self.mean.to(normalized)


def masked_family_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    family: str,
    beta: float = 1.0,
) -> Tensor:
    """Family-aware loss with structure- and target-balanced reductions."""

    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    element_loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    element_loss = torch.where(mask, element_loss, torch.zeros_like(element_loss))

    if family == "electronic_scalars":
        if prediction.ndim != 2 or prediction.shape[1] != 4:
            raise ValueError("electronic_scalars expects [batch, 4]")
        counts = mask.sum(dim=0)
        valid_targets = counts > 0
        if not torch.any(valid_targets):
            raise ValueError("batch contains no valid electronic scalar labels")
        per_target = element_loss.sum(dim=0) / counts.clamp_min(1)
        return per_target[valid_targets].mean()

    if family in {"dos_curve", "band_structure"}:
        flat_loss = element_loss.flatten(start_dim=1)
        flat_mask = mask.flatten(start_dim=1)
        counts = flat_mask.sum(dim=1)
        valid_samples = counts > 0
        if not torch.any(valid_samples):
            raise ValueError("batch contains no valid vector labels")
        per_sample = flat_loss.sum(dim=1) / counts.clamp_min(1)
        return per_sample[valid_samples].mean()

    if family == "formation":
        count = mask.sum()
        if count == 0:
            raise ValueError("batch contains no valid formation labels")
        return element_loss.sum() / count

    raise ValueError(f"unknown family {family!r}")
