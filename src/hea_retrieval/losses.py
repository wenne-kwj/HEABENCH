"""Contrastive objectives for the HEA retrieval task."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _check_embeddings(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"embeddings must be [B,D], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape != b.shape:
        raise ValueError(f"paired embeddings must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] < 1:
        raise ValueError("contrastive batch must not be empty")
    return F.normalize(a, dim=-1), F.normalize(b, dim=-1)


def symmetric_infonce(
    structure: Tensor,
    charge: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric in-batch InfoNCE (structure→charge and charge→structure)."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    structure, charge = _check_embeddings(structure, charge)
    logits = structure @ charge.transpose(0, 1) / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.transpose(0, 1), target))




class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, structure: Tensor, charge: Tensor) -> Tensor:
        return symmetric_infonce(structure, charge, self.temperature)


__all__ = ["symmetric_infonce", "SymmetricInfoNCE"]
