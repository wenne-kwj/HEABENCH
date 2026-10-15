"""Small, dependency-light utilities shared by all HEA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def masked_mean(values: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Mean-pool ``values`` while ignoring false entries in ``mask``.

    ``mask`` may have one fewer dimension than ``values``.  Empty rows are
    rejected explicitly; silently returning a zero vector would hide malformed
    patches and make the contrastive loss unstable.
    """

    if mask.dtype is not torch.bool:
        mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=dim)
    if torch.any(denom <= 0):
        raise ValueError("masked_mean received an empty row")
    return (values * weights).sum(dim=dim) / denom


def assert_finite(name: str, value: Tensor) -> None:
    """Raise a useful error when a model emits NaN/Inf."""

    if not torch.isfinite(value).all():
        bad = (~torch.isfinite(value)).sum().item()
        raise FloatingPointError(f"{name} contains {bad} non-finite values")


class ProjectionHead(nn.Module):
    """Independent modality projection followed by optional normalization.

    The upstream classification heads use a large dropout/BN stack.  For this
    relatively small HEA dataset we use LayerNorm and a modest dropout so that
    batch statistics do not depend on variable-size structure padding.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class RoleEmbedding(nn.Module):
    """Map HEA roles to a learned node feature.

    The stored shard uses ``role=0`` for ordinary neighbours and ``1/2`` for
    the two pair endpoints.  Adapters reserve index 0 for padding and map
    valid values to ``1/2/3`` via :func:`encode_roles`.
    """

    PAD = 0
    OTHER = 1
    ENDPOINT_A = 2
    ENDPOINT_B = 3

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, dim, padding_idx=self.PAD)

    @staticmethod
    def encode_roles(role: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        role = role.long()
        # Stored role values are 0, 1, 2.  Preserve padded zeros when a caller
        # supplies a padded tensor; valid ordinary neighbours become index 1.
        encoded = role + 1
        if valid_mask is not None:
            encoded = torch.where(valid_mask.bool(), encoded, torch.zeros_like(encoded))
        if encoded.numel() and int(encoded.max()) > 3:
            raise ValueError("role values must be 0, 1, or 2 (plus padding)")
        return encoded

    def forward(self, role: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        return self.embedding(self.encode_roles(role, valid_mask))


@dataclass
class EncoderOutput:
    """Optional debugging payload returned by adapters when requested."""

    embedding: Tensor
    backbone_dim: int
    metadata: dict[str, Any]


class RetrievalModel(nn.Module):
    """Two-tower model with independently projected normalized embeddings."""

    def __init__(self, structure_encoder: nn.Module, charge_encoder: nn.Module) -> None:
        super().__init__()
        self.structure_encoder = structure_encoder
        self.charge_encoder = charge_encoder

    def encode_structure(self, batch: dict[str, Tensor]) -> Tensor:
        return self.structure_encoder(
            batch["structure_z"],
            batch["structure_xyz"],
            batch.get("structure_role"),
            batch.get("structure_mask"),
        )

    def encode_charge_tensors(
        self,
        xyz: Tensor,
        rho_total: Tensor,
        rho_magnetization: Optional[Tensor] = None,
        *,
        rho_total_raw: Optional[Tensor] = None,
        point_mask: Optional[Tensor] = None,
        volume_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.charge_encoder(
            xyz,
            rho_total,
            rho_magnetization,
            rho_total_raw=rho_total_raw,
            point_mask=point_mask,
            volume_mask=volume_mask,
        )

    def encode_charge(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encode_charge_tensors(
            batch["charge_xyz"],
            batch["rho_total"],
            batch.get("rho_magnetization"),
            rho_total_raw=batch.get("rho_total_raw"),
            point_mask=batch.get("charge_mask"),
            volume_mask=batch.get("charge_volume_mask"),
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        return self.encode_structure(batch), self.encode_charge(batch)
