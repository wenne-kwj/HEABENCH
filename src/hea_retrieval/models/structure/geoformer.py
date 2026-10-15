"""HEA adapter for the official GeoFormer encoder.

Only the upstream ``GeoformerEncoder`` is instantiated here.  Its energy
decoder is intentionally not used: node features are masked-mean pooled and
then passed through the HEA projection head.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from ..common import ProjectionHead, RoleEmbedding, assert_finite, masked_mean


REPO_ROOT = Path(__file__).resolve().parents[3]
GEOFORMER_ROOT = Path(os.environ.get("HEABENCH_GEOFORMER_ROOT", str(REPO_ROOT / "third_party" / "geoformer")))


def _load_geoformer():
    root = str(GEOFORMER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Import lazily so data-only tools and the PointVector experiment do not
    # require HuggingFace/PyG at module import time.
    from geoformer.model.configuration_geoformer import GeoformerConfig
    from geoformer.model.modeling_geoformer import GeoformerEncoder

    return GeoformerConfig, GeoformerEncoder


class GeoFormerStructureEncoder(nn.Module):
    """Encode a padded HEA structure patch into a normalized 256-D vector."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        ffn_embedding_dim: int = 1024,
        num_layers: int = 9,
        num_heads: int = 8,
        cutoff: float = 5.0,
        num_rbf: int = 32,
        max_z: int = 100,
        projection_hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.0,
        use_role_embedding: bool = True,
    ) -> None:
        super().__init__()
        GeoformerConfig, GeoformerEncoder = _load_geoformer()
        config = GeoformerConfig(
            max_z=max_z,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            cutoff=cutoff,
            num_rbf=num_rbf,
            rbf_trainable=True,
            norm_type="max_min",
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            activation_function="silu",
            decoder_type="scalar",
            num_classes=embedding_dim,
            pad_token_id=0,
        )
        self.backbone = GeoformerEncoder(config)
        self.backbone_dim = embedding_dim
        self.use_role_embedding = use_role_embedding
        self.role_embedding = RoleEmbedding(embedding_dim) if use_role_embedding else None
        self.projection = ProjectionHead(
            embedding_dim, hidden_dim=projection_hidden_dim, out_dim=output_dim, dropout=0.1
        )
        self.output_dim = output_dim

    def forward(
        self,
        z: Tensor,
        xyz: Tensor,
        role: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if z.ndim != 2 or xyz.ndim != 3 or xyz.shape[:2] != z.shape or xyz.shape[-1] != 3:
            raise ValueError(f"expected z [B,N], xyz [B,N,3], got {tuple(z.shape)}, {tuple(xyz.shape)}")
        if mask is None:
            mask = z.ne(0)
        else:
            mask = mask.bool()
        # Padding must be represented by the official pad token, even if a
        # caller supplied a mask with a nonzero placeholder.
        z = z.masked_fill(~mask, 0).long()
        x, _edge = self.backbone(z=z, pos=xyz)
        if self.role_embedding is not None and role is not None:
            x = x + self.role_embedding(role, mask)
        pooled = masked_mean(x, mask)
        out = self.projection(pooled)
        assert_finite("GeoFormer embedding", out)
        return out


__all__ = ["GeoFormerStructureEncoder"]
