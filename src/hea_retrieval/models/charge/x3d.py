"""HEA adapter for the official X-3D (PointMetaBase) backbone."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from ..common import ProjectionHead, assert_finite
from .pointvector import _activate_openpoints, _easy_dict


REPO_ROOT = Path(__file__).resolve().parents[3]
X3D_ROOT = Path(os.environ.get("HEABENCH_X3D_ROOT", str(REPO_ROOT / "third_party" / "x3d" / "openpoints")))


class _GroupAllWithIdx(nn.Module):
    """Compatibility wrapper for the frozen X-3D GroupAll implementation.

    ``SetAbstraction.forward`` always calls ``forward_idx`` and passes the
    returned index as a fourth argument, while upstream ``GroupAll`` exposes
    only ``forward(new_xyz, xyz, features)``.  The wrapper preserves the
    upstream computation and accepts that harmless extra argument locally.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward_idx(self, query_xyz, support_xyz, features=None):
        return None

    def forward(self, new_xyz, xyz, features=None, idx=None):
        return self.inner(new_xyz, xyz, features)


def _patch_group_all(backbone: nn.Module) -> None:
    for module in backbone.modules():
        grouper = getattr(module, "grouper", None)
        if grouper is not None and grouper.__class__.__name__ == "GroupAll":
            module.grouper = _GroupAllWithIdx(grouper)


def load_x3d_class():
    _activate_openpoints(X3D_ROOT)
    module = importlib.import_module("openpoints.models.backbone.pointmetabase_X3D")
    return module.PointMetaBaseEncoder_X3D


class X3DChargeEncoder(nn.Module):
    """Encode a normalized charge point cloud with X-3D."""

    def __init__(
        self,
        *,
        output_dim: int = 256,
        projection_hidden_dim: int = 512,
        in_channels: int = 4,
        use_magnetization: bool = False,
        blocks: list[int] | None = None,
        strides: list[int] | None = None,
        width: int = 32,
        radius: float = 0.15,
        radius_scaling: float = 1.5,
        nsample: int = 32,
        sa_layers: int = 1,
        sa_use_res: bool = False,
        expansion: int = 1,
        x3d_layers: list[bool] | None = None,
        ncp: str = "false",
    ) -> None:
        super().__init__()
        if use_magnetization and in_channels != 5:
            in_channels = 5
        if in_channels not in (4, 5):
            raise ValueError("X3D adapter supports 4 or 5 input channels")
        if ncp != "false":
            raise ValueError("NCP must be the exact string 'false' for this adapter")
        PointMetaBaseEncoder_X3D = load_x3d_class()
        blocks = list(blocks or [1, 1, 1, 1, 1, 1])
        strides = list(strides or [1, 2, 2, 2, 2, 1])
        x3d_layers = list(x3d_layers or [False, False, False, True, True, False])
        if len(blocks) != len(strides) or len(x3d_layers) != len(blocks):
            raise ValueError("X3D blocks/strides/x3d_layers lengths must agree")
        self.backbone = PointMetaBaseEncoder_X3D(
            blocks=blocks,
            strides=strides,
            width=width,
            in_channels=in_channels,
            radius=radius,
            radius_scaling=radius_scaling,
            sa_layers=sa_layers,
            sa_use_res=sa_use_res,
            nsample=nsample,
            expansion=expansion,
            x3d_layers=x3d_layers,
            NCP="false",
            aggr_args=_easy_dict({
                "feature_type": "dp_fj",
                "reduction": "max",
                "es_type": "pca",
                "structure_kernel_args": {
                    "hidden_dim": 32,
                    "denoise": "true",
                    "modu": "false",
                },
            }),
            group_args=_easy_dict({"NAME": "ballquery", "normalize_dp": True}),
            conv_args={"order": "conv-norm-act"},
            act_args={"act": "relu"},
            norm_args={"norm": "bn"},
        )
        _patch_group_all(self.backbone)
        self.backbone_dim = int(self.backbone.out_channels)
        self.in_channels = in_channels
        self.use_magnetization = use_magnetization
        self.projection = ProjectionHead(
            self.backbone_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=output_dim,
            dropout=0.1,
        )
        self.output_dim = output_dim

    def forward(
        self,
        xyz: Tensor,
        rho_total: Tensor,
        rho_magnetization: Optional[Tensor] = None,
        *,
        rho_total_raw: Optional[Tensor] = None,
        point_mask: Optional[Tensor] = None,
        volume_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [B,N,3], got {tuple(xyz.shape)}")
        if rho_total.shape != xyz.shape[:2]:
            raise ValueError("rho_total must have shape [B,N] matching xyz")
        channels = [xyz, rho_total.unsqueeze(-1)]
        if self.use_magnetization:
            if rho_magnetization is None or rho_magnetization.shape != xyz.shape[:2]:
                raise ValueError("magnetization channel requested but not supplied")
            channels.append(rho_magnetization.unsqueeze(-1))
        f0 = torch.cat(channels, dim=-1).transpose(1, 2).contiguous()
        p0 = (xyz / 4.5).contiguous()
        features = self.backbone.forward_cls_feat(p0, f0)
        if features.ndim == 3:
            features = features.mean(dim=-1)
        if features.ndim != 2 or features.shape[1] != self.backbone_dim:
            raise RuntimeError(f"unexpected X3D feature shape: {tuple(features.shape)}")
        out = self.projection(features)
        assert_finite("X3D embedding", out)
        return out


__all__ = ["X3DChargeEncoder", "load_x3d_class"]
