"""HEA adapter for the official PointVector backbone."""

from __future__ import annotations

import os
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from ..common import ProjectionHead, assert_finite


REPO_ROOT = Path(__file__).resolve().parents[3]
POINTVECTOR_ROOT = Path(os.environ.get("HEABENCH_POINTVECTOR_ROOT", str(REPO_ROOT / "third_party" / "openpoints_pointvector")))

_ACTIVE_OPENPOINTS_ROOT: Path | None = None


def _easy_dict(value):
    """Recursively convert dictionaries to the attribute-style config used by OpenPoints."""
    from easydict import EasyDict
    if isinstance(value, dict):
        return EasyDict({key: _easy_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_easy_dict(item) for item in value]
    return value


def _purge_openpoints_modules() -> None:
    for name in list(sys.modules):
        if name == "openpoints" or name.startswith("openpoints."):
            del sys.modules[name]


def _activate_openpoints(root: Path) -> None:
    """Load one frozen OpenPoints checkout as the ``openpoints`` package.

    PointVector and X-3D are independent upstream checkouts with the same
    package name.  Loading the package from its explicit ``__init__.py`` keeps
    them isolated and avoids installing either checkout into system Python.
    Only one checkout is active in a process at a time; a model experiment is
    intentionally one point backbone at a time.
    """

    global _ACTIVE_OPENPOINTS_ROOT
    root = root.resolve()
    if _ACTIVE_OPENPOINTS_ROOT == root and "openpoints" in sys.modules:
        return
    if _ACTIVE_OPENPOINTS_ROOT is not None:
        _purge_openpoints_modules()
    init_file = root / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"OpenPoints package init not found: {init_file}")
    spec = importlib.util.spec_from_file_location(
        "openpoints", init_file, submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["openpoints"] = module
    spec.loader.exec_module(module)
    _ACTIVE_OPENPOINTS_ROOT = root


def load_pointvector_class():
    _activate_openpoints(POINTVECTOR_ROOT)
    module = importlib.import_module("openpoints.models.backbone.pointvector")
    return module.PointVectorEncoder


class PointVectorChargeEncoder(nn.Module):
    """Encode ``(xyz, rho_total, rho_magnetization)`` into 256 dimensions.

    The upstream backbone consumes ``p0=[B,N,3]`` and ``f0=[B,4,N]``.  The
    fourth channel is total valence density; magnetization is exposed as an
    optional fifth input only when explicitly requested, because the official
    small PointVector configuration is a 4-channel model.
    """

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
        sa_layers: int = 2,
        sa_use_res: bool = True,
        expansion: int = 4,
        flag: int = 0,
    ) -> None:
        super().__init__()
        if use_magnetization and in_channels != 5:
            in_channels = 5
        if in_channels not in (4, 5):
            raise ValueError("PointVector adapter supports 4 or 5 input channels")
        PointVectorEncoder = load_pointvector_class()
        blocks = list(blocks or [1, 1, 1, 1, 1, 1])
        strides = list(strides or [1, 2, 2, 2, 2, 1])
        if len(blocks) != len(strides):
            raise ValueError("PointVector blocks and strides must have equal length")
        # Every nested dictionary is fresh: the upstream constructor mutates
        # group_args while expanding stage radii and neighbour counts.
        self.backbone = PointVectorEncoder(
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
            flag=flag,
            aggr_args={"feature_type": "dp_fj", "reduction": "max"},
            group_args=_easy_dict({"NAME": "ballquery", "normalize_dp": True}),
            conv_args={"order": "conv-norm-act"},
            act_args={"act": "relu"},
            norm_args={"norm": "bn"},
        )
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
        # The extraction coordinates are in Å.  The upstream radius schedule
        # is defined on a unit-scale point cloud, hence the fixed 4.5 Å sphere
        # normalization here (also used by X-3D).
        p0 = (xyz / 4.5).contiguous()
        features = self.backbone.forward_cls_feat(p0, f0)
        if features.ndim == 3:
            features = features.mean(dim=-1)
        if features.ndim != 2 or features.shape[1] != self.backbone_dim:
            raise RuntimeError(f"unexpected PointVector feature shape: {tuple(features.shape)}")
        out = self.projection(features)
        assert_finite("PointVector embedding", out)
        return out


__all__ = ["PointVectorChargeEncoder", "load_pointvector_class"]
