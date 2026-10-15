"""Factory for all HEA structure--charge retrieval combinations."""

from __future__ import annotations

from torch import nn

from .common import RetrievalModel


def _charge_encoder_class(name: str) -> type[nn.Module]:
    if name in {"pointvector", "p"}:
        from .charge.pointvector import PointVectorChargeEncoder

        return PointVectorChargeEncoder
    if name in {"x3d", "x"}:
        from .charge.x3d import X3DChargeEncoder

        return X3DChargeEncoder
    if name in {"ptv3", "pointtransformerv3", "pointtransformerv3hea", "t3"}:
        from .charge.ptv3_hea import PointTransformerV3ChargeEncoder

        return PointTransformerV3ChargeEncoder
    if name in {"lshce", "sphericalharmonic", "sh"}:
        from .charge.lshce import LearnableSphericalHarmonicChargeEncoder

        return LearnableSphericalHarmonicChargeEncoder
    raise ValueError(f"unknown charge encoder: {name}")


def _non_mace_structure(
    name: str,
    kwargs: dict,
) -> nn.Module:
    if name in {"geoformer", "g"}:
        from .structure.geoformer import GeoFormerStructureEncoder

        return GeoFormerStructureEncoder(**kwargs)
    if name in {"equiformer", "equiformerv2", "e"}:
        from .structure.equiformer import EquiformerStructureEncoder

        return EquiformerStructureEncoder(**kwargs)
    if name in {"chgnet", "chg", "c"}:
        from .structure.chgnet import CHGNetStructureEncoder

        return CHGNetStructureEncoder(**kwargs)
    raise ValueError(f"unknown structure encoder: {name}")


def build_model(
    structure_name: str = "geoformer",
    charge_name: str = "pointvector",
    *,
    output_dim: int = 256,
    structure_kwargs: dict | None = None,
    charge_kwargs: dict | None = None,
) -> RetrievalModel:
    """Build any member of the 4x4 HEA encoder matrix.

    The ordinary retrieval environment uses e3nn 0.5.x, while MACE requires
    e3nn 0.4.4.  MACE must therefore be constructed before importing its
    selected charge adapter.  For every non-MACE experiment the charge class
    is imported first, matching the historical eager-import environment and
    ensuring CHGNet's auxiliary site-packages cannot replace an already
    selected e3nn runtime needed by LSHCE.
    """

    s_name = structure_name.lower().replace("-", "")
    c_name = charge_name.lower().replace("-", "")
    s_kwargs = dict(structure_kwargs or {})
    c_kwargs = dict(charge_kwargs or {})
    s_kwargs.setdefault("output_dim", output_dim)
    c_kwargs.setdefault("output_dim", output_dim)

    if s_name in {"mace", "m"}:
        from .structure.mace import MACEStructureEncoder

        structure = MACEStructureEncoder(**s_kwargs)
        charge_class = _charge_encoder_class(c_name)
    else:
        # Importing LSHCE here deliberately selects the ordinary e3nn 0.5.x
        # runtime before CHGNet adds its isolated dependency directory.
        charge_class = _charge_encoder_class(c_name)
        structure = _non_mace_structure(s_name, s_kwargs)

    charge = charge_class(**c_kwargs)
    return RetrievalModel(structure, charge)


__all__ = ["build_model"]
