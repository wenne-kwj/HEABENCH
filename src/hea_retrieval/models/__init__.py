"""Lazy public exports for HEABench retrieval encoders."""

from __future__ import annotations

import importlib

from .common import RetrievalModel

_LAZY_EXPORTS = {
    "EquiformerStructureEncoder": (
        ".structure.equiformer",
        "EquiformerStructureEncoder",
    ),
    "GeoFormerStructureEncoder": (
        ".structure.geoformer",
        "GeoFormerStructureEncoder",
    ),
    "CHGNetStructureEncoder": (".structure.chgnet", "CHGNetStructureEncoder"),
    "MACEStructureEncoder": (".structure.mace", "MACEStructureEncoder"),
    "PointVectorChargeEncoder": (
        ".charge.pointvector",
        "PointVectorChargeEncoder",
    ),
    "X3DChargeEncoder": (".charge.x3d", "X3DChargeEncoder"),
    "PointTransformerV3ChargeEncoder": (
        ".charge.ptv3_hea",
        "PointTransformerV3ChargeEncoder",
    ),
    "LearnableSphericalHarmonicChargeEncoder": (
        ".charge.lshce",
        "LearnableSphericalHarmonicChargeEncoder",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["RetrievalModel", *_LAZY_EXPORTS]
