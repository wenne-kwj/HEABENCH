"""Public condition names for Low-Label Magnetic Transfer."""

from __future__ import annotations


PUBLIC_CONDITIONS = ("s0", "s_dft", "s_gauss")
_ALIASES = {"s1": "s_dft"}


def normalize_condition(value: str) -> str:
    """Return a canonical condition name while accepting the legacy ``s1``."""

    name = str(value).strip().lower().replace("-", "_")
    name = _ALIASES.get(name, name)
    if name not in PUBLIC_CONDITIONS:
        allowed = ", ".join(PUBLIC_CONDITIONS)
        raise ValueError(f"unknown transfer condition {value!r}; choose one of {allowed}")
    return name


def uses_pretrained_encoder(value: str) -> bool:
    """Whether the downstream structure encoder is initialized from pretraining."""

    return normalize_condition(value) != "s0"


__all__ = ["PUBLIC_CONDITIONS", "normalize_condition", "uses_pretrained_encoder"]
