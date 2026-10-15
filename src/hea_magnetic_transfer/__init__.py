"""Data and training utilities for HEA Task 3 magnetic transfer."""

from .conditions import PUBLIC_CONDITIONS, normalize_condition
from .data_contract import (
    DEFAULT_RADIUS_A,
    DEFAULT_CHARGE_POINTS,
    validate_dataset,
)

__all__ = [
    "PUBLIC_CONDITIONS",
    "DEFAULT_RADIUS_A",
    "DEFAULT_CHARGE_POINTS",
    "normalize_condition",
    "validate_dataset",
]
