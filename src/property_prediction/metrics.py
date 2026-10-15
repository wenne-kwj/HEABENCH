from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .spec import TargetFamily


DISPLAY_NAMES = {
    "formation_energy_internal_eV_per_atom": "E_form",
    "N_EF_states_per_eV_atom": "DOS_EF",
    "fermi_crossing_count": "N_cross",
    "fermi_crossing_event_count": "N_event",
    "fermi_crossing_slope_eV_A": "cross_slope",
    "dos_curve_E_minus_EF_states_per_eV_atom": "DOS_curve",
    "near_fermi_band_energies_E_minus_EF_eV": "Band_EF",
}


def regression_metrics(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    family: TargetFamily,
) -> dict[str, Any]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    error = prediction - target
    result: dict[str, Any] = {}
    if len(family.targets) > 1:
        for column, target_name in enumerate(family.targets):
            valid = mask[:, column]
            values = error[:, column][valid]
            label = DISPLAY_NAMES[target_name]
            result[label] = _metrics(values)
        return result

    target_name = family.targets[0]
    label = DISPLAY_NAMES[target_name]
    valid_values = error[mask]
    result[label] = _metrics(valid_values)
    if target.ndim > 2 or (target.ndim == 2 and target.shape[1] > 1):
        flat_error = error.flatten(start_dim=1)
        flat_mask = mask.flatten(start_dim=1)
        valid_samples = flat_mask.any(dim=1)
        abs_sum = torch.where(flat_mask, flat_error.abs(), 0.0).sum(dim=1)
        sq_sum = torch.where(flat_mask, flat_error.square(), 0.0).sum(dim=1)
        count = flat_mask.sum(dim=1).clamp_min(1)
        sample_mae = abs_sum / count
        sample_rmse = torch.sqrt(sq_sum / count)
        result[label]["sample_count"] = int(valid_samples.sum())
        result[label]["sample_macro_mae"] = float(sample_mae[valid_samples].mean())
        result[label]["sample_macro_rmse"] = float(sample_rmse[valid_samples].mean())
    return result


def _metrics(error: Tensor) -> dict[str, float | int | None]:
    if error.numel() == 0:
        return {"count": 0, "mae": None, "rmse": None}
    mae = float(error.abs().mean())
    rmse = float(torch.sqrt(error.square().mean()))
    return {
        "count": int(error.numel()),
        "mae": mae if math.isfinite(mae) else None,
        "rmse": rmse if math.isfinite(rmse) else None,
    }
