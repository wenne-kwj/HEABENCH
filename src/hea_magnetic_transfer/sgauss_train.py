#!/usr/bin/env python3
"""Train the Task 3 S-Gauss NSP control with portable dependency paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hea_retrieval.models.charge import pointvector

from . import pretrain
from .gaussian_view import make_gaussian_dataset_class


DEFAULT_OPENPOINTS = pointvector.POINTVECTOR_ROOT


def main(argv: list[str] | None = None) -> None:
    wrapper = argparse.ArgumentParser(description=__doc__)
    wrapper.add_argument("gaussian_view_root", type=Path)
    wrapper.add_argument("--data-root", default=None)
    wrapper.add_argument("--output-dir", default=None)
    wrapper.add_argument("--device", default="cuda:0")
    wrapper.add_argument("--resume", default=None)
    known = wrapper.parse_args(argv)
    view_root = known.gaussian_view_root.expanduser().resolve()
    openpoints_root = DEFAULT_OPENPOINTS.expanduser().resolve()
    if not (view_root / "_SUCCESS").is_file():
        raise RuntimeError(f"Gaussian view is incomplete: {view_root}")
    if not (openpoints_root / "__init__.py").is_file():
        raise FileNotFoundError(openpoints_root / "__init__.py")

    pointvector.POINTVECTOR_ROOT = openpoints_root
    original_dataset = pretrain.Task3NSPSiteDataset
    original_validate = pretrain.validate_pretraining_contract
    gaussian_dataset = make_gaussian_dataset_class(original_dataset, view_root)
    stats_path = view_root / "density_stats.json"
    def load_gaussian_stats(_root):
        return json.loads(stats_path.read_text(encoding="utf-8"))

    def validate_gaussian_contract(data_root):
        audit = original_validate(data_root)
        dataset = gaussian_dataset(data_root, cache_size=1, use_center_indicator=True)
        sample = dataset[0]
        if sample["rho_total"].shape != (4096,) or sample["rho_total"].dtype != np.float32:
            raise ValueError("Gaussian sample contract mismatch")
        audit.update(
            {
                "field_source": "Gaussian structure-derived control; contains no DFT charge",
                "gaussian_view_root": str(view_root),
                "gaussian_sample_finite": bool(np.isfinite(sample["rho_total"]).all()),
            }
        )
        return audit

    pretrain.Task3NSPSiteDataset = gaussian_dataset
    pretrain.load_density_stats = load_gaussian_stats
    pretrain.validate_pretraining_contract = validate_gaussian_contract
    forwarded = ["--device", known.device]
    for option, value in (
        ("--data-root", known.data_root),
        ("--output-dir", known.output_dir),
        ("--resume", known.resume),
    ):
        if value is not None:
            forwarded.extend((option, str(value)))
    args = pretrain.parse_args(forwarded)
    args.gaussian_view_root = str(view_root)
    args.field_source = "S-Gauss periodic ZVAL Gaussian sigma=0.35 A"
    args.contains_dft_charge = False
    args.openpoints_root = str(openpoints_root)
    pretrain.train(args)


if __name__ == "__main__":
    main()
