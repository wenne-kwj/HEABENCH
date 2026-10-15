"""Unified paper-facing Task 1 training entry point.

The frozen runs were produced by condition-specific trainers. This dispatcher
keeps those tested engines unchanged while presenting one stable public
interface and one condition vocabulary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Sequence


CONDITIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "c": ("train_composition.py", ()),
    "c_s_cgcnn": ("train_cgcnn.py", ()),
    "c_s_alignn": ("train_alignn.py", ()),
    "c_s_ecomformer": ("train_comformer.py", ()),
    "c_rho_mcdc": ("train_charge.py", ("mcdc",)),
    "c_rho_pointvector": (
        "train_charge.py",
        ("pointvector",),
    ),
    "c_s_rho": (
        "train_fusion.py",
        ("dft",),
    ),
    "c_s_gaussian": (
        "train_fusion.py",
        ("gaussian",),
    ),
}


def _task_train_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "global_probing"
        / "train"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one Global Multimodal Probing model.")
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument(
        "family",
        choices=("formation", "electronic_scalars", "dos_curve", "band_structure"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parsed = parser.parse_args(argv)
    filename, injected = CONDITIONS[parsed.condition]
    script = _task_train_root() / filename
    if not script.is_file():
        raise FileNotFoundError(script)
    forwarded = [
        *injected,
        parsed.family,
        "--output-dir",
        str(parsed.output_dir),
        "--device",
        parsed.device,
    ]
    if parsed.seed is not None:
        forwarded.extend(("--seed", str(parsed.seed)))
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
