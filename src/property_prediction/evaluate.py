"""Stable Task 1 final-test evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Sequence


CONDITIONS = {
    "c": "C",
    "c_s_cgcnn": "C_S_cgcnn",
    "c_s_alignn": "C_S_alignn",
    "c_s_ecomformer": "C_S_ecomformer",
    "c_rho_mcdc": "C_rho_mcdc",
    "c_rho_pointvector": "C_rho_pointvector",
    "c_s_rho": "C_S_rho",
    "c_s_gaussian": "C_S_gaussian",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Global Multimodal Probing model.")
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument(
        "family",
        choices=("formation", "electronic_scalars", "dos_curve", "band_structure"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    script = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "global_probing"
        / "train"
        / "evaluate.py"
    )
    if not script.is_file():
        raise FileNotFoundError(script)
    forwarded = [CONDITIONS[args.condition], args.family, "--device", args.device]
    if args.seed is not None:
        forwarded.extend(("--seed", str(args.seed)))
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
