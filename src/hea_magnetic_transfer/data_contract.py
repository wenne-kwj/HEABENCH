"""Small public contract for the released magnetic-transfer bundle.

Raw VASP parsing and internal dataset-building utilities are intentionally not
part of the release. Training consumes the sanitized arrays and manifests in
the published bundle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RADIUS_A = 4.5
DEFAULT_CHARGE_POINTS = 4096
PAPER_BUDGETS = (50, 300, 620)
REQUIRED_FILES = (
    "dataset_contract.json",
    "validation_report.json",
    "nsp_manifest.csv",
    "sp_manifest.csv",
    "sp_budgets.json",
)


def _manifest_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return {"rows": len(rows), "columns": fields}


def validate_dataset(root: str | Path) -> dict[str, Any]:
    """Check the portable files required before starting Task 3 training."""

    root = Path(root).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete magnetic-transfer bundle: {missing}")
    budgets = json.loads((root / "sp_budgets.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    with (root / "sp_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        sp_rows = list(csv.DictReader(handle))
    train_ids = {row["parent_id"] for row in sp_rows if row["split"] == "train"}
    budget_map = budgets.get("budgets", budgets)
    previous: set[str] = set()
    budget_counts: dict[str, int] = {}
    for size in PAPER_BUDGETS:
        ids = list(budget_map.get(str(size), []))
        current = set(ids)
        if len(ids) != size or len(current) != size:
            raise ValueError(f"budget {size} is missing, duplicated, or has the wrong size")
        if not previous.issubset(current) or not current.issubset(train_ids):
            raise ValueError(f"budget {size} is not nested and train-only")
        previous = current
        budget_counts[str(size)] = len(ids)
    return {
        "root": str(root),
        "nsp_manifest": _manifest_summary(root / "nsp_manifest.csv"),
        "sp_manifest": _manifest_summary(root / "sp_manifest.csv"),
        "validation_passed": validation.get("valid") is True,
        "paper_budget_counts": budget_counts,
        "budgets_nested_train_only": True,
    }


__all__ = [
    "DEFAULT_RADIUS_A",
    "DEFAULT_CHARGE_POINTS",
    "PAPER_BUDGETS",
    "REQUIRED_FILES",
    "validate_dataset",
]
