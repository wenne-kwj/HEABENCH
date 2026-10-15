"""Portable filesystem defaults for Task 3."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / "tasks" / "magnetic_transfer"
CONFIG_PATH = TASK_ROOT / "config.json"


def _path_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


DEFAULT_TASK3_DATA_ROOT = _path_from_env(
    "HEABENCH_T3_DATA_ROOT", REPOSITORY_ROOT / "data" / "t3_magnetic_transfer"
)
DEFAULT_PRETRAIN_RUN_ROOT = _path_from_env(
    "HEABENCH_T3_PRETRAIN_RUN_ROOT",
    REPOSITORY_ROOT / "artifacts" / "t3_magnetic_transfer" / "pretraining",
)
DEFAULT_DOWNSTREAM_RUN_ROOT = _path_from_env(
    "HEABENCH_T3_RUN_ROOT",
    REPOSITORY_ROOT / "artifacts" / "t3_magnetic_transfer" / "downstream",
)
DEFAULT_EVALUATION_ROOT = _path_from_env(
    "HEABENCH_T3_EVAL_ROOT",
    REPOSITORY_ROOT / "artifacts" / "t3_magnetic_transfer" / "evaluations",
)
DEFAULT_PRETRAINED_CHECKPOINT = _path_from_env(
    "HEABENCH_T3_PRETRAINED_CHECKPOINT",
    REPOSITORY_ROOT / "artifacts" / "t3_magnetic_transfer" / "pretraining" / "best.pt",
)
