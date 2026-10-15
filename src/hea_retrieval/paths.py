"""Portable filesystem defaults for Task 2.

Every path can be overridden by a CLI argument.  Environment variables are
only convenience defaults; importing this module never creates directories.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / "tasks" / "local_matching"
CONFIG_PATH = TASK_ROOT / "config.json"


def _path_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


DEFAULT_DATA_ROOT = _path_from_env(
    "HEABENCH_T2_DATA_ROOT", REPOSITORY_ROOT / "data" / "t2_retrieval"
)
DEFAULT_RUN_ROOT = _path_from_env(
    "HEABENCH_T2_RUN_ROOT", REPOSITORY_ROOT / "artifacts" / "t2_retrieval" / "runs"
)
DEFAULT_EVAL_ROOT = _path_from_env(
    "HEABENCH_T2_EVAL_ROOT", REPOSITORY_ROOT / "artifacts" / "t2_retrieval" / "evaluations"
)
