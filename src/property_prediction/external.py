from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def upstream_path(environment: str, relative: str) -> Path:
    """Resolve an optional external encoder checkout without machine paths."""

    configured = os.environ.get(environment)
    return Path(configured).expanduser().resolve() if configured else REPOSITORY_ROOT / relative
