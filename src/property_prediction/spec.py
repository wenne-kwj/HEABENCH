from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping


EXPECTED_CONDITIONS = {
    "C": ("composition",),
    "C+S": ("composition", "structure"),
    "C+rho": ("composition", "charge_density"),
    "C+S+rho": ("composition", "structure", "charge_density"),
}

EXPECTED_FAMILIES = {
    "formation": (("formation_energy_internal_eV_per_atom",), (1,)),
    "electronic_scalars": (
        (
            "N_EF_states_per_eV_atom",
            "fermi_crossing_count",
            "fermi_crossing_event_count",
            "fermi_crossing_slope_eV_A",
        ),
        (4,),
    ),
    "dos_curve": (("dos_curve_E_minus_EF_states_per_eV_atom",), (1001,)),
    "band_structure": (
        ("near_fermi_band_energies_E_minus_EF_eV",),
        (80, 16),
    ),
}


@dataclass(frozen=True)
class TargetFamily:
    name: str
    targets: tuple[str, ...]
    output_shape: tuple[int, ...]

    @property
    def flat_output_dim(self) -> int:
        size = 1
        for value in self.output_shape:
            size *= value
        return size


@dataclass(frozen=True)
class BenchmarkSpec:
    source: Path
    benchmark_root: Path
    raw_proxy_root: Path
    nas_write_root: Path
    local_project_root: Path
    population: int
    required_encut_eV: float
    split_counts: Mapping[str, int]
    seed: int
    latent_dim: int
    head_hidden_dim: int
    dropout: float
    information_conditions: Mapping[str, tuple[str, ...]]
    target_families: Mapping[str, TargetFamily]


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    source = Path(path).resolve()
    repository_root = source.parents[2]
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {"data_root", "artifact_root", "seed"}
    unknown = sorted(set(payload) - required)
    missing = sorted(required - set(payload))
    if missing or unknown:
        raise ValueError(f"invalid probing config: missing={missing}, unknown={unknown}")

    conditions = dict(EXPECTED_CONDITIONS)
    families = {
        name: TargetFamily(name=name, targets=targets, output_shape=shape)
        for name, (targets, shape) in EXPECTED_FAMILIES.items()
    }
    split_counts = {"train": 2487, "validation": 311, "test": 311}
    population = 3109

    def resolved(value: str, environment: str | None = None) -> Path:
        candidate = os.environ.get(environment, value) if environment else value
        result = Path(candidate).expanduser()
        return (repository_root / result).resolve() if not result.is_absolute() else result.resolve()

    return BenchmarkSpec(
        source=source,
        benchmark_root=resolved(payload["data_root"], "HEABENCH_T1_DATA_ROOT"),
        raw_proxy_root=resolved(
            str(Path(payload["data_root"]) / "raw"), "HEABENCH_T1_RAW_ROOT"
        ),
        nas_write_root=resolved(payload["artifact_root"], "HEABENCH_T1_ARTIFACT_ROOT"),
        local_project_root=repository_root,
        population=population,
        required_encut_eV=550.0,
        split_counts=split_counts,
        seed=int(payload["seed"]),
        latent_dim=256,
        head_hidden_dim=128,
        dropout=0.1,
        information_conditions=conditions,
        target_families=families,
    )
