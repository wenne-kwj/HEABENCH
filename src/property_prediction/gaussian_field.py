from __future__ import annotations

import re
from pathlib import Path

import numpy as np


ELEMENTS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


def read_potcar_zval(path: str | Path) -> dict[str, float]:
    """Return element -> ZVAL for the exact POTCAR concatenation used by VASP."""

    result: dict[str, float] = {}
    current: str | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "TITEL" in line and "=" in line:
            tokens = line.split("=", 1)[1].split()
            if len(tokens) < 2:
                raise ValueError(f"malformed TITEL in {path}")
            match = re.match(r"([A-Z][a-z]?)", tokens[1])
            if match is None:
                raise ValueError(f"could not parse POTCAR element from {tokens[1]!r}")
            current = match.group(1)
        elif current is not None and "ZVAL" in line:
            match = re.search(r"ZVAL\s*=\s*([-+0-9.Ee]+)", line)
            if match is not None:
                value = float(match.group(1))
                if current in result and result[current] != value:
                    raise ValueError(f"conflicting ZVAL for {current} in {path}")
                result[current] = value
                current = None
    if not result:
        raise ValueError(f"no TITEL/ZVAL pairs found in {path}")
    return result


def atomic_zval(atomic_numbers: np.ndarray, potcar_path: str | Path) -> np.ndarray:
    mapping = read_potcar_zval(potcar_path)
    values = []
    for number in np.asarray(atomic_numbers).astype(int):
        symbol = ELEMENTS[number]
        if symbol not in mapping:
            raise ValueError(f"POTCAR {potcar_path} has no ZVAL for {symbol}")
        values.append(mapping[symbol])
    return np.asarray(values, dtype=np.float32)


def gaussian_atomic_field(
    fractional_positions: np.ndarray,
    cell: np.ndarray,
    weights: np.ndarray,
    *,
    grid_shape: tuple[int, int, int] = (56, 40, 40),
    sigma_A: float = 0.35,
) -> np.ndarray:
    """Periodic species/position field with integral equal to sum(weights)."""

    frac = np.mod(np.asarray(fractional_positions, dtype=np.float64), 1.0)
    lattice = np.asarray(cell, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if frac.shape != (len(weight), 3) or lattice.shape != (3, 3):
        raise ValueError("malformed structure arrays")
    shape = np.asarray(grid_shape, dtype=np.int64)
    volume = float(abs(np.linalg.det(lattice)))
    if volume <= 0 or sigma_A <= 0:
        raise ValueError("cell volume and sigma must be positive")
    impulses = np.zeros(tuple(shape), dtype=np.float64)
    scaled = frac * shape[None, :]
    base = np.floor(scaled).astype(np.int64)
    remainder = scaled - base
    factor = float(np.prod(shape, dtype=np.float64)) / volume
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.asarray((dx, dy, dz))
                indices = (base + offset[None, :]) % shape[None, :]
                interpolation = np.prod(
                    np.where(offset[None, :] == 1, remainder, 1.0 - remainder), axis=1
                )
                np.add.at(
                    impulses,
                    (indices[:, 0], indices[:, 1], indices[:, 2]),
                    weight * interpolation * factor,
                )

    frequencies = np.meshgrid(
        *(np.fft.fftfreq(int(size)) * int(size) for size in shape), indexing="ij"
    )
    integer_wave = np.stack(frequencies, axis=-1)
    reciprocal = 2.0 * np.pi * integer_wave @ np.linalg.inv(lattice).T
    squared_norm = np.sum(reciprocal * reciprocal, axis=-1)
    gaussian_filter = np.exp(-0.5 * sigma_A * sigma_A * squared_norm)
    field = np.fft.ifftn(np.fft.fftn(impulses) * gaussian_filter).real
    desired_mean = float(weight.sum() / volume)
    field += desired_mean - float(field.mean())
    return field.astype(np.float32, copy=False)
