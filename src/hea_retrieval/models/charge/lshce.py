"""Learnable spherical-harmonic charge encoder for HEA retrieval.

The encoder consumes every stored charge point without building a graph.  Its
learned density channels are expanded in radial spherical-Bessel functions and
real spherical harmonics; per-l Gram matrices then provide O(3)-invariant,
permutation-invariant features.  Physical integrals use only the 3,968
quasi-uniform volume samples, while the learned descriptor intentionally sees
all 4,096 samples, including the 128 pair-focused points.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from e3nn import o3
from torch import Tensor, nn

from ..common import assert_finite


PHYSICAL_FEATURE_NAMES = (
    "total_charge_e",
    "log1p_charge_over_reference",
    "radial_moment_1",
    "radial_moment_2",
    "radial_moment_3",
    "radial_moment_4",
    "volume_density_rms_over_reference",
    "log1p_density_max_over_reference",
)


@lru_cache(maxsize=8)
def _bessel_root_table(l_max: int, num_radial: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return positive spherical-Bessel roots and L2 normalizers.

    This executes once at model construction on CPU.  SciPy is already part of
    the isolated HEA retrieval environment and is not involved in forward or
    backward execution.
    """

    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    roots: list[float] = []
    normalizers: list[float] = []
    step = math.pi / 8.0
    for degree in range(l_max + 1):
        degree_roots: list[float] = []
        left = 1.0e-6
        left_value = float(spherical_jn(degree, left))
        right = left + step
        limit = float((num_radial + degree + 8) * math.pi)
        while right <= limit and len(degree_roots) < num_radial:
            right_value = float(spherical_jn(degree, right))
            if left_value * right_value < 0.0:
                root = float(
                    brentq(
                        lambda value: float(spherical_jn(degree, value)),
                        left,
                        right,
                        xtol=1.0e-13,
                        rtol=1.0e-13,
                    )
                )
                if not degree_roots or abs(root - degree_roots[-1]) > 1.0e-8:
                    degree_roots.append(root)
            left = right
            left_value = right_value
            right += step
        if len(degree_roots) != num_radial:
            raise RuntimeError(
                f"found {len(degree_roots)} roots for l={degree}; expected {num_radial}"
            )
        for root in degree_roots:
            # Integral_0^1 x^2 j_l(root*x)^2 dx = j_(l+1)(root)^2 / 2.
            next_value = abs(float(spherical_jn(degree + 1, root)))
            if next_value <= 0.0:
                raise RuntimeError(f"invalid Bessel normalizer for l={degree}, root={root}")
            roots.append(root)
            normalizers.append(math.sqrt(2.0) / next_value)
    return tuple(roots), tuple(normalizers)


def _odd_double_factorial(value: int) -> int:
    result = 1
    for term in range(value, 0, -2):
        result *= term
    return result


def spherical_bessel_j(degree: int, value: Tensor) -> Tensor:
    """Differentiable spherical Bessel j_l for l <= 4 with a stable origin."""

    if degree < 0 or degree > 4:
        raise ValueError("spherical_bessel_j supports degrees 0 through 4")
    absolute = value.abs()
    small = absolute < 1.0e-3
    safe = torch.where(small, torch.ones_like(value), value)
    j0 = torch.sin(safe) / safe
    if degree == 0:
        regular = j0
    else:
        j1 = torch.sin(safe) / safe.square() - torch.cos(safe) / safe
        if degree == 1:
            regular = j1
        else:
            previous, current = j0, j1
            for ell in range(1, degree):
                following = (2 * ell + 1) * current / safe - previous
                previous, current = current, following
            regular = current
    leading = value.pow(degree) / float(_odd_double_factorial(2 * degree + 1))
    series = leading * (
        1.0
        - value.square() / float(2 * (2 * degree + 3))
        + value.pow(4) / float(8 * (2 * degree + 3) * (2 * degree + 5))
    )
    return torch.where(small, series, regular)


class LearnableSphericalHarmonicChargeEncoder(nn.Module):
    """Encode a local scalar charge field into a normalized retrieval vector."""

    def __init__(
        self,
        *,
        output_dim: int = 256,
        cutoff: float = 4.5,
        l_max: int = 4,
        num_radial: int = 12,
        density_hidden_dim: int = 32,
        density_channels: int = 8,
        compressed_multiplicity: int = 32,
        projection_hidden_dim: int = 1024,
        projection_mid_dim: int = 512,
        dropout: float = 0.10,
        rho_ref: float = 1.0,
        physical_mean: Sequence[float] | None = None,
        physical_std: Sequence[float] | None = None,
        uniform_points: int = 3968,
        power_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or cutoff <= 0 or num_radial <= 0:
            raise ValueError("output_dim, cutoff, and num_radial must be positive")
        if l_max != 4:
            raise ValueError("the frozen LSHCE configuration requires l_max=4")
        if density_channels != 8:
            raise ValueError("the frozen LSHCE configuration requires 8 density channels")
        if density_hidden_dim <= 0 or compressed_multiplicity <= 0:
            raise ValueError("density and multiplicity widths must be positive")
        if not math.isfinite(float(rho_ref)) or float(rho_ref) <= 0:
            raise ValueError("rho_ref must be a positive finite training-set statistic")
        if uniform_points <= 0:
            raise ValueError("uniform_points must be positive")
        if power_epsilon <= 0:
            raise ValueError("power_epsilon must be positive")

        self.output_dim = int(output_dim)
        self.cutoff = float(cutoff)
        self.l_max = int(l_max)
        self.num_radial = int(num_radial)
        self.density_channels = int(density_channels)
        self.compressed_multiplicity = int(compressed_multiplicity)
        self.uniform_points = int(uniform_points)
        self.power_epsilon = float(power_epsilon)
        self.sphere_volume = 4.0 * math.pi * self.cutoff**3 / 3.0
        self.charge_reference = float(rho_ref) * self.sphere_volume

        # One exact linear density channel plus seven learned density-gated
        # channels.  Multiplication by u guarantees f(rho=0)=0, preventing the
        # encoder from matching patches solely from pair-focused support points.
        self.density_gates = nn.Sequential(
            nn.Linear(2, density_hidden_dim),
            nn.SiLU(),
            nn.Linear(density_hidden_dim, density_channels - 1),
        )
        raw_multiplicity = density_channels * num_radial
        self.multiplicity_mix = nn.ModuleList(
            nn.Linear(raw_multiplicity, compressed_multiplicity, bias=False)
            for _ in range(l_max + 1)
        )

        roots, normalizers = _bessel_root_table(l_max, num_radial)
        self.register_buffer(
            "bessel_roots",
            torch.tensor(roots, dtype=torch.float32).reshape(l_max + 1, num_radial),
            persistent=True,
        )
        self.register_buffer(
            "bessel_normalizers",
            torch.tensor(normalizers, dtype=torch.float32).reshape(l_max + 1, num_radial),
            persistent=True,
        )
        self.register_buffer("rho_reference", torch.tensor(float(rho_ref), dtype=torch.float32))

        mean = list(physical_mean or [0.0] * len(PHYSICAL_FEATURE_NAMES))
        std = list(physical_std or [1.0] * len(PHYSICAL_FEATURE_NAMES))
        if len(mean) != len(PHYSICAL_FEATURE_NAMES) or len(std) != len(PHYSICAL_FEATURE_NAMES):
            raise ValueError("physical_mean and physical_std must each contain 8 values")
        if any((not math.isfinite(float(x))) for x in mean + std):
            raise ValueError("physical feature statistics must be finite")
        if any(float(x) <= 0 for x in std):
            raise ValueError("physical_std values must be positive")
        self.register_buffer("physical_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("physical_std", torch.tensor(std, dtype=torch.float32))

        triangular = torch.triu_indices(compressed_multiplicity, compressed_multiplicity)
        self.register_buffer("triangular_row", triangular[0], persistent=False)
        self.register_buffer("triangular_col", triangular[1], persistent=False)
        scale = torch.where(
            triangular[0].eq(triangular[1]),
            torch.ones(triangular.shape[1]),
            torch.full((triangular.shape[1],), math.sqrt(2.0)),
        )
        self.register_buffer("svec_scale", scale, persistent=False)
        per_l_invariants = compressed_multiplicity * (compressed_multiplicity + 1) // 2
        self.invariant_dim = (l_max + 1) * per_l_invariants
        if self.invariant_dim != 2640:
            raise RuntimeError(f"frozen LSHCE invariant dimension is not 2640: {self.invariant_dim}")

        self.power_norm = nn.LayerNorm(self.invariant_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.invariant_dim + len(PHYSICAL_FEATURE_NAMES), projection_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(projection_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, projection_mid_dim),
            nn.SiLU(),
            nn.Linear(projection_mid_dim, output_dim),
        )

    def _masks(
        self,
        xyz: Tensor,
        point_mask: Optional[Tensor],
        volume_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        shape = xyz.shape[:2]
        if point_mask is None:
            point_mask = torch.ones(shape, dtype=torch.bool, device=xyz.device)
        if point_mask.shape != shape:
            raise ValueError(f"point_mask must have shape {shape}, got {tuple(point_mask.shape)}")
        point_mask = point_mask.bool()
        if volume_mask is None:
            indices = torch.arange(shape[1], device=xyz.device)
            volume_mask = indices.lt(min(self.uniform_points, shape[1])).expand(shape[0], -1)
        if volume_mask.shape != shape:
            raise ValueError(f"volume_mask must have shape {shape}, got {tuple(volume_mask.shape)}")
        volume_mask = volume_mask.bool() & point_mask
        if torch.any(point_mask.sum(dim=1) <= 0):
            raise ValueError("every charge patch must contain at least one valid point")
        if torch.any(volume_mask.sum(dim=1) <= 0):
            raise ValueError("every charge patch must contain at least one volume point")
        return point_mask, volume_mask

    def compute_physical_features(
        self,
        xyz: Tensor,
        rho_raw: Tensor,
        point_mask: Tensor,
        volume_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return eight raw physical features, dimensionless q, and Q in e."""

        radius = xyz.norm(dim=-1).div(self.cutoff).clamp(min=0.0, max=1.0)
        rho_positive = rho_raw.clamp_min(0.0)
        volume_weight = volume_mask.to(rho_positive.dtype)
        volume_count = volume_weight.sum(dim=1).clamp_min(1.0)
        density_sum = (rho_positive * volume_weight).sum(dim=1)
        density_mean = density_sum / volume_count
        total_charge = density_mean * self.sphere_volume
        q = density_mean / self.rho_reference.clamp_min(1.0e-12)
        density_denominator = density_sum.clamp_min(1.0e-12)
        moments = [
            (rho_positive * volume_weight * radius.pow(order)).sum(dim=1)
            / density_denominator
            for order in range(1, 5)
        ]
        density_rms = (
            (rho_positive.square() * volume_weight).sum(dim=1) / volume_count
        ).sqrt() / self.rho_reference.clamp_min(1.0e-12)
        masked_density = rho_positive.masked_fill(~point_mask, -torch.inf)
        density_max = masked_density.amax(dim=1).clamp_min(0.0)
        physical = torch.stack(
            [
                total_charge,
                torch.log1p(q),
                *moments,
                density_rms,
                torch.log1p(density_max / self.rho_reference.clamp_min(1.0e-12)),
            ],
            dim=-1,
        )
        return physical, q, total_charge

    @staticmethod
    def _quintic_envelope(radius: Tensor) -> Tensor:
        x = radius.clamp(min=0.0, max=1.0)
        return 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)

    def invariant_features(
        self,
        xyz: Tensor,
        rho_raw: Tensor,
        point_mask: Optional[Tensor] = None,
        volume_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized power features, raw physical features, and q."""

        point_mask, volume_mask = self._masks(xyz, point_mask, volume_mask)
        xyz = xyz.float()
        rho_raw = rho_raw.float()
        point_mask = point_mask.to(device=xyz.device)
        volume_mask = volume_mask.to(device=xyz.device)
        physical, q, _ = self.compute_physical_features(
            xyz, rho_raw, point_mask, volume_mask
        )

        radius_angstrom = xyz.norm(dim=-1)
        radius = radius_angstrom.div(self.cutoff).clamp(min=0.0, max=1.0)
        safe_radius = radius_angstrom.clamp_min(1.0e-8)
        direction = xyz / safe_radius.unsqueeze(-1)
        origin_direction = torch.zeros_like(direction)
        origin_direction[..., 2] = 1.0
        direction = torch.where(
            radius_angstrom.lt(1.0e-8).unsqueeze(-1), origin_direction, direction
        )
        harmonics = o3.spherical_harmonics(
            list(range(self.l_max + 1)),
            direction,
            normalize=False,
            normalization="integral",
        ).float()

        rho_positive = rho_raw.clamp_min(0.0)
        u = rho_positive / self.rho_reference.clamp_min(1.0e-12)
        density_input = torch.stack([u, torch.log1p(u)], dim=-1)
        gates = self.density_gates(density_input)
        density_channels = torch.cat([u.unsqueeze(-1), u.unsqueeze(-1) * gates], dim=-1)
        descriptor_weight = point_mask.to(xyz.dtype)
        descriptor_weight = descriptor_weight / descriptor_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        envelope = self._quintic_envelope(radius)

        power_vectors: list[Tensor] = []
        for degree in range(self.l_max + 1):
            argument = radius.unsqueeze(-1) * self.bessel_roots[degree]
            radial = spherical_bessel_j(degree, argument)
            radial = radial * self.bessel_normalizers[degree] * envelope.unsqueeze(-1)
            density_radial = (
                density_channels.unsqueeze(-1) * radial.unsqueeze(-2)
            ).flatten(start_dim=-2)
            harmonic_slice = harmonics[..., degree * degree : (degree + 1) ** 2]
            weighted_harmonics = harmonic_slice * descriptor_weight.unsqueeze(-1)
            coefficient = torch.bmm(
                density_radial.transpose(1, 2), weighted_harmonics
            )
            compressed = self.multiplicity_mix[degree](
                coefficient.transpose(1, 2)
            ).transpose(1, 2)
            gram = torch.bmm(compressed, compressed.transpose(1, 2))
            gram = gram / float(2 * degree + 1)
            gram = gram / (q.square() + self.power_epsilon).view(-1, 1, 1)
            vector = gram[:, self.triangular_row, self.triangular_col]
            vector = vector * self.svec_scale
            power_vectors.append(torch.sign(vector) * torch.log1p(vector.abs()))
        power = self.power_norm(torch.cat(power_vectors, dim=-1))
        assert_finite("LSHCE power spectrum", power)
        assert_finite("LSHCE physical features", physical)
        return power, physical, q

    def forward(
        self,
        xyz: Tensor,
        rho_total: Tensor,
        rho_magnetization: Optional[Tensor] = None,
        *,
        rho_total_raw: Optional[Tensor] = None,
        point_mask: Optional[Tensor] = None,
        volume_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [B,N,3], got {tuple(xyz.shape)}")
        if rho_total.shape != xyz.shape[:2]:
            raise ValueError("rho_total must have shape [B,N] matching xyz")
        rho_raw = rho_total if rho_total_raw is None else rho_total_raw
        if rho_raw.shape != xyz.shape[:2]:
            raise ValueError("rho_total_raw must have shape [B,N] matching xyz")

        device_type = xyz.device.type if xyz.device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            power, physical, _ = self.invariant_features(
                xyz.float(),
                rho_raw.float(),
                point_mask=point_mask,
                volume_mask=volume_mask,
            )
            standardized_physical = (
                physical - self.physical_mean
            ) / self.physical_std.clamp_min(1.0e-8)
            features = torch.cat([power, standardized_physical], dim=-1)
            embedding = F.normalize(self.projection(features), dim=-1, eps=1.0e-8)
        assert_finite("LSHCE embedding", embedding)
        return embedding


__all__ = [
    "LearnableSphericalHarmonicChargeEncoder",
    "PHYSICAL_FEATURE_NAMES",
    "spherical_bessel_j",
]
