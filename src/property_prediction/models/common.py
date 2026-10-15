from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn


MODALITIES = ("composition", "structure", "charge_density")


class ProjectionAdapter(nn.Module):
    """Map an encoder-specific pooled representation to the common latent."""

    def __init__(self, input_dim: int, latent_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"adapter expects [batch, features], got {inputs.shape}")
        return self.network(inputs)


class GatedModalityFusion(nn.Module):
    """Simple composition-anchored fusion shared by every information condition.

    Structure and charge receive element-wise gates computed from a fixed-width
    context. Missing modalities are represented by zeros, keeping the downstream
    network definition identical for C, C+S, C+rho, and C+S+rho.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        context_dim = len(MODALITIES) * latent_dim
        self.latent_dim = latent_dim
        self.gates = nn.ModuleDict(
            {
                name: nn.Linear(context_dim, latent_dim)
                for name in ("structure", "charge_density")
            }
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        latents: Mapping[str, Tensor],
        active_modalities: set[str] | frozenset[str],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        unknown = set(active_modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        if "composition" not in active_modalities:
            raise ValueError("composition is the required anchor modality")
        if "composition" not in latents:
            raise ValueError("composition latent is missing")

        reference = latents["composition"]
        if reference.ndim != 2 or reference.shape[-1] != self.latent_dim:
            raise ValueError("composition latent has the wrong shape")
        zeros = torch.zeros_like(reference)
        ordered: list[Tensor] = []
        for name in MODALITIES:
            value = latents.get(name, zeros)
            if value.shape != reference.shape:
                raise ValueError(f"{name} latent shape does not match composition")
            ordered.append(value if name in active_modalities else zeros)
        context = torch.cat(ordered, dim=-1)

        fused = reference
        gate_values: dict[str, Tensor] = {}
        for name in ("structure", "charge_density"):
            gate = torch.sigmoid(self.gates[name](context))
            gate_values[name] = gate
            if name in active_modalities:
                fused = fused + gate * latents[name]
        return self.output_norm(fused), gate_values


class ControlledFamilyHead(nn.Module):
    """The shared prediction-network template used for every modality condition."""

    def __init__(
        self,
        output_shape: tuple[int, ...],
        latent_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not output_shape or any(value <= 0 for value in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        self.output_shape = tuple(int(value) for value in output_shape)
        output_dim = math.prod(self.output_shape)
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2:
            raise ValueError(f"head expects [batch, features], got {latent.shape}")
        output = self.network(latent)
        return output.reshape(output.shape[0], *self.output_shape)
