"""One paper-facing implementation of every Task 1 information condition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .common import ControlledFamilyHead, GatedModalityFusion, ProjectionAdapter
from .composition import ElementFractionSetEncoder


STRUCTURE_ENCODERS = ("cgcnn", "ecomformer", "alignn")
CHARGE_ENCODERS = ("mcdc", "mcdc_high", "pointvector")


class MultimodalPropertyModel(nn.Module):
    """Composition-anchored model shared by all reported T1 conditions.

    Only requested encoder branches are constructed. Every branch is projected
    to the same latent width, then combined by the same gated residual fusion
    and target-family head. This makes the paper's controlled information
    conditions explicit without duplicating their forward logic.
    """

    def __init__(
        self,
        output_shape: tuple[int, ...],
        *,
        structure_encoder_name: str | None = None,
        charge_encoder_name: str | None = None,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.structure_encoder_name = structure_encoder_name
        self.charge_encoder_name = charge_encoder_name

        self.composition_encoder = ElementFractionSetEncoder(128, 128)
        self.composition_adapter = ProjectionAdapter(128, latent_dim, dropout)

        self.structure_encoder: nn.Module | None = None
        self.structure_adapter: nn.Module | None = None
        if structure_encoder_name is not None:
            self.structure_encoder, width = self._build_structure_encoder(
                structure_encoder_name
            )
            self.structure_adapter = ProjectionAdapter(width, latent_dim, dropout)

        self.charge_encoder: nn.Module | None = None
        self.charge_adapter: nn.Module | None = None
        if charge_encoder_name is not None:
            self.charge_encoder, width = self._build_charge_encoder(
                charge_encoder_name
            )
            self.charge_adapter = ProjectionAdapter(width, latent_dim, dropout)

        active = {"composition"}
        if self.structure_encoder is not None:
            active.add("structure")
        if self.charge_encoder is not None:
            active.add("charge_density")
        self.active_modalities = frozenset(active)

        self.fusion = GatedModalityFusion(latent_dim)
        self.head = ControlledFamilyHead(
            output_shape, latent_dim, hidden_dim, dropout
        )

    @staticmethod
    def _build_structure_encoder(name: str) -> tuple[nn.Module, int]:
        if name == "cgcnn":
            from .cgcnn import CGCNNEncoder

            return CGCNNEncoder(92, 31), 128
        if name == "ecomformer":
            from .comformer import EComFormerEncoder

            return EComFormerEncoder(), EComFormerEncoder.output_dim
        if name == "alignn":
            from .alignn import ALIGNNEncoder

            return ALIGNNEncoder(), ALIGNNEncoder.output_dim
        raise ValueError(
            f"unknown structure encoder {name!r}; expected one of "
            f"{STRUCTURE_ENCODERS}"
        )

    @staticmethod
    def _build_charge_encoder(name: str) -> tuple[nn.Module, int]:
        if name in {"mcdc", "mcdc_high"}:
            from .mcdc_charge import MCDCChargeEncoder

            shape = (56, 40, 40) if name == "mcdc" else (56, 80, 80)
            return MCDCChargeEncoder(input_shape=shape), MCDCChargeEncoder.output_dim
        if name == "pointvector":
            from .pointvector_charge import PointVectorPBCEncoder

            return PointVectorPBCEncoder(), PointVectorPBCEncoder.output_dim
        raise ValueError(
            f"unknown charge encoder {name!r}; expected one of {CHARGE_ENCODERS}"
        )

    def _encode_structure(self, batch: Mapping[str, Any]) -> Tensor:
        if self.structure_encoder is None or self.structure_adapter is None:
            raise RuntimeError("this model has no structure branch")
        if self.structure_encoder_name == "cgcnn":
            encoded = self.structure_encoder(
                batch["atom_features"],
                batch["neighbor_features"],
                batch["neighbor_indices"],
                batch["crystal_atom_indices"],
            )
        elif self.structure_encoder_name == "ecomformer":
            encoded = self.structure_encoder(batch["structure_graph"])
        elif self.structure_encoder_name == "alignn":
            graph, line_graph = batch["structure_graph"]
            encoded = self.structure_encoder(graph, line_graph)
        else:
            raise RuntimeError("unreachable structure encoder dispatch")
        return self.structure_adapter(encoded)

    def encode_modalities(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        latents = {
            "composition": self.composition_adapter(
                self.composition_encoder(
                    batch["composition_z"], batch["composition_fraction"]
                )
            )
        }
        if self.structure_encoder is not None:
            latents["structure"] = self._encode_structure(batch)
        if self.charge_encoder is not None:
            if self.charge_adapter is None:
                raise RuntimeError("charge adapter is missing")
            latents["charge_density"] = self.charge_adapter(
                self.charge_encoder(batch["charge_density"])
            )
        return latents

    def predict_from_latents(
        self,
        latents: Mapping[str, Tensor],
        *,
        active_modalities: set[str] | frozenset[str] | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        active = self.active_modalities if active_modalities is None else frozenset(
            active_modalities
        )
        if not active.issubset(self.active_modalities):
            unavailable = sorted(active - self.active_modalities)
            raise ValueError(f"requested unconfigured modalities: {unavailable}")
        fused, gates = self.fusion(latents, active)
        prediction = self.head(fused)
        if not return_diagnostics:
            return prediction

        anchor_norm = latents["composition"].norm(dim=-1).clamp_min(1.0e-8)
        contributions: dict[str, Tensor] = {}
        ratios: dict[str, Tensor] = {}
        for name in ("structure", "charge_density"):
            if name not in latents:
                continue
            contribution = gates[name] * latents[name]
            contributions[name] = contribution
            ratios[name] = contribution.norm(dim=-1) / anchor_norm
        return prediction, {
            "gates": gates,
            "contributions": contributions,
            "contribution_ratio_to_composition": ratios,
            "fused_latent": fused,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        active_modalities: set[str] | frozenset[str] | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        return self.predict_from_latents(
            self.encode_modalities(batch),
            active_modalities=active_modalities,
            return_diagnostics=return_diagnostics,
        )


def permute_modality_latent(
    latents: Mapping[str, Tensor], modality: str, permutation: Tensor
) -> dict[str, Tensor]:
    """Return a shallow latent copy with one non-composition branch permuted."""

    if modality not in {"structure", "charge_density"}:
        raise ValueError("only structure or charge_density may be permuted")
    if modality not in latents:
        raise KeyError(modality)
    if permutation.ndim != 1 or len(permutation) != len(latents[modality]):
        raise ValueError("permutation must have one index per batch sample")
    result = dict(latents)
    result[modality] = latents[modality].index_select(
        0, permutation.to(latents[modality].device)
    )
    return result


__all__ = [
    "CHARGE_ENCODERS",
    "STRUCTURE_ENCODERS",
    "MultimodalPropertyModel",
    "permute_modality_latent",
]
