"""Backward-compatible full T1 fusion interface."""

from __future__ import annotations

from .multimodal import MultimodalPropertyModel, permute_modality_latent


STRUCTURE_ENCODERS = ("cgcnn", "ecomformer", "alignn")
CHARGE_ENCODERS = ("mcdc", "pointvector")
ALL_MODALITIES = frozenset({"composition", "structure", "charge_density"})


class CompositionStructureChargeModel(MultimodalPropertyModel):
    def __init__(
        self,
        output_shape: tuple[int, ...],
        *,
        structure_encoder_name: str,
        charge_encoder_name: str,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            output_shape,
            structure_encoder_name=structure_encoder_name,
            charge_encoder_name=charge_encoder_name,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


__all__ = [
    "ALL_MODALITIES",
    "CHARGE_ENCODERS",
    "STRUCTURE_ENCODERS",
    "CompositionStructureChargeModel",
    "permute_modality_latent",
]
