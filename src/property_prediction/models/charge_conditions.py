"""Backward-compatible name for T1 composition-plus-field conditions."""

from __future__ import annotations

from .multimodal import MultimodalPropertyModel


class CompositionChargeModel(MultimodalPropertyModel):
    def __init__(
        self,
        output_shape: tuple[int, ...],
        *,
        encoder_name: str,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            output_shape,
            charge_encoder_name=encoder_name,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
