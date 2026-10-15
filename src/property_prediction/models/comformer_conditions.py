"""Backward-compatible name for the eComFormer T1 structure condition."""

from __future__ import annotations

from .multimodal import MultimodalPropertyModel


class CompositionComFormerModel(MultimodalPropertyModel):
    def __init__(
        self,
        output_shape: tuple[int, ...],
        *,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            output_shape,
            structure_encoder_name="ecomformer",
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
