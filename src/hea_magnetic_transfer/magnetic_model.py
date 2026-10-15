"""Structure-only S0, S-DFT, and S-Gauss magnetic-transfer models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hea_retrieval.models.common import assert_finite
from hea_retrieval.models.structure.equiformer import EquiformerStructureEncoder

from .conditions import normalize_condition, uses_pretrained_encoder
from .pretrain_data import EXPECTED_ATOMS


from .paths import DEFAULT_PRETRAINED_CHECKPOINT

DEFAULT_STRUCTURE_KWARGS: dict[str, Any] = {
    "sphere_channels": 128,
    "num_layers": 8,
    "lmax_list": [4],
    "mmax_list": [2],
    "max_radius": 5.0,
    "max_neighbors": 64,
    "max_num_elements": 100,
    "attn_hidden_channels": 128,
    "ffn_hidden_channels": 512,
    "grid_resolution": 18,
}


class MomentHead(nn.Module):
    """Small scalar head shared by all controlled initialization conditions."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("moment-head dimensions must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("moment-head dropout must be in [0,1)")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


class MagneticMomentModel(nn.Module):
    """Predict 100 site moments and an independent cell moment per atom.

    The encoder sees one atom-centred local patch at a time.  A parent batch is
    flattened to ``B_parent * 100`` patches by the collate function, then
    restored here to obtain one site embedding per atom and a mean-pooled cell
    embedding.  The cell head is intentionally independent of the sum of site
    predictions because OSZICAR includes interstitial magnetization.
    """

    def __init__(
        self,
        structure_encoder: nn.Module,
        *,
        embedding_dim: int = 256,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
        encoder_site_batch_size: int | None = None,
        encoder_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.structure_encoder = structure_encoder
        self.embedding_dim = int(embedding_dim)
        if encoder_site_batch_size is not None and int(encoder_site_batch_size) <= 0:
            raise ValueError("encoder_site_batch_size must be positive or None")
        self.encoder_site_batch_size = (
            int(encoder_site_batch_size)
            if encoder_site_batch_size is not None
            else None
        )
        self.encoder_gradient_checkpointing = bool(encoder_gradient_checkpointing)
        self.site_head = MomentHead(
            self.embedding_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
        )
        self.cell_head = MomentHead(
            self.embedding_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
        )

    def encode_sites(self, batch: dict[str, Any]) -> Tensor:
        z = batch["structure_z"]
        xyz = batch["structure_xyz"]
        role = batch.get("structure_role")
        mask = batch.get("structure_mask")
        edge_index = batch.get("structure_edge_index")
        node_offsets = batch.get("structure_node_offsets")
        edge_offsets = batch.get("structure_edge_offsets")
        edge_parts = (edge_index, node_offsets, edge_offsets)
        if any(value is not None for value in edge_parts) and not all(
            value is not None for value in edge_parts
        ):
            raise ValueError("precomputed graph requires edge index and both offsets")
        if node_offsets is not None and tuple(node_offsets.shape) != (z.shape[0] + 1,):
            raise ValueError("structure_node_offsets has an unexpected shape")
        if edge_offsets is not None and tuple(edge_offsets.shape) != (z.shape[0] + 1,):
            raise ValueError("structure_edge_offsets has an unexpected shape")
        if edge_index is not None and (edge_index.ndim != 2 or edge_index.shape[0] != 2):
            raise ValueError("structure_edge_index must have shape [2,E]")
        site_batch_size = self.encoder_site_batch_size or int(z.shape[0])
        chunks: list[Tensor] = []
        for start in range(0, int(z.shape[0]), site_batch_size):
            stop = min(start + site_batch_size, int(z.shape[0]))
            chunk_args = (
                z[start:stop],
                xyz[start:stop],
                role[start:stop] if role is not None else None,
                mask[start:stop] if mask is not None else None,
            )
            encoder_args = chunk_args
            if edge_index is not None:
                node_start = int(node_offsets[start])
                edge_start = int(edge_offsets[start])
                edge_stop = int(edge_offsets[stop])
                chunk_edge_index = edge_index[:, edge_start:edge_stop] - node_start
                encoder_args = (*chunk_args, chunk_edge_index)
            if (
                self.encoder_gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and role is not None
                and mask is not None
            ):
                encoded = checkpoint(
                    self.structure_encoder,
                    *encoder_args,
                    use_reentrant=False,
                )
            else:
                encoded = self.structure_encoder(*encoder_args)
            chunks.append(encoded)
        flat = torch.cat(chunks, dim=0)
        parent_count = len(batch["parent_id"])
        expected = parent_count * EXPECTED_ATOMS
        if tuple(flat.shape) != (expected, self.embedding_dim):
            raise RuntimeError(
                f"unexpected flat site embedding shape {tuple(flat.shape)}; "
                f"expected {(expected, self.embedding_dim)}"
            )
        embeddings = flat.reshape(parent_count, EXPECTED_ATOMS, self.embedding_dim)
        assert_finite("magnetic site embeddings", embeddings)
        return embeddings

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        site_embeddings = self.encode_sites(batch)
        site_moments = self.site_head(site_embeddings)
        cell_embedding = site_embeddings.mean(dim=1)
        cell_moment_per_atom = self.cell_head(cell_embedding)
        assert_finite("predicted site moments", site_moments)
        assert_finite("predicted cell moment per atom", cell_moment_per_atom)
        return {
            "site_moments": site_moments,
            "cell_moment_per_atom": cell_moment_per_atom,
            "site_embeddings": site_embeddings,
            "cell_embedding": cell_embedding,
        }


def build_magnetic_model(
    *,
    output_dim: int = 256,
    structure_kwargs: dict[str, Any] | None = None,
    head_hidden_dim: int = 128,
    head_dropout: float = 0.1,
    encoder_site_batch_size: int | None = None,
    encoder_gradient_checkpointing: bool = False,
) -> MagneticMomentModel:
    kwargs = dict(DEFAULT_STRUCTURE_KWARGS)
    kwargs.update(structure_kwargs or {})
    encoder = EquiformerStructureEncoder(output_dim=output_dim, **kwargs)
    return MagneticMomentModel(
        encoder,
        embedding_dim=output_dim,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        encoder_site_batch_size=encoder_site_batch_size,
        encoder_gradient_checkpointing=encoder_gradient_checkpointing,
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid NSP pretraining checkpoint: {path}")
    return payload


def load_pretrained_structure_encoder(
    model: MagneticMomentModel,
    checkpoint: str | Path = DEFAULT_PRETRAINED_CHECKPOINT,
) -> dict[str, Any]:
    """Strictly load only ``structure_encoder.*`` and leave both heads untouched."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_checkpoint(path)
    prefix = "structure_encoder."
    state = {
        key[len(prefix) :]: value
        for key, value in payload["model"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"checkpoint contains no {prefix} state: {path}")
    incompatible = model.structure_encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict encoder load failed: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    args = payload.get("args", {})
    return {
        "checkpoint": str(path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_step": int(payload.get("global_step", -1)),
        "checkpoint_best_score": float(payload.get("best_score", float("nan"))),
        "state_tensors": len(state),
        "source_objective": args.get("objective"),
        "source_from_scratch": args.get("from_scratch"),
        "source_field": args.get("field_source"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def initialize_magnetic_condition(
    mode: str,
    *,
    seed: int,
    checkpoint: str | Path = DEFAULT_PRETRAINED_CHECKPOINT,
    output_dim: int = 256,
    structure_kwargs: dict[str, Any] | None = None,
    head_hidden_dim: int = 128,
    head_dropout: float = 0.1,
    encoder_site_batch_size: int | None = None,
    encoder_gradient_checkpointing: bool = False,
) -> tuple[MagneticMomentModel, dict[str, Any]]:
    """Create one controlled downstream initialization.

    S0 is random. S-DFT and S-Gauss strictly load only the structure encoder
    from their respective pretraining checkpoint; both magnetic heads remain
    identically random.
    """

    condition = normalize_condition(mode)
    torch.manual_seed(int(seed))
    model = build_magnetic_model(
        output_dim=output_dim,
        structure_kwargs=structure_kwargs,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        encoder_site_batch_size=encoder_site_batch_size,
        encoder_gradient_checkpointing=encoder_gradient_checkpointing,
    )
    metadata: dict[str, Any] = {
        "mode": condition,
        "condition": condition,
        "seed": int(seed),
        "encoder_initialization": "random",
        "heads_initialization": "random",
        "optimizer_state_loaded": False,
        "charge_encoder_used": False,
        "encoder_site_batch_size": encoder_site_batch_size,
        "encoder_gradient_checkpointing": bool(encoder_gradient_checkpointing),
    }
    if uses_pretrained_encoder(condition):
        metadata.update(load_pretrained_structure_encoder(model, checkpoint))
        metadata["encoder_initialization"] = {
            "s_dft": "nsp_dft_charge_alignment_best",
            "s_gauss": "nsp_gaussian_field_alignment_best",
        }[condition]
    return model, metadata


def initialize_s0_or_s1(mode: str, **kwargs):
    """Compatibility alias for historical callers and checkpoints."""

    return initialize_magnetic_condition(mode, **kwargs)


__all__ = [
    "DEFAULT_PRETRAINED_CHECKPOINT",
    "DEFAULT_STRUCTURE_KWARGS",
    "MomentHead",
    "MagneticMomentModel",
    "build_magnetic_model",
    "initialize_magnetic_condition",
    "initialize_s0_or_s1",
    "load_pretrained_structure_encoder",
]
