"""Role-free MACE adapter for HEA local structure patches.

The retrieval data already contains the periodic images required by a local
patch.  This adapter therefore builds a *non-periodic* radius graph over the
valid patch atoms, runs the feature-producing part of the official MACE
backbone, and mean-pools the final invariant (``l=0``) node features.

The upstream MACE package is deliberately not modified.  MACE is normally
used as an energy/force model, so its public ``forward`` also constructs an
energy readout and (by default) force derivatives.  Retrieval does not need
either of those paths; calling the embedding, radial, interaction, and
product modules directly keeps the adapter feature-only and avoids treating
an arbitrary zero-energy readout as an embedding.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..common import ProjectionHead, assert_finite


# The structure shard contains exactly these 60 Z values.  Keeping the table
# in the model (rather than deriving it from each batch) makes checkpoints
# self-describing and prevents a validation/test element from silently
# changing the one-hot basis.
DEFAULT_HEA_SPECIES_Z: tuple[int, ...] = (
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MACE_SITE_PACKAGES = REPO_ROOT / ".external" / "mace-site-packages"


def _load_mace_modules():
    """Import MACE lazily and return its modules plus e3nn's ``o3``.

    The ordinary retrieval environment intentionally uses e3nn 0.5.x for the
    other encoders, while mace-torch 0.3.x pins e3nn 0.4.4.  Importing MACE
    lazily lets G/E/P/X data-only code continue to work, and the explicit
    path check gives a useful error instead of silently mixing incompatible
    e3nn versions.  A caller can set ``HEA_MACE_SITE_PACKAGES`` to an audited
    environment; that optional path is only a
    fallback.
    """

    # e3nn 0.4.4 loads its packaged, trusted ``constants.pt`` without an
    # explicit ``weights_only`` argument.  PyTorch >=2.6 defaults that load to
    # weights-only mode; the file contains only tensors plus builtin slices.
    # Allowlisting exactly ``slice`` is sufficient even when this adapter is
    # imported before MACE's own legacy PyTorch compatibility hook executes.
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([slice])

    # If MACE is already importable, use it.  This is the normal path inside
    # the dedicated M/P launcher.
    try:
        mace = importlib.import_module("mace")
        modules = importlib.import_module("mace.modules")
        o3 = importlib.import_module("e3nn.o3")
    except Exception as first_error:
        candidates: list[Path] = []
        configured = os.environ.get("HEA_MACE_SITE_PACKAGES", "").strip()
        if configured:
            candidates.append(Path(configured))
        candidates.append(DEFAULT_MACE_SITE_PACKAGES)
        last_error: BaseException = first_error
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                mace = importlib.import_module("mace")
                modules = importlib.import_module("mace.modules")
                o3 = importlib.import_module("e3nn.o3")
                break
            except Exception as exc:  # pragma: no cover - diagnostic branch
                last_error = exc
        else:
            raise ImportError(
                "MACE is unavailable. Run through the dedicated M/P launcher "
                "or set HEA_MACE_SITE_PACKAGES to a mace-torch installation."
            ) from last_error

    # MACE 0.3.x was built against e3nn 0.4.4.  A different e3nn version can
    # fail much later inside a tensor product, so reject it at construction.
    try:
        e3nn_version = importlib.metadata.version("e3nn")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        e3nn_version = getattr(importlib.import_module("e3nn"), "__version__", "unknown")
    if str(e3nn_version) != "0.4.4":
        raise RuntimeError(
            "MACE adapter requires e3nn==0.4.4 (the tested mace-torch 0.3.x "
            f"contract), but imported {e3nn_version!r}. Use the isolated M/P "
            "environment; do not install MACE into the E/P retrieval env."
        )
    return mace, modules, o3


def _as_irreps(o3: Any, value: Any):
    """Convert a config value to an e3nn Irreps object without eager imports."""

    return value if isinstance(value, o3.Irreps) else o3.Irreps(str(value))


def _validate_species(species_z: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(z) for z in species_z)
    if not values:
        raise ValueError("species_z must contain at least one atomic number")
    if values != tuple(sorted(set(values))):
        raise ValueError("species_z must be sorted and contain no duplicates")
    if any(z <= 0 for z in values):
        raise ValueError("species_z must contain positive atomic numbers")
    return values


def build_radius_graph(
    z: Tensor,
    xyz: Tensor,
    mask: Tensor,
    cutoff: float,
    max_neighbors: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a deterministic directed radius graph for padded patch tensors.

    Returns ``(edge_index, batch_index, ptr, neighbor_counts)``.  The graph
    contains no self loops and no periodic shifts: periodic images are already
    explicit rows in the patch shard.  ``edge_index[0]`` is the sender and
    ``edge_index[1]`` is the receiver, matching MACE's interaction blocks.

    Patch sizes are only about 6--42 atoms, so per-graph ``cdist`` is both
    simpler and more memory-safe than allocating a padded batch-wide distance
    matrix.  Ties are resolved by the original local index; when a cap is
    active this makes selection reproducible without changing the physical
    aggregation.
    """

    if z.ndim != 2 or xyz.ndim != 3 or xyz.shape[:2] != z.shape or xyz.shape[-1] != 3:
        raise ValueError(
            f"expected z [B,N] and xyz [B,N,3], got {tuple(z.shape)} and {tuple(xyz.shape)}"
        )
    if mask.shape != z.shape:
        raise ValueError("mask must have the same shape as z")
    if cutoff <= 0 or not np.isfinite(float(cutoff)):
        raise ValueError("cutoff must be a positive finite value")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")

    mask = mask.bool()
    edge_src: list[Tensor] = []
    edge_dst: list[Tensor] = []
    batch_parts: list[Tensor] = []
    ptr_values = [0]
    neighbor_parts: list[Tensor] = []
    offset = 0
    for batch_id in range(int(z.shape[0])):
        local_indices = torch.where(mask[batch_id])[0]
        n = int(local_indices.numel())
        if n <= 0:
            raise ValueError(f"patch {batch_id} has no valid atoms")
        pos = xyz[batch_id, local_indices]
        if not torch.isfinite(pos).all():
            raise ValueError(f"patch {batch_id} contains non-finite coordinates")
        z_local = z[batch_id, local_indices]
        distances = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        local_arange = torch.arange(n, device=pos.device)
        counts = torch.zeros(n, dtype=torch.long, device=pos.device)
        for receiver in range(n):
            candidates = torch.where(
                (distances[:, receiver] <= float(cutoff))
                & (local_arange != receiver)
            )[0]
            if candidates.numel() > max_neighbors:
                # torch.where returns ascending local indices; stable sorting by
                # distance therefore gives distance/local-index tie breaking.
                order = torch.argsort(
                    distances[candidates, receiver], stable=True
                )[:max_neighbors]
                candidates = candidates[order]
            counts[receiver] = candidates.numel()
            if candidates.numel() == 0:
                continue
            edge_src.append(candidates + offset)
            edge_dst.append(torch.full_like(candidates, offset + receiver))
        batch_parts.append(torch.full((n,), batch_id, dtype=torch.long, device=pos.device))
        neighbor_parts.append(counts)
        offset += n
        ptr_values.append(offset)

    if not edge_src:
        raise ValueError(
            "no radius edges were found; increase r_max or inspect the local patch"
        )
    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    batch_index = torch.cat(batch_parts, dim=0)
    neighbor_counts = torch.cat(neighbor_parts, dim=0)
    ptr = torch.tensor(ptr_values, dtype=torch.long, device=z.device)
    # Explicitly assert the graph invariant here.  This catches accidental
    # self-loops before they reach a tensor product and is cheap at patch size.
    if torch.any(edge_index[0] == edge_index[1]):
        raise RuntimeError("MACE radius graph unexpectedly contains self-loops")
    return edge_index, batch_index, ptr, neighbor_counts


class MACEStructureEncoder(nn.Module):
    """Encode a role-free HEA structure patch with randomly initialized MACE.

    ``role`` remains an optional positional argument solely for compatibility
    with :class:`~hea_retrieval.models.common.RetrievalModel`; it is never
    read.  The returned vector is a normalized ``output_dim``-dimensional
    embedding suitable for the existing symmetric in-batch InfoNCE loss.
    """

    def __init__(
        self,
        *,
        output_dim: int = 256,
        projection_hidden_dim: int = 512,
        r_max: float = 5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 6,
        max_ell: int = 1,
        num_interactions: int = 3,
        correlation: int = 3,
        hidden_irreps: str = "64x0e + 64x1o",
        MLP_irreps: str = "16x0e",
        species_z: Sequence[int] | None = None,
        avg_num_neighbors: float = 15.1,
        max_neighbors: int = 64,
        interaction: str = "RealAgnosticResidualInteractionBlock",
        interaction_first: str = "RealAgnosticResidualInteractionBlock",
        gate: str = "silu",
        radial_MLP: Sequence[int] | None = None,
        use_reduced_cg: bool = True,
        use_agnostic_product: bool = False,
    ) -> None:
        super().__init__()
        if num_interactions < 2:
            raise ValueError(
                "MACE retrieval adapter requires at least two interactions so "
                "the final block has an invariant scalar output"
            )
        if num_bessel <= 0 or num_polynomial_cutoff <= 0 or max_ell < 0:
            raise ValueError("MACE basis and max_ell values must be non-negative/positive")
        if correlation <= 0:
            raise ValueError("correlation must be positive")
        if not np.isfinite(float(avg_num_neighbors)) or avg_num_neighbors <= 0:
            raise ValueError("avg_num_neighbors must be a positive finite value")

        _mace, modules, o3 = _load_mace_modules()
        species = _validate_species(species_z or DEFAULT_HEA_SPECIES_Z)
        self.register_buffer(
            "species_z", torch.tensor(species, dtype=torch.long), persistent=True
        )
        self.r_max = float(r_max)
        self.max_neighbors = int(max_neighbors)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.num_elements = len(species)
        self.mace_version = importlib.metadata.version("mace-torch")

        hidden = _as_irreps(o3, hidden_irreps)
        mlp = _as_irreps(o3, MLP_irreps)
        if len(hidden) == 0 or hidden[0][1].l != 0 or hidden[0][1].p != 1:
            raise ValueError(
                "hidden_irreps must start with an even scalar (for example 64x0e)"
            )
        # The official MACE constructor selects hidden_irreps[0] for its final
        # interaction.  Validate that this really is scalar before pooling.
        first_scalar_mul = int(hidden[0][0])
        if first_scalar_mul <= 0:
            raise ValueError("the leading scalar multiplicity must be positive")

        interactions = getattr(modules, "interaction_classes", {})
        try:
            interaction_cls = interactions[interaction]
            interaction_cls_first = interactions[interaction_first]
            gate_fn = getattr(modules, "gate_dict")[gate]
        except KeyError as exc:
            raise ValueError(f"unknown MACE interaction or gate: {exc.args[0]!r}") from exc

        radial_mlp = list(radial_MLP or [64, 64, 64])
        if not radial_mlp or any(int(value) <= 0 for value in radial_mlp):
            raise ValueError("radial_MLP must contain positive widths")

        # ``atomic_energies`` and readouts are retained only because the
        # official MACE class requires them.  Retrieval never uses the energy
        # path; all energy baselines are zero and no foundation checkpoint is
        # loaded.
        self.backbone = modules.MACE(
            r_max=self.r_max,
            num_bessel=int(num_bessel),
            num_polynomial_cutoff=int(num_polynomial_cutoff),
            max_ell=int(max_ell),
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=int(num_interactions),
            num_elements=self.num_elements,
            hidden_irreps=hidden,
            MLP_irreps=mlp,
            atomic_energies=np.zeros(self.num_elements, dtype=np.float64),
            avg_num_neighbors=self.avg_num_neighbors,
            atomic_numbers=list(species),
            correlation=int(correlation),
            gate=gate_fn,
            use_reduced_cg=bool(use_reduced_cg),
            use_last_readout_only=True,
            use_agnostic_product=bool(use_agnostic_product),
            radial_MLP=radial_mlp,
        )

        # The final product is deliberately scalar-only in the official MACE
        # constructor (it uses ``str(hidden_irreps[0])``).  Read its public
        # irreps metadata instead of hard-coding a channel count.
        final_irreps = self.backbone.products[-1].linear.irreps_out
        if any(ir.l != 0 or ir.p != 1 for _, ir in final_irreps):
            raise RuntimeError(
                f"MACE final product is not an even scalar representation: {final_irreps}"
            )
        self.backbone_dim = int(final_irreps.dim)
        if self.backbone_dim != first_scalar_mul:
            raise RuntimeError(
                "unexpected MACE final scalar width: "
                f"metadata={self.backbone_dim}, hidden_irreps={first_scalar_mul}"
            )
        self.projection = ProjectionHead(
            self.backbone_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=output_dim,
            dropout=0.1,
        )
        self.output_dim = int(output_dim)

    def _species_indices(self, z: Tensor) -> Tensor:
        indices = torch.searchsorted(self.species_z, z)
        invalid = (indices >= self.species_z.numel()) | (
            self.species_z[indices.clamp_max(self.species_z.numel() - 1)] != z
        )
        if torch.any(invalid):
            bad = z[invalid][:8].detach().cpu().tolist()
            raise ValueError(
                f"structure patch contains elements outside the fixed MACE species table: {bad}"
            )
        return indices

    def _feature_forward(
        self,
        node_attrs: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Run only MACE's node-feature-producing path."""

        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender]
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        if not torch.isfinite(lengths).all() or torch.any(lengths <= 0):
            raise ValueError("MACE graph has invalid edge lengths")

        backbone = self.backbone
        node_feats = backbone.node_embedding(node_attrs)
        edge_attrs = backbone.spherical_harmonics(vectors)
        edge_feats, cutoff = backbone.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            backbone.atomic_numbers,
        )
        final: Optional[Tensor] = None
        for index, (interaction, product) in enumerate(
            zip(backbone.interactions, backbone.products)
        ):
            node_feats, skip = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=index == 0,
                lammps_class=None,
                lammps_natoms=(0, 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=skip,
                node_attrs=node_attrs,
            )
            if index == len(backbone.products) - 1:
                final = node_feats
        if final is None:  # pragma: no cover - guarded by constructor
            raise RuntimeError("MACE produced no final node features")
        if final.ndim != 2 or final.shape[1] != self.backbone_dim:
            raise RuntimeError(
                f"unexpected final MACE feature shape: {tuple(final.shape)}"
            )
        return final

    def forward(
        self,
        z: Tensor,
        xyz: Tensor,
        role: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # ``role`` is intentionally unused.  Keep the argument for the common
        # RetrievalModel interface, but do not even validate/read it.
        del role
        if z.ndim != 2 or xyz.ndim != 3 or xyz.shape[:2] != z.shape or xyz.shape[-1] != 3:
            raise ValueError(
                f"expected z [B,N], xyz [B,N,3], got {tuple(z.shape)} and {tuple(xyz.shape)}"
            )
        if mask is None:
            mask = z.ne(0)
        elif mask.shape != z.shape:
            raise ValueError("mask must have the same shape as z")
        mask = mask.bool()
        if torch.any(mask & z.eq(0)):
            raise ValueError("valid MACE nodes cannot have the padding atomic number 0")
        if not torch.isfinite(xyz).all():
            raise ValueError("MACE input coordinates contain NaN/Inf")

        # Keep all graph and feature tensors in the backbone's parameter dtype;
        # this also makes a future float64 diagnostic explicit instead of relying on
        # e3nn's implicit casts.
        parameter = next(self.backbone.parameters())
        positions = xyz.to(device=parameter.device, dtype=parameter.dtype)
        z = z.to(device=parameter.device, dtype=torch.long)
        mask = mask.to(device=parameter.device)
        flat_z = z[mask]
        flat_positions = positions[mask]
        batch_index = (
            torch.arange(z.shape[0], device=z.device)
            .unsqueeze(1)
            .expand_as(z)[mask]
        )
        counts = mask.sum(dim=1).long()
        if torch.any(counts <= 0):
            raise ValueError("MACE received an empty structure patch")

        species_indices = self._species_indices(flat_z)
        node_attrs = F.one_hot(
            species_indices, num_classes=self.num_elements
        ).to(dtype=flat_positions.dtype)
        edge_index, graph_batch, _ptr, _neighbor_counts = build_radius_graph(
            z,
            positions,
            mask,
            cutoff=self.r_max,
            max_neighbors=self.max_neighbors,
        )
        # ``graph_batch`` is constructed from the same mask/order as
        # ``batch_index``.  Check it once in debug-friendly code; using the
        # graph helper's result for pooling avoids relying on padded indices.
        if graph_batch.shape != batch_index.shape or not torch.equal(graph_batch, batch_index):
            raise RuntimeError("MACE graph batch index is inconsistent with the mask")

        node_features = self._feature_forward(node_attrs, flat_positions, edge_index)
        pooled = node_features.new_zeros((z.shape[0], self.backbone_dim))
        pooled.index_add_(0, graph_batch, node_features)
        pooled = pooled / counts.to(dtype=pooled.dtype).unsqueeze(-1)
        out = self.projection(pooled)
        assert_finite("MACE embedding", out)
        return out


__all__ = [
    "DEFAULT_HEA_SPECIES_Z",
    "MACEStructureEncoder",
    "build_radius_graph",
]
