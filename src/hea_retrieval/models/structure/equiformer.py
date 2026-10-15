"""HEA adapter for the official EquiformerV2 OC20 implementation.

The upstream class is kept untouched.  This adapter supplies a non-periodic
graph over the already-expanded local images and exposes the post-norm scalar
(``l=0,m=0``) node features before the energy head.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import Tensor, nn

from ..common import ProjectionHead, RoleEmbedding, assert_finite, masked_mean


REPO_ROOT = Path(__file__).resolve().parents[3]
EQUIFORMER_ROOT = Path(os.environ.get("HEABENCH_EQUIFORMER_ROOT", str(REPO_ROOT / "third_party" / "equiformer_v2")))
OCP_ROOT = Path(os.environ.get("HEABENCH_OCP_ROOT", str(
    REPO_ROOT
    / "third_party"
    / "EDBench"
    / "src"
    / "retrival_and_generation_tasks"
    / "Retrival"
    / "geoformer"
    / "fairchem-f83d150b5fcb940b814f25043fb00914945b8708"
)))


class GraphData(SimpleNamespace):
    """Attribute and mapping access compatible with both OCP code paths."""

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)


def _load_equiformer_module():
    for path in (str(OCP_ROOT), str(EQUIFORMER_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        return importlib.import_module("nets.equiformer_v2.equiformer_v2_oc20")
    except Exception as exc:  # pragma: no cover - exercised in env diagnostics
        raise ImportError(
            "EquiformerV2 dependencies/import failed. Install torch-geometric, "
            "torch-scatter, e3nn and expose the bundled OCP compatibility tree."
        ) from exc


def build_radius_edges(
    z: Tensor,
    xyz: Tensor,
    mask: Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[Tensor, Tensor]:
    """Build deterministic directed edges for explicit local-image patches."""

    if z.ndim != 2 or xyz.ndim != 3 or mask.shape != z.shape:
        raise ValueError("z/xyz/mask shapes are inconsistent")
    src_all: list[int] = []
    dst_all: list[int] = []
    flat_offset = 0
    for b in range(z.shape[0]):
        valid = mask[b].bool()
        pos = xyz[b, valid]
        z_valid = z[b, valid]
        n = int(pos.shape[0])
        if n == 0:
            continue
        dist = torch.cdist(pos, pos)
        for target in range(n):
            # Explicitly remove the diagonal.  ``torch.cdist`` may return a
            # tiny positive value for a self-distance on GPU due to its
            # matrix-multiplication implementation, so a numeric threshold
            # alone is not sufficient.
            candidates = torch.where((dist[:, target] <= cutoff) & (torch.arange(n, device=dist.device) != target))[0]
            if candidates.numel() == 0:
                continue
            # Stable tie-breaking follows the catalog convention as closely as
            # possible: distance, atomic number, local atom index.
            order = sorted(
                candidates.tolist(),
                key=lambda source: (float(dist[source, target]), int(z_valid[source]), int(source)),
            )
            for source in order[:max_neighbors]:
                src_all.append(flat_offset + int(source))
                dst_all.append(flat_offset + int(target))
        flat_offset += n
    if not src_all:
        raise ValueError("no graph edges were found; check cutoff and local patches")
    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long, device=xyz.device)
    # Sort by destination, as expected by the OCP neighbor-count helper and for
    # deterministic aggregation.
    packed_nodes = int(mask.sum())
    order = torch.argsort(edge_index[1] * (packed_nodes + 1) + edge_index[0])
    return edge_index[:, order], xyz.new_zeros((edge_index.shape[1], 3))


class _EquiformerFeatureBackbone(nn.Module):
    """Thin wrapper around the official class with a feature-only forward."""

    def __init__(
        self,
        *,
        sphere_channels: int = 128,
        num_layers: int = 8,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        max_radius: float = 5.0,
        max_neighbors: int = 64,
        max_num_elements: int = 100,
        attn_hidden_channels: int = 128,
        ffn_hidden_channels: int = 512,
        grid_resolution: int = 18,
    ) -> None:
        super().__init__()
        module = _load_equiformer_module()
        self._module = module
        self.model = module.EquiformerV2_OC20(
            num_atoms=0,
            bond_feat_dim=0,
            num_targets=1,
            use_pbc=False,
            regress_forces=False,
            otf_graph=False,
            max_neighbors=max_neighbors,
            max_radius=max_radius,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=8,
            attn_alpha_channels=32,
            attn_value_channels=16,
            ffn_hidden_channels=ffn_hidden_channels,
            norm_type="rms_norm_sh",
            lmax_list=list(lmax_list or [4]),
            mmax_list=list(mmax_list or [2]),
            grid_resolution=grid_resolution,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            attn_activation="scaled_silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="scaled_silu",
            use_gate_act=False,
            use_grid_mlp=False,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,
            weight_init="normal",
        )
        self.cutoff = float(max_radius)
        self.max_neighbors = int(max_neighbors)
        self.sphere_channels = int(sphere_channels)
        self.max_num_elements = int(max_num_elements)
        self.role_embedding = RoleEmbedding(self.model.sphere_channels_all)

    @staticmethod
    def _neighbors(data: GraphData, edge_index: Tensor) -> Tensor:
        return torch.bincount(edge_index[1], minlength=data.pos.shape[0]).to(dtype=torch.long)

    def _generate_graph(self, data: GraphData):
        edge_index = data.edge_index
        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]
        edge_distance = distance_vec.norm(dim=-1)
        cell_offsets = data.pos.new_zeros((edge_index.shape[1], 3))
        return (
            edge_index,
            edge_distance,
            distance_vec,
            cell_offsets,
            cell_offsets,
            self._neighbors(data, edge_index),
        )

    def forward_features(
        self,
        z: Tensor,
        xyz: Tensor,
        role: Optional[Tensor],
        mask: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        model = self.model
        valid = mask.bool()
        flat_z = z[valid].long()
        flat_pos = xyz[valid]
        batch = torch.arange(z.shape[0], device=z.device).unsqueeze(1).expand_as(z)[valid]
        natoms = valid.sum(dim=1).long()
        if edge_index is None:
            edge_index, _ = build_radius_edges(
                z, xyz, valid, self.cutoff, self.max_neighbors
            )
        else:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("precomputed edge_index must have shape [2,E]")
            edge_index = edge_index.to(device=xyz.device, dtype=torch.long)
        data = GraphData(pos=flat_pos, atomic_numbers=flat_z, batch=batch, natoms=natoms, edge_index=edge_index)

        model.batch_size = int(z.shape[0])
        model.dtype = flat_pos.dtype
        model.device = flat_pos.device
        atomic_numbers = flat_z
        num_atoms = int(atomic_numbers.shape[0])
        edge_index, edge_distance, edge_distance_vec, cell_offsets, _, neighbors = self._generate_graph(data)
        edge_rot_mat = model._init_edge_rot_mat(data, edge_index, edge_distance_vec)
        for i in range(model.num_resolutions):
            model.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = self._module.SO3_Embedding(
            num_atoms,
            model.lmax_list,
            model.sphere_channels,
            model.device,
            model.dtype,
        )
        atom_embedding = model.sphere_embedding(atomic_numbers)
        if role is not None:
            flat_role = role[valid]
            atom_embedding = atom_embedding + self.role_embedding(flat_role, torch.ones_like(flat_role, dtype=torch.bool))
        offset = 0
        offset_res = 0
        for resolution in range(model.num_resolutions):
            if model.num_resolutions == 1:
                x.embedding[:, offset_res, :] = atom_embedding
            else:
                x.embedding[:, offset_res, :] = atom_embedding[:, offset : offset + model.sphere_channels]
            offset += model.sphere_channels
            offset_res += int((model.lmax_list[resolution] + 1) ** 2)

        edge_distance = model.distance_expansion(edge_distance)
        edge_degree = model.edge_degree_embedding(atomic_numbers, edge_distance, edge_index)
        x.embedding = x.embedding + edge_degree.embedding
        for block in model.blocks:
            x = block(x, atomic_numbers, edge_distance, edge_index, batch=data.batch)
        x.embedding = model.norm(x.embedding)
        # The first coefficient is the invariant l=0,m=0 scalar channel.
        return x.embedding[:, 0, :], data.batch


class EquiformerStructureEncoder(nn.Module):
    """Encode local structure patches with EquiformerV2 and project to 256-D."""

    def __init__(self, *, output_dim: int = 256, projection_hidden_dim: int = 512, **kwargs) -> None:
        super().__init__()
        self.backbone = _EquiformerFeatureBackbone(**kwargs)
        self.backbone_dim = self.backbone.sphere_channels
        self.projection = ProjectionHead(
            self.backbone_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=output_dim,
            dropout=0.1,
        )
        self.output_dim = output_dim

    def forward(
        self,
        z: Tensor,
        xyz: Tensor,
        role: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is None:
            mask = z.ne(0)
        scalar, batch = self.backbone.forward_features(z, xyz, role, mask, edge_index)
        pooled = scalar.new_zeros((z.shape[0], scalar.shape[-1]))
        pooled.index_add_(0, batch, scalar)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1).to(pooled.dtype)
        out = self.projection(pooled)
        assert_finite("EquiformerV2 embedding", out)
        return out


__all__ = ["EquiformerStructureEncoder", "build_radius_edges", "GraphData"]
