"""HEA adapter for a randomly initialized CHGNet structure tower.

The upstream CHGNet model is an energy/force/magnetic-moment predictor built
around atom, bond, and angle graphs.  HEA retrieval needs a structure feature,
not an energy prediction, so this adapter keeps the CHGNet message-passing
backbone and replaces its crystal readout with a mean-pooled feature followed
by the common HEA projection head.

Important HEA conventions
--------------------------
* The local structure shard already contains explicit periodic images.  The
  adapter therefore builds a *non-periodic* graph in the supplied Cartesian
  coordinates and never asks CHGNet/pymatgen to expand the original cell.
* ``structure_role`` is intentionally ignored.  This is the same role-free
  convention used by the HEA MACE structure adapter and keeps the CHGNet/P
  experiment focused on chemical/geometry features.
* No CHGNet checkpoint is loaded.  ``CHGNet`` is instantiated with
  ``composition_model=None`` and all feature parameters are trained from
  scratch using the retrieval loss.

The adapter lazily imports CHGNet from an audited isolated environment.  This
keeps the ordinary E/G/P/X environment free of CHGNet/pymatgen dependency
conflicts; set ``HEA_CHGNET_SITE_PACKAGES`` to override the default path.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..common import ProjectionHead, assert_finite


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHGNET_SITE_PACKAGES = REPO_ROOT / ".external" / "chgnet-site-packages"
CHGNET_VERSION = "0.4.2"


def _purge_chgnet_modules() -> None:
    """Remove a failed partial import before trying an isolated path."""

    for name in list(sys.modules):
        if name == "chgnet" or name.startswith("chgnet."):
            del sys.modules[name]


def _load_chgnet_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load CHGNet and its graph classes without importing it eagerly.

    The HEA retrieval environment intentionally does not install CHGNet.  The
    tested CHGNet 0.4.2 package is already present in the dedicated
    a separate CHGNet environment; adding that site-packages directory
    to ``sys.path`` is sufficient because both environments use the same
    PyTorch 2.12 ABI on this workstation.
    """

    candidates: list[Path] = []
    candidates.append(Path(sysconfig.get_paths()["purelib"]))
    configured = os.environ.get("HEA_CHGNET_SITE_PACKAGES", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(DEFAULT_CHGNET_SITE_PACKAGES)

    last_error: BaseException | None = None
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            importlib.invalidate_caches()
            _purge_chgnet_modules()
            chgnet = importlib.import_module("chgnet")
            model_module = importlib.import_module("chgnet.model.model")
            graph_module = importlib.import_module("chgnet.graph.crystalgraph")
            graph_impl = importlib.import_module("chgnet.graph.graph")
            model_cls = getattr(chgnet.model, "CHGNet")
            crystal_graph_cls = getattr(graph_module, "CrystalGraph")
            graph_cls = getattr(graph_impl, "Graph")
            node_cls = getattr(graph_impl, "Node")
            batched_graph_cls = getattr(model_module, "BatchedGraph")
            version = importlib.metadata.version("chgnet")
            if version != CHGNET_VERSION:
                raise RuntimeError(
                    f"HEA CHGNet adapter was validated with {CHGNET_VERSION}, "
                    f"but imported {version!r}"
                )
            return (
                chgnet,
                model_cls,
                crystal_graph_cls,
                graph_cls,
                node_cls,
                batched_graph_cls,
            )
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc

    raise ImportError(
        "CHGNet is unavailable. Use the pinned CHGNet environment "
        "environment or set HEA_CHGNET_SITE_PACKAGES to a CHGNet 0.4.2 "
        "site-packages directory."
    ) from last_error


def _make_crystal_graph(
    z: Tensor,
    xyz: Tensor,
    *,
    atom_cutoff: float,
    bond_cutoff: float,
    cell_size: float,
    crystal_graph_cls: Any,
    graph_cls: Any,
    node_cls: Any,
) -> Any:
    """Build one CHGNet ``CrystalGraph`` from an explicit local patch.

    CHGNet's normal converter starts from a periodic pymatgen ``Structure``.
    Here the patch already contains periodic images, so we construct the graph
    directly.  A large artificial cell is used only as a coordinate container;
    all graph images are zero and no artificial periodic neighbour can fall
    within the local cutoff.
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3 or z.ndim != 1 or z.shape[0] != xyz.shape[0]:
        raise ValueError("one CHGNet patch must have z [N] and xyz [N,3]")
    n_atoms = int(z.shape[0])
    if n_atoms <= 0:
        raise ValueError("CHGNet cannot encode an empty structure patch")
    if atom_cutoff <= 0 or bond_cutoff <= 0 or cell_size <= 2 * atom_cutoff:
        raise ValueError("invalid CHGNet graph cutoffs or artificial cell size")

    # Topology is an input-side discrete operation.  Detach only this graph
    # construction; the fractional coordinates below retain the input tensor
    # so the feature path remains differentiable if a caller requests it.
    xyz_cpu = xyz.detach().to(device="cpu", dtype=torch.float32)
    distances = torch.cdist(xyz_cpu, xyz_cpu)
    graph = graph_cls([node_cls(index=i) for i in range(n_atoms)])
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = float(distances[i, j])
            if distance <= 1e-7 or distance > atom_cutoff:
                continue
            image = np.zeros(3, dtype=np.float32)
            # CHGNet's Graph requires both directed edges for every
            # undirected bond.  It then derives the angle/bond graph from
            # these paired edges.
            graph.add_edge(i, j, image, distance)
            graph.add_edge(j, i, image, distance)

    atom_graph_list, directed2undirected_list = graph.adjacency_list()
    bond_graph_list, undirected2directed_list = graph.line_graph_adjacency_list(
        cutoff=bond_cutoff
    )
    device = xyz.device
    dtype = xyz.dtype
    lattice = torch.eye(3, device=device, dtype=dtype) * float(cell_size)
    # Put the local midpoint at the centre of the artificial cell.  This is
    # numerically stable and guarantees that the cell boundary is irrelevant.
    atom_frac_coord = xyz / float(cell_size) + 0.5

    atom_graph = torch.as_tensor(
        atom_graph_list, dtype=torch.int32, device=device
    ).reshape(-1, 2)
    bond_graph = torch.as_tensor(
        bond_graph_list, dtype=torch.int32, device=device
    ).reshape(-1, 5)
    directed2undirected = torch.as_tensor(
        directed2undirected_list, dtype=torch.int32, device=device
    )
    undirected2directed = torch.as_tensor(
        undirected2directed_list, dtype=torch.int32, device=device
    )
    neighbor_image = torch.zeros(
        (atom_graph.shape[0], 3), dtype=dtype, device=device
    )
    return crystal_graph_cls(
        atomic_number=z.to(device=device, dtype=torch.int32),
        atom_frac_coord=atom_frac_coord,
        atom_graph=atom_graph,
        atom_graph_cutoff=float(atom_cutoff),
        neighbor_image=neighbor_image,
        directed2undirected=directed2undirected,
        undirected2directed=undirected2directed,
        bond_graph=bond_graph,
        bond_graph_cutoff=float(bond_cutoff),
        lattice=lattice,
    )


def _feature_forward(backbone: nn.Module, graph: Any) -> Tensor:
    """Run CHGNet's atom/bond/angle path and return mean-pooled features.

    This mirrors the feature-producing part of CHGNet 0.4.2's private
    ``_compute`` method, but deliberately does not execute its energy MLP or
    force/stress derivatives.  The backbone is constructed with
    ``mlp_first=False`` and ``read_out='ave'`` so its pooling module is a true
    mean over valid patch atoms.
    """

    atom_features = backbone.atom_embedding(graph.atomic_numbers - 1)
    bond_features = backbone.bond_embedding(graph.bond_bases_ag)
    bond_weights_ag = backbone.bond_weights_ag(graph.bond_bases_ag)
    bond_weights_bg = backbone.bond_weights_bg(graph.bond_bases_bg)
    has_angles = graph.angle_bases.numel() != 0
    if has_angles:
        angle_features = backbone.angle_embedding(graph.angle_bases)

    for atom_layer, bond_layer, angle_layer in zip(
        backbone.atom_conv_layers[:-1],
        backbone.bond_conv_layers,
        backbone.angle_layers,
        strict=False,
    ):
        atom_features = atom_layer(
            atom_feas=atom_features,
            bond_feas=bond_features,
            bond_weights=bond_weights_ag,
            atom_graph=graph.batched_atom_graph,
            directed2undirected=graph.directed2undirected,
        )
        if has_angles and bond_layer is not None:
            bond_features = bond_layer(
                atom_feas=atom_features,
                bond_feas=bond_features,
                bond_weights=bond_weights_bg,
                angle_feas=angle_features,
                bond_graph=graph.batched_bond_graph,
            )
            if angle_layer is not None:
                angle_features = angle_layer(
                    atom_feas=atom_features,
                    bond_feas=bond_features,
                    angle_feas=angle_features,
                    bond_graph=graph.batched_bond_graph,
                )

    atom_features = backbone.atom_conv_layers[-1](
        atom_feas=atom_features,
        bond_feas=bond_features,
        bond_weights=bond_weights_ag,
        atom_graph=graph.batched_atom_graph,
        directed2undirected=graph.directed2undirected,
    )
    if backbone.readout_norm is not None:
        atom_features = backbone.readout_norm(atom_features)
    pooled = backbone.pooling(atom_features, graph.atom_owners)
    return pooled


class CHGNetStructureEncoder(nn.Module):
    """Encode a variable-size HEA structure patch with random-init CHGNet.

    The public forward signature intentionally matches the other structure
    adapters: ``(z, xyz, role, mask)``.  ``role`` is accepted for interface
    compatibility and intentionally ignored.
    """

    def __init__(
        self,
        *,
        output_dim: int = 256,
        projection_hidden_dim: int = 512,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: int | list[int] = 64,
        bond_conv_hidden_dim: int | list[int] = 64,
        angle_layer_hidden_dim: int | list[int] = 0,
        atom_graph_cutoff: float = 5.0,
        bond_graph_cutoff: float = 3.0,
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        conv_dropout: float = 0.0,
        non_linearity: str = "silu",
        cell_size: float = 100.0,
        max_atomic_number: int = 94,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError(
                "CHGNetStructureEncoder is the from-scratch HEA variant; "
                "pretrained checkpoints are intentionally not supported"
            )
        if atom_graph_cutoff <= 0 or bond_graph_cutoff <= 0:
            raise ValueError("CHGNet graph cutoffs must be positive")
        if bond_graph_cutoff > atom_graph_cutoff:
            raise ValueError("bond_graph_cutoff cannot exceed atom_graph_cutoff")
        if cell_size <= 2 * atom_graph_cutoff:
            raise ValueError("cell_size must leave the local graph away from PBC")
        if max_atomic_number <= 0 or max_atomic_number > 94:
            raise ValueError("CHGNet 0.4.2 supports atomic numbers up to 94")

        (
            _chgnet,
            model_cls,
            crystal_graph_cls,
            graph_cls,
            node_cls,
            batched_graph_cls,
        ) = _load_chgnet_modules()
        self.chgnet_version = CHGNET_VERSION
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.bond_graph_cutoff = float(bond_graph_cutoff)
        self.cell_size = float(cell_size)
        self.max_atomic_number = int(max_atomic_number)
        self._crystal_graph_cls = crystal_graph_cls
        self._graph_cls = graph_cls
        self._node_cls = node_cls
        self._batched_graph_cls = batched_graph_cls

        # composition_model=None is essential: it prevents loading CHGNet's
        # atom-reference table and makes this an entirely random-init tower.
        self.backbone = model_cls(
            atom_fea_dim=int(atom_fea_dim),
            bond_fea_dim=int(bond_fea_dim),
            angle_fea_dim=int(angle_fea_dim),
            composition_model=None,
            num_radial=int(num_radial),
            num_angular=int(num_angular),
            n_conv=int(n_conv),
            atom_conv_hidden_dim=atom_conv_hidden_dim,
            bond_conv_hidden_dim=bond_conv_hidden_dim,
            angle_layer_hidden_dim=angle_layer_hidden_dim,
            conv_dropout=float(conv_dropout),
            read_out="ave",
            mlp_first=False,
            atom_graph_cutoff=float(atom_graph_cutoff),
            bond_graph_cutoff=float(bond_graph_cutoff),
            cutoff_coeff=int(cutoff_coeff),
            learnable_rbf=bool(learnable_rbf),
            non_linearity=non_linearity,
            graph_converter_algorithm="legacy",
        )
        if self.backbone.composition_model is not None:
            raise RuntimeError("CHGNet composition model unexpectedly loaded")
        self.backbone_dim = int(atom_fea_dim)
        self.projection = ProjectionHead(
            self.backbone_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=output_dim,
            dropout=0.1,
        )
        self.output_dim = int(output_dim)

    def _make_batch_graph(self, z: Tensor, xyz: Tensor, mask: Tensor) -> Any:
        graphs: list[Any] = []
        for batch_index in range(int(z.shape[0])):
            valid = mask[batch_index].bool()
            graphs.append(
                _make_crystal_graph(
                    z[batch_index, valid],
                    xyz[batch_index, valid],
                    atom_cutoff=self.atom_graph_cutoff,
                    bond_cutoff=self.bond_graph_cutoff,
                    cell_size=self.cell_size,
                    crystal_graph_cls=self._crystal_graph_cls,
                    graph_cls=self._graph_cls,
                    node_cls=self._node_cls,
                )
            )
        return self._batched_graph_cls.from_graphs(
            graphs,
            self.backbone.bond_basis_expansion,
            self.backbone.angle_basis_expansion,
        )

    def forward(
        self,
        z: Tensor,
        xyz: Tensor,
        role: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        del role  # role-free CHGNet contract
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
            raise ValueError("valid CHGNet nodes cannot have atomic number 0")
        if torch.any(mask & (z < 1)) or torch.any(mask & (z > self.max_atomic_number)):
            bad = z[mask & ((z < 1) | (z > self.max_atomic_number))][:8]
            raise ValueError(f"atomic numbers outside CHGNet range: {bad.tolist()}")
        if not torch.isfinite(xyz).all():
            raise ValueError("CHGNet input coordinates contain NaN/Inf")
        counts = mask.sum(dim=1).long()
        if torch.any(counts <= 0):
            raise ValueError("CHGNet received an empty structure patch")

        parameter = next(self.backbone.parameters())
        z_work = z.to(device=parameter.device, dtype=torch.long)
        xyz_work = xyz.to(device=parameter.device, dtype=parameter.dtype)
        mask_work = mask.to(device=parameter.device)
        graph = self._make_batch_graph(z_work, xyz_work, mask_work)
        pooled = _feature_forward(self.backbone, graph)
        if pooled.ndim != 2 or pooled.shape != (z.shape[0], self.backbone_dim):
            raise RuntimeError(f"unexpected CHGNet pooled feature shape: {tuple(pooled.shape)}")
        out = self.projection(pooled)
        assert_finite("CHGNet embedding", out)
        return out


__all__ = [
    "CHGNET_VERSION",
    "CHGNetStructureEncoder",
    "DEFAULT_CHGNET_SITE_PACKAGES",
]
