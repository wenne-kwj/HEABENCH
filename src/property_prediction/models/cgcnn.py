from __future__ import annotations

from pathlib import Path
import sys

from torch import Tensor, nn

from ..external import upstream_path


CGCNN_ROOT = upstream_path("HEABENCH_CGCNN_ROOT", "third_party/CGCNN")
if str(CGCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CGCNN_ROOT))

from cgcnn.model import CrystalGraphConvNet  # noqa: E402


class CGCNNEncoder(nn.Module):
    """Upstream CGCNN trunk with its scalar prediction layer removed."""

    def __init__(self, atom_feature_dim: int = 92, neighbor_feature_dim: int = 31):
        super().__init__()
        self.trunk = CrystalGraphConvNet(
            atom_feature_dim,
            neighbor_feature_dim,
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            classification=False,
        )
        self.trunk.fc_out = nn.Identity()

    def forward(
        self,
        atom_features: Tensor,
        neighbor_features: Tensor,
        neighbor_indices: Tensor,
        crystal_atom_indices: list[Tensor],
    ) -> Tensor:
        return self.trunk(
            atom_features, neighbor_features, neighbor_indices,
            crystal_atom_indices,
        )
