from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from torch import Tensor, nn

from ..external import upstream_path


ALIGNN_ROOT = upstream_path("HEABENCH_ALIGNN_ROOT", "third_party/ALIGNN")
if str(ALIGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(ALIGNN_ROOT))

from alignn.models.alignn import ALIGNN, ALIGNNConfig  # noqa: E402


class ALIGNNEncoder(nn.Module):
    """Official ALIGNN trunk with only its native property head removed."""

    output_dim = 256

    def __init__(self):
        super().__init__()
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            embedding_features=64,
            hidden_features=self.output_dim,
            output_features=1,
            link="identity",
        )
        self.trunk = ALIGNN(config)
        self.trunk.fc = nn.Identity()

    def forward(self, graph: Any, line_graph: Any) -> Tensor:
        pooled = self.trunk((graph, line_graph, None))
        # Upstream uses torch.squeeze on its property output. Restore the batch
        # axis for a final batch of size one after replacing fc with Identity.
        return pooled.unsqueeze(0) if pooled.ndim == 1 else pooled
