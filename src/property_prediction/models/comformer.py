from __future__ import annotations

from pathlib import Path
import sys
import types
import typing

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter as pyg_scatter

from ..external import upstream_path


COMFORMER_ROOT = upstream_path(
    "HEABENCH_COMFORMER_ROOT", "third_party/ComFormer/OpenMat/ComFormer"
)
if str(COMFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFORMER_ROOT))


def _install_upstream_compatibility() -> None:
    import pydantic
    import pydantic.typing
    import pydantic_settings

    pydantic.typing.Literal = typing.Literal
    pydantic.BaseSettings = pydantic_settings.BaseSettings
    scatter_module = types.ModuleType("torch_scatter")
    scatter_module.scatter = pyg_scatter
    scatter_module.gather_csr = lambda *args, **kwargs: None
    scatter_module.segment_csr = lambda *args, **kwargs: None
    sys.modules.setdefault("torch_scatter", scatter_module)
    sparse_module = types.ModuleType("torch_sparse")
    sparse_module.SparseTensor = object
    sys.modules.setdefault("torch_sparse", sparse_module)

    original_script = torch.jit.script
    torch.jit.script = lambda function: function
    try:
        import comformer.models.utils as model_utils
    finally:
        torch.jit.script = original_script
    typo_root = types.ModuleType("comforemr")
    typo_models = types.ModuleType("comforemr.models")
    sys.modules.setdefault("comforemr", typo_root)
    sys.modules.setdefault("comforemr.models", typo_models)
    sys.modules["comforemr.models.utils"] = model_utils
    import comformer.models.transformer as transformer
    transformer.MatformerConv = transformer.ComformerConv


_install_upstream_compatibility()

from comformer.models.comformer import eComformer, eComformerConfig  # noqa: E402


class EComFormerEncoder(nn.Module):
    """Official eComFormer trunk with the property layer removed."""

    output_dim = 256

    def __init__(self) -> None:
        super().__init__()
        config = eComformerConfig(
            name="eComformer",
            atom_input_features=92,
            edge_features=256,
            node_features=256,
            fc_features=self.output_dim,
            output_features=1,
            conv_layers=3,
        )
        self.trunk = eComformer(config)
        self.trunk.fc_out = nn.Identity()

    def forward(self, graph) -> Tensor:
        encoded = self.trunk((graph, None, None))
        return encoded.unsqueeze(0) if encoded.ndim == 1 else encoded
