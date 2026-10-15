from __future__ import annotations

import torch
from torch import Tensor, nn


class ElementFractionSetEncoder(nn.Module):
    """Permutation-invariant five-element composition encoder."""

    def __init__(self, embedding_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.element_embedding = nn.Embedding(119, embedding_dim, padding_idx=0)
        self.token_network = nn.Sequential(
            nn.Linear(embedding_dim + 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        self.attention_score = nn.Linear(embedding_dim, 1)
        self.output_network = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, atomic_numbers: Tensor, fractions: Tensor) -> Tensor:
        if atomic_numbers.ndim != 2 or fractions.shape != atomic_numbers.shape:
            raise ValueError("composition inputs must both have shape [batch, elements]")
        fractions = fractions.clamp_min(0.0)
        fractions = fractions / fractions.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        embedded = self.element_embedding(atomic_numbers)
        fraction_features = torch.stack(
            [fractions, torch.log(fractions.clamp_min(1.0e-8))], dim=-1
        )
        tokens = self.token_network(torch.cat([embedded, fraction_features], dim=-1))
        weighted = (tokens * fractions.unsqueeze(-1)).sum(dim=1)
        logits = self.attention_score(tokens).squeeze(-1) + torch.log(
            fractions.clamp_min(1.0e-8)
        )
        attention = torch.softmax(logits, dim=1)
        attended = (tokens * attention.unsqueeze(-1)).sum(dim=1)
        return self.output_network(torch.cat([weighted, attended], dim=-1))
