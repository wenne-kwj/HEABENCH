from __future__ import annotations

import torch
from torch import Tensor, nn

from property_prediction.charge_sampling import uniform_fractional_positions


def _periodic_fps(points: Tensor, count: int) -> Tensor:
    selected = torch.empty(count, dtype=torch.long)
    distance = torch.full((len(points),), float("inf"))
    farthest = torch.tensor(0, dtype=torch.long)
    for position in range(count):
        selected[position] = farthest
        delta = points - points[farthest]
        delta -= torch.round(delta)
        distance = torch.minimum(distance, delta.square().sum(dim=-1))
        farthest = torch.argmax(distance)
    return selected


def _periodic_neighbors(support: Tensor, query: Tensor, radius: float,
                        count: int = 32) -> tuple[Tensor, Tensor]:
    delta = support.unsqueeze(0) - query.unsqueeze(1)
    delta -= torch.round(delta)
    distance = delta.square().sum(dim=-1)
    values, indices = torch.topk(distance, k=min(count, len(support)), dim=-1,
                                  largest=False, sorted=True)
    indices = torch.where(
        values <= radius * radius,
        indices,
        indices[:, :1].expand_as(indices),
    )
    row = torch.arange(len(query)).unsqueeze(-1)
    offsets = delta[row, indices].permute(2, 0, 1).contiguous() / radius
    return indices.long(), offsets


def _gather(features: Tensor, indices: Tensor) -> Tensor:
    return features[:, :, indices]


class PointVectorBlock(nn.Module):
    """VPSA block transcribed from the official EDBench PointVector encoder."""

    def __init__(self, input_dim: int, output_dim: int, offsets: Tensor):
        super().__init__()
        self.register_buffer("offsets", offsets, persistent=False)
        self.theta_alpha = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1, bias=False),
            nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True),
        )
        self.theta_beta = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1, bias=False),
            nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True),
        )
        self.z = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.position = nn.Sequential(
            nn.Conv2d(3, input_dim, 1, bias=False), nn.BatchNorm2d(input_dim)
        )
        self.delta_norm = nn.BatchNorm2d(input_dim)
        self.vector_mix = nn.Sequential(
            nn.Conv1d(input_dim * 3, input_dim, 1, groups=input_dim, bias=False),
            nn.BatchNorm1d(input_dim), nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1, bias=False), nn.BatchNorm1d(output_dim)
        )
        self.residual = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1, bias=False), nn.BatchNorm1d(output_dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, features: Tensor, centers: Tensor, neighbors: Tensor) -> Tensor:
        identity = features[:, :, centers]
        grouped = _gather(features, neighbors)
        offsets = self.offsets.unsqueeze(0).expand(features.shape[0], -1, -1, -1)
        value = self.activation(
            self.delta_norm(grouped - identity.unsqueeze(-1)) + self.position(offsets)
        )
        alpha = self.theta_alpha(value); beta = self.theta_beta(value)
        magnitude = self.z(value)
        vector = torch.cat([
            magnitude * torch.sin(beta) * torch.cos(alpha),
            magnitude * torch.sin(beta) * torch.sin(alpha),
            magnitude * torch.cos(beta),
        ], dim=1)
        vector = torch.max(vector, dim=-1).values
        return self.activation(self.output(self.vector_mix(vector)) + self.residual(identity))


class PointVectorPBCEncoder(nn.Module):
    """EDBench PointVector adapted to a translation-invariant periodic field.

    Absolute molecular coordinates are removed from node/global features.
    Periodic relative offsets remain in every VPSA block. Input density values
    are sampled directly from native CHGCAR grids before any Fourier resampling.
    """

    output_dim = 512
    point_shape = (8, 16, 16)
    point_count = 2048

    def __init__(self):
        super().__init__()
        positions = torch.from_numpy(uniform_fractional_positions(self.point_shape))
        self.register_buffer("point_positions", positions, persistent=True)
        widths = [32, 64, 128, 256, 512]
        radii = [0.15, 0.225, 0.3375, 0.50625]
        self.input_projection = nn.Conv1d(1, widths[0], 1)
        hierarchy = [positions]
        stages: list[nn.Module] = []
        for stage in range(4):
            support = hierarchy[-1]
            centers = _periodic_fps(support, len(support) // 2)
            query = support[centers]
            neighbors, offsets = _periodic_neighbors(
                support, query, radii[stage], count=32
            )
            hierarchy.append(query)
            self.register_buffer(f"centers_{stage}", centers, persistent=False)
            self.register_buffer(f"neighbors_{stage}", neighbors, persistent=False)
            stages.append(PointVectorBlock(widths[stage], widths[stage + 1], offsets))
        self.stages = nn.ModuleList(stages)
        self.global_projection = nn.Sequential(
            nn.Conv1d(widths[-1], self.output_dim, 1, bias=False),
            nn.BatchNorm1d(self.output_dim), nn.ReLU(inplace=True),
        )

    def forward(self, density: Tensor) -> Tensor:
        if density.ndim != 2 or density.shape[1] != self.point_count:
            raise ValueError(f"PointVector density must be [B,{self.point_count}]")
        features = self.input_projection(density.unsqueeze(1))
        for stage, block in enumerate(self.stages):
            features = block(
                features,
                getattr(self, f"centers_{stage}"),
                getattr(self, f"neighbors_{stage}"),
            )
        return torch.max(self.global_projection(features), dim=-1).values
