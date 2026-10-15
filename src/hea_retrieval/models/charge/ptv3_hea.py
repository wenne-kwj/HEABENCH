"""HEA adapter for a dependency-light Point Transformer V3 charge tower.

The upstream Pointcept PTv3 implementation is vendored in
``third_party/pointcept_ptv3`` at a pinned commit for provenance.  Its public
backbone, however, requires ``spconv`` for the sparse stem and the convolutional
positional encoding.  The HEA retrieval environment uses PyTorch 2.12/CUDA 13
and has no compatible ``spconv`` wheel.  This module therefore keeps the core
PTv3 ideas (Morton serialization, serialized window attention, relative
position bias, and serialized pooling) in a small pure-PyTorch implementation.

This is intentionally a retrieval adapter, not a drop-in Pointcept
segmentation model:

* all 4096 charge samples in a patch are retained;
* the only input feature is standardized total valence density;
* coordinates are local Cartesian coordinates (normalised by 4.5 A);
* no atom/role information is passed to the charge tower;
* there is no decoder or classifier head; batch-wise mean pooling feeds the
  common HEA projection head.

The implementation is kept here rather than editing the frozen third-party
checkout.  ``PTv3-HEA`` should consequently be interpreted as a faithful,
from-scratch PTv3-style encoder with the sparse-convolution pieces replaced by
continuous point-wise positional projections suitable for these small dense
charge patches.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from ..common import ProjectionHead, assert_finite


PTV3_UPSTREAM_COMMIT = "2b247ed555e96eae11710dfde79a3fc16c8c74b3"


class _PointState(dict):
    """Small attribute/mapping hybrid used by the PTv3 blocks."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - normal Python attribute error
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


def _offset_to_batch(offset: Tensor) -> Tensor:
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.arange(
        counts.numel(), device=offset.device, dtype=torch.long
    ).repeat_interleave(counts)


def _batch_to_offset(batch: Tensor) -> Tensor:
    if batch.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=batch.device)
    counts = torch.bincount(batch.long())
    return torch.cumsum(counts, dim=0).long()


def _morton3d(grid_coord: Tensor, depth: int) -> Tensor:
    """Vectorised non-negative 3-D Morton code compatible with Pointcept."""

    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError("grid_coord must have shape [N, 3]")
    if depth <= 0 or depth > 16:
        raise ValueError(f"Morton depth must be in [1, 16], got {depth}")
    coord = grid_coord.long()
    limit = 1 << depth
    if coord.numel() and int(coord.min()) < 0:
        raise ValueError("grid_coord must be non-negative")
    if coord.numel() and int(coord.max()) >= limit:
        raise ValueError("grid_coord exceeds the selected Morton depth")
    code = torch.zeros((coord.shape[0],), dtype=torch.long, device=coord.device)
    for bit in range(depth):
        code |= ((coord[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((coord[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((coord[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def _serialize_from_grid(
    grid_coord: Tensor,
    batch: Tensor,
    *,
    order: Sequence[str],
    depth: int | None = None,
    shuffle: bool = False,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Create serialized codes, order and inverse arrays.

    Batch ids occupy the high bits exactly as in Pointcept, so serialized
    windows cannot cross patch boundaries.
    """

    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError("grid_coord must have shape [N, 3]")
    if batch.shape != (grid_coord.shape[0],):
        raise ValueError("batch and grid_coord lengths disagree")
    if not order:
        raise ValueError("at least one serialization order is required")
    if any(name not in {"z", "z-trans"} for name in order):
        raise ValueError("HEA PTv3 supports only z and z-trans serialization")
    max_coord = int(grid_coord.max().item()) if grid_coord.numel() else 0
    needed = max(1, max_coord.bit_length())
    depth = int(depth if depth is not None else needed)
    if depth < needed or depth > 16:
        raise ValueError(f"invalid serialization depth {depth}; needed {needed}")
    base_codes: list[Tensor] = []
    for name in order:
        coords = grid_coord[:, [1, 0, 2]] if name == "z-trans" else grid_coord
        code = _morton3d(coords, depth)
        code = code | (batch.long() << (3 * depth))
        base_codes.append(code)
    codes = torch.stack(base_codes, dim=0)
    serialized_order = torch.argsort(codes, dim=1, stable=True)
    if shuffle and serialized_order.shape[0] > 1:
        permutation = torch.randperm(serialized_order.shape[0], device=grid_coord.device)
        codes = codes[permutation]
        serialized_order = serialized_order[permutation]
    inverse = torch.zeros_like(serialized_order)
    source = torch.arange(
        grid_coord.shape[0], device=grid_coord.device, dtype=torch.long
    ).expand_as(serialized_order)
    inverse.scatter_(1, serialized_order, source)
    return codes, serialized_order, inverse, depth


def _initial_grid(coord: Tensor, grid_size: float, grid_origin: float) -> Tensor:
    if grid_size <= 0 or not math.isfinite(float(grid_size)):
        raise ValueError("grid_size must be a positive finite number")
    if not math.isfinite(float(grid_origin)):
        raise ValueError("grid_origin must be finite")
    # Use a fixed origin in the normalised [-1.25, 1.25] local sphere rather
    # than a batch-dependent or patch-dependent min.  This makes an embedding
    # independent of which other patches happen to share its batch and keeps
    # the quantisation frame stable under the shared rotation augmentation.
    origin = torch.full(
        (1, 3),
        float(grid_origin),
        dtype=coord.dtype,
        device=coord.device,
    )
    grid = torch.floor((coord.detach() - origin) / float(grid_size)).long()
    if grid.numel() and int(grid.min()) < 0:
        raise RuntimeError("initial grid coordinates became negative")
    return grid


def _make_state(
    coord: Tensor,
    feat: Tensor,
    batch: Tensor,
    *,
    grid_coord: Tensor,
    order: Sequence[str],
    shuffle: bool,
) -> _PointState:
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError("coord must have shape [N, 3]")
    if feat.ndim != 2 or feat.shape[0] != coord.shape[0]:
        raise ValueError("feat and coord lengths disagree")
    if batch.shape != (coord.shape[0],):
        raise ValueError("batch and coord lengths disagree")
    offset = _batch_to_offset(batch)
    codes, serialized_order, serialized_inverse, depth = _serialize_from_grid(
        grid_coord,
        batch,
        order=order,
        shuffle=shuffle,
    )
    return _PointState(
        coord=coord,
        feat=feat,
        batch=batch,
        offset=offset,
        grid_coord=grid_coord,
        serialized_code=codes,
        serialized_order=serialized_order,
        serialized_inverse=serialized_inverse,
        serialized_depth=depth,
    )


def _padding_indices(
    offset: Tensor,
    order: Tensor,
    batch: Tensor,
    patch_size_max: int,
) -> tuple[Tensor, Tensor, int]:
    """Return padded sorted-window positions and original positions.

    The first ``count`` positions of each sample are always the real points;
    any appended positions repeat the beginning of the final window.  This is
    equivalent to the upstream PTv3 padding contract for the unpadded outputs,
    while avoiding a dependency on flash-attn's cumulative sequence lengths.
    """

    if patch_size_max <= 0:
        raise ValueError("patch_size must be positive")
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    if counts.numel() == 0 or int(counts.min()) <= 0:
        raise ValueError("each PTv3 patch must contain at least one point")
    patch_size = min(int(counts.min().item()), int(patch_size_max))
    if order.shape != (int(offset[-1].item()),):
        raise ValueError("serialized order and offset lengths disagree")
    sorted_batch = batch[order]
    pad_parts: list[Tensor] = []
    original_parts: list[Tensor] = []
    padded_cursor = 0
    for sample_index, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        real = torch.where(sorted_batch == sample_index)[0]
        if int(real.numel()) != count:
            raise RuntimeError("serialized order is not grouped by batch")
        pad_count = 0
        if count > patch_size:
            pad_count = (-count) % patch_size
        if pad_count:
            # Repeat positions from the preceding window, matching the
            # upstream implementation's harmless duplicate-padding behavior.
            extra = real[max(0, count - patch_size) : max(0, count - patch_size) + pad_count]
            if extra.numel() < pad_count:
                extra = real[:1].expand(pad_count)
            padded = torch.cat([real, extra], dim=0)
        else:
            padded = real
        pad_parts.append(padded)
        original_parts.append(
            torch.arange(padded_cursor, padded_cursor + count, device=offset.device)
        )
        padded_cursor += int(padded.numel())
    return torch.cat(pad_parts), torch.cat(original_parts), patch_size


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_path must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


class _RelativePositionEncoding(nn.Module):
    """The discrete per-axis RPE table used by the official PTv3 block."""

    def __init__(self, patch_size: int, num_heads: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_heads = int(num_heads)
        self.pos_bnd = int((4 * self.patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.table = nn.Parameter(
            torch.zeros(3 * self.rpe_num, self.num_heads)
        )
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, relative_grid: Tensor) -> Tensor:
        # relative_grid: [windows, K, K, 3]
        idx = (
            relative_grid.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(
                3, device=relative_grid.device, dtype=relative_grid.dtype
            )
            * self.rpe_num
        ).long()
        out = self.table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(dim=3)
        return out.permute(0, 3, 1, 2)


class _SerializedAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        *,
        order_index: int,
        enable_rpe: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.patch_size_max = int(patch_size)
        self.order_index = int(order_index)
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_rpe = bool(enable_rpe)
        self.rpe = (
            _RelativePositionEncoding(patch_size, num_heads)
            if enable_rpe
            else None
        )

    def forward(self, point: _PointState) -> Tensor:
        order = point.serialized_order[self.order_index]
        pad, original, patch_size = _padding_indices(
            point.offset, order, point.batch, self.patch_size_max
        )
        sorted_indices = order[pad]
        qkv = self.qkv(point.feat)[sorted_indices]
        windows = qkv.reshape(
            -1, patch_size, 3, self.num_heads, self.channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = windows.unbind(dim=0)
        # The pure torch path deliberately upcasts attention logits for stable
        # training with the density values and does not require flash-attn.
        logits = (q.float() * self.scale) @ k.float().transpose(-2, -1)
        if self.rpe is not None:
            rel = point.grid_coord[sorted_indices].reshape(-1, patch_size, 3)
            rel = rel.unsqueeze(2) - rel.unsqueeze(1)
            logits = logits + self.rpe(rel)
        weights = torch.softmax(logits, dim=-1).to(v.dtype)
        weights = self.attn_drop(weights)
        attended = (weights @ v).transpose(1, 2).reshape(-1, self.channels)
        attended = attended[original]
        # ``original`` is in padded-sequence coordinates; the corresponding
        # entries are in sorted order.  Scatter them back to original indices.
        # Under CUDA autocast, the attention projection can return a lower
        # precision tensor while the state that entered this block remains in
        # FP32 (the residual path intentionally stays numerically stable).
        # Allocate the scatter target from the value being scattered so that
        # ``index_put`` does not reject a Float/BFloat16 combination.
        restored = attended.new_zeros(point.feat.shape)
        restored[order] = attended
        return self.proj_drop(self.proj(restored))


class _ContinuousPositionalEncoding(nn.Module):
    """Pure PyTorch replacement for PTv3's sparse CPE on small local patches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, coord: Tensor) -> Tensor:
        return self.net(coord)


class _SerializedBlock(nn.Module):
    """PTv3 block with pre-norm serialized attention and dense CPE."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        *,
        order_index: int,
        enable_rpe: bool,
        drop_path: float,
        attn_drop: float,
        proj_drop: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.cpe = _ContinuousPositionalEncoding(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = _SerializedAttention(
            channels,
            num_heads,
            patch_size,
            order_index=order_index,
            enable_rpe=enable_rpe,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )
        self.drop_path = _DropPath(drop_path)

    def forward(self, point: _PointState) -> _PointState:
        point.feat = point.feat + self.cpe(point.coord)
        residual = point.feat
        attn_point = _PointState(point)
        attn_point.feat = self.norm1(point.feat)
        point.feat = residual + self.drop_path(self.attn(attn_point))
        point.feat = point.feat + self.drop_path(self.mlp(self.norm2(point.feat)))
        return point


def _segment_mean(values: Tensor, cluster: Tensor, n_cluster: int) -> Tensor:
    result = values.new_zeros((n_cluster, values.shape[-1]))
    result.index_add_(0, cluster, values)
    counts = torch.bincount(cluster, minlength=n_cluster).to(values.dtype).clamp_min(1)
    return result / counts.unsqueeze(-1)


class _SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 2,
        reduce: str = "mean",
        order: Sequence[str],
        shuffle_orders: bool,
    ) -> None:
        super().__init__()
        if stride not in {2, 4, 8}:
            raise ValueError("PTv3 serialized pooling stride must be 2, 4, or 8")
        if reduce != "mean":
            raise ValueError("HEA PTv3 currently uses mean serialized pooling")
        self.stride = int(stride)
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.order = tuple(order)
        self.shuffle_orders = bool(shuffle_orders)

    def forward(self, point: _PointState) -> _PointState:
        depth_drop = (math.ceil(self.stride) - 1).bit_length()
        if depth_drop > point.serialized_depth:
            depth_drop = 0
        code = point.serialized_code[0] >> (depth_drop * 3)
        _, cluster, counts = torch.unique(
            code, sorted=True, return_inverse=True, return_counts=True
        )
        n_cluster = int(counts.numel())
        head_indices = torch.argsort(cluster, stable=True)
        # ``head_indices`` sorted by cluster; first item of each run indexes
        # the representative grid coordinate and batch id.
        starts = torch.cat(
            [counts.new_zeros(1), torch.cumsum(counts, dim=0)[:-1]]
        )
        heads = head_indices[starts]
        feat = _segment_mean(self.proj(point.feat)[head_indices], cluster[head_indices], n_cluster)
        coord = _segment_mean(point.coord[head_indices], cluster[head_indices], n_cluster)
        grid_coord = point.grid_coord[heads] >> depth_drop
        batch = point.batch[heads]
        new = _make_state(
            coord,
            self.act(self.norm(feat)),
            batch,
            grid_coord=grid_coord,
            order=self.order,
            shuffle=self.shuffle_orders and self.training,
        )
        return new


class _HEAPT3Backbone(nn.Module):
    """Small encoder-only PTv3 core for fixed-size HEA charge patches."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        order: Sequence[str] = ("z", "z-trans"),
        stride: Sequence[int] = (2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 2),
        enc_channels: Sequence[int] = (48, 96, 192, 256),
        enc_num_head: Sequence[int] = (3, 6, 12, 16),
        enc_patch_size: Sequence[int] = (128, 128, 128, 128),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_rpe: bool = True,
        shuffle_orders: bool = True,
        grid_size: float = 0.125 / 4.5,
        grid_origin: float = -1.25,
    ) -> None:
        super().__init__()
        values = [enc_depths, enc_channels, enc_num_head, enc_patch_size]
        if len({len(v) for v in values}) != 1:
            raise ValueError("PTv3 encoder tuple lengths must agree")
        if len(enc_depths) != len(stride) + 1:
            raise ValueError("stride must have one fewer item than encoder stages")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = int(in_channels)
        self.grid_size = float(grid_size)
        self.grid_origin = float(grid_origin)
        self.order = tuple(order)
        self.shuffle_orders = bool(shuffle_orders)
        self.enc_channels = tuple(int(x) for x in enc_channels)
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, self.enc_channels[0]),
            nn.LayerNorm(self.enc_channels[0]),
            nn.GELU(),
        )
        total_blocks = sum(int(x) for x in enc_depths)
        drop_values = torch.linspace(0.0, float(drop_path), total_blocks).tolist()
        cursor = 0
        stages: list[nn.Module] = []
        for stage, depth in enumerate(enc_depths):
            modules: list[nn.Module] = []
            if stage > 0:
                modules.append(
                    _SerializedPooling(
                        self.enc_channels[stage - 1],
                        self.enc_channels[stage],
                        stride=int(stride[stage - 1]),
                        order=self.order,
                        shuffle_orders=self.shuffle_orders,
                    )
                )
            for block_index in range(int(depth)):
                modules.append(
                    _SerializedBlock(
                        self.enc_channels[stage],
                        int(enc_num_head[stage]),
                        int(enc_patch_size[stage]),
                        order_index=block_index % len(self.order),
                        enable_rpe=enable_rpe,
                        drop_path=drop_values[cursor],
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_ratio=mlp_ratio,
                    )
                )
                cursor += 1
            stages.append(nn.ModuleList(modules))
        self.stages = nn.ModuleList(stages)
        self.out_channels = self.enc_channels[-1]

    def forward(self, coord: Tensor, feat: Tensor, batch: Tensor) -> _PointState:
        grid_coord = _initial_grid(coord, self.grid_size, self.grid_origin)
        point = _make_state(
            coord,
            self.embedding(feat),
            batch,
            grid_coord=grid_coord,
            order=self.order,
            shuffle=self.shuffle_orders and self.training,
        )
        for stage in self.stages:
            for module in stage:
                point = module(point)
        return point


class PointTransformerV3ChargeEncoder(nn.Module):
    """Encode a 4096-point HEA charge patch into a normalized embedding."""

    def __init__(
        self,
        *,
        output_dim: int = 256,
        projection_hidden_dim: int = 512,
        in_channels: int = 1,
        use_magnetization: bool = False,
        coord_scale: float = 4.5,
        grid_size_A: float = 0.125,
        grid_origin: float = -1.25,
        order: Sequence[str] = ("z", "z-trans"),
        stride: Sequence[int] = (2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 2),
        enc_channels: Sequence[int] = (48, 96, 192, 256),
        enc_num_head: Sequence[int] = (3, 6, 12, 16),
        enc_patch_size: Sequence[int] = (128, 128, 128, 128),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_rpe: bool = True,
        shuffle_orders: bool = True,
        pool_reduce: str = "mean",
    ) -> None:
        super().__init__()
        if use_magnetization and in_channels != 2:
            in_channels = 2
        if in_channels not in (1, 2):
            raise ValueError("PTv3 adapter supports one or two density channels")
        if coord_scale <= 0 or not math.isfinite(float(coord_scale)):
            raise ValueError("coord_scale must be positive and finite")
        if grid_size_A <= 0 or not math.isfinite(float(grid_size_A)):
            raise ValueError("grid_size_A must be positive and finite")
        if pool_reduce != "mean":
            raise ValueError("PTv3 HEA adapter currently supports pool_reduce='mean'")
        self.coord_scale = float(coord_scale)
        self.grid_size_A = float(grid_size_A)
        self.grid_origin = float(grid_origin)
        self.in_channels = int(in_channels)
        self.use_magnetization = bool(use_magnetization)
        self.backbone = _HEAPT3Backbone(
            in_channels=self.in_channels,
            order=tuple(order),
            stride=tuple(stride),
            enc_depths=tuple(enc_depths),
            enc_channels=tuple(enc_channels),
            enc_num_head=tuple(enc_num_head),
            enc_patch_size=tuple(enc_patch_size),
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_rpe=enable_rpe,
            shuffle_orders=shuffle_orders,
            grid_size=self.grid_size,
            grid_origin=self.grid_origin,
        )
        self.backbone_dim = self.backbone.out_channels
        self.projection = ProjectionHead(
            self.backbone_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=output_dim,
            dropout=0.1,
        )
        self.output_dim = int(output_dim)

    @property
    def grid_size(self) -> float:
        """Normalized grid size seen by the backbone."""

        return self.grid_size_A / self.coord_scale

    def forward(
        self,
        xyz: Tensor,
        rho_total: Tensor,
        rho_magnetization: Optional[Tensor] = None,
        *,
        rho_total_raw: Optional[Tensor] = None,
        point_mask: Optional[Tensor] = None,
        volume_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [B,N,3], got {tuple(xyz.shape)}")
        if rho_total.shape != xyz.shape[:2]:
            raise ValueError("rho_total must have shape [B,N] matching xyz")
        if self.use_magnetization:
            if rho_magnetization is None or rho_magnetization.shape != xyz.shape[:2]:
                raise ValueError("magnetization channel requested but not supplied")
            density = torch.stack([rho_total, rho_magnetization], dim=-1)
        else:
            density = rho_total.unsqueeze(-1)
        bsz, n_points = xyz.shape[:2]
        coord = (xyz / self.coord_scale).reshape(bsz * n_points, 3).contiguous()
        feat = density.reshape(bsz * n_points, self.in_channels).contiguous()
        batch = torch.arange(bsz, device=xyz.device, dtype=torch.long).repeat_interleave(
            n_points
        )
        point = self.backbone(coord, feat, batch)
        # ``point.feat`` may be BF16 under AMP while ``coord`` remains FP32.
        # Match the accumulator to the feature dtype before ``index_add_``.
        pooled = point.feat.new_zeros((bsz, point.feat.shape[-1]))
        pooled.index_add_(0, point.batch, point.feat)
        counts = torch.bincount(point.batch, minlength=bsz).to(pooled.dtype).clamp_min(1)
        pooled = pooled / counts.unsqueeze(-1)
        out = self.projection(pooled)
        assert_finite("PointTransformerV3 embedding", out)
        return out


__all__ = [
    "PTV3_UPSTREAM_COMMIT",
    "PointTransformerV3ChargeEncoder",
]
