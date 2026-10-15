from __future__ import annotations

from torch import Tensor, nn


class MCDCChargeEncoder(nn.Module):
    """PBC adaptation of the MCDC HEA 3-D CNN representation trunk.

    The published three anisotropic convolution/pooling stages are retained.
    Its complete Dense(2048)-Dense(1024)-property prediction network is removed;
    the flattened convolution map is passed to the benchmark common adapter.
    Circular padding replaces image-style zero padding. An adaptive pool is
    inactive for 56x40x40 and permits the controlled 56x80x80 sensitivity run
    without changing encoder or downstream parameter counts.
    """

    output_dim = 64 * 7 * 5 * 5
    canonical_shape = (56, 40, 40)
    sensitivity_shape = (56, 80, 80)

    def __init__(self, *, input_shape: tuple[int, int, int] = canonical_shape,
                 dropout: float = 0.3):
        super().__init__()
        if input_shape not in {self.canonical_shape, self.sensitivity_shape}:
            raise ValueError(f"unsupported MCDC input shape {input_shape}")
        self.input_shape = tuple(input_shape)
        kernel = (3, 3, 11)
        padding = tuple(value // 2 for value in kernel)
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel, padding=padding, padding_mode="circular"),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel, padding=padding, padding_mode="circular"),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel, padding=padding, padding_mode="circular"),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
        )
        self.fixed_pool = nn.AdaptiveAvgPool3d((7, 5, 5))

    def forward(self, density: Tensor) -> Tensor:
        if density.ndim == 4:
            density = density.unsqueeze(1)
        if density.ndim != 5 or density.shape[1] != 1:
            raise ValueError("MCDC density must be [B,D,H,W] or [B,1,D,H,W]")
        if tuple(density.shape[-3:]) != self.input_shape:
            raise ValueError(f"expected {self.input_shape}, received {density.shape[-3:]}")
        return self.fixed_pool(self.features(density)).flatten(1)
