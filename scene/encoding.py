import torch
from torch import nn
from torch import Tensor
from typing import Literal

# import tinycudann as tcnn
# class HashGridEncoding(torch.nn.Module):
#     def __init__(self, input_dim=3, n_levels=16, log2_hashmap_size=19, base_resolution=16, per_level_scale=1.5):
#         super().__init__()
#         # 定义Hash Grid编码
#         self.encoder = tcnn.Encoding(
#             n_input_dims=input_dim,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": n_levels,
#                 "n_features_per_level": 2,  # 每个level生成的特征数
#                 "log2_hashmap_size": log2_hashmap_size,
#                 "base_resolution": base_resolution,
#                 "per_level_scale": per_level_scale
#             }
#         )
#         self.output_dim = self.encoder.n_output_dims
#
#     def forward(self, x):
#         return self.encoder(x)
#
#     def get_output_n_channels(self) -> int:
#         return self.output_dim


class PositionalEncoding(torch.nn.Module):
    def __init__(self, input_channels: int, n_frequencies: int, log_sampling: bool = True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_channels = input_channels
        self.funcs = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.funcs) * n_frequencies + 1)

        max_frequencies = n_frequencies - 1
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_frequencies, steps=n_frequencies)

        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_frequencies, steps=n_frequencies)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)

    def get_output_n_channels(self) -> int:
        return self.output_channels

class SHEncoding(nn.Module):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, levels: int = 4, implementation: Literal["torch"] = "torch") -> None:
        super().__init__()

        if levels <= 0 or levels > 5:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 5 levels, requested {levels}")

        self.levels = levels

    def get_out_dim(self) -> int:
        """Returns the output dimension of the spherical harmonics encoding."""
        return self.levels ** 2  # Number of components in the spherical harmonics encoding

    @torch.no_grad()
    def pytorch_fwd(self, in_tensor: Tensor) -> Tensor:
        """Forward pass using PyTorch."""
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)

    def forward(self, in_tensor: Tensor) -> Tensor:
        """Main forward pass."""
        return self.pytorch_fwd(in_tensor)


def components_from_spherical_harmonics(
        levels: int, directions: Tensor
) -> Tensor:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Input tensor with shape (*batch, 3), representing the directions to encode.

    Returns:
        A tensor containing the spherical harmonic components with shape (*batch, components).
    """
    num_components = levels ** 2  # Number of spherical harmonic components
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    # Ensure levels are within valid range
    assert 1 <= levels <= 5, f"SH levels must be in [1,5], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    # Extract x, y, z components of the direction
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x ** 2
    yy = y ** 2
    zz = z ** 2

    # l0
    components[..., 0] = 0.28209479177387814  # The 0th spherical harmonic component (constant term)

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz ** 2 - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components
