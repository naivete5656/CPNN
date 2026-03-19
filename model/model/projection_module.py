import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import trunc_normal_, DropPath

from .utils import LayerNorm, GRN


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Adapter(nn.Module):
    """Adapter module for ConvNeXtV2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the convolution.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        num_layer=2,
        hidden_dims=[512, 512],
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.projection_layers = nn.ModuleList()
        last_dim = input_dim
        for i in range(num_layer):
            middle_layer = nn.Sequential(
                nn.Linear(last_dim, hidden_dims[i]),
                nn.GELU(),
            )
            self.projection_layers.append(middle_layer)
            last_dim = hidden_dims[i]
        self.head = nn.Linear(last_dim, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x.clone()

        for layer in self.projection_layers:
            x = layer(x)
        x = x + identity
        x = self.head(x)
        return x


class ConsNexProjectionModule(nn.Module):

    def __init__(
        self,
        input_dim,
        out_dim,
        num_stages=2,
        depths=[1, 1],
        dims=[512, 512, 512, 512],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.middle_layers = nn.ModuleList()
        dims[0] = input_dim  # Ensure the first dimension matches input_dim
        cur = 0
        for i in range(num_stages):
            middle_layer = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            cur += depths[i]
            self.middle_layers.append(middle_layer)

        self.head = nn.Linear(out_dim, out_dim)

    def forward(self, x, coords=None):
        for layer in self.middle_layers:
            x = layer(x)
        return self.conv(x)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


class LongProjectionModule(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        num_layer=2,
        hidden_dims=[512, 512],
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.prj = nn.Identity()
        self.projector = nn.Linear(input_dim, out_dim)

    def forward(self, patches, coords):
        logits_per_image = self.prj(patches, coords)
        x = self.projector(logits_per_image)

        return x


# # --- Example usage ---
# hidden_size = 512
# num_heads = 8
# window_size = 129  # center token + 64 tokens on each side

# # Instantiate model
# local_attn_model = SimpleLocalAttention(hidden_size, num_heads, window_size)

# # Dummy input data (batch_size=1, sequence_length=1024, hidden_size=512)
# input_tensor = torch.randn(1, 1024, hidden_size)

# # Run
# output = local_attn_model(input_tensor)

# print("Input shape:", input_tensor.shape)
# print("Output shape:", output.shape)
