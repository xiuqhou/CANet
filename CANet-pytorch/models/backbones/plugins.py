import math
from typing import List

from torch import nn
import torch


class PluginConfig:
    def __init__(self, block: nn.Module, stages: List[int], conv_pos: List[int]):
        assert max(stages) < 4, "stage index should be less than 4"
        assert max(conv_pos) < 3, "conv_pos should be less than 3"
        self.block = block
        self.stages = stages
        self.conv_pos = conv_pos


class SpatialAttentionModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        reduce_fct: int = 1,
        num_heads: int = 8,
        kv_stride: int = 2,
        pixel_shuffle: bool = False,
    ):
        super().__init__()
        hidden_channels = in_channels // reduce_fct
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.kv_stride = kv_stride
        self.qk_embed_dim = hidden_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        self.key_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_c, kernel_size=1, bias=False
        )

        self.v_dim = hidden_channels // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False
        )

        stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
        appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
        self.appr_bias = nn.Parameter(appr_bias_value)

        if pixel_shuffle:
            self.proj_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.v_dim * num_heads,
                    out_channels=in_channels * self.kv_stride**2,
                    kernel_size=1,
                    bias=True,
                ),
                nn.PixelShuffle(upscale_factor=self.kv_stride),
            )
        else:
            self.proj_conv = nn.Conv2d(
                in_channels=self.v_dim * num_heads,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            )
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.kv_stride > 1:
            if pixel_shuffle:
                self.kv_downsample = nn.Sequential(
                    nn.PixelUnshuffle(downscale_factor=self.kv_stride),
                    nn.Conv2d(
                        in_channels=in_channels * self.kv_stride**2,
                        out_channels=in_channels,
                        kernel_size=1,
                    )
                )
            else:
                self.kv_downsample = nn.AvgPool2d(kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F
        num_heads = self.num_heads

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        n, _, h_kv, w_kv = x_kv.shape

        proj_key = self.key_conv(x_kv).view((n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        # attention_type[0]: appr - appr
        # attention_type[1]: appr - position
        # attention_type[2]: bias - appr
        # attention_type[3]: bias - position
        appr_bias = self.appr_bias.\
            view(1, num_heads, 1, self.qk_embed_dim).\
            repeat(n, 1, h_kv * w_kv, 1)  # TODO: 将维度n也加入随机初始化，或者将维度h_kv*w_kv随机初始化

        energy = torch.matmul(appr_bias, proj_key)

        attention = F.softmax(energy, 3)

        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, h_kv, w_kv)

        out = self.proj_conv(out)

        out = F.interpolate(out, size=x_input.shape[2:], mode='bilinear', align_corners=False)

        out = self.gamma * out + x_input
        return out
