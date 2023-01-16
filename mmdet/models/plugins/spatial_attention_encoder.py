# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS
from mmcv.cnn.utils import kaiming_init

import math
eps = 1e-6


@PLUGIN_LAYERS.register_module()
class SpatialAttentionEncoder(nn.Module):
    _abbr_ = 'spatial_attention_encoder'

    def __init__(self,
                 in_channels: int,
                 num_heads: int = 9,
                 kv_stride: int = 2,
                 q_stride: int = 1):

        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        self.key_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_c,
            kernel_size=1,
            bias=False)
        self.key_conv.kaiming_init = True

        self.v_dim = in_channels // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False)
        self.value_conv.kaiming_init = True

        stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
        appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
        self.appr_bias = nn.Parameter(appr_bias_value)

        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_channels,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None
            
        self.init_weights()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        num_heads = self.num_heads

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        n, _, h_kv, w_kv = x_kv.shape

        proj_key = self.key_conv(x_kv).view(
            (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

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

        out = F.interpolate(
            out,
            size=x_input.shape[2:],
            mode='bilinear',
            align_corners=False)

        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)

if __name__ == '__main__':
    SAE = SpatialAttentionEncoder(in_channels=256, num_heads=8)
    X = torch.randn(3, 256, 40, 40)
    result = SAE(X)
    pass

from torchvision.models import VGG, AlexNet, GoogLeNet