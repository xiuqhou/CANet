from collections import OrderedDict
from functools import partial
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from models.bricks.basic import SqueezeAndExcitation


class FusionBasedFeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        fusion_module: nn.Module,
        extra_block: bool = False,
        norm_layer: nn.Module = None,
    ):
        """A general fusion based feature pyramid network module, which performs fusion operation
        based on the given fusion_module rather than summation in FPN for adjacent feature maps.

        :param in_channels_list: input channels. Example: [256, 512, 1024, 2048]
        :param out_channels: output channel. Example: 256
        """
        super(FusionBasedFeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
            )
            layer_block_module = Conv2dNormActivation(
                out_channels,
                out_channels,
                kernel_size=3,
                norm_layer=norm_layer,
                activation_layer=None,
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # build fusion module
        self.fusion_modules = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.fusion_modules.append(fusion_module(out_channels))

        self.extra_block = extra_block
        self.num_channels = [out_channels] * (len(in_channels_list) + int(extra_block))

        # add extra_block name, used for collect featmap names for MultiScaleRoIAlign
        if self.extra_block:
            self.extra_block_name = ["pool"]
        else:
            self.extra_block_name = []

    def forward(self, x: OrderedDict) -> List[Tensor]:
        keys = list(x.keys())
        x = list(x.values())
        assert len(x) == len(self.inner_blocks)

        # inner_lateral
        results = []
        for idx in range(len(x)):
            results.append(self.inner_blocks[idx](x[idx]))

        # top down path
        for idx in range(len(x) - 1, 0, -1):
            feat_shape = results[idx - 1].shape[-2:]
            results[idx - 1] = self.fusion_modules[idx - 1](
                low_feature=results[idx - 1],
                high_feature=F.interpolate(results[idx], size=feat_shape, mode="nearest"),
            )

        # output layer
        output = OrderedDict()
        for idx in range(len(x)):
            output[keys[idx]] = self.layer_blocks[idx](results[idx])

        # extra block
        if self.extra_block:
            output["pool"] = F.max_pool2d(list(output.values())[-1], 1, 2, 0)

        return output


class LaplacianFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_low = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv_high = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.se_sum = SqueezeAndExcitation(in_channels, reduction=1)
        self.se_sub = SqueezeAndExcitation(in_channels, reduction=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, low_feature, high_feature):
        low_feature = self.conv_low(low_feature)
        high_feature = self.conv_high(high_feature)
        sum_context = low_feature + high_feature
        sub_context = low_feature - high_feature

        sum_context = self.se_sum(sum_context) + sum_context
        sub_context = self.se_sub(sub_context) + sub_context

        result = self.alpha * sum_context + self.beta * sub_context
        return result


class LaplacianFeaturePyramidNetwork(FusionBasedFeaturePyramidNetwork):
    """Laplacian Feature Pyramid Network proposed in paper `CANet: Contextual Information and
        Spatial Attention Based Network for Detecting Small Defects in Manufacturing Industry`_.

    :param in_channels_list: input channels. Example: [256, 512, 1024, 2048]
    :param out_channels: output channel. Example: 256
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_block: bool = False,
        norm_layer: nn.Module = None,
    ):
        super(LaplacianFeaturePyramidNetwork, self).__init__(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_block=extra_block,
            fusion_module=LaplacianFusion,
            norm_layer=norm_layer,
        )
