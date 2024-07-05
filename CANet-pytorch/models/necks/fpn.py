from collections import OrderedDict
from typing import List

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_block: bool = False,
        norm_layer: nn.Module = None,
    ):
        """
        The implementation of paper `Feature Pyramid Networks for Object
        Detection <https://arxiv.org/abs/1612.03144>`_.
        :param in_channels_list: input channels. Example: [256, 512, 1024, 2048]
        :param out_channels: output channel. Example: 256
        """
        super(FeaturePyramidNetwork, self).__init__()
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
        self.extra_block = extra_block
        self.num_channels = [out_channels] * (len(in_channels_list) + int(extra_block))

        # add extra_block name, used for collect featmap names for MultiScaleRoIAlign
        if self.extra_block:
            self.extra_block_name = ["pool"]
        else:
            self.extra_block_name = []

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            results[idx - 1] = results[idx - 1] + F.interpolate(
                results[idx], size=feat_shape, mode="nearest"
            )
        # output layer
        output = OrderedDict()
        for idx in range(len(x)):
            output[keys[idx]] = self.layer_blocks[idx](results[idx])
        # extra block
        if self.extra_block:
            output["pool"] = F.max_pool2d(list(output.values())[-1], 1, 2, 0)

        return output
