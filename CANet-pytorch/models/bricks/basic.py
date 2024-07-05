from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, pooling_type: str = "avg"):
        super().__init__()
        assert pooling_type in ["avg", "attn"]
        self.pooling_type = pooling_type
        if pooling_type == "attn":
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
            nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in", nonlinearity="relu")
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        if self.pooling_type == "attn":
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        return self.se_module(context) * x


class ContextBlock(nn.Module):
    """ContextBlock module in GCNet.

    See 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            Options are 'attn' and 'avg', stand for attention pooling and
            average pooling respectively. Default: 'attn'.
        fusion_types (Sequence[str]): Fusion method for feature fusion,
            Options are 'channels_add', 'channel_mul', stand for channelwise
            addition and multiplication respectively. Default: ('channel_add',)
    """
    def __init__(
        self,
        in_channels: int,
        ratio: float,
        pooling_type: str = "attn",
        fusion_types: tuple = ("channel_add",),
    ):
        super().__init__()
        assert pooling_type in ["avg", "attn"]
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ["channel_add", "channel_mul"]
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, "at least one fusion should be used"
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == "attn":
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == "attn":
            nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in")
            nn.init.constant_(self.conv_mask.bias, 0)

        if self.channel_add_conv is not None:
            nn.init.constant_(self.channel_add_conv[-1].weight, 0)
            nn.init.constant_(self.channel_add_conv[-1].bias, 0)
        if self.channel_mul_conv is not None:
            nn.init.constant_(self.channel_mul_conv[-1].weight, 0)
            nn.init.constant_(self.channel_mul_conv[-1].bias, 0)

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        if self.pooling_type == "attn":
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class RepVggBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        inplace: bool = True,
        alpha: bool = False,
    ):
        super().__init__()
        if activation_layer is None:
            activation_layer = nn.ReLU
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_layer(inplace=True)

        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation_layer=None,
            inplace=inplace,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=None,
            inplace=inplace,
        )
        self.alpha = nn.Parameter(torch.tensor(1.0)) if alpha else 1.0

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x) + self.alpha * self.conv2(x)
        return self.activation(y)


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.SiLU,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True,
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, activation_layer=activation_layer)
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottlenecks(self.conv1(x)) + self.conv2(x)
        x = self.conv3(x)
        return x
