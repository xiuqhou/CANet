# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class LaplacianFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 attention=True, # 是否引入注意力机制
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LaplacianFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels # 输入通道列表，backbone每一个阶段的输出
        self.out_channels = out_channels  # 输出int
        self.num_ins = len(in_channels) # 输入特征图数目
        self.num_outs = num_outs # 输出特征图数目
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        """新加的"""
        self.attention = attention

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins  # backbone最后一张特征图的stage_level
            assert num_outs >= self.num_ins - start_level # 
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cross_convs = nn.ModuleList()
        # 这部分是4 -> 4
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            # 新加的
            cross_conv = LaplacianFusion(out_channels)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.cross_convs.append(cross_conv)
        """增加层注意力和空间注意力模块"""
        if self.attention == True:
            self.attention_aggregation = MutualAttention(in_channels=[out_channels] * self.num_ins, norm_cfg=norm_cfg)
        # 这部分多一个额外的卷积，4 -> 5
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1: # 如果金字塔再往上扩展
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input': # 如果对于输入卷积，则对应backbone最后一层
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:   # 否则是对现有金字塔最顶层进行卷积
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = self.cross_convs[i](laterals[i - 1], F.interpolate(
                    laterals[i], **self.upsample_cfg))
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.cross_convs[i](laterals[i - 1], F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg))

        """先用4->4注意力试试效果，好的话改成4->5"""
        # print("执行注意力权重")
        if self.attention == True:
            laterals = self.attention_aggregation(laterals)
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

# class AttentionAggregation(BaseModule):
#     def __init__(self, in_channels, conv_cfg=None, norm_cfg=None, act_cfg=None, init_cfg = None):
#         super().__init__(init_cfg)  # [b, 128, h1, w2], [b, 256, h1, w1]
#         self.in_channels = in_channels[0]
#         self.num_ins = len(in_channels)
#         self.base_feat_index = 1
#         self.v = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1)
#         self.k = nn.Conv3d(self.in_channels, 1, kernel_size=1)
#         # from mmcv.cnn.builder import build_model_from_cfg
#         # self.non_local = build_model_from_cfg(dict(type='NonLocal2d'))
#         print("执行注意力聚集")

#     @auto_fp16()
#     def forward(self, inputs):
#         assert inputs[0].shape[1] == inputs[1].shape[1], "all inputs of ABP must have the same channel numbers"
#         assert len(inputs) == self.num_ins, "length of inputs must have the same number as in_channels configuration"
#         # inputs可能有两种可能，奇数选中间，偶数向上取整，所有inputs通道数都相同
#         original_shape = [input.shape for input in inputs]
#         base_shape = original_shape[self.base_feat_index]
#         # 将所有输入变换到相同维度，这里的resize方式可能不合理，选择最大的？
#         resized_feat = []
#         for i in range(self.num_ins):
#             if i == self.base_feat_index:
#                 resized_feat.append(inputs[i])
#             elif i < self.base_feat_index:
#                 down_sample_rate = 2**(self.base_feat_index - i)
#                 resized_feat.append(F.max_pool2d(inputs[i], kernel_size=down_sample_rate, stride=down_sample_rate))
#             else:
#                 up_sample_rate = 2**(i - self.base_feat_index)
#                 resized_feat.append(F.interpolate(inputs[i], scale_factor=up_sample_rate))
#         assert resized_feat[0].shape == base_shape and resized_feat[-1].shape == base_shape
#         resized_feat = torch.stack(resized_feat, 2) # [batch, c, level, h, w]
#         B, C, L, H, W = resized_feat.shape
#         # 变换维度
#         K = self.k(resized_feat).reshape(B,1,L*H*W).permute(0,2,1) # [B, L*H*W, 1]
#         # 计算权重
#         K = torch.softmax(K, 1)
#         V = self.v(resized_feat).reshape(B,C,L*H*W) # [B, C, L*H*W]
#         # 执行查询
#         attention = torch.matmul(V, K) # [B, C, 1]
#         weighted_resized_feat = resized_feat + attention[:, :, :, None, None] # [B, C, L, H, W]
#         weighted_resized_feat = weighted_resized_feat.permute(2, 0, 1, 3, 4)  # [L, B, C, H, W]
#         # 最后维度变回去
#         outputs = []
#         for i in range(self.num_ins):
#             if i == self.base_feat_index:
#                 outputs.append(weighted_resized_feat[i] + inputs[i])
#             elif i < self.base_feat_index:
#                 down_sample_rate = 2**(self.base_feat_index - i)
#                 outputs.append(F.interpolate(weighted_resized_feat[i], scale_factor=down_sample_rate) + inputs[i])
#             else:
#                 up_sample_rate = 2**(i - self.base_feat_index)
#                 outputs.append(F.max_pool2d(weighted_resized_feat[i], kernel_size=up_sample_rate, stride=up_sample_rate) + inputs[i])
#         assert outputs[0].shape == original_shape[0] and outputs[-1].shape == original_shape[-1]
#         return outputs

class LaplacianFusion(BaseModule):
    def __init__(self, channel, init_cfg = None):
        super().__init__(init_cfg)
        self.pre_low_conv = ConvModule(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=None,
            act_cfg=None,
            inplace=False
        )
        self.pre_high_conv = ConvModule(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=None,
            act_cfg=None,
            inplace=False
        )
        # self.intermedia_context_conv = ConvModule(
        #     in_channels=channel,
        #     out_channels=channel,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     norm_cfg=None,
        #     act_cfg=None,
        #     inplace=False
        # )
        # self.intermedia_edge_conv = ConvModule(
        #     in_channels=channel,
        #     out_channels=channel,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     norm_cfg=None,
        #     act_cfg=None,
        #     inplace=False
        # )
        self.SE_context = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.Sigmoid()
        )
        self.SE_edge = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
    def forward(self, low_feature, high_feature):
        # [B,C,H,Wl] + [B,C,H,W] -> [B,C,H,W]
        pre_low_feature = self.pre_low_conv(low_feature) # 先经过3*3卷积
        pre_high_feature = self.pre_high_conv(high_feature)
        fused_context = pre_low_feature + pre_high_feature # 计算侧面连接和边缘
        laplacian_edge = pre_low_feature - pre_high_feature

        intermedia_fused_context = fused_context # 中间变换也可能不需要
        intermedia_laplacian_edge = laplacian_edge
        # intermedia_fused_context = self.intermedia_context_conv(fused_context) # 中间变换也可能不需要
        # intermedia_laplacian_edge = self.intermedia_edge_conv(laplacian_edge)
        # 这里可能要进行一定修改，把交叉的SE模块给去掉，后面也不能加SE模块
        SE_fused_context = intermedia_fused_context * self.SE_edge(torch.mean(intermedia_fused_context, dim=[2,3], keepdims=True)) + intermedia_fused_context

        SE_laplacian_edge = intermedia_laplacian_edge * self.SE_context(torch.mean(intermedia_laplacian_edge, dim=[2,3], keepdims=True)) + intermedia_laplacian_edge

        result_low_feature = self.alpha * SE_fused_context + self.beta * SE_laplacian_edge
        return result_low_feature

class MutualAttention(BaseModule):
    def __init__(self, in_channels, conv_cfg=None, norm_cfg=None, act_cfg=None, init_cfg = None):
        super().__init__(init_cfg)  # [b, 128, h1, w2], [b, 256, h1, w1]
        self.in_channels = in_channels
        self.num_ins = len(self.in_channels)
        self.base_feat_index = (self.num_ins + 1) // 2
        self.conv_list1 = nn.ModuleList()
        self.conv_list2 = nn.ModuleList()
        self.conv_list3 = nn.ModuleList()
        self.conv_list4 = nn.ModuleList()
        for i in range(len(self.in_channels)):
            # self.conv_list1.append(nn.Conv2d(self.in_channels[i], self.in_channels[i], 1, 1, 0))
            # self.conv_list2.append(nn.Conv2d(self.in_channels[i], 1, 1, 1, 0))
            self.conv_list1.append(ConvModule( # 用1*1或许比较好一些
                    self.in_channels[i],
                    self.in_channels[i],
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))
            self.conv_list2.append(ConvModule( # 用1*1或许比较好一些
                    self.in_channels[i],
                    1,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))
            self.conv_list3.append(ConvModule( # 用1*1或许比较好一些
                    self.in_channels[i],
                    self.in_channels[i]//2,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))
            self.conv_list4.append(ConvModule( # 用1*1或许比较好一些
                    self.in_channels[i]//2,
                    self.in_channels[i],
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False))

    @auto_fp16()
    def forward(self, inputs):
        # FPNGC模块
        # print("执行FPNGC")
        assert inputs[0].shape[1] == inputs[1].shape[1], "all inputs of ABP must have the same channel numbers"
        assert len(inputs) == self.num_ins, "length of inputs must have the same number as in_channels configuration"
        layer_attention_aggregation = []
        for lvl in range(self.num_ins):
            conv_feat1 = self.conv_list1[lvl](inputs[lvl]) # 先做1*1卷积 [B,C,Hl,Wl]
            conv_feat2 = self.conv_list2[lvl](inputs[lvl]) # 先做1*1卷积 [B,1,Hl,Wl]
            B, _, H, W = conv_feat2.shape
            conv_feat2 = torch.softmax(conv_feat2.reshape(B, 1, H*W), -1).reshape(B, 1, H, W) # 沿着空间维度做softmax
            layer_attention = torch.einsum('bchw, blhw -> bcl', conv_feat1, conv_feat2) # [BC1]空间注意力
            layer_attention = layer_attention[:,:,:,None] # [B, C, 1, 1]
            layer_attention = self.conv_list4[lvl](self.conv_list3[lvl](layer_attention))
            layer_attention_aggregation.append(layer_attention)
        layer_attention_aggregation = torch.stack(layer_attention_aggregation) # [level, B, C, 1]

        
        outputs = []
        for i in range(self.num_ins):
            # outputs.append(inputs[i] + soft_weight[i][:, :, :, None])
            outputs.append(layer_attention_aggregation[i] + inputs[i])
        return outputs
        
if __name__ == '__main__':
    # ag = AttentionAggregation(in_channels=[256, 256, 256])
    # x = [torch.randn(4, 256, 40, 40), torch.randn(4, 256, 20, 20), torch.randn(4, 256, 10, 10)]
    # output = ag(x)
    # in_channels=[256, 512, 1024, 2048]
    # out_channels=256
    # num_outs=5
    # abfpn1 = LaplacianFPN(in_channels, out_channels, num_outs, attention=True)
    # abfpn2 = LaplacianFPN(in_channels, out_channels, num_outs, attention=False)
    # x = [torch.randn(4,256,40,80), torch.randn(4,512,20,40),torch.randn(4,1024,10,20),torch.randn(4,2048,5,10)]
    # result1 = abfpn1(x)
    # result2 = abfpn2(x)
    a = torch.randn(10, 256, 1024, 1024)
    b = torch.randn(10, 256, 1024, 1024)
    lp_attn = LaplacianFusion(256)
    result = lp_attn(a, b)
    pass