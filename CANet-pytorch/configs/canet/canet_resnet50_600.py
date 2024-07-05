from functools import partial
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d

from models.anchor.anchor_generator import AnchorGenerator
from models.backbones.plugins import PluginConfig, SpatialAttentionModule
from models.backbones.resnet import ResNetBackbone
from models.bricks.basic import ContextBlock
from models.bricks.faster_rcnn_predictor import FastRCNNPredictor
from models.detectors.faster_rcnn import FasterRCNN
from models.heads.roi_head import RoIHeads
from models.heads.rpn_head import RegionProposalNetwork, RPNHead
from models.matcher.max_iou_matcher import MaxIoUMatcher
from models.necks.fusion_fpn import LaplacianFeaturePyramidNetwork

num_classes = 91
embed_dim = 256
representation_size = 1024
rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}
rpn_post_nms_top_n = {"training": 1000, "testing": 1000}
# feature map stride: (4, 8, 16, 32, 64), each produces anchors with base size 8
# therefore anchor_sizes is (32, 64, 128, 256, 512), each has three variants aspect ratios
anchor_sizes = ((32, 16, 8), (64, 32, 16), (128, 64, 32), (256, 128, 64), (512, 256, 128))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

# different from DETR series, FasterRCNN uses all the feature levels
backbone = ResNetBackbone(
    arch="resnet50",
    norm_layer=FrozenBatchNorm2d,
    freeze_indices=(0,),
    plugins=[
        PluginConfig(
            block=partial(SpatialAttentionModule),
            stages=(1, 2, 3),  # (0, 1, 2, 3)
            conv_pos=(1,),  # (0, 1, 2)
        ),
        PluginConfig(
            block=partial(ContextBlock, ratio=1.0 / 16),
            stages=(1, 2, 3),  # (0, 1, 2, 3)
            conv_pos=(2,),  # (0, 1, 2)
        )
    ],
)
neck = LaplacianFeaturePyramidNetwork(
    in_channels_list=backbone.num_channels,
    out_channels=embed_dim,
    extra_block=True,
)
# collect featmap_names of neck output, used in MultiScaleRoIAlign
feat_names = backbone.return_layers + neck.extra_block_name
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

model = FasterRCNN(
    backbone=backbone,
    neck=neck,
    rpn=RegionProposalNetwork(
        anchor_generator=rpn_anchor_generator,
        head=RPNHead(
            in_channels=embed_dim, num_anchors=rpn_anchor_generator.num_anchors_per_location()[0]
        ),
        proposal_matcher=MaxIoUMatcher(
            high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=True
        ),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n=rpn_pre_nms_top_n,
        post_nms_top_n=rpn_post_nms_top_n,
        nms_thresh=0.7,
        score_thresh=0.0,
    ),
    roi_heads=RoIHeads(
        box_roi_pool=MultiScaleRoIAlign(featmap_names=feat_names, output_size=7, sampling_ratio=2),
        box_head=TwoMLPHead(in_channels=embed_dim * 7**2, representation_size=representation_size),
        box_predictor=FastRCNNPredictor(in_channels=representation_size, num_classes=num_classes),
        proposal_matcher=MaxIoUMatcher(
            high_threshold=0.5, low_threshold=0.5, allow_low_quality_matches=False
        ),
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    ),
    min_size=600,
    max_size=600,
)
