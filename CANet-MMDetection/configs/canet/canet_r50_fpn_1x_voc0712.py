_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

plugins = [
        dict(
            cfg=dict(
                type='SpatialAttentionEncoder',
                num_heads=8,
                kv_stride=2),
            position='after_conv2'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]

# model settings
model = dict(
    backbone=dict(plugins=plugins),
    neck=dict(
        type='LaplacianFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
)
