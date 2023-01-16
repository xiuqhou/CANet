_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100e.py'
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

model = dict(
    type='FasterRCNN',
    backbone=dict(plugins=plugins),
    neck=dict(
        type='LaplacianFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[16, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1 / 9, loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2)),
        bbox_head=dict(
            num_classes=6)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1)),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=500, # 这里原本为1000
            nms=dict(type='nms', iou_threshold=0.6), # 这里原本为0.5
            min_bbox_size=0),
        rcnn=dict(
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
            ),
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(600, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale = (600, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset settings
data_root = 'data/NEU-DET-scale_VOC/'
dataset_type = 'VOCDataset'
# classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')
data = dict(
    train=dict(
        dataset=dict(
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root +'VOC2007'],
            pipeline=train_pipeline)),
    val=dict(
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007',
        pipeline=test_pipeline))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
evaluation = dict(interval=1, metric='mAP')



