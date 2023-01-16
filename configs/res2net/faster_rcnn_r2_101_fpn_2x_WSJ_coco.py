_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 4, 2],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            num_classes=2))
)


# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale = (1333,800),
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
data_root = 'data/WSJ_2_coco/'
dataset_type = 'CocoDataset'
classes = ('scratch', 'spot')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix=data_root+'train2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_train2014.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix=data_root+'val2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_val2014.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix=data_root+'val2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_val2014.json',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# 学习率调度器和动量调度器
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook。
    policy='step',
    warmup='linear',  # 预热(warmup)策略，也支持 `exp` 和 `constant`。
    warmup_iters=1000,  # 预热的迭代次数
    warmup_ratio=
    0.001,  # 用于热身的起始学习率的比率
    step=[8, 11])  # 衰减学习率的起止回合数

# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=100)  # actual epoch = 4 * 3 = 12

evaluation = dict(interval=1, metric='bbox')

