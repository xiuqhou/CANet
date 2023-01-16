_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[3, 4, 8, 3],
        head_dim=64,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        pretrained_path='model_pth/uniformer_small_in1k.pth',
        num_classes=2,
        img_size=832),
    neck=dict(
        type='LaplacianFPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 4, 2],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1 / 9, loss_weight=1.0)),  # 损失函数改为SmoothL1Loss，参数和xxy保持一致
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2)),  # 这里sampling_ratio跟xxy保持一致
        bbox_head=dict(
            num_classes=2)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,  # 这里又改回了默认值，之前0.5
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1)),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,   # 改回1000
            nms=dict(type='nms', iou_threshold=0.5),
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
        img_scale = (1333, 800),
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
    samples_per_gpu=1,
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
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# 学习率调度器和动量调度器
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook。
    policy='step',
    # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。请从 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 参考 LrUpdater 的细节。
    warmup='linear',  # 预热(warmup)策略，也支持 `exp` 和 `constant`。
    warmup_iters=1000,  # 预热的迭代次数
    warmup_ratio=
    0.001,  # 用于热身的起始学习率的比率
    step=[8, 11])  # 衰减学习率的起止回合数

# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=100)  # actual epoch = 4 * 3 = 12

evaluation = dict(interval=1, metric='bbox')



