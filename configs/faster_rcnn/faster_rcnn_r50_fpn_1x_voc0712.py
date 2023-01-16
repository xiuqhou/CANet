_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

plugins = [     # 占用显存很大，慎用
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            position='after_conv2'),
        # dict(cfg=dict(type='NonLocal2d'), position='after_conv2'), # 去掉这个能跑4batch_size
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]
# plugins = None

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(plugins=plugins), # 后续加上空间金字塔池化
    neck=dict(
        type='FPN',  # 之后改为UFPN
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_input', # 可能直接上采样效果不好，需要经过卷积
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],  # 1333的情况下减小基础锚框的尺寸[8,4,2]->[6,4] # 2太小，8太大，1000的情况下8, 4, 2
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]), # 默认值[4,8,16,32,64] 
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1 / 9, loss_weight=1.0)),  # 损失函数改为SmoothL1Loss，参数和xxy保持一致
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2)),  # 这里sampling_ratio跟xxy保持一致
        bbox_head=dict(
            num_classes=20)),
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
            nms=dict(type='nms', iou_threshold=0.7), # 这里VOC和WSJ的设置不一样，VOC默认为0.7
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
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 1256)],
        multiscale_mode='range',
        keep_ratio=True,
        backend='pillow'), # 多尺度训练只有短边的上下限
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
        img_scale=[(1333, 800), (1333, 640), (1333, 1024)], # 多尺度测试
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
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # 学习率原本是0.01~4
optimizer_config = dict(grad_clip=None)
# learning policy
# 学习率调度器和动量调度器
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook。
    policy='step',
    step=[3])  # 衰减学习率的起止回合数

# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 4 * 3 = 12

# evaluation = dict(interval=1, metric='map')

