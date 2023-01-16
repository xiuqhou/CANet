_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

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
        # img_scale=[(1333, 800), (1333, 640), (1333, 1024)], # 多尺度测试
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

data_root = 'data/test_data/'
dataset_type = 'CocoDataset'
# classes = ('scratch', 'spot')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix=data_root+'val2017/',
        # classes=classes,
        ann_file=data_root+'annotations/instances_val2017.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix=data_root+'val2017/',
        # classes=classes,
        ann_file=data_root+'annotations/instances_val2017.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix=data_root+'val2017/',
        # classes=classes,
        ann_file=data_root+'annotations/instances_val2017.json',
        pipeline=test_pipeline))