_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100e.py'
]

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[16, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(bbox_head=dict(num_classes=6)),
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
data_root = 'data/NEU-DET-scale_coco/'
dataset_type = 'CocoDataset'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')

data = dict(
    samples_per_gpu=4,
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
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
evaluation = dict(interval=1, metric='bbox')



