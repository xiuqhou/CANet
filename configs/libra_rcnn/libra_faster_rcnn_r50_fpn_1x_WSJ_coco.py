_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            num_classes=2,
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8,4,2],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))))

data_root = 'data/WSJ_2_coco/'
dataset_type = 'CocoDataset'
classes = ('scratch', 'spot')
data = dict(
    train=dict(
        img_prefix=data_root+'train2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_train2014.json'),
    val=dict(
        img_prefix=data_root+'val2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_val2014.json'),
    test=dict(
        img_prefix=data_root+'val2014/',
        classes=classes,
        ann_file=data_root+'annotations/instances_val2014.json'))

runner = dict(type='EpochBasedRunner', max_epochs=100)