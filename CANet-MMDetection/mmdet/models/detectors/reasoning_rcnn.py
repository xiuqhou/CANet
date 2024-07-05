import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseDetector
from ..builder import DETECTORS
from .. import builder
from mmdet.core import (bbox2roi, bbox2result, multi_apply, merge_aug_masks, build_assigner, build_sampler)

import numpy as np
import pickle
from mmdet.core.bbox.samplers import SamplingResult

from mmcv.cnn import ConvModule

def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    if len(sampling_result.bboxes) < cfg.sampler.num:
        cfg_num_neg = cfg.sampler.num - len(sampling_result.bboxes) # 不够的用负样本来补充
        add_index = np.random.randint(0, len(sampling_result.neg_bboxes), cfg_num_neg)
        sampling_result.neg_inds = torch.cat([sampling_result.neg_inds, sampling_result.neg_inds[add_index]], 0)
        sampling_result.neg_bboxes = torch.cat([sampling_result.neg_bboxes, sampling_result.neg_bboxes[add_index]], 0)
    return assign_result, sampling_result

@DETECTORS.register_module
class ReasoningRCNN(BaseDetector):
    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 adj_gt=None,
                 graph_out_channels=256,
                 normalize=None,
                 roi_feat_size=7,
                 shared_num_fc=2):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(ReasoningRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn) # 更新训练策略和测试策略
            self.rpn_head = builder.build_head(rpn_head_)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))

        self.normalize = normalize
        self.with_bias = normalize is None
        if adj_gt is not None:
            self.adj_gt = pickle.load(open(adj_gt, 'rb'))
            self.adj_gt = np.float32(self.adj_gt)
            self.adj_gt = nn.Parameter(torch.from_numpy(self.adj_gt), requires_grad=False)
        # init cmp attention
        self.cmp_attention = nn.ModuleList()
        self.cmp_attention.append(
            ConvModule(1024, 1024 // 16,
            3, stride=2, padding=1,
            norm_cfg=self.normalize,
            bias=self.with_bias))
        self.cmp_attention.append(
            nn.Linear(1024 // 16, bbox_head[0]['in_channels'] + 1))
        # init graph w
        self.graph_out_channels = graph_out_channels
        self.graph_weight_fc = nn.Linear(bbox_head[0]['in_channels'] + 1, self.graph_out_channels)
        self.relu = nn.ReLU(inplace=True)

        # shared upper neck
        in_channels = rpn_head['in_channels']
        if shared_num_fc > 0:
            in_channels *= (roi_feat_size * roi_feat_size)
        self.branch_fcs = nn.ModuleList()
        for i in range(shared_num_fc):
            fc_in_channels = (in_channels
                              if i == 0 else bbox_head[0]['in_channels'])
            self.branch_fcs.append(
                nn.Linear(fc_in_channels, bbox_head[0]['in_channels']))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(ReasoningRCNN, self).init_weights()
        self.backbone.init_weights()
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        # precmp attention
        if len(x) > 1:
            base_feat = []
            for b_f in x[1:]:
                base_feat.append(
                    F.interpolate(b_f, scale_factor=(x[2].size(2) / b_f.size(2), x[2].size(3) / b_f.size(3))))
            base_feat = torch.cat(base_feat, 1)
        else:
            base_feat = torch.cat(x, 1)

        for ops in self.cmp_attention:
            base_feat = ops(base_feat)
            if len(base_feat.size()) > 2:
                base_feat = base_feat.mean(3).mean(2)
            else:
                base_feat = self.relu(base_feat)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train( # NOTE: 这里有问题！！config没有传进来！采样不够数目
                x, 
                img_meta, 
                gt_bboxes, 
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg, 
                **kwargs)

            # rpn_outs = self.rpn_head(x)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            # rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            # proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            # proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # add reasoning process
            if i > 0:
                # 1.build global semantic pool
                global_semantic_pool = torch.cat((bbox_head.fc_cls.weight, bbox_head.fc_cls.bias.unsqueeze(1)), 1).detach() # 81 * 1025
                # 2.compute graph attention N*D D*C -> N*C
                attention_map = nn.Softmax(1)(torch.mm(base_feat, torch.transpose(global_semantic_pool, 0, 1))) 
                # 1 * 81
                # 3.adaptive global reasoning # TODO: 这里图还是用的COCO
                # N*C*1  1*C*D -> N*C*D
                alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0) 
                # 1 * 81 * 1025
                alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1)) # (N*C, D)
                # 81 * 1025
                alpha_em = self.graph_weight_fc(alpha_em)
                # 81 * 256
                alpha_em = self.relu(alpha_em)
                # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
                n_classes = bbox_head.fc_cls.weight.size(0) # 81 # TODO: cls_score 512 * 81
                cls_prob = nn.Softmax(1)(cls_score).view(len(img_meta), -1, n_classes) 
                enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_meta), -1, self.graph_out_channels))
                enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)

            if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(len(img_meta))]
            # assign gts and sample proposals
            assign_results, sampling_results = multi_apply(
                assign_and_sample,
                proposal_list,
                gt_bboxes,
                gt_bboxes_ignore,
                gt_labels,
                cfg=rcnn_train_cfg)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            # rois = bbox2roi([(res.pos_bboxes, res.neg_bboxes) for res in sampling_results])
            rois = bbox2roi([res.bboxes for res in sampling_results]) # 512 * 5
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            # without upperneck
            bbox_feats = bbox_feats.view(bbox_feats.size(0), -1) # 512 * 256 * 7 * 7
            for fc in self.branch_fcs:
                bbox_feats = self.relu(fc(bbox_feats))

            # cat with enhanced feature
            if i > 0:
                try:
                    bbox_feats = torch.cat([bbox_feats, enhanced_feat], 1)
                except:
                    print("self.graph_out_channels", self.graph_out_channels)
                    print("bbox_feats shape", bbox_feats.shape)
                    print("enhanced_feat shape", enhanced_feat.shape)
                    print("global semantic pool", global_semantic_pool.shape)
                    print("attention map", attention_map.shape)
                    print("n_classes", n_classes)
                    print("alpha_em", alpha_em.shape)
                    print("cls_prob", cls_prob.shape)
                    print("x shape", x.shape)

            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, rois, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(
                    i, name)] = (value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)

        # precmp attention
        if len(x) > 1:
            base_feat = []
            for b_f in x[1:]:
                base_feat.append(
                    F.interpolate(b_f, scale_factor=(x[2].size(2) / b_f.size(2), x[2].size(3) / b_f.size(3))))
            base_feat = torch.cat(base_feat, 1)
        else:
            base_feat = torch.cat(x, 1)

        for ops in self.cmp_attention:
            base_feat = ops(base_feat)
            if len(base_feat.size()) > 2:
                base_feat = base_feat.mean(3).mean(2)
            else:
                base_feat = self.relu(base_feat)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            # add reasoning process
            if i > 0:
                # transform CxC classes graph to region
                # 1.build global semantic pool
                global_semantic_pool = torch.cat((bbox_head.fc_cls.weight, bbox_head.fc_cls.bias.unsqueeze(1)), 1).detach()
                # 2.compute graph attention
                attention_map = nn.Softmax(1)(torch.mm(base_feat, torch.transpose(global_semantic_pool, 0, 1)))
                # 3.adaptive global reasoning
                alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
                alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
                alpha_em = self.graph_weight_fc(alpha_em)
                alpha_em = self.relu(alpha_em)
                n_classes = bbox_head.fc_cls.weight.size(0)
                cls_prob = nn.Softmax(1)(torch.cat(cls_score, 0)).view(len(img_metas), -1, n_classes)
                enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_metas), -1, self.graph_out_channels))
                enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)

            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            # without upperneck
            bbox_feats = bbox_feats.view(bbox_feats.size(0), -1)
            for fc in self.branch_fcs:
                bbox_feats = self.relu(fc(bbox_feats))
            # cat with enhanced feature
            if i > 0:
                bbox_feats = torch.cat([bbox_feats, enhanced_feat], 1)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            # TODO: 这里进行划分
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(ReasoningRCNN, self).show_result(data, result, img_norm_cfg,
                                                           **kwargs)
