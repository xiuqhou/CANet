from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OHEMSampler:
    def __init__(self, batch_size_per_image: int, positive_fraction: float, context: nn.Module):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.context = context

    @torch.no_grad()
    def calculate_loss(
        self,
        matched_idxs: List[Tensor],
        proposals: List[Tensor],
        features: Dict[str, Tensor],
        image_shapes: List[Tuple[int, int]],
    ):

        class_logits = self.context.bbox_forward(features, proposals, image_shapes)[0]
        classification_loss = F.cross_entropy(
            class_logits, torch.cat(matched_idxs), reduction="none", ignore_index=-1
        )
        return classification_loss

    def __call__(
        self,
        matched_idxs: List[Tensor],
        proposals: List[Tensor],
        features: Dict[str, Tensor],
        image_shapes: List[Tuple[int, int]],
        **kwargs,
    ):
        # predict losses using a batch mode
        num_proposals = [len(p) for p in proposals]
        classification_loss = self.calculate_loss(matched_idxs, proposals, features, image_shapes)

        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image, loss in zip(
            matched_idxs, classification_loss.split(num_proposals)
        ):
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # select top-k hardest examples from positive/negative
            pos_idx_per_image = positive[loss[positive].topk(num_pos)[1]]
            neg_idx_per_image = negative[loss[negative].topk(num_neg)[1]]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
