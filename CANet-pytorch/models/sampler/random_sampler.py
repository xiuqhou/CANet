from typing import List, Tuple
from torch import Tensor
import torchvision.models.detection._utils as det_utils


class BalancedPositiveNegativeSampler(det_utils.BalancedPositiveNegativeSampler):
    # This is a wrapper for BalancedPositiveNegativeSampler from torchvision
    # so that it can accepet **kwargs in the __call__
    def __call__(self, matched_idxs: List[Tensor], **kwargs) -> Tuple[List[Tensor], List[Tensor]]:
        return super(BalancedPositiveNegativeSampler, self).__call__(matched_idxs)
