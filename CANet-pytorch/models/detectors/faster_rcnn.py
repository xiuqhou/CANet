from typing import Dict, List, Tuple

from torch import Tensor, nn
from torchvision.models.detection.transform import resize_boxes

from models.detectors.base_detector import BaseDetector


class FasterRCNN(BaseDetector):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
        min_size=None,
        max_size=None,
        size_divisible=32,
    ) -> None:
        super().__init__(min_size, max_size, size_divisible)
        self.backbone = backbone
        self.neck = neck
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets = self.preprocess(images, targets)

        # extract features
        features = self.backbone(images.tensors)
        features = self.neck(features)

        # get predictions and losses
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        if self.training:
            return {**detector_losses, **proposal_losses}

        detections = self.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    @staticmethod
    def postprocess(
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: Tensor,
    ) -> List[Dict[str, Tensor]]:
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result
