import warnings
from typing import Any

import cv2
import torch
from torch import nn

from util import datapoints
from datasets.coco import CocoDetection


class AlbumentationsWrapper(nn.Module):
    def __init__(self, albumentation_transforms):
        """

        :param albumentation_transforms: albumentations transformation for data augmentation. For example:
        """
        super().__init__()
        self.albumentation_transforms = albumentation_transforms

    def forward(self, input: Any) -> Any:
        # get image, box, mask, label from input
        labels = input[-1]
        not_allowed_data = list(
            filter(
                lambda x: not isinstance(
                    x, (datapoints.Image, datapoints.BoundingBox, datapoints.Mask)
                ),
                input,
            )
        )
        not_allowed_data_type = set(list(map(lambda x: type(x), not_allowed_data)))
        if len(not_allowed_data) != 1:
            warnings.warn(
                f"current we only support images,  bounding boxes and masks"
                f"transformation for albumentations, but got {not_allowed_data_type}"
            )
        images = list(filter(lambda x: isinstance(x, datapoints.Image), input))
        boxes = list(filter(lambda x: isinstance(x, datapoints.BoundingBox), input))
        masks = list(filter(lambda x: isinstance(x, datapoints.Mask), input))
        if len(images) != 1 or len(boxes) != 1:
            raise ValueError

        # prepare albumentations input format
        images = images[0].data.numpy().transpose(1, 2, 0)
        boxes = boxes[0].data.numpy()
        keep = (boxes[:, 2] > boxes[:, 0]) & (
            boxes[:, 3] > boxes[:, 1]
        )  # TODO: change into a function
        input_dict = {
            "image": images,
            "bboxes": boxes[keep],
            "labels": labels.numpy()[keep],
        }
        if len(masks) != 0:
            masks = masks[0].data.numpy()
            if masks.ndim == 3:
                masks = masks.transpose(1, 2, 0)[keep]
            input_dict.update({"mask": masks})

        # perform albumentations transforms
        transformed = self.albumentation_transforms(**input_dict)
        images, boxes, labels = (
            transformed["image"],
            transformed["bboxes"],
            transformed["labels"],
        )
        if "mask" in transformed:
            masks = transformed["mask"]
            if masks.ndim == 3:
                masks = masks.transpose(2, 0, 1)
            masks = datapoints.Mask(masks)
        else:
            masks = None

        # prepare output data format
        images = datapoints.Image(images.transpose(2, 0, 1))
        boxes = datapoints.BoundingBox(
            torch.as_tensor(boxes).reshape(-1, 4),  # in case of empty boxes after transforms
            dtype=torch.float,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=images.shape[-2:],
        )
        output = [images, boxes]
        if masks is not None:
            output.append(masks)
        labels = torch.as_tensor(labels, dtype=torch.long)
        output.append(labels)
        return tuple(output)

    def __str__(self):
        return str(self.albumentation_transforms)


if __name__ == "__main__":
    import albumentations as A

    albumentation_transform = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.3), contrast_limit=(0.1, 0.3), p=0.2
            ),
            A.OneOf(
                [
                    A.RGBShift(
                        r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
            A.ChannelShuffle(p=0.1),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.1,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.0
        ),
    )

    transform = AlbumentationsWrapper(albumentation_transform)

    dataset = CocoDetection(
        img_folder=f"/data1/houxiuquan/detection/data/mobile_screen_reviewer_2_coco/train2017",
        ann_file=f"/data1/houxiuquan/detection/data/mobile_screen_reviewer_2_coco/annotations/instances_train2017.json",
        transforms=transform,
        filter_empty_img=True,
    )

    print(dataset[0])
