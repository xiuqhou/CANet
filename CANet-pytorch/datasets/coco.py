import json
import logging
import os
import tempfile
from typing import Callable, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torchvision
from terminaltables import AsciiTable
from tqdm import tqdm

from transforms import v2 as T
from transforms.convert_coco_polys_to_mask import ConvertCocoPolysToMask
from util import datapoints
from util.misc import deepcopy


def _filter_coco_ann_dict(
    ann_file,
    class_ids: List[Union[int, str]] = None,
    min_size: float = 0.0,
    min_box_area: float = 0.0,
    filter_empty_img: bool = False,
    save_file_name: str = None,
    filter_ignore: bool = False,
    filter_crowd: bool = False,
):
    # set up a logger to print useful information
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    # open coco annotations json file
    with open(ann_file, "r") as f:
        ann_dict = json.load(f)
    logger.info(f"Process COCO annotation json file: {ann_file}.")

    # collect statistical information
    data_metas = ["Images", "Annotations", "Categories"]
    statis_info = [["Filtering", *data_metas]]
    statis_info.append(["before", *[str(len(ann_dict[key.lower()])) for key in data_metas]])

    # filter class_ids for category
    if class_ids is None:
        class_ids = [c["id"] for c in ann_dict["categories"]]
    if isinstance(class_ids[0], str):
        categories = {c["name"]: c["id"] for c in ann_dict["categories"]}
        class_ids = [categories[c] for c in class_ids]

    # filter ann_dict["categories"] according to class_ids
    class_ids = list(set(class_ids))  # make sure no duplicate class_ids
    class_id_to_idx = {c["id"]: i for i, c in enumerate(ann_dict["categories"])}
    class_idx = [class_id_to_idx[i] for i in class_ids]
    ann_dict["categories"] = [ann_dict["categories"][i] for i in class_idx]

    # for annotations, filter class_id, min_size, min_box_area
    _filter_ann = lambda x: ((not filter_ignore or not x.get("ignore", False)) and
                             (not filter_crowd or x.get("iscrowd", 0) != 1) and
                             (x["category_id"] in class_ids) and
                             (min(x["bbox"][-2:]) > min_size) and
                             (x["bbox"][-2] * x["bbox"][-1] > min_box_area))
    ann_dict["annotations"] = list(filter(_filter_ann, tqdm(ann_dict["annotations"])))

    # for images
    if filter_empty_img:
        valid_image_id = list(set([a["image_id"] for a in ann_dict["annotations"]]))
        image_id_to_idx = {img["id"]: i for i, img in enumerate(ann_dict["images"])}
        valid_image_idx = [image_id_to_idx[i] for i in valid_image_id]
        ann_dict["images"] = [ann_dict["images"][i] for i in valid_image_idx]

    # update and show collected statistical information
    statis_info.append(["after", *[str(len(ann_dict[key.lower()])) for key in data_metas]])
    table = AsciiTable(statis_info)
    table.inner_footing_row_border = True
    logger.info("\n" + table.table)

    # save the filtered ann_dict
    temp_file = None
    if save_file_name is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        save_file_name = temp_file.name
    with open(save_file_name, "w") as f:
        f.write(json.dumps(ann_dict))
    logger.info(f"Annotation dict saved to {save_file_name}")

    # set automatically close
    if temp_file is not None:
        temp_file.delete = True

    return save_file_name


class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO Dataset for object detection."""
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms: Optional[Callable] = None,
        class_ids: Optional[List[Union[int, str]]] = None,
        min_size: float = 0,
        min_box_area: float = 0,
        filter_empty_img: bool = False,
        filter_ignore: bool = False,
        filter_crowd: bool = False,
        save_file_name: Optional[str] = None,
    ):
        """init COCO Dataset for object detection.

        :param img_folder: folder for coco images, for example: coco/train2017
        :param ann_file: path to json file of coco annotations, end with `.json`
        :param transforms: data augmentation for images, see `transforms.presets.py`
        :param class_ids: only annotations with categories in class_ids will be kept.
            If not given, all annotations will be kept, defaults to None.
        :param min_size: given a float, only annotations with box height and width
            larger than it will be kept, defaults to 0.0.
        :param min_box_area: given a float, only annotations with box area larger than
            it will be kept, defaults to 0.0.
        :param filter_empty_img: whether to ignore images without annotations, defaults to False.
        :param filter_ignore: whether to ignore annotations with `ignore` attribute is True.
        :param filter_crowd: whether to ignore annotations with `iscrowd` attribute is True.
        :param save_file_name: save filtered annotations to the given file path, defaults to None.
        """
        # filter annotation file
        ann_file = _filter_coco_ann_dict(
            ann_file=ann_file,
            class_ids=class_ids,
            min_size=min_size,
            min_box_area=min_box_area,
            filter_empty_img=filter_empty_img,
            filter_ignore=filter_ignore,
            filter_crowd=filter_crowd,
            save_file_name=save_file_name,
        )
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask()
        self._transforms = transforms
        self._transforms = self.update_dataset(self._transforms)

    def update_dataset(self, transform):
        if isinstance(transform, (T.Compose, A.Compose)):
            processed_transforms = []
            for trans in transform.transforms:
                trans = self.update_dataset(trans)
                processed_transforms.append(trans)
            return type(transform)(processed_transforms)
        if hasattr(transform, "update_dataset"):
            transform.update_dataset(self)
        return transform

    def load_image(self, image_name):
        # after comparing the speed and the compatibility of PIL, torchvision and cv2,
        # torchvision is chosen as the default backend to load images,
        # uncomment the following code to switch among them.

        # 1. PIL backends, low speed
        # image = Image.open(os.path.join(self.root, image_name)).convert('RGB')

        # 2. cv2 backends, sometimes not compatible with multi-process on Windows system
        # # To avoid deadlock between DataLoader and OpenCV
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        # image = cv2.imdecode(np.fromfile(os.path.join(self.root, image_name), dtype=np.uint8), -1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        # 3. torchvision backends, only support .jpg and .png
        image = torchvision.io.read_image(os.path.join(self.root, image_name))

        return image

    def get_image_id(self, item: int):
        return self.ids[item]

    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def data_augmentation(self, image, target):
        # preprocess
        image = datapoints.Image(image)
        bounding_boxes = datapoints.BoundingBox(
            target["boxes"],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.shape[-2:],
        )
        labels = target["labels"]
        if self._transforms is not None:
            image, bounding_boxes, labels = self._transforms(image, bounding_boxes, labels)

        return image.data, bounding_boxes.data, labels

    def __getitem__(self, item):
        image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)

    def __len__(self):
        return len(self.indices) if hasattr(self, "indices") else len(self.ids)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class Object365Detection(CocoDetection):
    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        # NOTE: Only for object 365
        image_name = os.path.join(*image_name.split(os.sep)[-2:])
        if self.train:
            image_name = os.path.join("images/train", image_name)
        else:
            image_name = os.path.join("images/val", image_name)
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def __getitem__(self, item):
        try:
            image, target = self.load_image_and_target(item)
        except:
            item += 1
            image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)
