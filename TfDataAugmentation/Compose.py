#
# Compose.py
#

import tensorflow as tf
from . import BaseAug
from .BboxParams import BboxParams
from . import gen_utils
from . import image_utils


class Compose(BaseAug):
    def __init__(self, transforms, bbox_params=None, p=1.0):
        super(Compose, self).__init__(p)
        self.transforms = gen_utils.check_transforms(transforms)
        self.bbox_processor = BboxParams.make_processor(bbox_params)

    def do_aug(self, **data):
        if "bboxes" in data.keys():
            data["bboxes"] = self._to_internal_bboxes(data)

        for trans in self.transforms:
            data = trans(**data)

        if "bboxes" in data.keys():
            data["bboxes"] = self._to_original_bboxes(data)
        return data

    def _to_internal_bboxes(self, data):
        original_bboxes = data["bboxes"]
        image = data["image"]
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        internal_bboxes = self.bbox_processor.to_internal_format(
            original_bboxes, image_height, image_width)
        return internal_bboxes

    def _to_original_bboxes(self, data):
        internal_bboxes = data["bboxes"]
        image = data["image"]
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        original_bboxes = self.bbox_processor.to_original_format(
            internal_bboxes, image_height, image_width)
        return original_bboxes

    def _do_aug_image(self, image):
        raise NotImplementedError(
            "_do_aug_image for Compose should not be called")
