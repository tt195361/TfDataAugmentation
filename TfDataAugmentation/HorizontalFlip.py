#
# HorizontalFlip.py
#

import tensorflow as tf
from . import BaseAug
from . import image_utils as iu


class HorizontalFlip(BaseAug):
    def __init__(self, p):
        super(HorizontalFlip, self).__init__(p)

    def _do_aug_image(self, image):
        aug_image = tf.image.flip_left_right(image)
        return aug_image

    def _do_aug_mask(self, mask):
        aug_mask = tf.image.flip_left_right(mask)
        return aug_mask

    def _do_aug_bboxes(self, bboxes):
        x1, y1, x2, y2 = iu.decompose_bboxes(bboxes)
        x1 = 1.0 - x1
        x2 = 1.0 - x2
        aug_bboxes = iu.compose_bboxes(x1, y1, x2, y2)
        return aug_bboxes
