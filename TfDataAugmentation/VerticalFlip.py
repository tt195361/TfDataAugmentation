#
# VerticalFlip.py
#

import tensorflow as tf
from . import BaseAug
from . import image_utils as iu


class VerticalFlip(BaseAug):
    def __init__(self, p=0.5):
        super(VerticalFlip, self).__init__(p)

    def _do_aug_image(self, image):
        aug_image = tf.image.flip_up_down(image)
        return aug_image

    def _do_aug_mask(self, mask):
        aug_mask = tf.image.flip_up_down(mask)
        return aug_mask

    def _do_aug_bboxes(self, bboxes):
        x1, y1, x2, y2 = iu.decompose_bboxes(bboxes)
        y1 = 1.0 - y1
        y2 = 1.0 - y2
        aug_bboxes = iu.compose_bboxes(x1, y1, x2, y2)
        return aug_bboxes
