#
# Transpose.py
#

import tensorflow as tf
from . import BaseAug
from . import image_utils as iu


class Transpose(BaseAug):
    def __init__(self, p=0.5):
        super(Transpose, self).__init__(p)

    def _do_aug_image(self, image):
        aug_image = tf.transpose(image, perm=[1, 0, 2])
        return aug_image

    def _do_aug_mask(self, mask):
        aug_mask = tf.transpose(mask, perm=[1, 0, 2])
        return aug_mask

    def _do_aug_bboxes(self, bboxes):
        x_min, y_min, x_max, y_max = iu.decompose_bboxes(bboxes)
        aug_bboxes = iu.compose_bboxes(y_min, x_min, y_max, x_max)
        return aug_bboxes
