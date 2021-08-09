#
# ShiftScaleRotate.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils


def make_image_map(height, width, trans_mat):
    h_rng = tf.range(height, dtype=tf.float32)
    w_rng = tf.range(width, dtype=tf.float32)
    x, y = tf.meshgrid(w_rng, h_rng)
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    ones = tf.ones_like(x)
    coord_mat = tf.stack([x, y, ones])

    res_mat = tf.linalg.matmul(trans_mat, coord_mat)
    map_x = res_mat[0]
    map_y = res_mat[1]
    return map_x, map_y


class ShiftScaleRotate(BaseAug):
    def __init__(
            self, shift_limit, scale_limit, rotate_limit,
            interpolation='bilinear', border_mode='constant',
            p=1.0):
        super(ShiftScaleRotate, self).__init__(p)
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation
        self.border_mode = border_mode

    def _make_params(self, image):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        rnd_shift = gen_utils.random_float(
            [2], -self.shift_limit, self.shift_limit)
        params = {
            "tx": image_width * rnd_shift[0],
            "ty": image_height * rnd_shift[1],
            "z": gen_utils.random_float(
                [], 1.0 - self.scale_limit, 1.0 + self.scale_limit),
            "theta": gen_utils.random_float(
                [], -self.rotate_limit, self.rotate_limit)
        }
        return params

    def _prepare_aug(self, image, params):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        image_trans_mat = image_utils.make_trans_mat(
            image_height, image_width,
            -params["tx"], -params["ty"],
            1.0/params["z"], -params["theta"])
        self.image_map_x, self.image_map_y = make_image_map(
            image_height, image_width, image_trans_mat)

    def _do_aug_image(self, image):
        aug_image = image_utils.remap(
            image, self.image_map_x, self.image_map_y,
            interpolation=self.interpolation,
            border_mode=self.border_mode)
        return aug_image

    def _do_aug_mask(self, mask):
        aug_mask = image_utils.remap(
            mask, self.image_map_x, self.image_map_y,
            interpolation=self.interpolation,
            border_mode=self.border_mode)
        return aug_mask

    def _do_aug_bboxes(self, bboxes, image):
        image_height, _ = image_utils.get_image_size(image, dtype=tf.float32)
        x1, y1, x2, y2 = image_utils.decompose_bboxes(bboxes)
        y1 = image_height - y1
        y2 = image_height - y2
        aug_bboxes = image_utils.compose_bboxes(x1, y1, x2, y2)
        return aug_bboxes
