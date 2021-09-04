#
# ShiftScaleRotate.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils


def make_image_inv_map(height, width, trans_mat):
    h_rng = tf.range(height, dtype=tf.float32)
    w_rng = tf.range(width, dtype=tf.float32)
    x, y = tf.meshgrid(w_rng, h_rng)
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    ones = tf.ones_like(x)
    coord_mat = tf.stack([x, y, ones])

    trans_mat = tf.linalg.inv(trans_mat)

    res_mat = tf.linalg.matmul(trans_mat, coord_mat)
    map_x = res_mat[0]
    map_y = res_mat[1]
    return map_x, map_y


class ShiftScaleRotate(BaseAug):
    def __init__(
            self,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation='bilinear',
            border_mode='constant',
            p=0.5):
        super(ShiftScaleRotate, self).__init__(p)
        self.shift_limit = gen_utils.check_range(
            shift_limit, 0.0, 1.0, "shift_limit")
        self.scale_limit = gen_utils.check_range(
            scale_limit, 0.0, 1.0, "scale_limit")
        self.rotate_limit = gen_utils.check_range(
            rotate_limit, 0.0, 180.0, "rotate_limit")
        self.interpolation = gen_utils.check_enum(
            interpolation, image_utils.SUPPORTED_INTERPOLATIONS,
            "interpolation")
        self.border_mode = gen_utils.check_enum(
            border_mode, image_utils.SUPPORTED_BORDER_MODE,
            "border_mode")

    def _make_params(self):
        rnd_shift = gen_utils.random_float(
            [2], -self.shift_limit, self.shift_limit)
        rnd_z = gen_utils.random_float(
            [], 1.0 - self.scale_limit, 1.0 + self.scale_limit)
        rnd_theta = gen_utils.random_float(
            [], -self.rotate_limit, self.rotate_limit)
        params = {
            "tx": rnd_shift[0],
            "ty": rnd_shift[1],
            "z": rnd_z,
            "theta": rnd_theta,
        }
        return params

    def _prepare_aug(self, image, params):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        image_trans_mat = image_utils.make_trans_mat(
            image_height, image_width,
            params["tx"], params["ty"],
            params["z"], params["theta"])
        self.image_map_x, self.image_map_y = make_image_inv_map(
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

    def _do_aug_bboxes(self, bboxes):
        # TODO: process the associated label if the bbox goes out
        # of the image.
        params = self.params
        bbox_trans_mat = image_utils.make_trans_mat(
            1.0, 1.0,
            params["tx"], params["ty"],
            params["z"], params["theta"])
        x_mins, y_mins, x_maxs, y_maxs = \
            image_utils.decompose_bboxes(bboxes)

        def _trans_one_bbox(i):
            x_min, y_min, x_max, y_max = \
                x_mins[i], y_mins[i], x_maxs[i], y_maxs[i]
            coord_mat = tf.convert_to_tensor([
                [x_min, x_min, x_max, x_max],
                [y_min, y_max, y_min, y_max],
                [1.0,   1.0,   1.0,   1.0]], dtype=tf.float32)
            res_mat = tf.linalg.matmul(bbox_trans_mat, coord_mat)

            aug_x_min = tf.math.reduce_min(res_mat[0])
            aug_x_min = tf.clip_by_value(aug_x_min, 0.0, 1.0)
            aug_y_min = tf.math.reduce_min(res_mat[1])
            aug_y_min = tf.clip_by_value(aug_y_min, 0.0, 1.0)
            aug_x_max = tf.math.reduce_max(res_mat[0])
            aug_x_max = tf.clip_by_value(aug_x_max, 0.0, 1.0)
            aug_y_max = tf.math.reduce_max(res_mat[1])
            aug_y_max = tf.clip_by_value(aug_y_max, 0.0, 1.0)
            aug_bbox = tf.convert_to_tensor(
                [aug_x_min, aug_y_min, aug_x_max, aug_y_max],
                dtype=tf.float32)
            return aug_bbox

        n_bboxes = tf.shape(bboxes)[0]
        bbox_range = tf.range(n_bboxes, dtype=tf.int32)
        aug_bboxes = tf.map_fn(
            _trans_one_bbox, bbox_range, dtype=tf.float32)
        return aug_bboxes
