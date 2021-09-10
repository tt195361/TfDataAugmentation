#
# OpticalDistortion.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils


class OpticalDistortion(BaseAug):
    def __init__(
            self,
            distort_limit=0.05,
            shift_limit=0.05,
            interpolation='bilinear',
            border_mode='mirror',
            p=1.0):
        super(OpticalDistortion, self).__init__(p)
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.interpolation = interpolation
        self.border_mode = border_mode

    def _make_params(self):
        rnd_k = gen_utils.random_float(
            [], -self.distort_limit, self.distort_limit)
        rnd_d = gen_utils.random_float(
            [2], -self.shift_limit, self.shift_limit)
        params = {
            "k": rnd_k,
            "dx": rnd_d[0],
            "dy": rnd_d[1],
        }
        return params

    def _prepare_aug(self, image, params):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        self.image_map_x, self.image_map_y = \
            self.init_undistort_rectify_map(
                image_height, image_width,
                params['k'], params['dx'], params['dy'])

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
        message = "OpticalDistortion does not support bbox."
        raise NotImplementedError(message)

    @staticmethod
    def init_undistort_rectify_map(height, width, k, dx, dy):
        f_x = width
        f_y = height
        c_x = width * 0.5 + dx
        c_y = height * 0.5 + dy

        f_dash_x = f_x
        c_dash_x = width * 0.5
        f_dash_y = f_y
        c_dash_y = height * 0.5

        h_rng = tf.range(height, dtype=tf.float32)
        w_rng = tf.range(width, dtype=tf.float32)
        v, u = tf.meshgrid(w_rng, h_rng)

        x = (v - c_dash_x) / f_dash_x
        y = (u - c_dash_y) / f_dash_y
        x_dash = x
        y_dash = y

        r_2 = x_dash * x_dash + y_dash * y_dash
        r_4 = r_2 * r_2
        x_dash_dash = x_dash * (1 + k * r_2 + k * r_4)
        y_dash_dash = y_dash * (1 + k * r_2 + k * r_4)

        map_x = x_dash_dash * f_x + c_x
        map_y = y_dash_dash * f_y + c_y
        return map_x, map_y
