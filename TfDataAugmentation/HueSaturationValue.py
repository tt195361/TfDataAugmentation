#
# HueSaturationValue.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils


class HueSaturationValue(BaseAug):
    def __init__(
            self,
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5):
        super(HueSaturationValue, self).__init__(p)
        self.hue_shift_limit = gen_utils.check_float_range(
            hue_shift_limit, 0, None, "hue_shift_limit")
        self.sat_shift_limit = gen_utils.check_float_range(
            sat_shift_limit, 0, None, "sat_shift_limit")
        self.val_shift_limit = gen_utils.check_float_range(
            val_shift_limit, 0, None, "val_shift_limit")

    def _make_params(self, image):
        hue_shift = gen_utils.random_float(
            [], -self.hue_shift_limit, self.hue_shift_limit)
        sat_shift = gen_utils.random_float(
            [], -self.sat_shift_limit, self.sat_shift_limit)
        val_shift = gen_utils.random_float(
            [], -self.val_shift_limit, self.val_shift_limit)
        params = {
            "hue_shift": hue_shift,
            "sat_shift": sat_shift,
            "val_shift": val_shift,
        }
        return params

    def _do_aug_image(self, image):
        hue_shift = self.params['hue_shift'] / 360.0
        sat_shift = self.params['sat_shift'] / 255.0
        val_shift = self.params['val_shift'] / 255.0

        hsv_image = tf.image.rgb_to_hsv(image)
        hue_values = (hsv_image[..., 0] + hue_shift) % 1.0
        sat_values = hsv_image[..., 1] + sat_shift
        sat_values = tf.clip_by_value(sat_values, 0.0, 1.0)
        val_values = hsv_image[..., 2] + val_shift
        val_values = tf.clip_by_value(val_values, 0.0, 1.0)
        hsv_image = tf.stack(
            [hue_values, sat_values, val_values], axis=-1)
        aug_image = tf.image.hsv_to_rgb(hsv_image)
        return aug_image
