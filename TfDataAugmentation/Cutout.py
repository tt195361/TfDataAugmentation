#
# Cutout.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils


class Cutout(BaseAug):
    def __init__(
            self,
            num_holes=8,
            max_h_size=8,
            max_w_size=8,
            p=0.5):
        super(Cutout, self).__init__(p)
        self.num_holes = gen_utils.check_int_range(
            num_holes, 1, None, "num_holes")
        self.max_h_size = gen_utils.check_int_range(
            max_h_size, 1, None, "max_h_size")
        self.max_w_size = gen_utils.check_int_range(
            max_w_size, 1, None, "max_w_size")

    def _make_params(self, image):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.int32)
        w_half = self.max_w_size // 2
        h_half = self.max_h_size // 2
        x_min = gen_utils.random_int(
            [self.num_holes], -w_half, image_width - w_half)
        y_min = gen_utils.random_int(
            [self.num_holes], -h_half, image_height - h_half)
        x_max = tf.math.minimum(
            x_min + self.max_w_size, image_width - 1)
        y_max = tf.math.minimum(
            y_min + self.max_h_size, image_height - 1)
        x_min = tf.math.maximum(x_min, 0)
        y_min = tf.math.maximum(y_min, 0)
        holes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        params = {
            "holes": holes,
        }
        return params

    def _do_aug_image(self, image):
        image_height, image_width = \
            image_utils.get_image_size(image, dtype=tf.int32)
        holes = self.params["holes"]

        def _make_one_mask(i):
            hole = holes[i]
            x_min = hole[0]
            y_min = hole[1]
            x_max = hole[2]
            y_max = hole[3]
            mask_height = y_max - y_min
            mask_width = x_max - x_min
            mask_shape = [mask_height, mask_width]
            mask = tf.ones(mask_shape, dtype=tf.bool)

            paddings = [
                [y_min, image_height - y_max],
                [x_min, image_width - x_max]]
            mask = tf.pad(mask, paddings, mode='CONSTANT')
            return mask

        num_cuts_rng = tf.range(self.num_holes, dtype=tf.int32)
        cut_masks = tf.map_fn(
            _make_one_mask, num_cuts_rng,
            fn_output_signature=tf.bool)
        cut_mask = tf.reduce_any(cut_masks, axis=0)
        cut_mask = cut_mask[..., tf.newaxis]

        mask_value = tf.constant(0.0, dtype=tf.float32)
        aug_image = tf.where(cut_mask, mask_value, image)
        return aug_image
