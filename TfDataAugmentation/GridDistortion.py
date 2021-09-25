#
# GridDistortion.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils


class GridDistortion(BaseAug):
    def __init__(
            self,
            num_steps=5,
            distort_limit=0.3,
            interpolation='bilinear',
            border_mode='mirror',
            p=0.5):
        super(GridDistortion, self).__init__(p)
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.interpolation = interpolation
        self.border_mode = border_mode

    def _make_params(self, image):
        stepsx = gen_utils.random_float(
            [self.num_steps + 1],
            minval=1.0 - self.distort_limit,
            maxval=1.0 + self.distort_limit)
        stepsy = gen_utils.random_float(
            [self.num_steps + 1],
            minval=1.0 - self.distort_limit,
            maxval=1.0 + self.distort_limit)
        params = {
            "stepsx": stepsx,
            "stepsy": stepsy,
        }
        return params

    def _prepare_aug(self, image, params):
        self.image_map_x, self.image_map_y = \
            self.make_grid_distorted_maps(
                image, self.num_steps,
                params['stepsx'], params['stepsy'])

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
        message = "GridDistortion does not support bbox."
        raise NotImplementedError(message)

    @staticmethod
    def make_grid_distorted_maps(
            image, num_steps, stepsx, stepsy):
        height, width = \
            image_utils.get_image_size(image, dtype=tf.int32)
        height_f, width_f = \
            image_utils.get_image_size(image, dtype=tf.float32)

        def _make_maps_before_last(step, steps):  # size=512, step=102,
            # steps.shape=[num_steps]
            step_rep = tf.repeat(step, num_steps)  # [102, 102, 102, 102, 102]
            step_rep_f = tf.cast(step_rep, dtype=tf.float32)
            step_inc = step_rep_f * steps  # [102*s_0, ..., 102*s_4]
            cur = tf.math.cumsum(step_inc)  # [si_0, si_0 + si_1, ... ]
            zero = tf.zeros([1], dtype=tf.float32)
            prev = tf.concat([zero, cur[:-1]], axis=0)  # [0, c_0, ..., c_3]
            prev_cur = tf.stack([prev, cur])  # [[p_0, p_1, ...], [c_0, c_1, ...]]
            ranges = tf.transpose(prev_cur)  # [[p_0, c_0], [p_1, c_1], ... ]

            def _linspace_range(rng):
                return tf.linspace(rng[0], rng[1], step)

            maps_stack = tf.map_fn(_linspace_range, ranges)
            maps = tf.reshape(maps_stack, [-1])  # [-1] flatten into 1-D
            return maps

        def _make_last_map(size, step, maps_before_last):
            last_start = maps_before_last[-1]
            last_step = size - step * num_steps  # 512 - 102*5 = 2
            size_f = tf.cast(size, dtype=tf.float32)
            last_map = tf.linspace(
                last_start + 1.0, size_f - 1.0, last_step)
            distorted_map = tf.concat(
                [maps_before_last, last_map], axis=0)
            return distorted_map

        def _make_distorted_map(size, steps):
            step = size // num_steps
            last_step = size % num_steps
            maps_before_last = _make_maps_before_last(step, steps[:-1])
            distorted_map = tf.cond(
                last_step == 0,
                lambda: maps_before_last,
                lambda: _make_last_map(size, step, maps_before_last))
            return distorted_map

        xx = _make_distorted_map(width, stepsx)
        xx = tf.clip_by_value(
            xx, clip_value_min=0.0, clip_value_max=width_f - 1.0)
        yy = _make_distorted_map(height, stepsy)
        yy = tf.clip_by_value(
            yy, clip_value_min=0.0, clip_value_max=height_f - 1.0)
        map_x, map_y = tf.meshgrid(xx, yy)
        return map_x, map_y
