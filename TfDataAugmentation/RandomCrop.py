#
# RandomCrop.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils
from . import image_utils
from . import BboxProcessor


class RandomCrop(BaseAug):
    def __init__(self, height, width, p=1.0):
        super(RandomCrop, self).__init__(p)
        self.crop_height = gen_utils.check_int_range(
            height, 0, None, "height")
        self.crop_width = gen_utils.check_int_range(
            width, 0, None, "width")

    def _make_params(self, image):
        starts = gen_utils.random_float([2])
        params = {
            "h_start": starts[0],
            "w_start": starts[1],
        }
        return params

    @staticmethod
    def _calc_start(orig, crop, ratio):
        diff = tf.cast(orig - crop, dtype=tf.float32)
        start = tf.cast(diff * ratio, dtype=tf.int32)
        return start

    @staticmethod
    def _calc_ratio(size, point):
        size_f = tf.cast(size, dtype=tf.float32)
        point_f = tf.cast(point, dtype=tf.float32)
        ratio = point_f / size_f
        return ratio

    def _prepare_aug(self, image, params):
        self.image_height, self.image_width = \
            image_utils.get_image_size(image, dtype=tf.float32)
        self.y1 = self._calc_start(
            self.image_height, self.crop_height, params['h_start'])
        self.y2 = self.y1 + self.crop_height
        self.x1 = self._calc_start(
            self.image_width, self.crop_width, params['w_start'])
        self.x2 = self.x1 + self.crop_width

    def _do_aug_image(self, image):
        aug_image = self._crop_image(image)
        return aug_image

    def _do_aug_mask(self, mask):
        aug_mask = self._crop_image(mask)
        return aug_mask

    def _crop_image(self, image):
        aug_image = image[self.y1:self.y2, self.x1:self.x2]
        return aug_image

    def _do_aug_bboxes(self, bboxes):
        original_bboxes = BboxProcessor.denormalize_bboxes(
            bboxes, self.image_height, self.image_width)
        x_min, y_min, x_max, y_max = \
            image_utils.decompose_bboxes(original_bboxes)

        def _minus_clip(val, minus_val, max_val):
            val = val - tf.cast(minus_val, dtype=tf.float32)
            val = tf.clip_by_value(val, 0.0, max_val)
            return val

        x_min = _minus_clip(x_min, self.x1, self.crop_width)
        y_min = _minus_clip(y_min, self.y1, self.crop_height)
        x_max = _minus_clip(x_max, self.x1, self.crop_width)
        y_max = _minus_clip(y_max, self.y1, self.crop_height)
        aug_bboxes = \
            image_utils.compose_bboxes(x_min, y_min, x_max, y_max)
        aug_bboxes = BboxProcessor.normalize_bboxes(
            aug_bboxes, self.crop_height, self.crop_width)
        return aug_bboxes
