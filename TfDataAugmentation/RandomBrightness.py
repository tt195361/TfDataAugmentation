#
# RandomBrightness.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils


class RandomBrightness(BaseAug):
    def __init__(
            self,
            limit=0.2,
            p=0.5):
        super(RandomBrightness, self).__init__(p)
        self.limit = limit

    def _make_params(self):
        alpha = gen_utils.random_float(
            [], 1.0 - self.limit, 1.0 + self.limit)
        params = {
            "alpha": alpha,
        }
        return params

    def _do_aug_image(self, image):
        aug_image = image * self.params['alpha']
        aug_image = tf.clip_by_value(
            aug_image, clip_value_min=0.0, clip_value_max=1.0)
        return aug_image
