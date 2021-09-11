#
# RandomBrightnessContrast.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils


class RandomBrightnessContrast(BaseAug):
    def __init__(
            self,
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5):
        super(RandomBrightnessContrast, self).__init__(p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def _make_params(self):
        alpha = gen_utils.random_float(
            [], 1.0 - self.contrast_limit, 1.0 + self.contrast_limit)
        beta = gen_utils.random_float(
            [], -self.brightness_limit, self.brightness_limit)
        params = {
            "alpha": alpha,
            "beta": beta,
        }
        return params

    def _do_aug_image(self, image):
        aug_image = image * self.params['alpha']
        mean_image = tf.reduce_mean(aug_image)
        aug_image += self.params['beta'] * mean_image
        aug_image = tf.clip_by_value(
            aug_image, clip_value_min=0.0, clip_value_max=1.0)
        return aug_image
