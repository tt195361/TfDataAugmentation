#
# MedianBlur.py
#

import tensorflow as tf
import tensorflow_addons as tfa
from . import BaseAug
from . import gen_utils


class MedianBlur(BaseAug):
    MIN_KSIZE = 3

    def __init__(
            self,
            blur_limit=7,
            p=0.5):
        super(MedianBlur, self).__init__(p)
        self.blur_limit = gen_utils.check_int_range(
            blur_limit, self.MIN_KSIZE, None, "blur_limit")

    def _make_params(self):
        ksize = gen_utils.random_int(
            [], self.MIN_KSIZE, self.blur_limit + 1)
        params = {
            "ksize": ksize,
        }
        return params

    def _do_aug_image(self, image):
        image_shape = tf.shape(image)
        # ksize for median_filter2d must be integer.
        ksize = int(self.params['ksize'])
        filter_shape = (ksize, ksize)
        aug_image = tfa.image.median_filter2d(
            image, filter_shape=filter_shape)
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image
