#
# Blur.py
#

import tensorflow as tf
import tensorflow_addons as tfa
from . import BaseAug
from . import gen_utils


class Blur(BaseAug):
    def __init__(
            self,
            blur_limit=7,
            p=0.5):
        super(Blur, self).__init__(p)
        self.blur_limit = blur_limit

    def _make_params(self):
        ksize = gen_utils.random_int(
            [], 3, self.blur_limit + 1)
        params = {
            "ksize": ksize,
        }
        return params

    def _do_aug_image(self, image):
        image_shape = tf.shape(image)
        ksize = self.params['ksize']
        filter_shape = (ksize, ksize)
        aug_image = tfa.image.gaussian_filter2d(
            image, filter_shape=filter_shape)
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image
