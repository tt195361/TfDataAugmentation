#
# MedianBlur.py
#

import tensorflow as tf
import tensorflow_addons as tfa # noqa: for PyCharm inspection
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
        # ksize for tf.image.median_filter2d must be integer,
        # not tensor, so needs some program...
        self.median_filter2d_method = eval(
            self.make_median_filter2d_method())

    def make_median_filter2d_method(self):
        s = "lambda ksize, image: "
        s += "   tf.switch_case("
        s += "       ksize, "
        s += "       branch_fns={"
        # branch index must start from 0
        for i in range(0, self.blur_limit + 1):
            s += "      {0}: lambda: tfa.image.median_filter2d(image, filter_shape=({0}, {0}))".format(i)
            if i < self.blur_limit:
                s += ", "
        s += "       }"
        s += "   )"
        return s

    def _make_params(self):
        ksize = gen_utils.random_int(
            [], self.MIN_KSIZE, self.blur_limit + 1)
        params = {
            "ksize": ksize,
        }
        return params

    def _do_aug_image(self, image):
        image_shape = tf.shape(image)
        ksize = self.params['ksize']
        aug_image = \
            self.median_filter2d_method(ksize, image)
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image
