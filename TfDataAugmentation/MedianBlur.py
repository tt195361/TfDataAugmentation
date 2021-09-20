#
# MedianBlur.py
#

import tensorflow as tf
import tensorflow_addons as tfa
from . import BaseAug
from . import gen_utils


class MedianBlur(BaseAug):
    MIN_KSIZE = 3
    MAX_KSIZE = 10

    def __init__(
            self,
            blur_limit=7,
            p=0.5):
        super(MedianBlur, self).__init__(p)
        self.blur_limit = gen_utils.check_int_range(
            blur_limit, self.MIN_KSIZE, self.MAX_KSIZE, "blur_limit")

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

        # filter_shape for tf.image.median_filter2d must be integer,
        # not tensor, so needs some program...
        def median_filter2d(filter_shape):
            def _do_filter():
                filter_image = tfa.image.median_filter2d(
                    image, filter_shape=filter_shape)
                return filter_image
            return _do_filter

        aug_image = tf.switch_case(
            ksize,
            branch_fns={
                # branch index must start from 0
                0: median_filter2d(0),
                1: median_filter2d(1),
                2: median_filter2d(2),
                3: median_filter2d(3),
                4: median_filter2d(4),
                5: median_filter2d(5),
                6: median_filter2d(6),
                7: median_filter2d(7),
                8: median_filter2d(8),
                9: median_filter2d(9),
                10: median_filter2d(10)
            }
        )
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image
