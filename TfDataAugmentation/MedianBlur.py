#
# MedianBlur.py
#

import tensorflow as tf
import tensorflow_addons as tfa
from . import BaseAug
from . import gen_utils


class MedianBlur(BaseAug):
    MIN_KSIZE = 3
    MAX_KSIZE = 20

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

        if ksize == 3:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[3, 3])
        elif ksize == 4:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[4, 4])
        elif ksize == 5:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[5, 5])
        elif ksize == 6:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[6, 6])
        elif ksize == 7:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[7, 7])
        elif ksize == 8:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[8, 8])
        elif ksize == 9:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[9, 9])
        elif ksize == 10:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[10, 10])
        elif ksize == 11:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[11, 11])
        elif ksize == 12:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[12, 12])
        elif ksize == 13:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[13, 13])
        elif ksize == 14:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[14, 14])
        elif ksize == 15:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[15, 15])
        elif ksize == 16:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[16, 16])
        elif ksize == 17:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[17, 17])
        elif ksize == 18:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[18, 18])
        elif ksize == 19:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[19, 19])
        else:
            aug_image = tfa.image.median_filter2d(image, filter_shape=[20, 20])

        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image
