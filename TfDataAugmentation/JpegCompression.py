#
# JpegCompression.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils


class JpegCompression(BaseAug):
    def __init__(
            self,
            quality_lower=99,
            quality_upper=100,
            p=0.5):
        super(JpegCompression, self).__init__(p)
        self.quality_lower = gen_utils.check_int_range(
            quality_lower, 0, 100, "quality_lower")
        self.quality_upper = gen_utils.check_int_range(
            quality_upper, 0, 100, "quality_upper")
        if quality_lower > quality_upper:
            message = \
                "'quality_lower' needs to be less than " \
                "or equal to 'quality_upper': " \
                "quality_lower={0}, quality_upper={1}" \
                .format(quality_lower, quality_upper)
            raise ValueError(message)

    def _make_params(self, image):
        quality = gen_utils.random_int(
            [], self.quality_lower, self.quality_upper + 1)
        params = {
            "quality": quality,
        }
        return params

    def _do_aug_image(self, image):
        jpeg_quality = self.params["quality"]
        aug_image = tf.image.adjust_jpeg_quality(image, jpeg_quality)
        return aug_image
