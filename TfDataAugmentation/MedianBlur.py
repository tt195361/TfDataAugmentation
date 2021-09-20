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
        # # filter_shape for tf.image.median_filter2d must be integer,
        # # not tensor, so needs some program...
        # def median_filter(ksz):
        #     filter_shape = (ksz, ksz)
        #     def _do_filter():
        #         filter_image = median_filter2d(
        #             image, filter_shape=filter_shape)
        #         return filter_image
        #     return _do_filter
        #
        image_shape = tf.shape(image)
        ksize = self.params['ksize']
        aug_image = tf.switch_case(
            ksize,
            branch_fns={
                # branch index must start from 0
                0: lambda: tfa.image.median_filter2d(image, filter_shape=[0, 0]),
                1: lambda: tfa.image.median_filter2d(image, filter_shape=[1, 1]),
                2: lambda: tfa.image.median_filter2d(image, filter_shape=[2, 2]),
                3: lambda: tfa.image.median_filter2d(image, filter_shape=[3, 3]),
                4: lambda: tfa.image.median_filter2d(image, filter_shape=[4, 4]),
                5: lambda: tfa.image.median_filter2d(image, filter_shape=[5, 5]),
                6: lambda: tfa.image.median_filter2d(image, filter_shape=[6, 6]),
                7: lambda: tfa.image.median_filter2d(image, filter_shape=[7, 7]),
                8: lambda: tfa.image.median_filter2d(image, filter_shape=[8, 8]),
                9: lambda: tfa.image.median_filter2d(image, filter_shape=[9, 9]),
                10: lambda: tfa.image.median_filter2d(image, filter_shape=[10, 10])
            }
        )

        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image


# Original one causes the error "ValueError: Paddings must be non-negative..."
# https://github.com/tensorflow/addons/blob/v0.14.0/tensorflow_addons/image/filters.py#L125-L200
def median_filter2d(
    image,
    filter_shape,
    padding="REFLECT",
    constant_values=0,
):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]

    # Explicitly pad the image
    image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

    area = filter_shape[0] * filter_shape[1]

    floor = (area + 1) // 2
    ceil = area // 2 + 1

    # tf.image.extract_patches needs at least 4D, so add 1D
    image = image[tf.newaxis, ...]
    patches = tf.image.extract_patches(
        image,
        sizes=[1, filter_shape[0], filter_shape[1], 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches = tf.reshape(patches, shape=[height, width, channels, area])

    # Note the returned median is casted back to the original type
    # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
    # It turns out to be int(6.5) = 6 if the original type is int
    top = tf.nn.top_k(patches, k=ceil).values
    if area % 2 == 1:
        median = top[:, :, :, floor - 1]
    else:
        median = (top[:, :, :, floor - 1] + top[:, :, :, ceil - 1]) / 2

    return median


def _pad(
    image,
    filter_shape,
    mode="CONSTANT",
    constant_values=0,
):
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)
