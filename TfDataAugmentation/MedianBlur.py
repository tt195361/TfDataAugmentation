#
# MedianBlur.py
#

import tensorflow as tf
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
        ksize = self.params['ksize']
        filter_shape = (ksize, ksize)

        aug_image = median_filter2d(image, filter_shape)
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image


# original is "tensorflow_addons/image/filters.py"
@tf.function
def median_filter2d(image, filter_shape):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]

    # Explicitly pad the image
    image = _pad(image, filter_shape, mode="REFLECT", constant_values=0)

    fh, fw = filter_shape
    height_rng = tf.range(height, dtype=tf.int32)
    width_rng = tf.range(width, dtype=tf.int32)
    channels_rng = tf.range(channels, dtype=tf.int32)
    height_idx, width_idx, channels_idx = tf.meshgrid(
        height_rng, width_rng, channels_rng, indexing="ij")
    height_idx = tf.reshape(height_idx, [-1])
    width_idx = tf.reshape(width_idx, [-1])
    channels_idx = tf.reshape(channels_idx, [-1])
    map_idx = tf.stack(
        [height_idx, width_idx, channels_idx])
    map_idx = tf.transpose(map_idx)

    area = filter_shape[0] * filter_shape[1]
    floor = (area + 1) // 2
    ceil = area // 2 + 1

    def get_median(mi):
        h = mi[0]
        w = mi[1]
        c = mi[2]
        patch = image[h:h + fh, w:w + fw, c]
        patch = tf.reshape(patch, [-1])  # flatten
        top = tf.nn.top_k(patch, k=ceil).values
        m = tf.cond(
            area % 2 == 1,
            lambda: top[floor - 1],
            lambda: (top[floor - 1] + top[ceil - 1]) / 2)
        return m
    median = tf.vectorized_map(get_median, map_idx)
    return median


@tf.function
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
