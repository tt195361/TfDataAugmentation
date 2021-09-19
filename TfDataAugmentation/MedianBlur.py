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
        aug_image = median_filter2d(
            image, filter_shape=filter_shape)
        aug_image = tf.reshape(aug_image, image_shape)
        return aug_image


# Original one doesn't work when filter_shape is tensor...
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

    patches = extract_patches(
        image,
        filter_shape
    )

    patches = tf.reshape(patches, shape=[height, width, channels, area])

    # Note the returned median is casted back to the original type
    # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
    # It turns out to be int(6.5) = 6 if the original type is int
    top = tf.nn.top_k(patches, k=ceil).values
    if area % 2 == 1:
        median = top[:, :, :, floor - 1]
    else:
        median = (top[:, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

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


def extract_patches(
        image, filter_shape):
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    image_channel = image_shape[2]

    filter_height = filter_shape[0]
    filter_width = filter_shape[1]

    image_height_rng = tf.range(image_height - filter_height + 1)
    image_width_rng = tf.range(image_width - filter_width + 1)
    image_channel_rng = tf.range(image_channel)
    filter_height_rng = tf.range(filter_height)
    filter_width_rng = tf.range(filter_width)

    def extract_height(h):
        def extract_width(w):
            def extract_channel(c):
                def extract_filter_height(fh):
                    def extract_filter_width(fw):
                        return image[h + fh, w + fw, c]
                    pfw = tf.map_fn(
                        extract_filter_width, filter_width_rng,
                        fn_output_signature=tf.float32)
                    return pfw
                pfh = tf.map_fn(
                    extract_filter_height, filter_height_rng,
                    fn_output_signature=tf.float32)
                return pfh
            pc = tf.map_fn(
                extract_channel, image_channel_rng,
                fn_output_signature=tf.float32)
            return pc
        pw = tf.map_fn(
            extract_width, image_width_rng,
            fn_output_signature=tf.float32)
        return pw
    patches = tf.map_fn(
        extract_height, image_height_rng,
        fn_output_signature=tf.float32)
    return patches
