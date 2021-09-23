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


# original is "tensorflow_addons/image/filters.py".
# original one accepts only 'int' for filter_shape,
# it doesn't accept tensor...  I'd like to use
# tensor for filter_shape.
@tf.function
def median_filter2d(image, filter_shape):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]

    # Explicitly pad the image
    image = _pad(image, filter_shape, mode="REFLECT", constant_values=0)

    height_rng = tf.range(height, dtype=tf.int32)
    width_rng = tf.range(width, dtype=tf.int32)
    channels_rng = tf.range(channels, dtype=tf.int32)
    height_idx, width_idx, channels_idx = tf.meshgrid(
        height_rng, width_rng, channels_rng, indexing="ij")
    height_idx = tf.reshape(height_idx, [-1])      # [h*w*c]
    width_idx = tf.reshape(width_idx, [-1])        # [h*w*c]
    channels_idx = tf.reshape(channels_idx, [-1])  # [h*w*c]
    map_idx = tf.stack(                            # [3, h*w*c]
        [height_idx, width_idx, channels_idx])
    map_idx = tf.transpose(map_idx)                # [h*w*c, 3]
    map_idx = map_idx[:, tf.newaxis, :]            # [h*w*c, 1, 3]

    area = filter_shape[0] * filter_shape[1]
    fh, fw = filter_shape
    fh_rng = tf.range(fh, dtype=tf.int32)
    fw_rng = tf.range(fw, dtype=tf.int32)
    fh_idx, fw_idx = tf.meshgrid(fh_rng, fw_rng, indexing="ij")
    fh_idx = tf.reshape(fh_idx, [-1])              # [fh*fw]
    fw_idx = tf.reshape(fw_idx, [-1])              # [fh*fw]
    fc_idx = tf.zeros_like(fh_idx, dtype=tf.int32)
    flt_idx = tf.stack([fh_idx, fw_idx, fc_idx])   # [3, fh*fw]
    flt_idx = tf.transpose(flt_idx)                # [fh*fw, 3]
    flt_idx = flt_idx[tf.newaxis, :, :]            # [1, fh*fw, 3]

    gat_idx = map_idx + flt_idx                    # [h*w*c, fh*fw, 3]
    gat_idx = tf.reshape(                          # [h, w, c, fh*fw, 3]
        gat_idx, [height, width, channels, area, 3])
    patches = tf.gather_nd(image, gat_idx)         # [h, w, c, fh*fw]

    floor = (area + 1) // 2
    ceil = area // 2 + 1
    top = tf.nn.top_k(patches, k=ceil).values      # [h, w, c, ceil]
    median = tf.cond(
        area % 2 == 1,
        lambda: top[:, :, :, floor - 1],
        lambda: (top[:, :, :, floor - 1] + top[:, :, :, ceil - 1]) / 2)
    return median                                  # [h, w, c]


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
