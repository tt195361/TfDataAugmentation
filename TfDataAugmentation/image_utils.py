#
# image_utils.py
#

import tensorflow as tf
import math


def decompose_bboxes(bboxes):
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]
    return x_min, y_min, x_max, y_max


def compose_bboxes(x1, y1, x2, y2):
    x_min = tf.math.minimum(x1, x2)
    y_min = tf.math.minimum(y1, y2)
    x_max = tf.math.maximum(x1, x2)
    y_max = tf.math.maximum(y1, y2)
    bboxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
    return bboxes


def get_image_size(image, dtype):
    image_shape = tf.cast(tf.shape(image), dtype=dtype)
    image_height = image_shape[0]
    image_width = image_shape[1]
    return image_height, image_width


def make_trans_mat(height, width, tx, ty, z, theta):
    cx = width * 0.5
    cy = height * 0.5

    center_shift_mat = tf.convert_to_tensor([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    trans_mat = center_shift_mat

    rot_rad = 2.0 * math.pi * theta / 360.0
    rotation_mat = tf.convert_to_tensor([
        [tf.math.cos(rot_rad), tf.math.sin(rot_rad), 0.0],
        [-tf.math.sin(rot_rad), tf.math.cos(rot_rad), 0.0],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    trans_mat = tf.linalg.matmul(rotation_mat, trans_mat)

    zoom_mat = tf.convert_to_tensor([
        [z, 0.0, 0.0],
        [0.0, z, 0.0],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    trans_mat = tf.linalg.matmul(zoom_mat, trans_mat)

    shift_mat = tf.convert_to_tensor([
        [1.0, 0.0, tx * width],
        [0.0, 1.0, ty * height],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    trans_mat = tf.linalg.matmul(shift_mat, trans_mat)

    center_back_mat = tf.convert_to_tensor([
        [1.0, 0.0, cx],
        [0.0, 1.0, cy],
        [0.0, 0.0, 1.0]], dtype=tf.float32)
    trans_mat = tf.linalg.matmul(center_back_mat, trans_mat)

    return trans_mat


def mirror_boundary(v, max_v):
    # v % (max_v*2.0-2.0) ==> v % (512*2-2) ==> [0..1022]
    # [0..1022] - (max_v-1.0) ==> [0..1022] - 511 ==> [-511..511]
    # -1.0 * abs([-511..511]) ==> [-511..0]
    # [-511..0] + max_v - 1.0 ==> [-511..0] + 511 ==> [0..511]
    mirror_v = -1.0 * tf.math.abs(
        v % (max_v*2.0-2.0) - (max_v-1.0)) + max_v-1.0
    return mirror_v


def clip_boundary(v, max_v):
    clip_v = tf.clip_by_value(v, 0.0, max_v-1.0)
    return clip_v


def gather_nd(image, map_x, map_y):
    map_stack = tf.stack([map_y, map_x])  # [ 2, height, width ]
    map_indices = tf.transpose(
        map_stack, perm=[1, 2, 0])  # [ height, width, 2 ]
    map_indices = tf.cast(map_indices, dtype=tf.int32)
    gather_image = tf.gather_nd(image, map_indices)
    return gather_image


# https://stackoverflow.com/questions/54816173/how-to-accurately-round-half-up-with-tensorflow
def classical_round(x):
    return tf.math.floor(x+0.5)


def interpolate_nearest(image, map_x, map_y):
    rounded_map_x = classical_round(map_x)
    rounded_map_y = classical_round(map_y)
    interpolate_image = gather_nd(
        image, rounded_map_x, rounded_map_y)
    return interpolate_image


def interpolate_bilinear(image, map_x, map_y):
    ll = gather_nd(image, tf.math.floor(map_x), tf.math.floor(map_y))
    lr = gather_nd(image, tf.math.ceil(map_x), tf.math.floor(map_y))
    ul = gather_nd(image, tf.math.floor(map_x), tf.math.ceil(map_y))
    ur = gather_nd(image, tf.math.ceil(map_x), tf.math.ceil(map_y))

    fraction_x = tf.expand_dims(map_x % 1.0, axis=-1)  # [h, w, 1]
    int_l = (lr - ll) * fraction_x + ll
    int_u = (ur - ul) * fraction_x + ul

    fraction_y = tf.expand_dims(map_y % 1.0, axis=-1)  # [h, w, 1]
    interpolate_image = (int_u - int_l) * fraction_y + int_l
    return interpolate_image


SUPPORTED_INTERPOLATIONS = ('nearest', 'bilinear')
SUPPORTED_BORDER_MODE = ('mirror', 'constant')


def remap(
        image, map_x, map_y,
        interpolation='bilinear',
        border_mode='constant'):
    assert interpolation in SUPPORTED_INTERPOLATIONS
    assert border_mode in SUPPORTED_BORDER_MODE

    height, width = get_image_size(image, dtype=tf.int32)
    height_f, width_f = get_image_size(image, dtype=tf.float32)
    map_x = tf.reshape(map_x, shape=[height, width])
    map_y = tf.reshape(map_y, shape=[height, width])
    if border_mode == 'mirror':
        b_map_x = mirror_boundary(map_x, width_f)
        b_map_y = mirror_boundary(map_y, height_f)
    else:
        b_map_x = clip_boundary(map_x, width_f)
        b_map_y = clip_boundary(map_y, height_f)

    if interpolation == 'bilinear':
        image_remap = interpolate_bilinear(image, b_map_x, b_map_y)
    else:
        image_remap = interpolate_nearest(image, b_map_x, b_map_y)

    if border_mode == 'constant':
        map_stack = tf.stack([map_y, map_x])  # [ 2, height, width ]
        map_indices = tf.transpose(           # [ height, width ,2 ]
            map_stack, perm=[1, 2, 0])
        y_ge_0 = (-0.5 <= map_indices[:, :, 0])
        y_lt_h = (map_indices[:, :, 0] < height_f - 0.5)
        x_ge_0 = (-0.5 <= map_indices[:, :, 1])
        x_lt_w = (map_indices[:, :, 1] < width_f - 0.5)
        inside_boundary = tf.math.reduce_all(
            tf.stack([y_ge_0, y_lt_h, x_ge_0, x_lt_w]), axis=0)  # [h, w]
        inside_boundary = inside_boundary[:, :, tf.newaxis]  # [h, w, 1]
        image_remap = tf.where(inside_boundary, image_remap, 0.0)

    return image_remap
