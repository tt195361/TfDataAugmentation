#
# test_utils.py
#

import tensorflow as tf
import numpy as np
import albumentations as A
from enum import Enum
from .context import TfDataAugmentation as Tfda

IMAGE_HEIGHT = 20
IMAGE_WIDTH = 16


def random_int(shape=None, minval=0, maxval=1):
    shape = [] if shape is None else shape
    rand = tf.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=tf.int32)
    return rand


def random_float(shape=None, minval=0.0, maxval=1.0):
    shape = [] if shape is None else shape
    rand = tf.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rand


def make_test_image(shape=None):
    shape = [IMAGE_HEIGHT, IMAGE_WIDTH, 3] if shape is None else shape
    test_image = random_float(shape, minval=0.0, maxval=1.0)
    # h_val = tf.range(IMAGE_HEIGHT, dtype=tf.float32)
    # w_val = tf.range(IMAGE_WIDTH, dtype=tf.float32) / 100.0
    # h_val = h_val[:, tf.newaxis] # [IMAGE_HEIGHT, 1]
    # w_val = w_val[tf.newaxis, :] # [1, IMAGE_WIDTH]
    # hw_val = h_val + w_val       # [IMAGE_HEIGHT, IMAGE_WIDTH]
    # test_image = tf.repeat(      # [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    #     hw_val[..., tf.newaxis], 3, axis=-1)
    return test_image


def make_test_bbox(_i):
    # https://stackoverflow.com/questions/32602902/explicitly-declaring-variable-as-unused-in-python-pycharm
    x_min = random_int([], minval=0, maxval=IMAGE_WIDTH-1)
    y_min = random_int([], minval=0, maxval=IMAGE_HEIGHT-1)
    x_max = random_int([], minval=x_min+1, maxval=IMAGE_WIDTH)
    y_max = random_int([], minval=y_min+1, maxval=IMAGE_HEIGHT)
    test_bbox = tf.convert_to_tensor(
        [x_min, y_min, x_max, y_max], dtype=tf.float32)
    return test_bbox


def make_test_bboxes(num_bboxes=4):
    num_bboxes_rng = tf.range(num_bboxes, dtype=tf.int32)
    test_bboxes = tf.map_fn(
        make_test_bbox, num_bboxes_rng, dtype=tf.float32)
    return test_bboxes


def to_alb_bboxes(bboxes, height, width):
    alb_bboxes = np.empty_like(bboxes)
    alb_bboxes[:, 0] = bboxes[:, 0] / width
    alb_bboxes[:, 1] = bboxes[:, 1] / height
    alb_bboxes[:, 2] = bboxes[:, 2] / width
    alb_bboxes[:, 3] = bboxes[:, 3] / height
    return alb_bboxes


def to_tfda_bboxes(bboxes, height, width):
    tfda_bboxes = np.empty_like(bboxes)
    clipped_bboxes = np.clip(bboxes, 0.0, 1.0)
    tfda_bboxes[:, 0] = clipped_bboxes[:, 0] * width
    tfda_bboxes[:, 1] = clipped_bboxes[:, 1] * height
    tfda_bboxes[:, 2] = clipped_bboxes[:, 2] * width
    tfda_bboxes[:, 3] = clipped_bboxes[:, 3] * height
    return tfda_bboxes


def make_labels(bboxes):
    num_bboxes = tf.shape(bboxes)[0]
    labels = random_int([num_bboxes], minval=1, maxval=5)
    return labels


def make_tgt_transform(transform):
    bbox_params = Tfda.BboxParams('pascal_voc')
    tgt_transform = Tfda.Compose(
        [transform], bbox_params=bbox_params, p=1.0)
    return tgt_transform


def make_ref_transform(transform):
    bbox_params = make_alb_bbox_params()
    ref_transform = A.Compose(
        [transform],
        p=1.0, bbox_params=bbox_params)
    return ref_transform


def make_alb_bbox_params():
    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['labels'])
    return bbox_params


def calc_abs_diff(expected_np, actual_tf, eps=1e-5):
    actual_np = actual_tf.numpy()
    abs_diff = np.abs(expected_np - actual_np)
    abs_diff_le_eps = (abs_diff <= eps)
    return abs_diff_le_eps


def assert_array(expected_np, actual_tf, message, eps=1e-5):
    abs_diff_le_eps = calc_abs_diff(expected_np, actual_tf, eps)
    assert np.all(abs_diff_le_eps), message


def partial_assert_array(
        expected_np, actual_tf, accept_ratio, message, eps=1e-5):
    abs_diff_le_eps = calc_abs_diff(expected_np, actual_tf, eps)
    true_elem_count = np.sum(abs_diff_le_eps)
    total_elem_count = np.size(expected_np)
    true_elem_ratio = true_elem_count / total_elem_count
    assert true_elem_ratio >= accept_ratio, message


class TestResult(Enum):
    OK = 0
    Error = 1
