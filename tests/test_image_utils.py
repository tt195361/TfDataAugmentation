#
# test_image_utils.py
#

import pytest
import tensorflow as tf
import cv2
import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils

HEIGHT = 4
WIDTH = 6


@pytest.mark.parametrize(
    "tx, ty, z, theta, condition", [
        (1.0 / WIDTH, 0.0, 1.0, 0.0, "tx=1"),
        (0.0, 1.0 / HEIGHT, 1.0, 0.0, "ty=1"),
        (0.0, 0.0, 0.5, 0.0, "z=0.5"),
        (0.0, 0.0, 2.0, 0.0, "z=2.0"),
        (0.0, 0.0, 0.0, 30.0, "theta=30.0"),
        (0.0, 0.0, 0.0, -45.0, "theta=-45.0"),
    ])
def test_make_trans_mat(tx, ty, z, theta, condition):
    actual = Tfda.make_trans_mat(HEIGHT, WIDTH, tx, ty, z, theta)
    actual = actual[:2, :]

    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/functional.py
    center = (WIDTH / 2, HEIGHT / 2)
    expected = cv2.getRotationMatrix2D(center, theta, z)
    expected[0, 2] += tx * WIDTH
    expected[1, 2] += ty * HEIGHT

    test_utils.assert_array(expected, actual, condition)


@pytest.mark.parametrize(
    "tx, ty, z, theta, condition", [
        (1.0 / WIDTH, 0.0, 1.0, 0.0, "tx=1"),
        (0.0, 1.0 / HEIGHT, 1.0, 0.0, "ty=1"),
        (0.0, 0.0, 0.5, 0.0, "z=0.5"),
        (0.0, 0.0, 2.0, 0.0, "z=2.0"),
        (0.0, 0.0, 1.0, 30.0, "theta=30.0"),
    ])
def test_remap(tx, ty, z, theta, condition):
    # image = test_utils.make_test_image([HEIGHT, WIDTH, 1])
    image = tf.range(1.0, WIDTH * HEIGHT + 1, dtype=tf.float32)
    image = tf.reshape(image, [HEIGHT, WIDTH, 1])

    trans_mat = Tfda.make_trans_mat(
        HEIGHT, WIDTH, -tx, -ty, 1.0/z, -theta)
    map_x, map_y = Tfda.make_image_map(HEIGHT, WIDTH, trans_mat)
    actual = Tfda.remap(
        image, map_x, map_y, interpolation='nearest',
        border_mode='constant')

    image_np = image.numpy()
    expected = A.shift_scale_rotate(
        image_np, theta, z, tx, ty,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT)

    test_utils.partial_assert_array(
        expected, actual, 0.75, condition)
