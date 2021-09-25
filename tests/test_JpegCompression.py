#
# test_JpegCompression.py
#

import pytest
import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils
from .test_utils import TestResult


@pytest.mark.parametrize(
    "quality_lower, quality_upper, expected, message", [
        # quality_lower
        (-1, 100, TestResult.Error,
         "quality_lower < min => Error"),
        (0, 100, TestResult.OK,
         "quality_lower == min => OK"),
        (100, 100, TestResult.OK,
         "quality_lower == max => OK"),
        (101, 100, TestResult.Error,
         "quality_lower >= max => Error"),

        # quality_upper
        (0, -1, TestResult.Error,
         "quality_upper < min => Error"),
        (0, 0, TestResult.OK,
         "quality_upper == min => OK"),
        (0, 100, TestResult.OK,
         "quality_upper == max => OK"),
        (0, 101, TestResult.Error,
         "quality_upper > max => Error"),

        # Relation
        (50, 50, TestResult.OK,
         "quality_lower == quality_upper => OK"),
        (51, 50, TestResult.Error,
         "quality_lower > quality_upper => Error"),
    ])
def test_hue_shift_limit_value(
        quality_lower, quality_upper, expected, message):
    try:
        Tfda.JpegCompression(
            quality_lower=quality_lower,
            quality_upper=quality_upper)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


def test_call():
    quality_lower = 50
    quality_upper = 100
    tgt_jpeg = Tfda.JpegCompression(
        quality_lower=quality_lower,
        quality_upper=quality_upper,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_jpeg)
    image = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image)
    actual_image = tgt_result['image']

    image_np = image.numpy()
    quality = float(tgt_jpeg.get_param('quality'))
    expected_image = A.image_compression(
        image_np, quality, image_type='.jpg')

    test_utils.partial_assert_array(
        expected_image, actual_image, 0.6, "image", eps=0.1)
