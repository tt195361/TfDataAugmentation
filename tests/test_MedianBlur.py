#
# test_MedianBlur.py
#

import pytest
import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils
from .test_utils import TestResult


@pytest.mark.parametrize(
    "blur_limit, expected, message", [
        (3, TestResult.OK, "integer => OK"),
        (3.0, TestResult.Error, "not integer => Error"),
    ])
def test_blur_limit_type(blur_limit, expected, message):
    try:
        Tfda.MedianBlur(blur_limit=blur_limit)
        actual = TestResult.OK
    except TypeError:
        actual = TestResult.Error
    assert expected == actual, message


@pytest.mark.parametrize(
    "blur_limit, expected, message", [
        (2, TestResult.Error, "< min => Error"),
        (3, TestResult.OK, "== min => OK"),
        (10, TestResult.OK, "== max => OK"),
        (11, TestResult.Error, "> max => Error"),
    ])
def test_blur_limit_value(blur_limit, expected, message):
    try:
        Tfda.MedianBlur(blur_limit=blur_limit)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


def test_call():
    tgt_median_blur = Tfda.MedianBlur(
        blur_limit=5,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_median_blur)
    image = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image)
    actual_image = tgt_result['image']

    image_np = image.numpy()
    ksize = tgt_median_blur.get_param('ksize')

    # ksize must be 3 or 5 for A.median_blur
    if ksize == 3 or ksize == 5:
        expected_image = A.median_blur(image_np, ksize)

        # Not exactly the same, but calculates much the same values
        # 80% points of abs diffs should be less than 0.1
        test_utils.partial_assert_array(
            expected_image, actual_image, 0.8, "image", eps=0.1)
