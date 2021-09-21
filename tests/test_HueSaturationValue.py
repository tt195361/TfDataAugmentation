#
# test_HueSaturationValue.py
#

import pytest
import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils
from .test_utils import TestResult


@pytest.mark.parametrize(
    "hue_shift_limit, expected, message", [
        (-1, TestResult.Error, "< min => Error"),
        (0, TestResult.OK, "== min => OK"),
        (10000, TestResult.OK, "no max limit => OK"),
    ])
def test_hue_shift_limit_value(
        hue_shift_limit, expected, message):
    try:
        Tfda.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=0,
            val_shift_limit=0)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


@pytest.mark.parametrize(
    "sat_shift_limit, expected, message", [
        (-1, TestResult.Error, "< min => Error"),
        (0, TestResult.OK, "== min => OK"),
        (10000, TestResult.OK, "no max limit => OK"),
    ])
def test_sat_shift_limit_value(
        sat_shift_limit, expected, message):
    try:
        Tfda.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=0)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


@pytest.mark.parametrize(
    "val_shift_limit, expected, message", [
        (-1, TestResult.Error, "< min => Error"),
        (0, TestResult.OK, "== min => OK"),
        (10000, TestResult.OK, "no max limit => OK"),
    ])
def test_val_shift_limit_value(
        val_shift_limit, expected, message):
    try:
        Tfda.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=0,
            val_shift_limit=val_shift_limit)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


@pytest.mark.parametrize(
    "hue_shift_limit, sat_shift_limit, val_shift_limit", [
        (90, 0,  0),
        (0, 20,  0),
        (0,  0, 30),
    ])
def test_call(
        hue_shift_limit, sat_shift_limit, val_shift_limit):
    tgt_hsv = Tfda.HueSaturationValue(
        hue_shift_limit=hue_shift_limit,
        sat_shift_limit=sat_shift_limit,
        val_shift_limit=val_shift_limit,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_hsv)
    image = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image)
    actual_image = tgt_result['image']

    image_np = image.numpy()
    hue_shift = float(tgt_hsv.get_param('hue_shift'))
    sat_shift = float(tgt_hsv.get_param('sat_shift')) / 255.0
    val_shift = float(tgt_hsv.get_param('val_shift')) / 255.0
    expected_image = A.shift_hsv(
        image_np, hue_shift, sat_shift, val_shift)

    test_utils.assert_array(
        expected_image, actual_image, "image")
