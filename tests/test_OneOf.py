#
# test_OneOf.py
#

import pytest
import tensorflow as tf
from .context import TfDataAugmentation as Tfda
from . import test_utils
from .test_utils import TestResult
from .TestLogger import TestLogger
from .MockTrans import MockTrans

logger = TestLogger()
transform_1 = MockTrans('transform_1', logger, p=0.1)
transform_2 = MockTrans('transform_2', logger, p=0.1)
transform_3 = MockTrans('transform_3', logger, p=0.1)
transforms = [transform_1, transform_2, transform_3]


@pytest.mark.parametrize(
    "arg, expected, message", [
        (None,
         TestResult.Error, "None => Error"),
        (transform_1,
         TestResult.Error, "not iterable => Error"),
        ([transform_1],
         TestResult.Error, "less than 2 => Error"),
        ([transform_1, transform_2],
         TestResult.OK, "2 transforms => OK"),
        ([logger],
         TestResult.Error, "element is not BaseAug => Error"),
    ])
def test_transforms_type(arg, expected, message):
    try:
        _ = Tfda.OneOf(arg, p=1.0)
        actual = TestResult.OK
    # https://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line-except-block
    except (TypeError, ValueError):
        actual = TestResult.Error
    assert expected == actual, message


# https://webbibouroku.com/Blog/Article/pytest-mock
@pytest.mark.parametrize(
    "rnd, transform, message", [
        (0.333, transform_1, "0.333 => transform_1"),
        (0.666, transform_2, "0.666 => transform_2"),
        (0.667, transform_3, "0.667 => transform_3"),
    ])
def test_select(mocker, rnd, transform, message):
    logger.reset()
    mocker.patch(
        'tensorflow.random.uniform',
        return_value=tf.constant(rnd, dtype=tf.float32))

    tgt_transform = Tfda.OneOf(transforms, p=1.0)
    image = test_utils.make_test_image()
    _ = tgt_transform(image=image)

    expected_log = transform.get_call_message()
    actual_log = logger.get()
    assert expected_log == actual_log, message


@pytest.mark.parametrize(
    "rnd, transform, message", [
        (0.5, transform_2, "0.5 => transform_2"),
        (0.51, None, "0.51 => not applied"),
    ])
def test_p(mocker, rnd, transform, message):
    logger.reset()
    mocker.patch(
        'tensorflow.random.uniform',
        return_value=tf.constant(rnd, dtype=tf.float32))

    tgt_transform = Tfda.OneOf(transforms, p=0.5)
    image = test_utils.make_test_image()
    _ = tgt_transform(image=image)

    expected_log = ""
    if transform is not None:
        expected_log = transform.get_call_message()
    actual_log = logger.get()
    assert expected_log == actual_log, message
