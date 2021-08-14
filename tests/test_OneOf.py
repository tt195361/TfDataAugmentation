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
transform_1 = MockTrans('transform_1', logger, p=1.0)
transform_2 = MockTrans('transform_2', logger, p=1.0)
transform_3 = MockTrans('transform_3', logger, p=1.0)
transforms = [transform_1, transform_2, transform_3]


@pytest.mark.parametrize(
    "arg, expected, message", [
        (transforms, TestResult.OK, "iterable => OK"),
        (None, TestResult.Error, "None => Error"),
        (transform_1, TestResult.Error, "not iterable => Error"),
        ([logger], TestResult.Error, "element is not BaseAug => Error"),
    ])
def test_transforms_type(arg, expected, message):
    actual = TestResult.OK
    try:
        _ = Tfda.OneOf(arg, p=1.0)
    except TypeError:
        actual = TestResult.Error
    assert expected == actual, message


# https://webbibouroku.com/Blog/Article/pytest-mock
@pytest.mark.parametrize(
    "rnd, transform, message", [
        (0, transform_1, "0 => transform_1"),
        (1, transform_2, "1 => transform_2"),
        (2, transform_3, "2 => transform_3"),
    ])
def test_call(mocker, rnd, transform, message):
    logger.reset()
    mocker.patch(
        'tensorflow.random.uniform',
        return_value=tf.constant(rnd, dtype=tf.int32))

    tgt_transform = Tfda.OneOf(transforms, p=1.0)
    image = test_utils.make_test_image()
    _ = tgt_transform(image=image)

    expected_log = transform.get_call_message()
    actual_log = logger.get()
    assert expected_log == actual_log, message
