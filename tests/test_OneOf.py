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

TRANSFORM_1 = 'transform_1'
TRANSFORM_2 = 'transform_2'
TRANSFORM_3 = 'transform_3'

logger = TestLogger()
transform_1 = MockTrans(TRANSFORM_1, logger, p=1.0)
transform_2 = MockTrans(TRANSFORM_2, logger, p=1.0)
transform_3 = MockTrans(TRANSFORM_3, logger, p=1.0)
transforms = [transform_1, transform_2, transform_3]


@pytest.mark.parametrize(
    "arg, expected, message", [
        (transforms, TestResult.OK, "iterable => OK"),
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
    "rnd, name, message", [
        (0, TRANSFORM_1, "0 => transform_1"),
        (1, TRANSFORM_2, "1 => transform_2"),
        (2, TRANSFORM_3, "2 => transform_3"),
    ])
def test_call(mocker, rnd, name, message):
    logger.reset()
    mocker.patch(
        'tensorflow.random.uniform',
        return_value=tf.constant(rnd, dtype=tf.int32))

    tgt_transform = Tfda.OneOf(transforms, p=1.0)
    image = test_utils.make_test_image()
    _ = tgt_transform(image=image)

    expected_log = "{0} called".format(name)
    actual_log = logger.get()
    assert expected_log == actual_log, message
