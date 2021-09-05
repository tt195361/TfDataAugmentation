#
# test_BaseAug.py
#

import tensorflow as tf
import pytest
from .context import TfDataAugmentation as Tfda
from . import test_utils
from .test_utils import TestResult
from .TestLogger import TestLogger
from .MockTrans import MockTrans


@pytest.mark.parametrize(
    "p, expected, message", [
        (-0.1, TestResult.Error, "< min => Error"),
        (0.0, TestResult.OK, "== min => OK"),
        (1.0, TestResult.OK, "== max => OK"),
        (1.1, TestResult.Error, "> max => Error"),
    ])
def test_init_param_p(p, expected, message):
    try:
        Tfda.BaseAug(p=p)
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message


@pytest.mark.parametrize(
    "rnd, called, message", [
        (0.5, True, "rnd <= p => called"),
        (0.51, False, "rnd > p => NOT called"),
    ])
def test_call_p(mocker, rnd, called, message):
    mocker.patch(
        'tensorflow.random.uniform',
        return_value=tf.constant(rnd, dtype=tf.float32))

    logger = TestLogger()
    tgt_transform = MockTrans('transform', logger, p=0.5)

    image = test_utils.make_test_image()
    _ = tgt_transform(image=image) # noqa: for PyCharm inspection

    expected_log = ""
    if called:
        expected_log = tgt_transform.get_call_message()
    actual_log = logger.get()
    assert expected_log == actual_log, message


@pytest.mark.parametrize(
    "data, expected, message", [
        ({"image": test_utils.make_test_image()},
         TestResult.OK, "contains image => OK"),
        ({},
         TestResult.Error, "no image => Error"),
        ({"image": test_utils.make_test_image(),
          "unknown": 1},
         TestResult.OK, "contains unknown => OK"),
    ])
def test_call_data(data, expected, message):
    logger = TestLogger()
    tgt_transform = MockTrans('transform', logger, p=1.0)
    try:
        _ = tgt_transform(**data)  # noqa: for PyCharm inspection
        actual = TestResult.OK
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message
