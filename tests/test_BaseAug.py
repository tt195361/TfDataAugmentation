#
# test_BaseAug.py
#

import pytest
from .context import TfDataAugmentation as Tfda
from .test_utils import TestResult


@pytest.mark.parametrize(
    "p, expected, message", [
        (-0.1, TestResult.Error, "< min => Error"),
        (0.0, TestResult.OK, "== min => OK"),
        (1.0, TestResult.OK, "== max => OK"),
        (1.1, TestResult.Error, "> max => Error"),
    ])
def test_init_param_p(p, expected, message):
    actual = TestResult.OK
    try:
        Tfda.BaseAug(p=p)
    except ValueError:
        actual = TestResult.Error
    assert expected == actual, message
