#
# test_BaseAug.py
#

import pytest
from .context import TfDataAugmentation as tfda


@pytest.mark.parametrize(
    "p, expected, message", [
        (-0.1, True, "< min => ValueError"),
        (0.0, False, "= min => OK, no error"),
        (1.0, False, "= max => OK, no error"),
        (1.1, True, "> max => ValueError"),
    ])
def test_init_param_p(p, expected, message):
    actual = False
    try:
        tfda.BaseAug(p=p)
    except ValueError:
        actual = True
    assert expected == actual, message
