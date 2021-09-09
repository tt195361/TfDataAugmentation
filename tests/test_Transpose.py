#
# test_Transpose.py
#

import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils
from . import TestCall


def test_call():
    tgt_transform = test_utils.make_tgt_transform(
        Tfda.Transpose(p=1.0))
    ref_transform = \
        test_utils.make_ref_transform(A.Transpose(p=1.0))

    TestCall.call(tgt_transform, ref_transform)
