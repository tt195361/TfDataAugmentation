#
# test_Compose.py
#

from .context import TfDataAugmentation as Tfda
from . import test_utils
from .TestLogger import TestLogger
from .MockTrans import MockTrans


def test_call():
    logger = TestLogger()
    transform_1 = MockTrans('transform_1', logger, p=1.0)
    transform_2 = MockTrans('transform_2', logger, p=1.0)
    transform_3 = MockTrans('transform_3', logger, p=1.0)
    transforms = [transform_1, transform_2, transform_3]

    tgt_transform = Tfda.Compose(transforms, p=1.0)
    image = test_utils.make_test_image()
    _ = tgt_transform(image=image)

    expected_messages = \
        [ t.get_call_message() for t in transforms ]
    expected_log = logger.make_log(expected_messages)
    actual_log = logger.get()
    assert \
        expected_log == actual_log, \
        "each transforms are called one by one"
