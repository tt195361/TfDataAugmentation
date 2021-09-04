#
# test_HorizontalFlip.py
#

import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils


def test_call():
    tgt_transform = test_utils.make_tgt_transform(
        Tfda.HorizontalFlip(p=1.0))

    image = test_utils.make_test_image()
    mask = test_utils.make_test_image()
    bboxes = test_utils.make_test_bboxes()
    labels = test_utils.make_labels(bboxes)

    tgt_result = tgt_transform(
        image=image, mask=mask, bboxes=bboxes)
    actual_image = tgt_result['image']
    actual_mask = tgt_result['mask']
    actual_bboxes = tgt_result['bboxes']

    ref_transform = \
        test_utils.make_ref_transform(A.HorizontalFlip(p=1.0))
    ref_result = ref_transform(
        image=image.numpy(), mask=mask.numpy(),
        bboxes=bboxes.numpy(), labels=labels.numpy())
    expected_image = ref_result['image']
    expected_mask = ref_result['mask']
    expected_bboxes = ref_result['bboxes']

    test_utils.assert_array(expected_image, actual_image, "image")
    test_utils.assert_array(expected_mask, actual_mask, "mask")
    test_utils.assert_array(expected_bboxes, actual_bboxes, "bboxes")
