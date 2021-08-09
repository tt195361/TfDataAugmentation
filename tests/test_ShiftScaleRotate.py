#
# test_ShiftScaleRotate.py
#

import pytest
import albumentations as A
import cv2
from .context import TfDataAugmentation as tfda
from . import test_utils


@pytest.mark.parametrize(
    "shift_limit, scale_limit, rotate_limit, condition", [
        (0.0, 0.0, 0.0, "all limits == 0"),
        (0.3, 0.0, 0.0, "shift_limit: 0.3"),
        (0.0, 0.3, 0.0, "scale_limit: 0.3"),
        (0.0, 0.0, 45.0, "rotate_limit: 45.0"),
        (0.3, 0.3, 45.0, "all_limits > 0"),
    ])
def test_call(shift_limit, scale_limit, rotate_limit, condition):
    tgt_transform = tfda.ShiftScaleRotate(
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        interpolation='nearest',
        border_mode='constant',
        p=1.0)

    image = test_utils.make_test_image()
    mask = test_utils.make_test_image()
    bboxes = test_utils.make_test_bboxes()
    # labels = test_utils.make_labels(bboxes)

    tgt_result = tgt_transform(
        image=image, mask=mask, bboxes=bboxes)
    actual_image = tgt_result['image']
    actual_mask = tgt_result['mask']
    # TODO:
    # actual_bboxes = tgt_result['bboxes']

    image_np = image.numpy()
    mask_np = mask.numpy()
    theta = tgt_transform.get_param('theta')
    z = tgt_transform.get_param('z')
    tx = tgt_transform.get_param('tx')
    ty = tgt_transform.get_param('ty')
    expected_image = A.shift_scale_rotate(
        image_np, theta, z, tx, ty,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT)
    expected_mask = A.shift_scale_rotate(
        mask_np, theta, z, tx, ty,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT)

    test_utils.almost_assert_array(
        expected_image, actual_image, 0.85, condition + ": image")
    test_utils.almost_assert_array(
        expected_mask, actual_mask, 0.85, condition + ": mask")
    # test_utils.assert_array(expected_bboxes, actual_bboxes, "bboxes")
