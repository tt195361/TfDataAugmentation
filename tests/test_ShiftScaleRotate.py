#
# test_ShiftScaleRotate.py
#

import pytest
import numpy as np
import albumentations as A
import cv2
from .context import TfDataAugmentation as Tfda
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
    tgt_transform = Tfda.ShiftScaleRotate(
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        interpolation='nearest',
        border_mode='constant',
        p=1.0)

    height = 16
    width = 20
    shape = [height, width, 1]
    image = test_utils.make_test_image(shape)
    mask = test_utils.make_test_image(shape)
    bboxes = test_utils.make_test_bboxes()

    tgt_result = tgt_transform(
        image=image, mask=mask, bboxes=bboxes)
    actual_image = tgt_result['image']
    actual_mask = tgt_result['mask']
    actual_bboxes = tgt_result['bboxes']

    image_np = image.numpy()
    mask_np = mask.numpy()
    bboxes_np = bboxes.numpy()
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

    bboxes_alb = test_utils.to_alb_bboxes(bboxes_np, height, width)
    expected_alb_bbox_list = []
    for bbox_alb in bboxes_alb:
        expected_alb_bbox = A.bbox_shift_scale_rotate(
            bbox_alb, theta, z, tx, ty, height, width)
        expected_alb_bbox_list.append(expected_alb_bbox)
    expected_alb_bboxes = np.array(expected_alb_bbox_list)
    expected_bboxes = test_utils.to_tfda_bboxes(
        expected_alb_bboxes, height, width)

    test_utils.partial_assert_array(
        expected_image, actual_image, 0.85, condition + ": image")
    test_utils.partial_assert_array(
        expected_mask, actual_mask, 0.85, condition + ": mask")
    test_utils.assert_array(
        expected_bboxes, actual_bboxes, "bboxes")
