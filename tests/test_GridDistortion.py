#
# test_GridDistortion.py
#

import albumentations as A
import cv2 # noqa: for PyCharm inspection
from .context import TfDataAugmentation as Tfda
from . import test_utils


def test_call():
    num_steps = 5

    tgt_gd = Tfda.GridDistortion(
        num_steps=num_steps,
        distort_limit=0.5,
        interpolation='nearest',
        border_mode='constant',
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_gd)
    image = test_utils.make_test_image()
    mask = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image, mask=mask)
    actual_image = tgt_result['image']
    actual_mask = tgt_result['mask']

    image_np = image.numpy()
    mask_np = mask.numpy()
    stepsx = tgt_gd.get_param('stepsx')
    stepsy = tgt_gd.get_param('stepsy')
    expected_image = A.grid_distortion(
        image_np, num_steps, stepsx, stepsy,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT)
    expected_mask = A.grid_distortion(
        mask_np, num_steps, stepsx, stepsy,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT)

    test_utils.partial_assert_array(
        expected_image, actual_image, 0.7, "image")
    test_utils.partial_assert_array(
        expected_mask, actual_mask, 0.7, "mask")
