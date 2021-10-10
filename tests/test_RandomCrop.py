#
# test_RandomCrop.py
#

import numpy as np
import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils


def test_call():
    image_height = test_utils.IMAGE_HEIGHT
    image_width = test_utils.IMAGE_WIDTH
    crop_height = int(image_height * np.random.rand())
    crop_width = int(image_width * np.random.rand())

    image = test_utils.make_test_image()
    mask = test_utils.make_test_image()
    bboxes = test_utils.make_test_bboxes()

    tgt_rc = Tfda.RandomCrop(
        height=crop_height,
        width=crop_width,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_rc)

    tgt_result = tgt_transform(
        image=image, mask=mask, bboxes=bboxes)
    actual_image = tgt_result['image']
    actual_mask = tgt_result['mask']
    actual_bboxes = tgt_result['bboxes']

    image_np = image.numpy()
    mask_np = mask.numpy()
    bboxes_np = bboxes.numpy()
    h_start = tgt_rc.get_param('h_start')
    w_start = tgt_rc.get_param('w_start')
    expected_image = A.random_crop(
        image_np, crop_height, crop_width, h_start, w_start)
    expected_mask = A.random_crop(
        mask_np, crop_height, crop_width, h_start, w_start)

    bboxes_alb = test_utils.to_alb_bboxes(
        bboxes_np, image_height, image_width)
    expected_alb_bbox_list = []
    for bbox_alb in bboxes_alb:
        expected_alb_bbox = A.bbox_random_crop(
            bbox_alb, crop_height, crop_width,
            h_start, w_start, image_height, image_width)
        expected_alb_bbox_list.append(expected_alb_bbox)
    expected_alb_bboxes = np.array(expected_alb_bbox_list)
    expected_bboxes = test_utils.to_tfda_bboxes(
        expected_alb_bboxes, crop_height, crop_width)

    test_utils.assert_array(
        expected_image, actual_image, "image")
    test_utils.assert_array(
        expected_mask, actual_mask, "mask")
    test_utils.assert_array(
        expected_bboxes, actual_bboxes, "bboxes")
