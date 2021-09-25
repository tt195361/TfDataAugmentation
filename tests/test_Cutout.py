#
# test_Cutout.py
#

import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils
import random


def test_call():
    num_holes = random.randint(1, 5)
    max_h_size = random.randint(1, 5)
    max_w_size = random.randint(1, 5)
    tgt_cutout = Tfda.Cutout(
        num_holes=num_holes,
        max_h_size=max_h_size,
        max_w_size=max_w_size,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_cutout)
    image = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image)
    actual_image = tgt_result['image']

    image_np = image.numpy()
    holes = tgt_cutout.get_param('holes')
    expected_image = A.cutout(
        image_np, holes, fill_value=0)

    test_utils.assert_array(
        expected_image, actual_image, "image")
