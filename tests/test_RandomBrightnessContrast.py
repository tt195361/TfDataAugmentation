#
# test_RandomBrightnessContrast.py
#

import albumentations as A
from .context import TfDataAugmentation as Tfda
from . import test_utils


def test_call():
    tgt_rb = Tfda.RandomBrightnessContrast(
        brightness_limit=0.5,
        contrast_limit=0.5,
        p=1.0)
    tgt_transform = \
        test_utils.make_tgt_transform(tgt_rb)
    image = test_utils.make_test_image()

    tgt_result = tgt_transform(image=image)
    actual_image = tgt_result['image']

    image_np = image.numpy()
    alpha = tgt_rb.get_param('alpha')
    beta = tgt_rb.get_param('beta')
    expected_image = A.brightness_contrast_adjust(
        image_np, alpha=alpha, beta=beta)

    test_utils.assert_array(
        expected_image, actual_image, "image")
