#
# Compose.py
#

from . import BaseAug
from . import gen_utils


class Compose(BaseAug):
    def __init__(self, transforms, p=1.0):
        super(Compose, self).__init__(p)
        self.transforms = gen_utils.check_transforms(transforms)

    def __call__(self, **data):
        for trans in self.transforms:
            data = trans(**data)
        return data

    def _do_aug_image(self, image):
        raise NotImplementedError(
            "_do_aug_image for Compose should not be called")
