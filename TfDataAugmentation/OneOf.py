#
# OneOf.py
#

from . import BaseAug
from . import gen_utils


class OneOf(BaseAug):
    def __init__(self, transforms, p):
        super(OneOf, self).__init__(p)
        self.transforms = gen_utils.check_transforms(transforms)
        self.n_transforms = len(transforms)

    def __call__(self, **data):
        idx = gen_utils.random_int(
            shape=[], minval=0, maxval=self.n_transforms)
        selected_transform = self.transforms[idx]
        data = selected_transform(**data)
        return data

    def _do_aug_image(self, image):
        raise NotImplementedError(
            "_do_aug_image for OneOf should not be called")
