#
# OneOf.py
#

import tensorflow as tf
from . import BaseAug
from . import gen_utils


class OneOf(BaseAug):
    def __init__(self, transforms, p):
        super(OneOf, self).__init__(p)
        self.transforms = gen_utils.check_transforms(transforms)
        self.n_transforms = len(transforms)
        self.select_transform_method = eval(
            self.make_select_transform_method(self.n_transforms))

    def make_select_transform_method(self, n_transforms):
        s = \
            "lambda idx, transforms: " \
            "    tf.switch_case(" \
            "        idx," \
            "        branch_fns={"
        for i in range(n_transforms):
            s += "{0}: lambda: transforms[{0}]".format(i)
            if i < n_transforms - 1:
                s += ","
        s += \
            "        }" \
            "    )"
        return s

    def __call__(self, **data):
        idx = gen_utils.random_int(
            shape=[], minval=0, maxval=self.n_transforms)
        selected_transform = \
            self.select_transform_method(idx, self.transforms)
        data = selected_transform(**data)
        return data

    def _do_aug_image(self, image):
        raise NotImplementedError(
            "_do_aug_image for OneOf should not be called")
