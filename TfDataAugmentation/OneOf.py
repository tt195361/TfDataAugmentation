#
# OneOf.py
#

import tensorflow as tf # noqa: for PyCharm inspection
import numpy as np
from . import BaseAug
from . import gen_utils


class OneOf(BaseAug):
    """
    Select one of transforms to apply.
    Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1,
    so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying this transform. Default: 0.5.
    """

    def __init__(self, transforms, p=0.5):
        super(OneOf, self).__init__(p)
        self.transforms = gen_utils.check_transforms(transforms)
        if len(transforms) < 2:
            message = "At least 2 transforms need to be specified for OneOf."
            raise ValueError(message)
        self.select_transform_method = eval(
            self.make_select_transform_method())

    def make_select_transform_method(self):
        p_list = [t.p for t in self.transforms]
        p_cumsum_list = np.cumsum(p_list)
        p_cumsum_list = p_cumsum_list / p_cumsum_list[-1]
        len_t = len(self.transforms)

        s = "lambda rnd, transforms, **data: "
        s += "   tf.cond("
        s += "       rnd <= {0}, ".format(p_cumsum_list[0])
        s += "       lambda: transforms[0](force_apply=True, **data), "
        for i in range(1, len_t - 1):
            s += "   lambda: tf.cond("
            s += "       rnd <= {0}, ".format(p_cumsum_list[i])
            s += "       lambda: transforms[{0}](force_apply=True, **data), ".format(i)
        s += "       lambda: transforms[{0}](force_apply=True, **data)".format(len_t - 1)
        s += ")" * (len_t - 1)
        return s

    def do_aug(self, **data):
        rnd = gen_utils.random_float()
        data = \
            self.select_transform_method(rnd, self.transforms, **data)
        return data

    def _do_aug_image(self, image):
        raise NotImplementedError(
            "_do_aug_image for OneOf should not be called")
