#
# BaseAug.py
#

import tensorflow as tf
import abc
from . import gen_utils


class BaseAug:
    def __init__(self, p=0.5):
        self.p = gen_utils.check_float_range(p, 0.0, 1.0, "p")
        self.params = {}

    # Must not override
    # @final # available Python 3.8
    def __call__(self, force_apply=False, **data):
        rnd = gen_utils.random_float()
        aug_result = tf.cond(
            tf.math.logical_or(rnd <= self.p, force_apply),
            lambda: self.do_aug(**data),
            lambda: self._no_aug(**data))
        return aug_result

    def do_aug(self, **data):
        if "image" not in data.keys():
            raise ValueError('No "image" in **data.')

        self.params = self._make_params()
        self._prepare_aug(data["image"], self.params)
        data["image"] = self._do_aug_image(data["image"])
        if "mask" in data.keys():
            data["mask"] = self._do_aug_mask(data["mask"])
        if "bboxes" in data.keys():
            data["bboxes"] = self._do_aug_bboxes(data["bboxes"])
        return data

    def _make_params(self):
        return {}

    def _prepare_aug(self, image, params):
        pass

    @abc.abstractmethod
    def _do_aug_image(self, image):
        raise NotImplementedError("_do_aug_image needs to implement")

    def _do_aug_mask(self, mask):
        return mask

    def _do_aug_bboxes(self, bboxes):
        return bboxes

    @staticmethod
    def _no_aug(**data):
        return data

    def get_param(self, name):
        return self.params[name].numpy()
