#
# BaseAug.py
#

import tensorflow as tf
import abc
from . import gen_utils


class BaseAug:
    def __init__(self, p=0.5):
        self.p = gen_utils.check_range(p, 0.0, 1.0, "p")
        self.params = {}

    def __call__(self, **data):
        supported_keys = {"image", "mask", "bboxes"}
        for key in data.keys():
            if key not in supported_keys:
                message = \
                    "The specified key '{0}' in **data is not supported." \
                    "Supported keys are '{1}'." \
                    .format(key, ', '.join(supported_keys))
                raise ValueError(message)

        if "image" not in data.keys():
            raise ValueError('No "image" in **data.')

        image = data["image"]
        mask = data["mask"] if "mask" in data.keys() else None
        bboxes = data["bboxes"] if "bboxes" in data.keys() else None

        rnd = gen_utils.random_float()
        image, mask, bboxes = tf.cond(
            rnd <= self.p,
            lambda: self.do_aug(image, mask, bboxes),
            lambda: self._no_aug(image, mask, bboxes))

        aug_result = {"image": image}
        if mask is not None:
            aug_result["mask"] = mask
        if bboxes is not None:
            aug_result["bboxes"] = bboxes
        return aug_result

    def do_aug(self, image, mask, bboxes):
        self.params = self._make_params(image)
        self._prepare_aug(image, self.params)
        aug_image = self._do_aug_image(image)
        aug_mask = self._do_aug_mask(mask) \
            if mask is not None else None
        aug_bboxes = self._do_aug_bboxes(bboxes, image) \
            if bboxes is not None else None
        return aug_image, aug_mask, aug_bboxes

    def _make_params(self, image):
        return {}

    def _prepare_aug(self, image, params):
        pass

    @abc.abstractmethod
    def _do_aug_image(self, image):
        raise NotImplementedError("_do_aug_image needs to implement")

    def _do_aug_mask(self, mask):
        return mask

    def _do_aug_bboxes(self, bboxes, image):
        return bboxes

    @staticmethod
    def _no_aug(image, mask, bboxes):
        return image, mask, bboxes

    def get_param(self, name):
        return self.params[name].numpy()
