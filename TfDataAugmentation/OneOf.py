#
# OneOf.py
#

from . import BaseAug
from . import gen_utils


class OneOf(BaseAug):
    def __init__(self, transforms, p):
        super(OneOf, self).__init__(p)
        # https://a-zumi.net/python-iterable-object/
        if not hasattr(transforms, '__iter__'):
            # Actually 'transforms' need indexing, so not sure
            # whether checking iterable is enough or not.
            message = \
                "The parameter 'transforms' needs to be an iterable. " \
                "Passed 'transforms': {0} ({1})" \
                .format(transforms, type(transforms))
            raise TypeError(message)
        for trans in transforms:
            if not isinstance(trans, BaseAug):
                message = \
                    "The element in 'transforms' needs to be " \
                    "an instance of 'BaseAug'. " \
                    "The element '{0} ({1})' is not an instance " \
                    "of 'BaseAug'." \
                    .format(trans, type(trans))
                raise TypeError(message)

        self.transforms = transforms
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
