#
# gen_utils.py
#

import tensorflow as tf
from . import BaseAug as Ba


def check_range(val, min_val, max_val, name):
    """
    Checks the specified val is in the range between min_val and max_val.
    """
    if val < min_val or max_val < val:
        message = \
            "{0} is ouf of range for {3}, " \
            "valid range is {1} <= {3} <= {2}." \
            .format(val, min_val, max_val, name)
        raise ValueError(message)
    return val


def check_enum(val, candidates, name):
    """
    Checks the specified val is in the specified candidates.
    """
    if val not in candidates:
        message = \
            "{0} can not be used for {1}. " \
            "Available values are {2}." \
            .format(val, name, candidates)
        raise ValueError(message)
    return val


def check_transforms(transforms):
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
        if not isinstance(trans, Ba.BaseAug):
            message = \
                "The element in 'transforms' needs to be " \
                "an instance of 'BaseAug'. " \
                "The element '{0} ({1})' is not an instance " \
                "of 'BaseAug'." \
                .format(trans, type(trans))
            raise TypeError(message)

    return transforms


def random_int(shape=None, minval=0, maxval=1):
    """
    Returns random integer number(s) of the specified shape and range.
    """
    shape = [] if shape is None else shape
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)
    return rnd


def random_float(shape=None, minval=0.0, maxval=1.0):
    """
    Returns random float number(s) of the specified shape and range.
    """
    shape = [] if shape is None else shape
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd
