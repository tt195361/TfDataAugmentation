#
# gen_utils.py
#

import tensorflow as tf
from . import BaseAug as Ba


def check_float_range(val, min_val, max_val, name):
    _check_type(val, float, name)
    _check_range(val, min_val, max_val, name)
    return val


def check_int_range(val, min_val, max_val, name):
    _check_type(val, int, name)
    _check_range(val, min_val, max_val, name)
    return val


def _check_type(val, expected_type, name):
    if not isinstance(val, expected_type):
        message = \
            "The type of '{0}' must be float. " \
            "Got {1} ({2})" \
            .format(name, val, type(val))
        raise TypeError(message)


def _check_range(val, min_val, max_val, name):
    """
    Checks the specified val is in the range between min_val and max_val.
    """
    assert \
        min_val is not None or max_val is not None, \
        "Either min_val or max_val is not None"

    if min_val is None:
        _check_max(val, max_val, name)
    elif max_val is None:
        _check_min(val, min_val, name)
    else:
        _check_in_between(val, min_val, max_val, name)


def _check_in_between(val, min_val, max_val, name):
    if val < min_val or max_val < val:
        message = \
            "{0} is ouf of range for '{3}', " \
            "valid range is {1} <= '{3}' <= {2}." \
            .format(val, min_val, max_val, name)
        raise ValueError(message)


def _check_min(val, min_val, name):
    if val < min_val:
        message = \
            "{0} is ouf of range for '{2}', " \
            "valid range is {1} <= '{2}'." \
            .format(val, min_val, name)
        raise ValueError(message)


def _check_max(val, max_val, name):
    if max_val < val:
        message = \
            "{0} is ouf of range for '{2}', " \
            "valid range is '{2}' <= {1}." \
            .format(val, max_val, name)
        raise ValueError(message)


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
                "Passed element '{0} ({1})'." \
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
