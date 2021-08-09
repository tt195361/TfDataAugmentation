#
# gen_utils.py
#

import tensorflow as tf


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


def random_float(shape=None, minval=0.0, maxval=1.0):
    """
    Returns random float number(s) of the specified shape and range.
    """
    shape = [] if shape is None else shape
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd
