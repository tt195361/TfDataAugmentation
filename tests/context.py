#
# context.py
#

import sys
import os

this_dir = os.path.dirname(__file__)
this_parent_dir = os.path.join(this_dir, '..')
this_parent_abs_dir = os.path.abspath(this_parent_dir)
sys.path.insert(0, this_parent_abs_dir)

# https://stackoverflow.com/questions/21139329/false-unused-import-statement-in-pycharm
# noinspection PyUnresolvedReferences
import TfDataAugmentation
# noinspection PyUnresolvedReferences
import TfDataAugmentation.image_utils
