#
# BboxProcessor.py
#

import abc
from . import image_utils


class BboxProcessor:
    @abc.abstractmethod
    def to_internal_format(
            self, original_bboxes, image_height, image_width):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_original_format(
            self, internal_bboxes, image_height, image_width):
        raise NotImplementedError()


class NoneBboxProcessor(BboxProcessor):
    message = "BboxParam must be specified to handle bboxes."

    def to_internal_format(
            self, original_bboxes, image_height, image_width):
        raise ValueError(self.message)

    def to_original_format(
            self, internal_bboxes, image_height, image_width):
        raise ValueError(self.message)


class PascalVocBboxProcessor(BboxProcessor):
    def to_internal_format(
            self, original_bboxes, image_height, image_width):
        internal_bboxes = normalize_bboxes(
            original_bboxes, image_height, image_width)
        return internal_bboxes

    def to_original_format(
            self, internal_bboxes, image_height, image_width):
        original_bboxes = denormalize_bboxes(
            internal_bboxes, image_height, image_width)
        return original_bboxes


def normalize_bboxes(original_bboxes, image_height, image_width):
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = \
        image_utils.decompose_bboxes(original_bboxes)

    # (image_width - 1) and (image_height - 1) ?
    int_x_min = orig_x_min / image_width
    int_y_min = orig_y_min / image_height
    int_x_max = orig_x_max / image_width
    int_y_max = orig_y_max / image_height

    internal_bboxes = image_utils.compose_bboxes(
        int_x_min, int_y_min, int_x_max, int_y_max)
    return internal_bboxes


def denormalize_bboxes(internal_bboxes, image_height, image_width):
    int_x_min, int_y_min, int_x_max, int_y_max = \
        image_utils.decompose_bboxes(internal_bboxes)

    # (image_width - 1) and (image_height - 1) ?
    orig_x_min = int_x_min * image_width
    orig_y_min = int_y_min * image_height
    orig_x_max = int_x_max * image_width
    orig_y_max = int_y_max * image_height

    original_bboxes = image_utils.compose_bboxes(
        orig_x_min, orig_y_min, orig_x_max, orig_y_max)
    return original_bboxes
