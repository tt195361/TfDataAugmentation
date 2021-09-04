#
# BboxParams.py
#

from . import gen_utils
from . import BboxProcessor

SUPPORTED_FORMAT = ('pascal_voc', )


class BboxParams:
    @staticmethod
    def make_processor(params):
        if params is None:
            return BboxProcessor.NoneBboxProcessor()

        if params.format == 'pascal_voc':
            return BboxProcessor.PascalVocBboxProcessor()

        message = \
            "The specified format '{0}' is not supported. " \
            "Supported format is '{1}'." \
            .format(params.format, SUPPORTED_FORMAT)
        raise ValueError(message)

    def __init__(self, bbox_format):
        self.format = gen_utils.check_enum(
            bbox_format, SUPPORTED_FORMAT, "bbox_format")
