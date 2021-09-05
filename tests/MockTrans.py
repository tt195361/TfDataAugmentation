#
# MockTrans.py
#

from .context import TfDataAugmentation as Tfda


class MockTrans(Tfda.BaseAug):
    def __init__(self, name, logger, p):
        super(MockTrans, self).__init__(p)
        self.name = name
        self.logger = logger

    def _do_aug_image(self, image):
        message = self.get_call_message()
        self.logger.add(message)
        return image

    def get_call_message(self):
        return "{0} called".format(self.name)
