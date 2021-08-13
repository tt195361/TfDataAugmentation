#
# MockTrans.py
#

from .context import TfDataAugmentation as Tfda


class MockTrans(Tfda.BaseAug):
    def __init__(self, name, logger, p):
        super(MockTrans, self).__init__(p)
        self.name = name
        self.logger = logger

    def __call__(self, **data):
        message = "{0} called".format(self.name)
        self.logger.add(message)


