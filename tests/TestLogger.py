#
# TestLogger.py
#

class TestLogger:
    def __init__(self):
        self.message_list = []

    def reset(self):
        # All instance variables should be declared in __init__().
        # So, call it to go back to the initial state.
        self.__init__()

    def add(self, message):
        self.message_list.append(message)

    def get(self):
        return TestLogger.make_log(self.message_list)

    @staticmethod
    def make_log(message_list):
        return '\n'.join(message_list)
