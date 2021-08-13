#
# TestLogger.py
#

class TestLogger:
    def __init__(self):
        self.log = []

    def reset(self):
        # All instance variables should be declared in __init__().
        # So, call it to go back to the initial state.
        self.__init__()

    def add(self, message):
        self.log.append(message)

    def get(self):
        return '\n'.join(self.log)
