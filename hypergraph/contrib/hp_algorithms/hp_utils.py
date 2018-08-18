from abc import ABC, abstractmethod


class Error(ABC, Exception):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        self.message = ''
        return self.message


class TypeMisMatchError(Error):
    def __init__(self, expected, given):
        self.expected = expected
        self.given = given

    def __str__(self):
        self.message = "Expected '%s' but received '%s'" % (self.expected, self.given)
        return self.message






