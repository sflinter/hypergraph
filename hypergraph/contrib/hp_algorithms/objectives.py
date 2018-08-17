from abc import ABC, abstractmethod


class MethodFactory(ABC):
    def __init__(self):
        pass

    @staticmethod
    def factory(method):
        if method == "max":
            return Maximizer()
        if method == "min":
            return Minimizer()

    def execute(self):
        pass

    @abstractmethod
    def target(self):
        pass


class Executor(ABC):
    pass


class Maximizer(Executor):
    pass


class Minimizer(Executor):
    pass