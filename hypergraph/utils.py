import sys
import abc


def export(f):
    """
    Decorator used to export functions and classes to the package level
    :param f:
    :return:
    """
    mod = sys.modules[f.__module__]
    if hasattr(mod, '__all__'):
        name, all_ = f.__name__, mod.__all__
        if name not in all_:
            all_.append(name)
    else:
        mod.__all__ = [f.__name__]
    return f


class StructFactory:
    @abc.abstractmethod
    def __call__(self, values):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class SequentialDictFactory(StructFactory):
    def __init__(self, keys):
        self.keys = list(keys)

    def __call__(self, values):
        return dict([(k, v) for k, v in zip(self.keys, values)])

    def __len__(self):
        return len(self.keys)


class ListFactory(StructFactory):
    def __init__(self, size=None):
        # TODO validate size
        self.size = size

    def __call__(self, values):
        size = self.size
        if size is not None:
            return list(values[:size])
        return list(values)

    def __len__(self):
        sz = self.size
        if sz is not None:
            return sz
        raise TypeError()
