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


def serializable_form(obj):
    if not isinstance(obj, dict):
        raise ValueError()
    # TODO support generic structure
    output = {}
    for key, value in obj.items():
        pass
        if hasattr(value, '_hg_serializer_descriptor'):
            value = getattr(value, '_hg_serializer_descriptor')
        output[key] = value
    return output


class StructFactory:
    # TODO create new output_factory where the scaling (and some tweaks) are included in one single component
    # the new OutputFactory (or StructFactory) should be a Node!

    """
    Generic struct factory given a list of values (iterable)
    """

    def __init__(self, input_size=None):
        if not isinstance(input_size, (int, type(None))):
            raise ValueError()
        if input_size < 0:
            raise ValueError()
        self._input_size = input_size

    @property
    def input_size(self):
        """
        The number of required values or None if the factory supports any number of them.
        :return:
        """
        return self._input_size

    @abc.abstractmethod
    def __call__(self, values):
        pass


class SequentialDictFactory(StructFactory):
    def __init__(self, keys):
        self.keys = list(keys)
        super().__init__(len(keys))

    def __call__(self, values):
        return dict([(k, v) for k, v in zip(self.keys, values)])


class ListFactory(StructFactory):
    def __init__(self, size):
        super().__init__(input_size=size)

    def __call__(self, values):
        size = self.input_size
        if size is not None:
            return list(values[:size])
        return list(values)


class SingleValueStructFactory(StructFactory):
    def __init__(self):
        super().__init__(input_size=1)

    def __call__(self, values):
        return values[0]
