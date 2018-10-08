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
    # TODO use a graph that emits a serialized format, save with msgpack or yaml.
    # TODO pass a SerializerContext
    # TODO create deserializer, use graph to connect parent -> children dependencies
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
