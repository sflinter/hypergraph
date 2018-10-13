import sys
import numpy as np


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


class MsgPackEncoders:
    @staticmethod
    def encode(obj):
        if isinstance(obj, np.int64):
            return {'__np_int64__': True, 'v': int(obj)}
        if isinstance(obj, np.int32):
            return {'__np_int32__': True, 'v': int(obj)}
        return obj

    @staticmethod
    def decode(obj):
        if '__np_int64__' in obj:
            return np.int64(obj['v'])
        if '__np_int32__' in obj:
            return np.int32(obj['v'])
        return obj
