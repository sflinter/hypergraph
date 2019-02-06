import sys
import numpy as np
import importlib


class HGError(Exception):
    """
    The generic hypergraph error
    """
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


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


def get_hg_module_obj(name: str):
    """
    Load and return a module or class relative to this framework.
    The identifier follows this form: <hg_module_name>.<class>.<...>
    :param name:
    :return: A module or class object
    """
    name = name.split('.')
    if len(name[0]) == 0:
        raise ValueError()
    obj = importlib.import_module('.' + name[0], package='hypergraph')
    if len(name) < 2:
        return obj
    name = name[1:]

    for n in name:
        if len(n) == 0:
            raise ValueError()
        obj = getattr(obj, n)
    return obj


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


class IntBinVecEncoder:
    def __init__(self, range, dim=None):
        if len(range) != 2:
            raise ValueError()
        if range[0] > range[1]:
            raise ValueError()

        self.range = tuple(range)
        self.dim = max([1, int(np.ceil(np.log(range[1]-range[0])/np.log(2)))]) if dim is None else int(dim)

    def decode(self, v: np.ndarray):
        v = np.where(np.array(v) >= 0.)
        if len(v) != 1:
            raise ValueError()
        v = np.sum(2 ** v[0]) + self.range[0]
        m = self.range[1]
        return v if v < m else m-1

    def encode(self, v: int):
        v = int(v) - self.range[0]
        v = np.fromstring(np.binary_repr(v), dtype=np.uint8) - ord('0')
        v1 = np.full((int(self.dim), ), -1, dtype=np.int)
        l = np.min([len(v), len(v1)])
        v1[:l] = v[::-1][:l]
        v1[np.where(v1 == 0)] = -1
        return v1


class BinDrivenTransferFunction:
    def __init__(self, dim, f=lambda x: x):
        self.dim = int(dim) + 1
        self.f = f

    def decode(self, v: np.ndarray):
        sign = v[-1] >= 0
        sign = -1. if sign else 1.
        v = np.where(np.array(v[:-1]) >= 0.)
        if len(v) != 1:
            raise ValueError()
        v = sign * np.sum(2 ** v[0])
        #if sign < 0:
        #    v = -1.*(2**self.dim - v)
        v = 0 if v == 0 else 1.0/v
        return self.f(v)
