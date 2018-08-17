import numpy as np
from functools import partial
from abc import ABC
from . import graph as hgg
from . import tweaks


class Operators(ABC):
    def __init__(self):
        self.runtime_ops = []

    def _get_op_methods(self):
        ops = map(partial(getattr, self), filter(lambda n: n.startswith('op_'), dir(self)))

        def run_func_factory(f):
            if hasattr(f, '_hg_cgp_func_factory'):
                return f()
            return f
        return map(run_func_factory, ops)

    def get_ops(self) -> list:
        return list(self.runtime_ops) + list(self._get_op_methods())


def unitary_adapter(func):
    """
    Adapt a unitary function to cgp function
    :param func: A unitary function
    :return: A new function which has the arguments: x, y, p. The argument x is used as input param.
    """
    return lambda x, y, p: func(x)


def func_factory(f):
    """
    A decorator to mark functions factories.
    :param f:
    :return:
    """
    f._hg_cgp_func_factory = True
    return f


class TensorOperators(Operators):
    def __init__(self, default_shape=(3, ), invalid_value=.0):
        # TODO default_axis
        self.default_shape = default_shape
        self.invalid_value = invalid_value
        self.runtime_ops += list(map(unitary_adapter, [np.ceil, np.floor]))

    @staticmethod
    def op_identity(x, y, p):
        return x

    @staticmethod
    def op_const(x, y, p):
        return p

    @func_factory
    def op_const_v(self):
        def f(x, y, p):
            v = np.empty(shape=self.default_shape)
            v[:] = p
            return v
        return f

    @func_factory
    def op_head(self):
        def f(x, y, p):
            if isinstance(x, np.ndarray):
                if np.size(x) == 0:
                    return self.invalid_value
                return x[0]
            return x
        return f

    @func_factory
    def op_last(self):
        def f(x, y, p):
            if isinstance(x, np.ndarray):
                if np.size(x) == 0:
                    return self.invalid_value
                return x[-1]
            return x
        return f

    @staticmethod
    def op_ravel(x, y, p):
        if isinstance(x, np.ndarray):
            return np.ravel(x)
        return x

    @staticmethod
    def op_transpose(x, y, p):
        if isinstance(x, np.ndarray):
            return np.transpose(x)
        return x

    # TODO sort or argsort? argmin and argmax?

    @staticmethod
    def op_shape(x, y, p):
        if isinstance(x, np.ndarray):
            return np.array(x.shape)    # TODO set dtype?
        return tuple()

    @staticmethod
    def op_size(x, y, p):
        if isinstance(x, np.ndarray):
            return np.size(x)
        return 1

    @staticmethod
    def op_sum(x, y, p):
        if isinstance(x, np.ndarray):
            return x.sum(axis=-1)
        return x

    @staticmethod
    def op_cumsum(x, y, p):
        if isinstance(x, np.ndarray):
            return x.cumsum(axis=-1)
        return x

    # TODO cumprod?


class Cell(hgg.Node):
    def __init__(self, operators: Operators, name=None):
        if not isinstance(operators, Operators):
            raise ValueError()
        self.operators = operators
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        prefix = self.fully_qualified_name

        return {
            prefix + '_f': tweaks.UniformChoice(values=self.operators.get_ops()),
            prefix + '_p': tweaks.Uniform()
        }

    def __call__(self, input, hpopt_config={}):
        x, y = input['x'], input['y']

        prefix = self.fully_qualified_name
        f = hpopt_config[prefix + '_f']
        p = hpopt_config[prefix + '_p']

        return f(x, y, p)
