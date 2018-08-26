import numpy as np
from functools import partial
import itertools
from abc import ABC
from . import graph as hgg
from . import tweaks
from .utils import StructFactory


class Operators(ABC):
    def __init__(self, default_value=0.0, input_count=2):
        if not isinstance(input_count, int):
            raise ValueError()
        if input_count < 1:
            raise ValueError()

        self._input_count = input_count
        self.default_value = default_value

    @property
    def input_count(self):
        return self._input_count

    def get_ops(self):
        ops = filter(lambda f: hasattr(f, '_hg_cgp_func_factory'), self.__dict__.values())

        def run_func_factory(f):
            # TODO are methods invoked correctly?

            desc = f._hg_cgp_func_factory
            if desc.multi:
                if desc.factory:
                    fs = f()
                    return [f() for f in fs]
                return f()  # TODO check returned value is iterable
            if desc.factory:
                return [f()]
            return [f]
        return itertools.chain.from_iterable(map(run_func_factory, ops))


def unary_adapter(func, activation=lambda a: a):
    """
    Adapt a unary function to cgp function
    :param func: A unary function
    :param activation: The function to be applied to the output
    :return: A new function which has the arguments: x, y, p. The argument x is used as input param.
    """
    return lambda x, y, p: activation(func(x))


class FuncMark:
    """
    A decorator to mark a function factory for the CGP' node.
    """

    def __init__(self, factory: bool=False, multi: bool=False):
        """
        Init func factory
        :param factory: boolean, when true the function is a factory
        :param multi: boolean, when true the function is expected to return a list of function factories.
        """
        self.factory = factory
        self.multi = multi

    def __call__(self, f):
        # TODO add f to a list?
        f._hg_cgp_func_factory = self
        return f


class TensorOperators(Operators):
    def __init__(self, default_shape=(3, ), default_axis=-1):
        if default_axis not in (0, -1):
            raise ValueError()

        self.default_shape = default_shape
        self.default_axis = default_axis

        if not (-1. <= self.default_value <= 1.):
            raise ValueError()

        # TODO pow, pow_int, tan?
        # TODO remove sqrt and add pow only?

    @FuncMark(multi=True)
    def op_simple_math_unary(self):
        return map(partial(unary_adapter, activation=self.activation),
                   [np.ceil, np.floor, np.abs, np.exp, np.sin, np.cos, np.tanh, np.arctanh, np.sqrt])

    def activation(self, v):
        v = np.array(v)
        isscalar = (v.ndim == 0)
        v = v[None] if isscalar else v

        np.copyto(v, 1, where=np.isposinf(v) or np.greater(v, 1))
        np.copyto(v, -1, where=np.isneginf(v) or np.less(v, -1))
        np.copyto(v, self.default_value, where=np.isnan(v))

        return v[0] if isscalar else v

    @staticmethod
    @FuncMark
    def op_identity(x, y, p):
        return x

    @staticmethod
    @FuncMark
    def op_const(x, y, p):
        return p

    @FuncMark
    def op_const_v(self, x, y, p):
        v = np.empty(shape=self.default_shape)
        v[:] = p
        return v

    @FuncMark
    def op_empty_v(self, x, y, p):
        v = np.empty(shape=self.default_shape)
        v[:] = self.default_value
        return v

    @FuncMark
    def op_head(self, x, y, p):
        if isinstance(x, np.ndarray):
            if np.size(x) == 0:
                return self.default_value
            return x[0]
        return x

    @FuncMark
    def op_last(self, x, y, p):
        if isinstance(x, np.ndarray):
            if np.size(x) == 0:
                return self.default_value
            return x[-1]
        return x

    @staticmethod
    @FuncMark
    def op_ravel(x, y, p):
        if isinstance(x, np.ndarray):
            return np.ravel(x)
        return x

    @staticmethod
    @FuncMark
    def op_transpose(x, y, p):
        if isinstance(x, np.ndarray):
            return np.transpose(x)
        return x

    # TODO sort or argsort? argmin and argmax?

    #@FuncMark
    #def op_shape(self, x, y, p):
    #    if isinstance(x, np.ndarray):
    #        return np.array(x.shape)    # TODO set dtype?
    #    return self.default_value

    #@FuncMark
    #def op_size(self, x, y, p):
    #    if isinstance(x, np.ndarray):
    #        return np.size(x)
    #    return self.default_value

    @FuncMark
    def op_sum(self, x, y, p):
        if isinstance(x, np.ndarray):
            return self.activation(x.sum(axis=self.default_axis))
        return x

    @FuncMark
    def op_cumsum(self, x, y, p):
        if isinstance(x, np.ndarray):
            return self.activation(x.cumsum(axis=self.default_axis))
        return x

    # TODO cumprod?
    # TODO more math ops


class StochasticOperators(Operators):
    @staticmethod
    @FuncMark
    def op_sample_uniform(x, y, p):
        return np.random.uniform() * p  # TODO better scaling

    @staticmethod
    @FuncMark
    def op_sample_normal(x, y, p):
        return np.random.normal() * p   # TODO better scaling

    @FuncMark
    def op_sample_poisson(self, x, y, p):
        if p == 0.0:
            return self.default_value
        return np.random.poisson(1.0/p)


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
            prefix + '_p': tweaks.Uniform(low=-1.0, high=1.0)     # TODO get distribution from operators
        }

    def __call__(self, input, hpopt_config={}):
        ops = self.operators
        direct_inputs = input[:ops.input_count]

        prefix = self.fully_qualified_name
        f = hpopt_config[prefix + '_f']
        p = hpopt_config[prefix + '_p']

        return f(*direct_inputs, p)


class RegularGrid:
    """
    Regular grid pattern factory
    """

    def __init__(self, input_range, shape, output_factory: StructFactory,
                 operators: Operators, backward_length=1, name=None):
        """
        Init the network factory
        :param input_range:
        :param shape:
        :param output_factory:
        :param operators:
        :param backward_length:
        :param name: The name associated to the returned graph
        """
        if not isinstance(backward_length, int):
            raise ValueError()
        if backward_length <= 0:
            raise ValueError()

        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError()

        # TODO validate params

        self.input_range = input_range
        self.shape = shape
        self.output_factory = output_factory
        self.operators = operators
        self.backward_length = backward_length
        self.name = name

    @classmethod
    def get_comp_name(cls, comp, i, j):
        """
        Return the name to be associated to a node given the component name and the coordinates (i,j).
        If i and j are of type range then a list of components relative to the sub-grid is returned.
        :param comp: The name of the component (eg. cell or permutation)
        :param i:
        :param j:
        :return:
        """

        def adapt_to_range(x):
            if isinstance(x, range):
                return x
            return range(x, x + 1)

        if isinstance(i, range) or isinstance(j, range):
            i = adapt_to_range(i)
            j = adapt_to_range(j)
            coords = cls.get_grid_coords_list((i, j))
            return [comp + '_' + str(i) + '_' + str(j) for i, j in coords]

        return comp + '_' + str(i) + '_' + str(j)

    @staticmethod
    def get_grid_coords_list(ranges):
        """
        Given a list of ranges return a list of coordinates for the mesh identified by the ranges.
        The coordinates are of the form: [[i1, j1], [i2, j2], ...]
        :param ranges:
        :return:
        """
        grid = np.meshgrid(*ranges, indexing='ij')
        grid = map(np.ravel, grid)
        grid = np.stack(grid).T  # grid: [[i1, j1], [i2, j2], ...]
        return grid

    def create_inputs(self):
        if self.input_range is not None:
            # TODO + [hgg.input_all()]? (if possible)
            return [hgg.input_key(key=k) for k in self.input_range]
        return [hgg.input_all()]

    def __call__(self):
        ops = self.operators
        cname = self.get_comp_name
        shape = self.shape
        backward_length = self.backward_length
        output_factory = self.output_factory

        grid = self.get_grid_coords_list(map(range, shape))
        rows_range = range(shape[0])

        output = g.Graph(name=self.name)
        with output.as_default():
            inputs = self.create_inputs()

            # iterate through the grid
            for i, j in grid:
                # TODO change permutation into combination! We want to allow this configuration: op(node1, node1)
                Cell(operators=ops,
                     name=cname('c', i, j)) << tweaks.permutation(size=ops.input_count, name=cname('p', i, j))

            for j in range(shape[1]+1):
                j0 = j-backward_length
                connections = []
                if j0 < 0:
                    # input layer link
                    connections += inputs
                j0 = max(j0, 0)
                connections += map(hgg.node_ref, self.get_comp_name('c', rows_range, range(j0, max(0, j-1))))

                if j == shape[1]:
                    # connect outputs
                    hgg.output() << output_factory([(tweaks.switch() << connections)
                                                    for _ in range(output_factory.input_size)])
                else:
                    for i in range(shape[0]):
                        hgg.link(hgg.node_ref(self.get_comp_name('p', i, j)), connections)

        return output
