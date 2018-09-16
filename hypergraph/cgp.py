import numpy as np
from functools import partial
import itertools
import collections
from abc import ABC
from . import graph as hgg
from . import tweaks
from .utils import StructFactory


class Operators(ABC):
    def __init__(self, input_count=2):
        if not isinstance(input_count, int):
            raise ValueError()
        if input_count < 1:
            raise ValueError()

        self._input_count = input_count
        self.include = None
        self.exclude = None

    @property
    def input_count(self):
        return self._input_count

    def set_selection(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def get_ops(self):
        """
        Get a list of operators. The operators are methods which name starts with 'op_'. The optional
        decorator FuncMark can be used to specify additional features.
        :return:
        """
        ops = filter(lambda n: n.startswith('op_'), dir(self))
        ops = map(lambda n: getattr(self, n), ops)

        def run_func_factory(f):
            if hasattr(f, '_hg_cgp_func_mark'):
                desc = f._hg_cgp_func_mark
                if desc.factory:
                    ret = f()
                    ret.__name__ = f.__name__
                    return ret
            return f

        def check_inc(f, *, include):
            if f.__name__ in include:
                return True
            if hasattr(f, '_hg_cgp_func_mark'):
                return f._hg_cgp_func_mark.group in include
            return False

        include = self.include
        exclude = self.exclude
        if include is not None:
            ops = filter(partial(check_inc, include=include), ops)
        if exclude is not None:
            ops = filter(lambda f: not check_inc(f, include=exclude), ops)

        ops = list(map(run_func_factory, ops))
        if len(ops) == 0:
            raise RuntimeError("The selected operator list is empty")
        return ops


class FuncMark:
    """
    A decorator to mark a function factory for the CGP' node.
    """

    def __init__(self, group: str=None, *, factory: bool=False, sym_exp=None):
        """
        Init func factory
        :param factory: boolean, when true the function is an operator factory that returns an operator or a list of them.
        :param sym_exp: a custom function to be used for symbolic expansion
        """
        self.factory = factory
        self.group = group
        self.sym_exp = sym_exp

    def __call__(self, f):
        f._hg_cgp_func_mark = self
        return f


class TensorOperators(Operators):
    clip_params = {'a_min': -1, 'a_max': 1}

    def __init__(self, default_axis=-1):
        if default_axis not in (0, -1):
            raise ValueError()
        self.default_axis = default_axis
        super().__init__(input_count=2)

    def test_ops(self, trials=10**7):
        ops = list(self.get_ops())
        shape_dim_count_max = 3
        shape_dim_max = 16

        def random_tensor():
            if np.random.choice(2) == 0:
                # scalar case
                return np.random.uniform(-1, 1)
            dim = np.random.randint(1, shape_dim_count_max + 1)
            shape = np.random.randint(1, shape_dim_max + 1, size=dim)
            return np.random.uniform(-1, 1, size=shape)

        for _ in range(trials):
            op = np.random.choice(ops)
            ret = op(x=random_tensor(), y=random_tensor(), p=np.random.uniform(-1, 1))
            if np.any(ret != np.clip(ret, -1, 1)):
                raise ValueError("Values outside bounds for operator " + op.__name__)

    #@staticmethod
    #def clip(v):
    #    v = np.array(v)
    #    isscalar = (v.ndim == 0)
    #    v = v[None] if isscalar else v
    #
    #    # clip values to [-1, 1]
    #    np.copyto(v, 1, where=np.isposinf(v) or np.greater(v, 1))
    #    np.copyto(v, -1, where=np.isneginf(v) or np.less(v, -1))
    #    np.copyto(v, 0, where=np.isnan(v))
    #
    #    return v[0] if isscalar else v

    @property
    def null_value(self):
        return 0.

    @staticmethod
    def to_scalar(value):
        value = np.array(value)
        if np.shape(value) != ():
            value = value.flat
            value = 0 if len(value) == 0 else value[0]
        return np.float(value)

    @staticmethod
    def tensor_shape(v):
        v = np.shape(v)
        return None if v == () else v

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: x)
    def op_identity(x, y, p):
        return x

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: y)
    def op_ywire(x, y, p):
        return y

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: p)
    def op_const(x, y, p):
        return p

    @staticmethod
    def create_const_v(x, p):
        v = np.empty_like(x)
        v.fill(p)
        return v

    @FuncMark('base')
    def op_const_v(self, x, y, p):
        return self.create_const_v(x, p)

    @FuncMark('base')
    def op_zeros(self, x, y, p):
        return self.create_const_v(x, 0.)

    @FuncMark('base')
    def op_ones(self, x, y, p):
        return self.create_const_v(x, 1.)

    @FuncMark('base')
    def op_head(self, x, y, p):
        # TODO head and tail should refer to the default_axis
        return self.unary_vec_op_or_ident(x, lambda v: v[0])

    @FuncMark('base')
    def op_last(self, x, y, p):
        return self.unary_vec_op_or_ident(x, lambda v: v[-1])

    @staticmethod
    @FuncMark('base')
    def op_ravel(x, y, p):
        if isinstance(x, np.ndarray):
            return np.ravel(x)
        return x

    @staticmethod
    @FuncMark('base')
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

    @staticmethod
    def binary_op_factory(op):
        def f(x, y, p):
            isarray = map(lambda v: isinstance(v, np.ndarray), (x, y))
            if all(isarray):
                min_dim = np.minimum(len(x.shape), len(y.shape))
                shape = np.minimum(x.shape[-min_dim:], y.shape[-min_dim:])
                shape = [...] + list(map(slice, shape))
                x, y = x[shape], y[shape]
            return op(x, y)
        return f

    @classmethod
    @FuncMark('math', factory=True)
    def op_add(cls):
        return cls.binary_op_factory(lambda a, b: (a + b)/2.0)

    @classmethod
    @FuncMark('math', factory=True)
    def op_aminus(cls):
        return cls.binary_op_factory(lambda a, b: np.abs(a - b)/2.0)

    @classmethod
    @FuncMark('math', factory=True)
    def op_mul(cls):
        return cls.binary_op_factory(lambda a, b: a * b)

    @classmethod
    @FuncMark('math', factory=True)
    def op_max2(cls):
        return cls.binary_op_factory(lambda a, b: np.maximum(a, b))

    @classmethod
    @FuncMark('math', factory=True)
    def op_min2(cls):
        return cls.binary_op_factory(lambda a, b: np.minimum(a, b))

    @classmethod
    @FuncMark('math', factory=True)
    def op_lt(cls):
        return cls.binary_op_factory(lambda a, b: (np.less(a, b)).astype(dtype=np.float))

    @classmethod
    @FuncMark('math', factory=True)
    def op_gt(cls):
        return cls.binary_op_factory(lambda a, b: (np.greater(a, b)).astype(dtype=np.float))

    @staticmethod
    @FuncMark('math')
    def op_max1(x, y, p):
        return np.max(x)

    @staticmethod
    @FuncMark('math')
    def op_min1(x, y, p):
        return np.min(x)

    @staticmethod
    @FuncMark('math')
    def op_round(x, y, p):
        return np.round(x)

    @staticmethod
    @FuncMark('math')
    def op_ceil(x, y, p):
        return np.ceil(x)

    @staticmethod
    @FuncMark('math')
    def op_floor(x, y, p):
        return np.floor(x)

    @classmethod
    @FuncMark('math', factory=True)
    def op_sqrtxy(cls):
        return cls.binary_op_factory(lambda a, b: np.sqrt(np.square(a)+np.square(b))/np.sqrt(2.))

    @staticmethod
    @FuncMark('math')
    def op_sqrt(x, y, p):
        return np.sqrt(np.abs(x))

    @staticmethod
    @FuncMark('math')
    def op_cpow(x, y, p):
        return np.power(np.abs(x), p + 1)

    @classmethod
    @FuncMark('math', factory=True)
    def op_ypow(cls):
        return cls.binary_op_factory(lambda a, b: np.power(np.abs(a), np.abs(b)))

    @staticmethod
    @FuncMark('math')
    def op_expx(x, y, p):
        return (np.exp(x)-1.)/(np.e-1.)

    @staticmethod
    @FuncMark('math')
    def op_sinx(x, y, p):
        return np.sin(x*np.pi/2.)

    @staticmethod
    @FuncMark('math')
    def op_acos(x, y, p):
        return np.arccos(x)/np.pi

    @staticmethod
    @FuncMark('math')
    def op_asin(x, y, p):
        return np.arcsin(x)*(2./np.pi)

    @staticmethod
    @FuncMark('math')
    def op_atan(x, y, p):
        return np.arctan(x)*(4./np.pi)

    @staticmethod
    @FuncMark('math')
    def op_cmul(x, y, p):
        return x*p

    @staticmethod
    @FuncMark('math')
    def op_abs(x, y, p):
        return np.abs(x)

    @FuncMark('math')
    def op_sum(self, x, y, p):
        if isinstance(x, np.ndarray):
            return np.clip(x.sum(axis=self.default_axis), **self.clip_params)
        return x

    @FuncMark('math')
    def op_cumsum(self, x, y, p):
        if isinstance(x, np.ndarray):
            return np.clip(x.cumsum(axis=self.default_axis), **self.clip_params)
        return x

    @FuncMark('math')
    def op_cumprod(self, x, y, p):
        return np.cumprod(x, axis=self.default_axis)

    @classmethod
    @FuncMark('stochastic')
    def op_sample_uniform(cls, x, y, p):
        return np.random.uniform(low=-1, high=1, size=cls.tensor_shape(x))

    @classmethod
    @FuncMark('stochastic')
    def op_sample_normal(cls, x, y, p):
        return np.clip(np.random.normal(size=cls.tensor_shape(x)) * p, **cls.clip_params)

    def get_soft_index(self, v, p_idx):
        """
        Return an index relative to the element of the default axis addressed by p_idx which is a value in the
        interval [-1, 1]
        :param v: A tensor
        :param p_idx:
        :return:
        """
        shape = self.tensor_shape(v)
        if shape is None:
            return None
        idx = (p_idx + 1.) / 2.
        idx = int(np.round(idx * (shape[self.default_axis] - 1)))
        slices = [slice(k) for k in shape]
        slices[self.default_axis] = idx
        return slices

    @FuncMark('base')
    def op_indexp(self, x, y, p):
        idx = self.get_soft_index(x, p)
        if idx is None:
            return 0.
        return x[idx]

    @staticmethod
    def unary_vec_op_or_ident(x, func):     # TODO use in all unary ops
        x = np.array(x)
        if np.size(x) == 0:
            return 0.
        if x.shape == ():
            return x
        return func(x)

    @staticmethod
    def unary_vec_op_or_zero(x, func):
        x = np.array(x)
        if (np.size(x) == 0) or (x.shape == ()):
            return 0.
        return func(x)

    @FuncMark('stat')
    def op_mean(self, x, y, p):
        return self.unary_vec_op_or_ident(x, partial(np.mean, axis=self.default_axis))

    @FuncMark('stat')
    def op_range(self, x, y, p):
        if not isinstance(x, np.ndarray):
            return x
        return np.max(x, axis=self.default_axis)-np.min(x, axis=self.default_axis)-1

    @FuncMark('stat')
    def op_stddev(self, x, y, p):
        def f(v):
            return np.clip(np.std(v, axis=self.default_axis), **self.clip_params)
        return self.unary_vec_op_or_zero(x, f)

    # TODO skew, kurtosis and the other list operations


class SymbolicEntity:
    pass


class SymbolicInvocation(SymbolicEntity):
    def __init__(self, f, params):
        self.f = f
        self.params = params

    def __str__(self):
        output = self.f.__name__ + '('
        params = self.params
        params = map(str, params)
        output += ', '.join(params)
        output += ')'
        return output

    def __repr__(self):
        return self.__str__()


class SymbolicVariable(SymbolicEntity):
    def __init__(self, name):
        self.name = name
        # TODO declare time step, eg a(0), a(1), ...
        # the "version" of the variable is so identified

    def __str__(self):
        return 'var(\'' + self.name + '\')'

    def __repr__(self):
        return self.__str__()


def exec_symbolically(graph: hgg.Graph, tweaks={}):     # TODO move all to graph?
    if not isinstance(graph, hgg.Graph):
        raise ValueError()

    tweaks = tweaks.copy()
    tweaks[Cell.SYMBOLIC_TWEAK] = True
    ctx = hgg.ExecutionContext(tweaks=tweaks)
    for var in graph.get_vars():
        ctx.set_var(var=var, value=SymbolicVariable(var.fq_var_name))

    output = {}
    with ctx.as_default():
        output['__output__'] = graph(SymbolicVariable('__input__'))

    for var in graph.get_vars():
        output[var.fq_var_name] = ctx.get_var_value(var=var)

    return output


class Cell(hgg.Node):
    SYMBOLIC_TWEAK = '__hg__.cgp.symbolic'

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

    def get_contextual_descriptor(self, desc_ctx):
        ctx = hgg.ExecutionContext.get_default(auto_init=False)
        if ctx is None:
            return super().get_contextual_descriptor(desc_ctx)

        if desc_ctx == 'hg.cgp.formula':
            f = ctx.tweaks.get(self.fully_qualified_name + '_f')
            if f is not None:
                return f.__name__
        return super().get_contextual_descriptor(desc_ctx)

    def __call__(self, input, hpopt_config={}):
        ops = self.operators
        direct_inputs = input[:ops.input_count]

        prefix = self.fully_qualified_name
        f = hpopt_config[prefix + '_f']
        p = hpopt_config[prefix + '_p']

        if bool(hpopt_config.get(self.SYMBOLIC_TWEAK, False)):
            if hasattr(f, '_hg_cgp_func_mark'):
                sym_exp_f = f._hg_cgp_func_mark.sym_exp
                if sym_exp_f is not None:
                    return sym_exp_f(*direct_inputs, p)
            return SymbolicInvocation(f, direct_inputs + [p])

        return f(*direct_inputs, p)


class RegularGrid:
    """
    Regular grid pattern factory
    """

    def __init__(self, input_range, shape, output_factory: StructFactory,
                 operators: Operators, backward_length=1, feedback=False, name=None):
        """
        Init the network factory
        :param input_range: A list of keys of a range
        :param shape:
        :param output_factory:
        :param operators:
        :param backward_length:
        :param feedback: When true a feedback connection is created
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
        self.feedback = bool(feedback)
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
        # TODO check that we have at least ops.input_count inputs
        ops = self.operators

        if self.feedback:
            hgg.var('feedback', initial_value=ops.null_value)

        def gen_extra_inputs(real_input_count):
            extra = [ops.null_value for _ in range(max(0, ops.input_count - (real_input_count + self.feedback)))]
            if self.feedback:
                extra += [hgg.var('feedback')]
            return extra

        if self.input_range is not None:
            # TODO + [hgg.input_all()]? (if possible)
            return [hgg.input_key(key=k) for k in self.input_range] + gen_extra_inputs(len(self.input_range))
        # we assure that we have at least ops.input_count inputs
        return [hgg.input_all()] + gen_extra_inputs(1)

    def __call__(self):
        ops = self.operators
        cname = self.get_comp_name
        shape = self.shape
        backward_length = self.backward_length
        output_factory = self.output_factory

        grid = self.get_grid_coords_list(map(range, shape))
        rows_range = range(shape[0])

        # TODO check grid's input range!

        output = hgg.Graph(name=self.name)
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
                connections += map(hgg.node_ref, self.get_comp_name('c', rows_range, range(j0, j)))

                if j == shape[1]:
                    # TODO different probability for output nodes

                    # connect outputs
                    hgg.output() << output_factory([(tweaks.switch('out_sw_' + str(out_idx)) << connections)
                                                    for out_idx in range(output_factory.input_size)])
                    if self.feedback:
                        # TODO to be tested!
                        set_feedback = hgg.set_var('feedback') << (tweaks.switch(name='feeback_sw') << connections)
                        hgg.add_event_handler('exit', set_feedback)
                else:
                    for i in range(shape[0]):
                        hgg.link(hgg.node_ref(self.get_comp_name('p', i, j)), connections)

        return output
