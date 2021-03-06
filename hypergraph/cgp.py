# Cartesian Genetic Programming algorithms for the Hypergraph framework

import numpy as np
from functools import partial
import itertools
import uuid
import types
from abc import ABC
from . import graph as hgg
from . import tweaks


class Operators(ABC):
    """
    Base class for the operators to be used in the grid's nodes.
    """

    def __init__(self, input_count=2):
        """
        Init the base class for the grid's operators. The number of inputs for each node is fixed for all nodes.
        :param input_count: The number of inputs for each node. By inputs we mean the parameters that are considered
        variables, note that each operator function will have a third parameter, a constant one that remain fixed
        for a certain node of the CGP grid.
        """
        if not isinstance(input_count, int):
            raise ValueError()
        if input_count < 1:
            raise ValueError()

        self._input_count = input_count
        self.include = None
        self.exclude = None
        self._obj_id = str(uuid.uuid4())  # used by serializer

    def _install_op_serializers(self, ser_type):
        """
        Install serializer descriptor on operators. This is an internal function.
        :param ser_type: The identification of the serializer
        :return:
        """
        ops = self.get_ops()
        for f in ops:
            if isinstance(f, types.MethodType):
                f = f.__func__
            if isinstance(f, types.FunctionType):
                # v = {'__hg_ser_type__': ser_type, 'op': f.__name__, 'parent': self._obj_id}
                setattr(f, '_hg_tweak_descriptor', f.__name__)

    @property
    def input_count(self):
        return self._input_count

    def set_selection(self, include=None, exclude=None):
        """
        Set the filter for the included/excluded operators. The filter can be set at single operator level
        or group of them. See the annotation FuncMark for more information on operators grouping.
        :param include:
        :param exclude:
        :return:
        """
        self.include = include
        self.exclude = exclude

    def get_op_by_name(self, name: str):
        """
        Return a specific operator's callable by name
        :param name: The string identifier of the operator
        :return: The operator's callable
        """
        if not name.startswith('op_'):
            raise ValueError()
        return getattr(self, name)

    def get_ops(self, *, apply_filters=True):
        """
        Get a list of operators. The operators are methods which name starts with 'op_'. The optional
        decorator FuncMark can be used to specify additional features.
        :return:
        """
        ops = filter(lambda n: n.startswith('op_'), dir(self))
        ops = map(lambda n: getattr(self, n), ops)

        def check_inc(f, *, include):
            if f.__name__ in include:
                return True
            if hasattr(f, '_hg_cgp_func_mark'):
                return f._hg_cgp_func_mark.group in include
            return False

        if apply_filters:
            include = self.include
            exclude = self.exclude
            if include is not None:
                ops = filter(partial(check_inc, include=include), ops)
            if exclude is not None:
                ops = filter(lambda f: not check_inc(f, include=exclude), ops)

        ops = list(ops)
        if len(ops) == 0:
            raise RuntimeError("The selected operator list is empty")
        return ops


class FuncMark:
    """
    A decorator to mark a function factory for the CGP' node.
    """

    def __init__(self, group: str=None, *, sym_exp=None):
        """
        Init func factory
        :param group: Name of the group that this operator belongs. The group name can be used to activate/deactivate
        groups of operators. See the method Operators.set_selection for more insights.
        :param sym_exp: a custom function to be used for symbolic expansion. This is particularly useful during
        the symbolic execution, because we can hide identity nodes that would not contribute with any information.
        """
        self.group = group
        self.sym_exp = sym_exp

    def __call__(self, f):
        f._hg_cgp_func_mark = self
        return f


class DelayOperators(Operators):
    """
    An operator that acts as identity (output=input) except that the output is delayed by a certain number
    of time steps. Actually there are just two delay available, the delay1 and delay2. The two operators
    postpone the output by one and two time steps respectively. These operators come as a separate class because
    their generality allows the association with other specific operators.
    """

    SERIALIZER_TYPE = 'hg.cgp.delay_ops.op'

    def __init__(self, parent: Operators):
        """
        Init the delay operators and install them into a parent set of operators
        :param parent: A parent set of operators
        """
        if not isinstance(parent, Operators):
            raise ValueError()
        if parent.input_count <= 0:
            raise ValueError()
        super().__init__(input_count=parent.input_count)

        graph = hgg.Graph()     # graph used as relative context for the internal variable storage
        initializer = parent.null_like
        self._delay1 = hgg.DelayProcess(graph=graph, units=1, initializer=initializer)
        self._delay2 = hgg.DelayProcess(graph=graph, units=2, initializer=initializer)

        # install operators into parent
        parent.op_delay1 = self.op_delay1
        parent.op_delay2 = self.op_delay2

        self._install_op_serializers(self.SERIALIZER_TYPE)

    @FuncMark('delay')
    def op_delay1(self, *inputs):
        return self._delay1(inputs[0])

    @FuncMark('delay')
    def op_delay2(self, *inputs):
        return self._delay2(inputs[0])


class TensorOperators(Operators):
    """
    CGP Operators for tensor data (numpy ndarray). These operators accept 3 input parameters, two tensors and one
    constant scalar. The constant scalar parameter p,is considered part of the operator, thus given a node in the
    CGP grid, then p remains constant after the evolution of the program.
    All parameters have scalar values with the following domain: [-1, 1].
    """

    clip_params = {'a_min': -1, 'a_max': 1}
    SERIALIZER_TYPE = 'hg.cgp.tensor_ops.op'

    def __init__(self, default_axis=-1):
        if default_axis not in (0, -1):
            raise ValueError()
        self.default_axis = default_axis
        super().__init__(input_count=2)
        self._install_op_serializers(self.SERIALIZER_TYPE)

        # operators to be applied to each output, usually aggregators used to transform tensor of vector
        # values to scalars
        self.output_ops = [self.op_max1, self.op_mean, self.op_indexp]  # TODO fix, a tensor can be still the result...

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
        """
        Return the scalar constant zero.
        """
        return 0.

    def null_like(self, x):
        """
        Return a tensor with the same shape as the parameter x but filled with zeros.
        """
        return self.create_const_v(x, 0.)

    @staticmethod
    def to_scalar(value):
        value = np.array(value)
        if np.shape(value) != ():
            value = value.flat
            value = 0 if len(value) == 0 else np.mean(value)    # previous version: value[0]
            # TODO various options possible here: value[0], max, mean, ...
            # TODO this shuold be a tweak!
        return np.float(value)

    @staticmethod
    def tensor_shape(v):
        v = np.shape(v)
        return None if v == () else v

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: x)
    def op_identity(x, y, p):
        """
        An operator that acts as identity with respect to the first parameter (x).
        """
        return x

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: y)
    def op_ywire(x, y, p):
        """
        An operator that acts as identity with respect to the second parameter (y).
        """
        return y

    @staticmethod
    @FuncMark('base', sym_exp=lambda x, y, p: p)
    def op_const(x, y, p):
        """
        Return the const param p. Note that beside p appears as an input variable, its value remain fixed for a
        specific node in the CGP grid.
        """
        return p

    @staticmethod
    def create_const_v(x, p):
        """
        Return a tensor with the same shape as x but filled with the value of the parameter p (p is a scalar).
        """
        v = np.empty_like(x)
        v.fill(p)
        return v

    @FuncMark('base')
    def op_const_v(self, x, y, p):
        """
        Return a tensor with the same shape as x but filled with the value of the parameter p.
        """
        return self.create_const_v(x, p)

    @FuncMark('base')
    def op_zeros(self, x, y, p):
        """
        Return a tensor with the same shape as x but filled with zeros.
        """
        return self.create_const_v(x, 0.)

    @FuncMark('base')
    def op_ones(self, x, y, p):
        """
        Return a tensor with the same shape as x but filled with ones.
        """
        return self.create_const_v(x, 1.)

    @FuncMark('base')
    def op_head(self, x, y, p):
        """
        Return the first element of x, this is equivalent to x[0]. Note that the first element may be a tensor
        as well.
        """
        # TODO head and tail should refer to the default_axis
        return self.unary_vec_op_or_ident(x, lambda v: v[0])

    @FuncMark('base')
    def op_last(self, x, y, p):
        return self.unary_vec_op_or_ident(x, lambda v: v[-1])

    @staticmethod
    @FuncMark('base')
    def op_ravel(x, y, p):
        """
        Return a tensor which is the reduction of the tensor passed through the parameter x to a 1D array.
        """
        if isinstance(x, np.ndarray):
            return np.ravel(x)
        return x

    @staticmethod
    @FuncMark('base')
    def op_transpose(x, y, p):
        """
        Return a transposed version of the tensor passed through the parameter x.
        """
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
    def binary_op(op, x, y):
        """
        Helper for binary operators.
        :param op: A function to be used as binary operator for the parameters x and y.
        :param x:
        :param y:
        :return:
        """

        isarray = map(lambda v: isinstance(v, np.ndarray), (x, y))
        if all(isarray):
            min_dim = np.minimum(len(x.shape), len(y.shape))
            shape = np.minimum(x.shape[-min_dim:], y.shape[-min_dim:])
            shape = [...] + list(map(slice, shape))
            x, y = x[shape], y[shape]
        return op(x, y)

    @classmethod
    @FuncMark('math')
    def op_add(cls, x, y, p):
        """
        Arithmetic addition normalized, (a + b)/2.0
        """
        return cls.binary_op(lambda a, b: (a + b)/2.0, x, y)

    @classmethod
    @FuncMark('math')
    def op_aminus(cls, x, y, p):
        return cls.binary_op(lambda a, b: np.abs(a - b)/2.0, x, y)

    @classmethod
    @FuncMark('math')
    def op_mul(cls, x, y, p):
        return cls.binary_op(lambda a, b: a * b, x, y)

    @classmethod
    @FuncMark('math')
    def op_max2(cls, x, y, p):
        return cls.binary_op(lambda a, b: np.maximum(a, b), x, y)

    @classmethod
    @FuncMark('math')
    def op_min2(cls, x, y, p):
        return cls.binary_op(lambda a, b: np.minimum(a, b), x, y)

    @classmethod
    @FuncMark('math')
    def op_lt(cls, x, y, p):
        return cls.binary_op(lambda a, b: (np.less(a, b)).astype(dtype=np.float), x, y)

    @classmethod
    @FuncMark('math')
    def op_gt(cls, x, y, p):
        return cls.binary_op(lambda a, b: (np.greater(a, b)).astype(dtype=np.float), x, y)

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
    @FuncMark('math')
    def op_sqrtxy(cls, x, y, p):
        return cls.binary_op(lambda a, b: np.sqrt(np.square(a)+np.square(b))/np.sqrt(2.), x, y)

    @staticmethod
    @FuncMark('math')
    def op_sqrt(x, y, p):
        return np.sqrt(np.abs(x))

    @staticmethod
    @FuncMark('math')
    def op_cpow(x, y, p):
        return np.power(np.abs(x), p + 1)

    @classmethod
    @FuncMark('math')
    def op_ypow(cls, x, y, p):
        return cls.binary_op(lambda a, b: np.power(np.abs(a), np.abs(b)), x, y)

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
        # TODO investigate why sometimes we have values outside the boundaries
        return np.arccos(np.clip(x, -1., 1.))/np.pi

    @staticmethod
    @FuncMark('math')
    def op_asin(x, y, p):
        return np.arcsin(np.clip(x, -1., 1.))*(2./np.pi)

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
    """
    The base class for the placeholder used for symbolic execution. Symbolic execution is a method implemented
    into our CGP implementation that allows the extraction of the formula that represents the evolved program.
    """
    pass


class SymbolicInvocation(SymbolicEntity):
    """
    A symbolic entity that represents a function invocation.
    """

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
    """
    A symbolic entity that represents an external variable identified by name.
    """

    def __init__(self, name):
        self.name = name
        self.indexes = None
        # TODO declare time step, eg a(0), a(1), ...
        # the "version" of the variable is so identified

    def __str__(self):
        output = 'var(\'' + self.name + '\')'
        if self.indexes is not None:
            for idx in self.indexes:
                output += '[' + str(idx) + ']'
        return output

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        indexes = self.indexes
        indexes = () if indexes is None else indexes
        indexes += (key, )
        output = SymbolicVariable(self.name)
        output.indexes = indexes
        return output


def exec_symbolically(graph: hgg.Graph, tweaks={}):     # TODO move all to graph?
    """
    Execute a graph symbolically. The nodes used in this graph must support the symbolic execution.
    The returned value is a syntax tree containing symbolic entities that represent the formula implemented by the
    graph. See the class SymbolicEntity and its subclasses for more details.
    :param graph:
    :param tweaks:
    :return:
    """
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
    """
    A node representing an operator's cell. This is one of the main components of the CGP grid.
    """

    SYMBOLIC_TWEAK = '__hg__.cgp.symbolic'

    def __init__(self, operators: Operators, op_distr=None, name=None):
        """
        Init the cell node.
        :param operators: The operators to be used for this node.
        :param op_distr:
        :param name: The name of the node, this is relative to the graph.
        """
        if not isinstance(operators, Operators):
            raise ValueError()
        if not isinstance(op_distr, (tweaks.Distribution, type(None))):
            raise ValueError()

        self.operators = operators
        self.op_distr = op_distr
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        prefix = self.fully_qualified_name

        f_distr = self.op_distr
        if f_distr is None:
            f_distr = tweaks.UniformChoice(values=self.operators.get_ops())
        return {
            prefix + '_f': f_distr,
            prefix + '_p': tweaks.Uniform(low=-1.0, high=1.0)     # TODO get distribution from operators
        }

    # def get_contextual_descriptor(self, desc_ctx):
    #    ctx = hgg.ExecutionContext.get_default(auto_init=False)
    #    if ctx is None:
    #        return super().get_contextual_descriptor(desc_ctx)
    #
    #    if desc_ctx == 'hg.cgp.formula':
    #        f = ctx.tweaks.get(self.fully_qualified_name + '_f')
    #        if f is not None:
    #            return f.__name__
    #    return super().get_contextual_descriptor(desc_ctx)

    def resolve_tweaks(self, tweaks):
        key = self.fully_qualified_name + '_f'
        f = tweaks[key]
        if isinstance(f, str):
            f = self.operators.get_op_by_name(f)
            tweaks[key] = f

    def __call__(self, input, hpopt_config={}):
        ops = self.operators
        direct_inputs = input[:ops.input_count]

        prefix = self.fully_qualified_name
        f = hpopt_config[prefix + '_f']
        p = hpopt_config[prefix + '_p']
        if not callable(f):
            raise ValueError()

        if bool(hpopt_config.get(self.SYMBOLIC_TWEAK, False)):
            if hasattr(f, '_hg_cgp_func_mark'):
                sym_exp_f = f._hg_cgp_func_mark.sym_exp
                if sym_exp_f is not None:
                    return sym_exp_f(*direct_inputs, p)
            return SymbolicInvocation(f, direct_inputs + [p])

        return f(*direct_inputs, p)


class Tensor2Inputs:
    @staticmethod
    def transform(tensor):
        if np.shape(tensor) != ():
            return list(itertools.chain((tensor, ), tensor.flat))
        return tensor

    @staticmethod
    def range(shape):
        if shape != ():
            return range(1 + np.array(shape).prod())
        return None


class RegularGrid:
    """
    Regular grid pattern factory
    """

    STRUCTURE_VERSION = 1

    def __init__(self, input_range, shape, operators: Operators,
                 output_size=None, backward_length=1, feedback=False, name=None):
        """
        Init the network factory
        :param input_range: A list of keys/indexes to be used to index the input variable. If None then there is a
        single input and no subscript is applied
        :param shape: The shape of the cgp grid
        :param output_size: The number of outputs or None if the graph's output is connected directly to the grid
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
        self.output_size = output_size
        self.operators = operators
        self.backward_length = backward_length
        self.feedback = bool(feedback)
        self.name = name
        self.custom_cells_op_distr = {}   # key is tuple (i, j)

    def set_cell_op_distr(self, pos, distr):
        # TODO param with cell_type? This would be useful for output nodes
        """
        Set a custom operator distribution for the cell identified by position pos
        :param pos: A tuple of the form (i, j)
        :param op: A distribution for the operators (typically a uniform choice) or an operator. In the latter case
        a Constant distribution is created.
        :return:
        """
        if not isinstance(pos, tuple):
            raise ValueError()
        if not isinstance(distr, (tweaks.Distribution, type(None))):
            distr = tweaks.Constant(distr)  # TODO not necessary or when sampling remove instances of this class... see genetic.py
        if len(pos) != 2:
            raise ValueError()
        self.custom_cells_op_distr[pos] = distr

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

    def _create_inputs(self):
        # TODO check that we have at least ops.input_count inputs
        ops = self.operators

        if self.feedback:
            hgg.var('feedback', initial_value=ops.null_value)

        def gen_extra_inputs(real_input_count):
            total_input_count = real_input_count + self.feedback
            extra = [ops.null_value for _ in range(max(0, ops.input_count - total_input_count))]
            if self.feedback:
                extra += [hgg.var('feedback')]
            return extra

        if self.input_range is not None:
            # TODO + [hgg.input_all()]? (if possible)
            return [hgg.input_key(key=k) for k in self.input_range] + gen_extra_inputs(len(self.input_range))
        # we assure that we have at least ops.input_count inputs
        return [hgg.input_all()] + gen_extra_inputs(1)

    def _pad_cell_input(self, v):
        ops = self.operators
        return v + [ops.null_value]*(ops.input_count-len(v))

    def __call__(self) -> hgg.Graph:
        """
        Execute the grid factory. A graph is returned.
        :return: The graph representing the CGP grid.
        """
        ops = self.operators
        cname = self.get_comp_name
        shape = self.shape
        backward_length = self.backward_length

        grid = self.get_grid_coords_list(map(range, shape))
        rows_range = range(shape[0])

        # TODO check grid's input range!

        output = hgg.Graph(name=self.name)
        with output.as_default():
            hgg.SignatureCheck(
                # TODO get signature from operators and input_range
                signature='v={},shape={},bl={},out_sz={},fb={}'.format(*map(str, [self.STRUCTURE_VERSION, shape, backward_length, self.output_size, self.feedback])),
                name='sign')
            inputs = self._create_inputs()

            # iterate through the grid
            for i, j in grid:
                op_distr = self.custom_cells_op_distr.get((i, j))
                # TODO change permutation into combination! We want to allow this configuration: op(node1, node1)
                Cell(operators=ops, op_distr=op_distr,
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
                    output_size = self.output_size
                    output_size = 1 if output_size is None else output_size

                    # connect outputs
                    # TODO different probability for output nodes, assign these to a specific group
                    output1 = [(tweaks.switch(name='out_sw_' + str(out_idx), distribution_group='cgp_output') << connections)
                               for out_idx in range(output_size)]

                    if hasattr(ops, 'output_ops'):
                        output_ops_distr = tweaks.UniformChoice(values=ops.output_ops)
                        output1 = [(Cell(operators=ops, op_distr=output_ops_distr,
                                         name='out_f_' + str(i1)) << self._pad_cell_input([n])) for i1, n in enumerate(output1)]

                    if self.output_size is None:
                        output1 = output1[0]
                    hgg.output() << output1
                    if self.feedback:
                        # TODO use distribution_group='cgp_output' here?
                        set_feedback = hgg.set_var('feedback') << (tweaks.switch(name='feedback_sw') << connections)
                        hgg.add_event_handler('exit', set_feedback)
                else:
                    for i in range(shape[0]):
                        hgg.link(hgg.node_ref(self.get_comp_name('p', i, j)), connections)

        return output
