from abc import ABC, abstractmethod
from . import graph as g
from .utils import export, MsgPackEncoders
import numpy as np
import copy
import math
import msgpack


@export
class Distribution(ABC):
    def __init__(self, space_descriptor):
        self._space_descriptor = space_descriptor
        # The group property is used to assemble different sets of distributions. This is used for example
        # in cgp to assign different probabilities to different groups of distributions.
        self.group = None

    @abstractmethod
    def sample(self):
        pass

    @property
    def space_descriptor(self):
        return copy.copy(self._space_descriptor)

    # TODO def sample_from_uniform(v) where v is a value in [0, 1]
    # TODO operator 'in'


# TODO g.Graph.adapters[FuncTweaksDecl] = None


@export
def decl_tweaks(**key_tweak_pairs):
    """
    A decorator to be used to associate tweaks to a function declaration. Also the function is 'marked' as graph's node
    :param key_tweak_pairs:
    :return:
    """
    def real_decorator(function):
        if isinstance(function, g.FuncTweaksDecl):
            wrapper = function
        else:
            wrapper = g.FuncTweaksDecl(function=function, prefix=function.__name__)
        wrapper.tweaks_config.update(key_tweak_pairs)
        return wrapper
    return real_decorator


class Aggregation(Distribution):
    """
    An aggregation of independent random variables of the same type
    """

    def __init__(self, base: Distribution, size):
        if isinstance(base, Aggregation):
            raise ValueError('The base distribution of an Aggregation cannot be an Aggregation itself')
        self.base = base
        self.size = size
        super().__init__(space_descriptor={'type': base.space_descriptor['type'], 'size': size})

    def sample(self):
        count = np.product(self.size)
        base = self.base
        output = [base.sample() for _ in range(count)]
        return np.reshape(output, self.size)

    def mutation(self, current_value, prob):
        """
        A specific helper for genetic algorithms. A subset of the aggregated variables is selected with
        probability prob and mutation is applied by resampling the generated values from the base distribution.
        :param current_value:
        :param prob:
        :return:
        """
        selection = np.where(np.random.uniform(size=self.size) < prob)
        current_value[selection] = self.base.sample_multiple(size=(len(selection[0]), ))
        return current_value


@export
class Constant(Distribution):
    def __init__(self, value):
        self.value = value
        super().__init__(space_descriptor={'type': 'categorical', 'size': 1})

    def sample(self):
        return self.value


@export
class UniformChoice(Distribution):
    # TODO random subset and different probs for each value

    def __init__(self, values=(), group=None):
        values = list(values)
        self.values = values
        self.gen = lambda: np.random.choice(values)
        super().__init__(space_descriptor={'type': 'categorical', 'size': len(values)})
        self.group = group

    def sample(self):
        return self.gen()

    def sample_multiple(self, size):
        return np.random.choice(self.values, size=size)


@export
class UniformPermutation(Distribution):
    def __init__(self, values, k=None):
        """
        Random permutation of elements from parameter values into groups of size k.
        :param values:
        :param k: The number of elements of a permutation. If k is None the k is assumed len(values).
        """

        values = list(values)
        n = len(values)
        if k is None:
            k = n
        if not isinstance(k, int):
            raise ValueError()
        if k > n:
            raise ValueError()

        self.k = k
        self.values = values
        size = math.factorial(n) // math.factorial(n-k)
        super().__init__(space_descriptor={'type': 'categorical', 'size': size})

    def sample(self):
        values = self.values
        if self.k == len(values):
            return list(np.random.permutation(values))

        idxs = np.random.permutation(len(values))[:self.k]
        return [values[idx] for idx in idxs]


def _qround(n, q):
    return int(np.round(n/q)*q)


@export
class QUniform(Distribution):
    def __init__(self, low=0, high=16, q=1.0, size=None, group=None):
        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError()

        self.range = (low, high)
        self.q = q
        self.size = size

        if q == 1.0:
            self.gen = lambda: np.random.randint(low=low, high=high, size=size)
        else:
            self.gen = lambda: int(np.round(np.random.uniform(low=low, high=high, size=size) / q) * q)
        super().__init__(space_descriptor={'type': 'discrete', 'boundaries': (_qround(low, q=q), _qround(high, q=q))})
        self.group = group

    def sample(self):
        return self.gen()


@export
class QNormal(Distribution):
    def __init__(self, mean=0, stddev=1.0, q=1.0, size=None):
        self.gen = lambda: int(np.round(np.random.normal(loc=mean, scale=stddev, size=size) / q) * q)
        super().__init__(space_descriptor={'type': 'discrete', 'boundaries': (-np.inf, np.inf)})

    def sample(self):
        return self.gen()


@export
class Uniform(Distribution):
    def __init__(self, low=.0, high=1.0, size=None):
        self.range = (low, high)
        self.size = size
        self.gen = lambda: np.random.uniform(low=low, high=high, size=size)
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (low, high)})

    def sample(self):
        return self.gen()

    def sample_multiple(self, size):
        r = self.range
        return np.random.uniform(low=r[0], high=r[1], size=size)


@export
class Normal(Distribution):
    def __init__(self, mean=0, stddev=1.0, size=None):
        self.mean = mean
        self.stddev = stddev
        self.gen = lambda: np.random.normal(loc=mean, scale=stddev, size=size)
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (-np.inf, np.inf)})

    def sample(self):
        return self.gen()


@export
class Poisson(Distribution):
    def __init__(self, lam=1.0, size=None):
        self.gen = lambda: np.random.poisson(lam=lam, size=size)
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (0, np.inf)})

    def sample(self):
        return self.gen()


@export
class LogUniform(Distribution):
    def __init__(self, low=np.finfo(float).tiny, high=1.0, size=None):
        # TODO validate params
        self.gen = lambda: np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (low, high)})

    def sample(self):
        return self.gen()


@export
class LogNormal(Distribution):
    def __init__(self, mean=0, stddev=1.0, size=None):
        self.gen = lambda: np.exp(np.random.normal(loc=mean, scale=stddev, size=size))
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (-np.inf, np.inf)})

    def sample(self):
        return self.gen()


@export
class QLogUniform(Distribution):
    def __init__(self, low=1, high=100, q=1.0, size=None):
        self.gen = lambda: int(np.round(np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))/q)*q)
        super().__init__(space_descriptor={'type': 'discrete', 'boundaries': (_qround(low, q=q), _qround(high, q=q))})

    def sample(self):
        return self.gen()


class Sample(g.Node):
    def __init__(self, distribution: Distribution, default=None, name=None):
        if not isinstance(distribution, Distribution):
            raise ValueError()
        self.distribution = distribution
        self.default = default
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        return {self.fully_qualified_name: self.distribution}

    def __call__(self, input, hpopt_config={}):
        return hpopt_config.get(self.fully_qualified_name, self.default)


class Switch(g.Node):
    """
    A node that switches between multiple inputs
    """

    def __init__(self, default=None, name=None, distribution_group=None):
        #TODO allow different probabilities for different inputs
        self.default = default
        self.distribution_group = distribution_group
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        if input_binding is None:
            return {}

        if isinstance(input_binding, dict):
            return {self.fully_qualified_name: UniformChoice(input_binding.keys(), group=self.distribution_group)}

        return {self.fully_qualified_name: QUniform(high=len(input_binding), group=self.distribution_group)}

    def get_input_binding(self, hpopt_config={}):
        choice = hpopt_config.get(self.fully_qualified_name, self.default)
        if choice is None:
            return None

        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        assert input_binding is not None
        return input_binding[choice]

    def __call__(self, input, hpopt_config={}):
        # the selection is performed in the get_input_binding so here we simply return the input
        return input


class Permutation(g.Node):
    def __init__(self, size=None, name=None):
        if size is not None and size <= 0:
            raise ValueError()

        self.size = size
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        if input_binding is None:
            return {}

        if isinstance(input_binding, dict):
            return {self.fully_qualified_name: UniformPermutation(k=self.size, values=input_binding.keys())}

        return {self.fully_qualified_name: UniformPermutation(k=self.size, values=range(len(input_binding)))}

    def get_input_binding(self, hpopt_config={}):
        selection = hpopt_config.get(self.fully_qualified_name)
        if selection is None:
            return None

        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        assert input_binding is not None
        return [input_binding[key] for key in selection]

    def __call__(self, input, hpopt_config={}):
        # the selection is performed in the get_input_binding so here we simply return the input
        return input

@export
def switch(*, default=None, name=None, distribution_group=None) -> g.Node:
    return Switch(name=name, default=default, distribution_group=distribution_group)
# TODO switch_call

@export
def permutation(*, size=None, name=None) -> g.Node:
    return Permutation(size=size, name=name)


@export
def tweak(value, *, default=None, name=None) -> g.Node:
    # TODO declare "level", that is, when the tweak is applied (eg. runtime)
    if isinstance(value, Distribution):
        return Sample(distribution=value, name=name, default=default)
    raise ValueError("Input type not supported")


class TweaksSerializer:
    @staticmethod
    def save(obj, stream):
        if not isinstance(obj, dict):
            raise ValueError()

        output = {}
        for key, value in obj.items():
            if hasattr(value, '_hg_tweak_descriptor'):
                value = getattr(value, '_hg_tweak_descriptor')
            output[key] = value

        msgpack.pack(output, stream, use_bin_type=True, default=MsgPackEncoders.encode)

    @staticmethod
    def load(stream, graph=None):
        tweaks = msgpack.unpack(stream, object_hook=MsgPackEncoders.decode, raw=False)
        if graph is not None:
            graph.resolve_tweaks(tweaks)
        return tweaks
