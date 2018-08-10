from abc import ABC, abstractmethod
from . import graph as g
from .utils import export
import numpy as np
import copy


@export
class Distribution(ABC):
    def __init__(self, space_descriptor):
        self._space_descriptor = space_descriptor

    @abstractmethod
    def sample(self):
        pass

    @property
    def space_descriptor(self):
        return copy.copy(self._space_descriptor)

    # TODO def sample_from_uniform(v) where v is a value in [0, 1]


@export
class UniformChoice(Distribution):
    # TODO random subset and different probs for each value

    def __init__(self, values=[]):
        self.gen = lambda: np.random.choice(values)
        super().__init__(space_descriptor={'type': 'categorical', 'size': len(values)})

    def sample(self):
        return self.gen()


def _qround(n, q):
    return int(np.round(n/q)*q)


@export
class QUniform(Distribution):
    def __init__(self, low=0, high=16, q=1.0, size=None):
        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError()
        if q == 1.0:
            self.gen = lambda: np.random.randint(low=low, high=high, size=size)
        else:
            self.gen = lambda: int(np.round(np.random.uniform(low=low, high=high, size=size) / q) * q)
        super().__init__(space_descriptor={'type': 'discrete', 'boundaries': (_qround(low, q=q), _qround(high, q=q))})

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
        self.gen = lambda: np.random.uniform(low=low, high=high, size=size)
        super().__init__(space_descriptor={'type': 'continuous', 'boundaries': (low, high)})

    def sample(self):
        return self.gen()


@export
class Normal(Distribution):
    def __init__(self, mean=0, stddev=1.0, size=None):
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

    def __init__(self, default=None, name=None):
        #TODO allow different probabilities for different inputs
        self.default = default
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        if input_binding is None:
            return {}

        if isinstance(input_binding, dict):
            return {self.fully_qualified_name: UniformChoice(input_binding.keys())}

        return {self.fully_qualified_name: QUniform(high=len(input_binding))}

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


@export
def switch(default=None, name=None) -> g.Node:
    return Switch(name=name, default=default)


@export
def tweak(value, default=None, name=None) -> g.Node:
    # TODO declare "level", that is, when the tweak is applied (eg. runtime)
    if isinstance(value, Distribution):
        return Sample(distribution=value, name=name, default=default)
    raise ValueError("Input type not supported")


class GeneticBase:
    """
    Base class for genetic algorithms applied to the Graph structure. This class contains a phenotype
    composed by a dictionary of key:distribution pairs.
    """

    def __init__(self, graph: g.Graph):
        """
        Init the object.
        :param graph: The graph used to initialize the phenotype.
        """
        self.phenotype = self.phenotype = graph.get_hpopt_config_ranges()

    def create_population(self, size) -> list:
        """
        Create and return a population of individuals.
        :param size: The number of individuals in the population
        :return:
        """
        return [dict([(k, d.sample()) for k, d in self.phenotype.items()]) for _ in range(size)]

    def crossover(self, parents):
        """
        Given a number of individuals (considered parents), return a new individual which is the result of the
        crossover between the parents' genes.
        :param parents:
        :return:
        """
        if len(parents) < 2:
            raise ValueError("At least two parents are necessary to crossover")
        phe = self.phenotype
        selectors = np.random.randint(low=0, high=len(parents), size=len(phe))
        child = {}
        for idx, gene_key in zip(selectors, phe.keys()):
            child[gene_key] = parents[idx][gene_key]
        return child

    def mutations(self, individual, prob):
        """
        Apply mutations to the provided individual. Every gene has the same probability of being mutated.
        :param individual:
        :param prob: Gene mutation probability
        :return: The individual
        """
        phe = self.phenotype
        gene_keys = np.array(list(phe.keys()))
        selection = np.where(np.random.uniform(size=len(gene_keys)) < prob)
        gene_keys = gene_keys[selection]
        for key in gene_keys:
            individual[key] = phe[key].sample()
        return individual
