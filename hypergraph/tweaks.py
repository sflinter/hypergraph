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
        self.phenotype = graph.get_hpopt_config_ranges()
        if len(self.phenotype.keys()) <= 2:
            raise ValueError("Insufficient number of genes")

    def create_population(self, size=None) -> list:
        """
        Create and return a population of individuals.
        :param size: The number of individuals in the population
        :return:
        """
        sample_f = lambda: dict([(k, d.sample()) for k, d in self.phenotype.items()])
        if size is None:
            return sample_f()
        return [sample_f() for _ in range(size)]

    def _select_genes_from_parents(self, parents, selectors):
        return dict([(gene_key, parents[idx][gene_key]) for idx, gene_key in zip(selectors, self.phenotype.keys())])

    def crossover_uniform_multi_parents(self, parents) -> dict:
        """
        Given a number of individuals (considered parents), return a new individual which is the result of the
        crossover between the parents' genes.
        :param parents:
        :return: The created individual
        """
        if len(parents) < 2:
            raise ValueError("At least two parents are necessary to crossover")
        return self._select_genes_from_parents(parents,
                                               np.random.randint(low=0, high=len(parents), size=len(self.phenotype)))

    def crossover_uniform(self, parents):
        """
        Given two parents, return two new individuals. These are the result of the
        crossover between the parents' genes.
        :param parents:
        :return: A list with two individuals
        """
        if len(parents) != 2:
            raise ValueError("Two parents are necessary to crossover")
        phe = self.phenotype
        selectors = np.random.randint(low=0, high=2, size=len(phe))
        f = self._select_genes_from_parents
        return [f(parents, selectors), f(parents, -selectors+1)]

    @staticmethod
    def _select_genes_by_name(source, keys):
        return [(k, source[k]) for k in keys]

    def crossover_1point(self, parents):
        """
        Given two parents, return two new individuals. These are the result of the 1-point
        crossover between the parents' genes.
        :param parents:
        :return: A list with two individuals
        """

        # TODO k-point cross over
        keys = list(self.phenotype.keys())
        assert len(keys) > 2
        k = np.random.randint(low=1, high=len(keys)-1)

        keys = (keys[:k], keys[k:])
        f = self._select_genes_by_name
        return [
            dict(f(parents[0], keys[0]) + f(parents[1], keys[1])),
            dict(f(parents[1], keys[0]) + f(parents[0], keys[1]))
        ]

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
