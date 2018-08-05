from abc import ABC, abstractmethod
from . import graph as g
from .utils import export
import numpy as np

# TODO optional encoding for category with binary values...


@export
class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    # TODO def sample_from_uniform(v) where v is a value in [0, 1]


@export
class UniformChoice(Distribution):
    def __init__(self, values=[]):
        self.values = values
        self.gen = lambda: np.random.choice(values)

    def sample(self):
        return self.gen()


@export
class UniformInt(Distribution):
    def __init__(self, low=0, high=16, size=None):
        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError()
        self.low = low
        self.high = high
        self.gen = lambda: np.random.randint(low=low, high=high, size=size)

    def sample(self):
        return self.gen()


@export
class Uniform(Distribution):
    def __init__(self, low=.0, high=1.0, size=None):
        self.low = low
        self.high = high
        self.gen = lambda: np.random.uniform(low=low, high=high, size=size)

    def sample(self):
        return self.gen()


@export
class LogUniform(Distribution):
    def __init__(self, low=1e-5, high=1.0, size=None):
        self.low = low
        self.high = high
        self.gen = lambda: np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))

    def sample(self):
        return self.gen()


@export
class IntLogUniform(Distribution):
    def __init__(self, low=1, high=100, size=None):
        self.low = low
        self.high = high
        self.gen = lambda: int(np.round(np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))))

    def sample(self):
        return self.gen()


class Sample(g.Node):
    def __init__(self, distribution: Distribution, default_output=None, name=None):
        if not isinstance(distribution, Distribution):
            raise ValueError()
        self.distribution = distribution
        self.default_output = default_output
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        return {self.fully_qualified_name: self.distribution}

    def __call__(self, input, hpopt_config={}):
        return hpopt_config.get(self.fully_qualified_name, self.default_output)


class Switch(g.Node):
    """
    A node that switches between multiple inputs
    """

    def __init__(self, default_choice=None, name=None):
        #TODO allow different probabilities for different inputs
        self.default_choice = default_choice
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(self)
        if input_binding is None:
            return {}

        if isinstance(input_binding, dict):
            return {self.fully_qualified_name: UniformChoice(input_binding.keys())}

        return {self.fully_qualified_name: UniformInt(high=len(input_binding))}

    def get_input_binding(self, hpopt_config={}):
        choice = hpopt_config.get(self.fully_qualified_name, self.default_choice)
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
def switch(default_choice=None, name=None) -> g.Node:
    return Switch(name=name, default_choice=default_choice)


@export
def tweak(value, name=None, default_output=None) -> g.Node:
    if isinstance(value, Distribution):
        return Sample(distribution=value, name=name, default_output=default_output)
