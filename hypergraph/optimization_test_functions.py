import numpy as np
import abc
from . import tweaks


class SingleObjectiveFunction(abc.ABC):
    def __init__(self):
        self.dim = None
        self.domain = None
        self.global_minima = None


class RastriginFunction(SingleObjectiveFunction):
    def __init__(self, dim=2):
        self.dim = dim
        self.domain = [tweaks.Uniform(low=-5.12, high=5.12) for _ in range(dim)]
        self.global_minima = [np.zeros(dim)]

    def __call__(self, x):
        a = 10
        return a*self.dim + np.sum(x**2-a*np.cos(2*np.pi*x))


class SphereFunction(SingleObjectiveFunction):
    def __init__(self, dim=2):
        self.dim = dim
        self.domain = [tweaks.Uniform(low=-5, high=5) for _ in range(dim)]
        self.global_minima = [np.zeros(dim)]

    def __call__(self, x):
        return np.sum(x**2)


class EasonFunction(SingleObjectiveFunction):
    def __init__(self):
        self.dim = 2
        self.domain = [tweaks.Uniform(low=-100, high=100) for _ in range(2)]
        self.global_minima = [np.full(2, np.pi)]

    def __call__(self, x):
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-(np.square(x[0]-np.pi)+np.square(x[1]-np.pi)))
