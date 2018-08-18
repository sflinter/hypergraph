from abc import ABC, abstractmethod
from objectives import MethodFactory as mf


class Tuner(ABC):
    @abstractmethod
    def __init__(self, objective, search_space=None, trials=None, run_config=None, **kwargs):
        self.search_space = search_space
        self.trials = trials
        self.objective = mf.factory(objective)
        self.run_config = run_config

    @abstractmethod
    def run(self):
        pass


class Search(Tuner):
    def __init__(self, objective, **kwargs):
        super().__init__(objective, **kwargs)
        self.objective = objective


class Bayesian(Tuner):
    pass


class Evolutionary(Tuner):
    pass


class Bandit(Bayesian):
    def run(self):
        pass


