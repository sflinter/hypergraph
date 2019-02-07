import hyperopt as hp
from . import graph as hgg
from . import optimizer as opt
from . import hpopt_adapter as hp_ada


class TreeParzenEstimator(opt.OptimizerBase):
    """
    Tree Parzen estimator strategy. This is based on the Hyperopt implementation.
    """

    def __init__(self, graph_or_config_ranges: hgg.Graph, objective, *, max_evals=100, callbacks=opt.ConsoleLog()):
        """
        Init the TPE strategy.
        :param graph_or_config_ranges: A graph or tweaks configs to be used as phenotype.
        :param objective: An objective function that given a dictionary of tweaks as argument returns a measure of
        performance.
        :param max_evals: The maximum number of evaluations to be performed.
        :param callbacks: A callback or a list of callbacks. The callbacks are instances of the
        class optimizer.Callback.
        """
        self.objective = objective
        self._best = None
        self.tweaks_config = hgg.Graph.copy_tweaks_config(graph_or_config_ranges)
        self.max_evals = max_evals
        super().__init__(callbacks=callbacks)

    def reset(self):
        """
        Reset the optimizer
        :return:
        """
        self._best = None

    @property
    def best(self):     # TODO return Individual
        """
        Return the genetic material of the best individual
        :return:
        """
        p = self._best
        return None if p is None else dict(p.gene)

    def __call__(self):
        """
        Run the optimizer.
        :return:
        """
        self.reset()

        for callback in self.callbacks:
            callback.set_model(self)

        for callback in self.callbacks:
            callback.on_strategy_begin()

        def my_objective(tweaks):
            # TODO
            ret = self.objective(tweaks)
            # return ret
            raise NotImplementedError()

        hp_tweaks = hp_ada.tweaks2hpopt(self.tweaks_config)
        hp.fmin(my_objective, space=hp_tweaks, algo=hp.tpe.suggest, max_evals=self.max_evals)
        raise NotImplementedError()
