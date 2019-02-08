import hyperopt
from . import graph as hgg
from . import optimizer as opt
from . import hpopt_adapter as hp_ada
import time
from datetime import datetime
import numpy as np
import copy


class TreeParzenEstimator(opt.OptimizerBase):
    """
    Tree Parzen estimator strategy. This is based on the Hyperopt implementation.
    """

    def __init__(self, graph_or_config_ranges: hgg.Graph, objective, *, max_evals=100, callbacks=opt.ConsoleLog(),
                 target_score=None):
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
        self.target_score = target_score
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

        class MyObjective:
            def __init__(self, objective, callbacks):
                self.objective = objective
                self.callbacks = callbacks

                self.idx = 0
                self.best_score = np.inf
                self.best = None

            def __call__(self, tweaks):
                start_time = time.monotonic()
                tweaks = hp_ada.expand_hpopt_adapters(tweaks)
                ret = self.objective(tweaks)
                hit = False
                if ret < self.best_score:
                    hit = True
                    self.best_score = ret
                    self.best = copy.copy(tweaks)
                # TODO use target_score

                rec = {'gen_idx': self.idx,
                       'gen_time': time.monotonic() - start_time,
                       'datetime': datetime.utcnow(),
                       'score': ret,
                       'best_score': self.best_score,
                       'hit': hit}
                for callback in self.callbacks:
                    callback.on_gen_end(rec)

                return {'loss': ret, 'status': hyperopt.STATUS_OK}

        hp_tweaks = hp_ada.tweaks2hpopt(self.tweaks_config)
        my_objective = MyObjective(objective=self.objective, callbacks=self.callbacks)
        hyperopt.fmin(my_objective, space=hp_tweaks, algo=hyperopt.tpe.suggest, max_evals=self.max_evals)
        self._best = my_objective.best
