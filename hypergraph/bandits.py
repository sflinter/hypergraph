# Multi-armed bandit-based algorithms

import numpy as np
from .genetic import GeneticBase


class HyperBand:
    def __init__(self, config_ranges, loss, max_resources_per_conf, eta=3):
        """
        Init the hyper band algorithm
        :param config_ranges: Config ranges eg taken from graph.get_hpopt_config_ranges()
        :param loss: A loss function with params (config, status, resources) which evaluates the configuration given
        the resources allocation. The returned value is the tuple (loss_value, status). The status is an internal value used
        by the function loss to store information during successive calls. The initial status (first execution) is None.
        The status parameter is optional.
        :param max_resources_per_conf: The maximum amount of resources allocated to evaluate a configuration
        :param eta: Param that controls the proportion of configurations discarded in each round of successive halving.
        """
        self.config_ranges = dict(config_ranges)
        self.loss = loss
        self.max_resources_per_conf = max_resources_per_conf
        self.eta = eta

    def __call__(self):
        """
        Execute the algorithm
        :return: The best config
        """
        # TODO statistics

        f_eta = float(self.eta)
        s_max = int(np.floor(np.log(self.max_resources_per_conf)/np.log(f_eta)))
        budget = (s_max+1.)*self.max_resources_per_conf
        gene = GeneticBase(self.config_ranges)
        best = (np.inf, None)   # the best observed config, the tuple is of the form (loss_value, config)

        for s in range(s_max, -1, -1):
            n = int(np.ceil((budget/self.max_resources_per_conf)*np.power(f_eta, s)/(s+1)))
            r = self.max_resources_per_conf*np.power(f_eta, -s)
            # begin SuccessiveHalving with (n, r) inner loop
            configs = gene.create_population(size=n)
            for i in range(s+1):
                assert len(configs) > 0
                n_i = int(np.floor(n*np.power(f_eta, -i)))
                r_i = r*np.power(f_eta, i)  # TODO is this supposed to be integer?
                results = [self.loss(config=config, resources=r_i)[0] for config in configs]
                k = int(np.floor(n_i/f_eta))
                if k == 0:
                    break
                # take top k performing configurations indexes (based on loss)
                selection = np.argsort(results)[:k]

                # update the best observed config
                local_best_loss = results[selection[0]]
                if local_best_loss < best[0]:
                    best = (local_best_loss, configs[selection[0]])

                configs = [configs[t] for t in selection]

        return best[1]
