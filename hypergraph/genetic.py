# Genetic algorithms for the Hypergraph platform.

import numpy as np
from . import graph as hgg
from . import tweaks
from . import optimizer as opt
import itertools
import time
from datetime import datetime


class GeneticBase:
    """
    Basic routines for genetic algorithms. This class contains a phenotype composed by a dictionary
    of <key>:<distribution> pairs.
    """

    def __init__(self, graph: [hgg.Graph, dict]):
        """
        Init the genetic basic routines by getting a phenotype from either a graph or a dictionary.
        :param graph: The graph used to initialize the phenotype.
        """

        self.phenotype = None
        if graph is not None:
            self.init_phenotype(graph)

    def init_phenotype(self, graph_or_config_ranges: [hgg.Graph, dict]):
        self.phenotype = hgg.Graph.copy_tweaks_config(graph_or_config_ranges)

    @staticmethod
    def _sample_distr_tweak(d):
        if isinstance(d, tweaks.Distribution):
            return d.sample()
        return d

    def create_population(self, size=None) -> list:
        """
        Create and return a population of individuals.
        :param size: The number of individuals in the population
        :return:
        """
        sample_f = lambda: dict([(k, self._sample_distr_tweak(d)) for k, d in self.phenotype.items()])
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
        :param individual: The individual as instance of class Individual or a dictionary containing its tweaks
        :param prob: Gene mutation probability. If this parameter is a callable then it has the form
        func(keys, size=None) and it is used to specify a custom probability for each gene.
        :return: The mutated individual as a dict of tweaks
        """

        # *** groups_prob removed ***
        # if callable(prob) and groups_prob is not None:
        #    raise ValueError('callable prob and groups probabilities are mutual exclusive')

        phe = self.phenotype

        # def get_custom_prob(key_):
        #    g = phe[key_].group
        #    return prob if g is None else groups_prob.get(g, prob)

        if isinstance(individual, opt.Individual):
            individual = individual.gene
        individual = dict(individual)

        def is_distr_not_aggr(item):
            if isinstance(item[1], tweaks.Aggregation):
                return False
            return isinstance(item[1], tweaks.Distribution)

        # select items with distributions
        # gene_keys = filter(lambda it: isinstance(it[1], tweaks.Distribution), phe.items())
        gene_keys = filter(is_distr_not_aggr, phe.items())
        gene_keys = map(lambda it: it[0], gene_keys)
        gene_keys = np.array(list(gene_keys))

        if callable(prob):
            # prob is callable, so we get specific probabilities by key
            prob = prob(gene_keys)

        probs = prob
        # if groups_prob is None:
        #    probs = prob
        # else:
        #    probs = list(map(get_custom_prob, gene_keys))

        selection = np.where(np.random.uniform(size=len(gene_keys)) < probs)
        gene_keys = gene_keys[selection]
        for key in gene_keys:
            individual[key] = phe[key].sample()

        # special handling for tweaks of type Aggregation
        gene_keys = filter(lambda it: isinstance(it[1], tweaks.Aggregation), phe.items())
        gene_keys = map(lambda it: it[0], gene_keys)
        gene_keys = np.array(list(gene_keys))

        probs = itertools.repeat(prob)
        # if groups_prob is None:
        #    probs = itertools.repeat(prob)
        # else:
        #    probs = list(map(get_custom_prob, gene_keys))

        for key, p in zip(gene_keys, probs):
            aggr = phe[key]
            if callable(p):
                p = p((key, ), size=aggr.size)[0]
            individual[key] = aggr.mutation(current_value=individual[key], prob=p)

        return individual


class TournamentSelection:
    def __init__(self, k=4, p=0.95):
        """
        Init tournament selection algorithm
        :param k: the arity of the selection
        :param p: the probability of the selection
        """

        if not isinstance(k, int):
            raise ValueError()
        if k <= 0:
            raise ValueError()
        self.k = k

        p = float(p)
        if p <= 0. or p > 1.:
            raise ValueError()
        self.p = p

    def select(self, fitness_ordered_pop):
        """
        Perform a tournament selection on the provided population. The population must be ordered by decreasing
        score (thus the best individual is the first).
        :param fitness_ordered_pop:
        :return: The index of the selected individual
        """
        if len(fitness_ordered_pop) <= self.k:
            return 0
        if isinstance(fitness_ordered_pop[0], opt.Individual):
            fitness_ordered_pop = list(opt.Individual.get_genes(fitness_ordered_pop))

        idxs_subset = sorted([np.random.randint(len(fitness_ordered_pop)) for _ in range(self.k)])
        # generate k random uniform numbers on [0,1] and check which is less than p, hence
        # return the index of the first.
        idx = np.argmax(np.random.uniform(0, 1, self.k) < self.p)   # idx relative to idxs_subset
        return idxs_subset[idx]


class MutationOnlyEvoStrategy(opt.OptimizerBase):
    """
    mu+lambda evolutionary strategy.
    """

    # TODO pass a 'scheduler' (eg. hyperband) and use it to assign the resources to each evaluation of the objective
    def __init__(self, graph_or_config_ranges: hgg.Graph, objective, *,
                 mutation_prob=(0.1, 0.8),
                 population_size=1, lambda_=4, elitism=1, generations=10**4, target_score=None,
                 selector=TournamentSelection(), callbacks=opt.ConsoleLog()):
        """
        Init the mu+lambda evolutionary strategy.
        :param graph_or_config_ranges: A graph or tweaks configs to be used as phenotype.
        :param objective: A function that given a dictionary of tweaks as argument returns a measure to be
        minimized.
        :param mutation_prob: The probability of having a mutation. If a tuple containing two probabilities is provided
        then these are considered the minimum and maximum probability. In this case the mutation probability is randomly
        selected within the specified interval each time we have a new mutation.
        :param population_size: The size of the population.
        :param lambda_: The number of children that each parent generates.
        :param elitism: The genetic algorithms elitism.
        :param generations: The number of generations to be evolved.
        :param target_score: If specified, when the objective value reaches this value then this determines
        a stop condition.
        :param selector: The selector that performs the selection of the individuals that pass to the next generation
        for successive reproduction.
        :param callbacks: A callback or a list of callbacks. The callbacks are instances of the
        class optimizer.Callback.
        """
        self.objective = objective
        self.mutation_prob = mutation_prob
        # self.mutation_groups_prob = None if mutation_groups_prob is None else dict(mutation_groups_prob)
        self.population_size = population_size
        self.lambda_ = lambda_
        self.elitism = elitism
        self.generations = generations
        self.target_score = None if target_score is None else float(target_score)
        self.selector = selector

        self.gene = GeneticBase(graph_or_config_ranges)

        self.population = None
        self.last_gen_id = -1
        self._best = None

        super().__init__(callbacks=callbacks)

    def reset(self):
        """
        Reset the optimizer
        :return:
        """
        self.population = None
        self.last_gen_id = -1
        self._best = None

    @property
    def best(self):     # TODO return Individual
        """
        Return the genetic material of the best individual
        :return:
        """
        p = self._best
        return None if p is None else dict(p.gene)

    def _apply_objective(self, population):
        """
        Given a population compute the objective for each individual.
        :param population:
        :return:
        """
        objective = self.objective
        for p in population:
            p.score = objective(p.gene)

    def _create_parent_offspring(self, parent, gen_id=None):
        """
        Given a parent generate the offspring through mutation only.
        :param parent: The individual to be used as parent.
        :param gen_id: The generation identifier.
        :return: A list of newly generated individuals.
        """

        p = self.mutation_prob
        if isinstance(p, (tuple, list)):
            def prob_gen(keys, size=None):
                if size is None:
                    size = (len(keys), )
                else:
                    size = (len(keys), ) + size
                return np.random.uniform(low=p[0], high=p[1], size=size)
            pf = prob_gen
        else:
            pf = p

        mut = self.gene.mutations

        max_trials = 1000
        for t in range(max_trials):     # check uniqueness of the children
            ret = [opt.Individual(mut(parent, prob=pf), gen_id=gen_id) for _ in range(self.lambda_)]
            # TODO if len(set([frozenset(k.gene.items()) for k in ret])) != self.lambda_:
            # TODO    continue
            # Problem here, some values may be unhashable
            return ret
        raise RuntimeError()

    def __call__(self):
        """
        Run the optimizer.
        :return:
        """
        self.reset()

        for callback in self.callbacks:
            callback.set_model(self)

        population = self.population
        target_score = self.target_score
        selector = self.selector

        def sort_min(a):
            a = list(a)
            return np.argsort(a)

        def apply_perm(lst, idxs):
            return [lst[i] for i in idxs]

        def update_best(candidate: opt.Individual):
            if (self._best is None) or (candidate.score <= self._best.score):
                self._best = candidate.copy()
                return True
            return False

        for callback in self.callbacks:
            callback.on_strategy_begin()

        # create initial population
        if population is None:
            # TODO if we restart then we need the id of the previous generation
            gen_id = self.last_gen_id + 1
            population = [opt.Individual(p, gen_id=gen_id) for p in self.gene.create_population(size=self.population_size)]
            self._apply_objective(population)
            population = apply_perm(population, sort_min(opt.Individual.get_scores(population)))
            self.population = population
            self.last_gen_id = gen_id

        base_gen_id = self.last_gen_id + 1
        for c in range(base_gen_id, base_gen_id+self.generations):
            start_time = time.monotonic()

            # create new offspring
            offspring = itertools.chain(*[self._create_parent_offspring(parent, gen_id=c) for parent in population])
            offspring = list(offspring)
            self._apply_objective(offspring)
            offspring.extend(population[:self.elitism])     # preserve the best parents
            offspring = apply_perm(offspring, sort_min(opt.Individual.get_scores(offspring)))

            hit = update_best(offspring[0])     # update the best ever

            # apply selection strategy and create a new population
            selection = set()
            while len(selection) != self.population_size:
                selection.add(selector.select(offspring))
                # TODO in certain cases we may have an infinite loop...
            population = apply_perm(offspring, sorted(selection))
            self.population = population
            self.last_gen_id = c
            # TODO check order
            del offspring

            target_achieved = False
            if hit:
                if (target_score is not None) and (self._best.score <= target_score):
                    target_achieved = True

            if len(self.callbacks):
                population_scores = list(opt.Individual.get_scores(population))
                rec = {'gen_idx': c,
                       'gen_time': time.monotonic() - start_time,
                       'datetime': datetime.utcnow(),
                       'best_score': self._best.score,
                       'population_scores': population_scores,
                       'population_mean_score': np.mean(population_scores),
                       'hit': hit}
                for callback in self.callbacks:
                    callback.on_gen_end(rec)
            if target_achieved:
                break
