import numpy as np
from . import graph as hgg
from . import tweaks
from . import optimizer as opt
import itertools
import time

from datetime import datetime


class GeneticBase:
    """
    Base class for genetic algorithms applied to the Graph structure. This class contains a phenotype
    composed by a dictionary of key:distribution pairs.
    """

    def __init__(self, graph: hgg.Graph):
        """
        Init the object.
        :param graph: The graph used to initialize the phenotype.
        """

        self.phenotype = None
        if graph is not None:
            self.init_phenotype(graph)

    def init_phenotype(self, graph: hgg.Graph):
        phenotype = graph.get_hpopt_config_ranges()
        if len(phenotype.keys()) <= 2:
            # raise ValueError("Insufficient number of genes")
            phenotype['_internal_placeholder_77177ce9d789'] = tweaks.Uniform()
        self.phenotype = phenotype

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

    def mutations(self, individual, prob, groups_prob=None):
        """
        Apply mutations to the provided individual. Every gene has the same probability of being mutated.
        :param individual:
        :param prob: Gene mutation probability, this is the default probability, that is the one applied to
        groups that don't have a specific value
        :param groups_prob: A dictionary containing items of the form group_name:prob. This map specifies a custom
        probability for each declared group. The group name matches the Distribution.group property.
        :return: The individual
        """

        phe = self.phenotype

        def get_custom_prob(key_):
            g = phe[key_].group
            return prob if g is None else groups_prob.get(g, prob)

        if isinstance(individual, opt.Individual):
            individual = individual.gene
        individual = dict(individual)

        # select items with distributions
        gene_keys = filter(lambda it: isinstance(it[1], tweaks.Distribution), phe.items())
        gene_keys = map(lambda it: it[0], gene_keys)
        gene_keys = np.array(list(gene_keys))

        if groups_prob is None:
            probs = prob
        else:
            probs = list(map(get_custom_prob, gene_keys))

        selection = np.where(np.random.uniform(size=len(gene_keys)) < probs)
        gene_keys = gene_keys[selection]
        for key in gene_keys:
            individual[key] = phe[key].sample()
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


class MutationOnlyEvoStrategy:
    """
    mu+lambda evolutionary strategy
    """

    def __init__(self, graph: hgg.Graph, fitness, *,
                 opt_mode='max', mutation_prob=(0.1, 0.8), mutation_groups_prob=None,
                 population_size=1, lambda_=4, elitism=1, generations=10**4, target_score=None,
                 selector=TournamentSelection(), callbacks=opt.ConsoleLog()):
        # TODO validate params
        if opt_mode not in ['min', 'max']:
            raise ValueError()

        self.fitness = fitness
        self.opt_mode = opt_mode
        self.mutation_prob = mutation_prob
        self.mutation_groups_prob = None if mutation_groups_prob is None else dict(mutation_groups_prob)
        self.population_size = population_size
        self.lambda_ = lambda_
        self.elitism = elitism
        self.generations = generations
        self.target_score = None if target_score is None else float(target_score)
        self.selector = selector

        self.callbacks = []
        if isinstance(callbacks, opt.Callback):
            self.callbacks.append(callbacks)
        elif callbacks is not None:
            self.callbacks.extend(callbacks)

        self.gene = GeneticBase(graph=graph)

        self.population = None
        self.last_gen_id = -1
        self._best = None

    def reset(self):
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

    def _apply_fitness(self, population):
        fitness = self.fitness
        for p in population:
            p.score = fitness(p.gene)

    def _create_parent_offspring(self, parent, gen_id=None):
        p = self.mutation_prob
        gp = self.mutation_groups_prob
        if isinstance(p, tuple):
            pf = lambda: np.random.uniform(p[0], p[1])
        else:
            pf = lambda: p

        mut = self.gene.mutations
        return [opt.Individual(mut(parent, prob=pf(), groups_prob=gp), gen_id=gen_id) for _ in range(self.lambda_)]

    def __call__(self):
        self.reset()

        for callback in self.callbacks:
            callback.set_model(self)

        population = self.population
        target_score = self.target_score
        selector = self.selector

        def score_cmp_min(a, b):
            return a <= b

        def sort_min(a):
            a = list(a)
            return np.argsort(a)

        def score_cmp_max(a, b):
            return a >= b

        def sort_max(a):
            a = list(a)
            return np.argsort(a)[::-1]

        def apply_perm(lst, idxs):
            return [lst[i] for i in idxs]

        score_cmp = score_cmp_min if self.opt_mode == 'min' else score_cmp_max
        sort_op = sort_min if self.opt_mode == 'min' else sort_max

        def update_best(candidate: opt.Individual):
            if (self._best is None) or score_cmp(candidate.score, self._best.score):
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
            self._apply_fitness(population)
            population = apply_perm(population, sort_op(opt.Individual.get_scores(population)))
            self.population = population
            self.last_gen_id = gen_id

        base_gen_id = self.last_gen_id + 1
        for c in range(base_gen_id, base_gen_id+self.generations):
            start_time = time.monotonic()

            # create new offspring
            offspring = itertools.chain(*[self._create_parent_offspring(parent, gen_id=c) for parent in population])
            offspring = list(offspring)
            self._apply_fitness(offspring)
            offspring.extend(population[:self.elitism])     # preserve the best parents
            offspring = apply_perm(offspring, sort_op(opt.Individual.get_scores(offspring)))

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
                if (target_score is not None) and score_cmp(self._best.score, target_score):
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
