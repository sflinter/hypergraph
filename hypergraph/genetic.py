import numpy as np
from . import graph as hgg
from . import tweaks


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
        phenotype = graph.get_hpopt_config_ranges()
        if len(phenotype.keys()) <= 2:
            # raise ValueError("Insufficient number of genes")
            phenotype['_internal_placeholder_77177ce9d789'] = tweaks.Uniform()
        self.phenotype = phenotype

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

        individual = dict(individual)
        phe = self.phenotype
        gene_keys = np.array(list(phe.keys()))
        selection = np.where(np.random.uniform(size=len(gene_keys)) < prob)
        gene_keys = gene_keys[selection]
        for key in gene_keys:
            individual[key] = phe[key].sample()
        return individual


class History:  # TODO callback
    def __init__(self):
        self.generations = []


class MutationOnlyEvoStrategy(GeneticBase):
    """
    1+lambda evolutionary strategy
    """

    def __init__(self, graph: hgg.Graph, fitness, *, opt_mode='max', lambda_=4, generations=10**4):
        # TODO callback
        # TODO validate params
        if opt_mode not in ['min', 'max']:
            raise ValueError()

        self.fitness = fitness
        self.opt_mode = opt_mode
        self.lambda_ = lambda_
        self.generations = generations
        super().__init__(graph=graph)
        self.parent = None
        self.parent_score = None
        self.history = None

    def reset(self):
        self.parent = None
        self.parent_score = None
        self.history = History()

    @property
    def best(self):
        """
        The best individual
        :return:
        """
        p = self.parent
        return None if p is None else dict(p)

    def __call__(self):
        self.reset()

        parent = self.parent
        parent_score = self.parent_score
        fitness = self.fitness
        opt_mode = self.opt_mode

        def score_cmp_min(a, b):
            return a <= b

        def score_cmp_max(a, b):
            return a >= b

        score_cmp = score_cmp_min if opt_mode == 'min' else score_cmp_max

        if parent is None:
            parent = self.create_population()
        if parent_score is None:
            parent_score = fitness(parent)
            if parent_score is None:
                raise ValueError()

        for c in range(self.generations):
            hit = 0
            offspring = [self.mutations(parent, prob=np.random.uniform()) for _ in range(self.lambda_)]
            for child in offspring:
                score = fitness(child)
                if score is None:
                    break
                # TODO define score_thr, when achieved, stop
                if score_cmp(score, parent_score):
                    parent = child
                    parent_score = score
                    hit = 1

            self.parent = parent
            self.parent_score = parent_score
            if hit:
                # TODO move to a specific callback
                self.history.generations.append({'idx': c, 'best_score': parent_score}) # TODO include datetime

            if c % 100 == 0:
                # TODO move to a specific callback
                print("**** **** ****")
                print("best: "+str(self.parent))
                print("best_score: "+str(self.parent_score)+", generation: "+str(c))
