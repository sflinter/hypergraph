import hypergraph as hg
from hypergraph import cgp, cgp_int_op
from hypergraph import utils as hg_utils
from hypergraph.genetic import MutationOnlyEvoStrategy
import numpy as np


def primes_gen(limit):
    table = [True] * limit
    table[0] = table[1] = False

    for (i, isprime) in enumerate(table):
        if isprime:
            yield i
            for n in range(i*i, limit, i):
                table[n] = False


primes = set([n for n in primes_gen(1024)])

op = cgp_int_op.IntOperators()

grid = cgp.RegularGrid(shape=(1, 5), input_range=None, output_size=1,
                       operators=op, backward_length=3, feedback=True, name='cgp')
grid = grid()
grid.dump()

max_sequence_length = 256


def fitness(individual):
    ctx = hg.ExecutionContext(tweaks=individual)
    reward = 0
    unique_primes = set()
    with ctx.as_default():
        for idx in range(max_sequence_length):
            p = np.ravel(np.array(grid(input=int(idx))))[0]
            p = abs(p)
            if p <= 1:  # ignore 0 and 1 but no reward
                continue
            if p not in primes:
                break
            if p in unique_primes:
                continue
            unique_primes.add(p)
            reward += 1
    return reward


strategy = MutationOnlyEvoStrategy(grid, fitness=fitness, generations=100*10**3)
strategy()
print("best:" + str(strategy.best))
