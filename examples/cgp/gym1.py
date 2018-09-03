import hypergraph as hg
from hypergraph import cgp
from hypergraph.genetic import MutationOnlyEvoStrategy
from hypergraph_test import gym_adapter
import gym
import pandas as pd
import matplotlib.pyplot as plt
import time

op = cgp.TensorOperators()

env = gym.make('CartPole-v1')
gymman = gym_adapter.GymManager(env, max_steps=250, trials_per_individual=3, action_prob=0.5)

grid = cgp.RegularGrid(shape=(3, 3), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=2)
# TODO investigate, shape: (1, 10) backward: 1; Note: row_count*backward_length>=2 otherwise not enough connections...

grid = grid()
grid.dump()

strategy = MutationOnlyEvoStrategy(grid, fitness=gymman.create_fitness(grid), generations=10*10**3, mutation_prob=0.2)
strategy()
print("best:" + str(strategy.best))

history = pd.DataFrame(strategy.history.generations)
#history.set_index('idx')
history.plot(x='idx', y='best_score')
plt.show()
print(history)

while True:
    gymman.test(grid, strategy.best, speed=0.5)
    time.sleep(1)
