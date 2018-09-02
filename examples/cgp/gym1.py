import hypergraph as hg
from hypergraph import cgp
from hypergraph.genetic import MutationOnlyEvoStrategy
from hypergraph_test import gym_adapter
import gym
import pandas as pd
import matplotlib.pyplot as plt

op = cgp.TensorOperators()

env = gym.make('CartPole-v0')
gymman = gym_adapter.GymManager(env, max_steps=250)

grid = cgp.RegularGrid(shape=(3, 3), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=2)
grid = grid()
grid.dump()

strategy = MutationOnlyEvoStrategy(grid, fitness=gymman.create_fitness(grid), generations=10*10**3)
strategy()
print("best:" + str(strategy.best))

history = pd.DataFrame(strategy.history.generations)
#history.set_index('idx')
history.plot(x='idx', y='best_score')
plt.show()
print(history)
