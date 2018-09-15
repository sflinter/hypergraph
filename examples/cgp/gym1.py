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
gymman = gym_adapter.GymManager(env, max_steps=250, trials_per_individual=3, action_prob=1.)

grid = cgp.RegularGrid(shape=(5, 5), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=3, feedback=True, name='cgp')
# TODO fix error when row_count*backward_length<2

grid = grid()
grid.dump()

strategy = MutationOnlyEvoStrategy(grid, fitness=gymman.create_fitness(grid), generations=100) # 10*10**3
strategy()
print("best:" + str(strategy.best))

history = pd.DataFrame(strategy.history.generations)
# history.set_index('idx')
history.plot(x='idx', y='best_score')
plt.show()
print(history)

print(cgp.exec_symbolically(grid, tweaks=strategy.best))

while True:
    gymman.test(grid, strategy.best, speed=0.5)
    time.sleep(1)
