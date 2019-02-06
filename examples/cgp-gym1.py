# An example of a graph simulating a Cartesian Genetic Programming grid (CGP). The program is evolved using
# genetic algorithms. Notably, we use the well-known OpenAI gym as "playground".

import hypergraph as hg
from hypergraph import cgp, tweaks
from hypergraph.genetic import MutationOnlyEvoStrategy
from hypergraph.optimizer import History, ConsoleLog, ModelCheckpoint
from hypergraph_test import gym_adapter
import gym
import pandas as pd
import time

# **** Begin of config section ****
graphics_enabled = True
model_file = None   # file containing the saved model, when provided the evolutionary strategy is not executed
# **** End of config section ****

op = cgp.TensorOperators()      # Instantiate the operators to be used in the nodes of the CGP grid
cgp.DelayOperators(parent=op)   # install "delay" operators in the list of actual operators

env = gym.make('CartPole-v1')   # CartPole-v1, MountainCar-v0, Acrobot-v1
gymman = gym_adapter.GymManager(env, max_steps=250, trials_per_individual=3, action_prob=1.)

grid = cgp.RegularGrid(shape=(5, 5), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=3, name='cgp')
# TODO fix error when row_count*backward_length<2

grid = grid()   # finalize the construction of the grid, an instance of the class hg.Graph is returned
# grid.dump()

if model_file is not None:
    with open(model_file, 'rb') as ins:
        model = tweaks.TweaksSerializer.load(ins, graph=grid)
else:
    history = History()
    best = hg.optimize(algo='genetic', graph=grid, fitness=gymman.create_fitness(grid),
                       generations=10**3, target_score=250, mutation_prob=0.1,  # these are algo specific params
                       mutation_groups_prob={'cgp_output': 0.6}, lambda_=9,
                       callbacks=[history, ConsoleLog(), ModelCheckpoint('/tmp/')])
    print("best:" + str(best))

    history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
    if graphics_enabled:
        import matplotlib.pyplot as plt
        history.plot(x='gen_idx', y=['best_score', 'population_mean_score'])
        plt.show()
    model = best

print('symbolic execution: ' + str(cgp.exec_symbolically(grid, tweaks=model)))

# Test the best model
if graphics_enabled:
    while True:
        gymman.test(grid, model, speed=1.)
        time.sleep(1)
