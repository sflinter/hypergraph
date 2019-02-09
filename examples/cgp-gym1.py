# An example of a graph simulating a Cartesian Genetic Programming grid (CGP). The program is evolved using
# genetic algorithms. Notably, we use the well-known OpenAI gym as "playground".

import hypergraph as hg
from hypergraph import cgp, tweaks
from hypergraph.optimizer import History, ConsoleLog, ModelCheckpoint
from hypergraph_test import gym_adapter
import matplotlib.pyplot as plt
import gym
import pandas as pd
import time

# **** Begin of config section ****
model_file = None   # file containing the saved model, when provided the evolutionary strategy is not executed
# **** End of config section ****

op = cgp.TensorOperators()      # Instantiate the operators to be used in the nodes of the CGP grid
cgp.DelayOperators(parent=op)   # install "delay" operators in the list of actual operators

env = gym.make('CartPole-v1')   # CartPole-v1, MountainCar-v0, Acrobot-v1
gymman = gym_adapter.GymManager(env, max_steps=250, trials_per_individual=3, action_prob=1.)

# Create a 5x5 CGP grid with backward length 3
grid = cgp.RegularGrid(shape=(5, 5), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=3, name='cgp')
grid = grid()   # finalize the construction of the grid, an instance of the class hg.Graph is returned
# grid.dump()

if model_file is not None:
    with open(model_file, 'rb') as ins:
        model = tweaks.TweaksSerializer.load(ins, graph=grid)
else:
    gen_history = History()
    tpe_history = History()
    # We run two optimization algorithms so that we can compare the performances
    # Run the genetic optimization
    gen_best = hg.optimize(algo='genetic', graph=grid, objective=gymman.create_objective(grid),
                           callbacks=[gen_history, ConsoleLog(), ModelCheckpoint('/tmp/')],
                           generations=10**3, target_score=-250, mutation_prob=0.1,  # these are algo specific params
                           mutation_groups_prob={'cgp_output': 0.6}, lambda_=9)
    # Run the Tree Parzen Estimator optimization
    tpe_best = hg.optimize(algo='tpe', graph=grid, objective=gymman.create_objective(grid),
                           callbacks=[tpe_history, ConsoleLog()],
                           target_score=-250,  max_evals=10**3)   # these are algo specific params
    print("best:" + str(gen_best))

    ax = plt.subplot(1, 2, 1)
    ax.set_title('Genetic algo evolution')
    history = pd.DataFrame(gen_history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
    history.plot(x='gen_idx', y=['best_score', 'population_mean_score'], ax=ax)

    ax = plt.subplot(1, 2, 2)
    ax.set_title('TPE algo evolution')
    history = pd.DataFrame(tpe_history.generations, columns=['gen_idx', 'best_score'])
    history.plot(x='gen_idx', y='best_score', ax=ax)

    plt.show()
    model = gen_best

print('symbolic execution: ' + str(cgp.exec_symbolically(grid, tweaks=model)))

# Test the best model
while True:
    gymman.test(grid, model, speed=1.)
    time.sleep(1)
