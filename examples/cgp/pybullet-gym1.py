import hypergraph as hg
from hypergraph import cgp, tweaks
from hypergraph.genetic import MutationOnlyEvoStrategy, History, ConsoleLog, ModelCheckpoint
from hypergraph_test import gym_adapter
import gym
import tempfile
import os
import uuid
import pandas as pd
import time

import pybullet as pb
pb.connect(pb.DIRECT)
import pybullet_envs


# **** Begin of config section ****
mode_delay = True   # when delay operators enabled feedback is disabled
graphics_enabled = True
model_file = None   # file containing the saved model, when provided the evolutionary strategy is not executed
# **** End of config section ****

op = cgp.TensorOperators()
if mode_delay:
    cgp.DelayOperators(parent=op)
# TODO try to reduce the set of op

env = gym.make('HopperBulletEnv-v0')
gymman = gym_adapter.GymManager(env, max_steps=2000, trials_per_individual=3, action_prob=1.)

grid = cgp.RegularGrid(shape=(40, 5), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=5, feedback=not mode_delay, name='cgp')
# TODO fix error when row_count*backward_length<2
if mode_delay:
    # We force the cell (1, 1) to a custom distribution of delay operators only
    #grid.set_cell_op_distr((1, 1), tweaks.UniformChoice([op.op_delay1, op.op_delay2]))
    #grid.set_cell_op_distr((1, 1), op.op_delay1)
    pass

grid = grid()
# grid.dump()

if model_file is not None:
    with open(model_file, 'rb') as ins:
        model = tweaks.TweaksSerializer.load(ins, graph=grid)
else:
    history = History()
    strategy = MutationOnlyEvoStrategy(grid, fitness=gymman.create_fitness(grid), generations=10*10**3,
                                       target_score=2000, mutation_prob=0.1, mutation_groups_prob={'cgp_output': 0.6},
                                       lambda_=9, callbacks=[history, ConsoleLog(), ModelCheckpoint('/tmp/')])
    strategy()
    print("best:" + str(strategy.best))

    history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
    # history.set_index('idx')
    if graphics_enabled:
        import matplotlib.pyplot as plt
        history.plot(x='gen_idx', y=['best_score', 'population_mean_score'])
        plt.show()
    # print(history)
    model = strategy.best

print('symbolic execution: ' + str(cgp.exec_symbolically(grid, tweaks=model)))

if graphics_enabled:
    while True:
        gymman.test(grid, model, speed=1., single_render_invocation=True)
        time.sleep(1)
