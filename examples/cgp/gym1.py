import hypergraph as hg
from hypergraph import cgp, tweaks
from hypergraph.genetic import MutationOnlyEvoStrategy
from hypergraph_test import gym_adapter
import gym
import tempfile
import os
import uuid
import pandas as pd
import time


def save_model(obj):
    f = os.path.join(tempfile.gettempdir(), 'cgp-' + str(uuid.uuid4()))
    with open(f, 'wb') as outs:
        tweaks.TweaksSerializer.save(obj, outs)
    print("Model saved, file: " + str(f))


mode_delay = True   # when delay operators enabled feedback is disabled
graphics_enabled = True
model_file = '/tmp/cgp-cfbc255b-6ea2-44fe-86fb-66c7a0f4890c'

if graphics_enabled:
    import matplotlib.pyplot as plt

op = cgp.TensorOperators()
if mode_delay:
    cgp.DelayOperators(parent=op)

env = gym.make('CartPole-v1')
gymman = gym_adapter.GymManager(env, max_steps=250, trials_per_individual=3, action_prob=1.)

grid = cgp.RegularGrid(shape=(5, 5), **gymman.get_cgp_net_factory_config(),
                       operators=op, backward_length=3, feedback=not mode_delay, name='cgp')
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
    strategy = MutationOnlyEvoStrategy(grid, fitness=gymman.create_fitness(grid), generations=100*10**3,
                                       target_score=250)
    strategy()
    print("best:" + str(strategy.best))
    save_model(strategy.best)

    history = pd.DataFrame(strategy.history.generations)
    # history.set_index('idx')
    if graphics_enabled:
        history.plot(x='idx', y='best_score')
        plt.show()
    print(history)
    model = strategy.best

print('symbolic execution: ' + str(cgp.exec_symbolically(grid, tweaks=model)))

if graphics_enabled:
    while True:
        gymman.test(grid, model, speed=0.5)
        time.sleep(1)
