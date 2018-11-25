import keras
import hypergraph as hg
from hypergraph.genetic import MutationOnlyEvoStrategy, History, ConsoleLog, ModelCheckpoint
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeHistory(keras.callbacks.Callback):
    def __init__(self):
        self.epoch_time_start = None
        self.times = None
        super().__init__()

    def on_train_begin(self, logs={}):
        self.times = []

    def on_batch_begin(self, batch, logs={}):
        self.epoch_time_start = time.monotonic()

    def on_batch_end(self, batch, logs={}):
        self.times.append(time.monotonic() - self.epoch_time_start)

    def get_avg_time(self):
        return np.mean(self.times)

    def get_total_time(self):
        return np.sum(self.times)


input_shape = (64, 64)


class MyGenerator(keras.utils.Sequence):
    def __init__(self):
        self.batch_size = 16

    def set_batch_size(self, v):
        self.batch_size = v

    def __getitem__(self, index):
        batch_size = self.batch_size
        x = np.random.uniform(size=(batch_size, ) + input_shape).astype(np.float32)
        y = np.random.uniform(size=(batch_size, )).astype(np.float32)
        return x, y

    def __len__(self):
        return 256


net_input = keras.layers.Input(shape=input_shape, dtype=np.float32)
net = keras.layers.Flatten()(net_input)
net = keras.layers.Dense(32, activation='relu')(net)
net = keras.layers.Dense(1, activation='sigmoid')(net)
model = keras.Model(net_input, net)
model.compile(optimizer='rmsprop', loss='mse')
generator = MyGenerator()


graph1 = hg.Graph(name="keras_exec_opt")
with graph1.as_default():
    time_history = TimeHistory()
    set_batch_size = hg.call(generator.set_batch_size) << [hg.tweak(hg.QUniform(low=2, high=64), name='batch_size')]
    fit = hg.call(model.fit_generator) << {'generator': generator, 'epochs': 1,
                                           'max_queue_size': hg.tweak(hg.QUniform(low=2, high=64), name='max_queue_size'),
                                           'workers': hg.tweak(hg.QLogUniform(low=1, high=16), name='workers'),
                                           'callbacks': [time_history], 'verbose': 0}
    fit << hg.deps(set_batch_size)
    hg.output() << hg.call1(lambda _: time_history.get_total_time()) << hg.deps(fit)


def fitness(individual):
    ctx = hg.ExecutionContext(tweaks=individual)
    with ctx.as_default():
        return graph1()


history = History()
strategy = MutationOnlyEvoStrategy(graph1, fitness=fitness, opt_mode='min',
                                   generations=50, mutation_prob=0.1, lambda_=4,
                                   callbacks=[ConsoleLog(), history])
strategy()
print()
print("best:" + str(strategy.best))

history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
history.plot(x='gen_idx', y=['best_score', 'population_mean_score'])
plt.show()
