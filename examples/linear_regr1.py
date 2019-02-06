# An example where we demonstrate the generalization we can reach with hypergraph. Here a simple linear model
# is fitted on a sample of data point. The parameters of the graph are extracted automatically and evolved using a
# genetic algorithm. Note that, the use of genetic algorithms for the linear model is just for demonstrative purposes.

import hypergraph as hg
from hypergraph.optimizer import History, ConsoleLog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Training data
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168,
                    9.779, 6.182, 7.59, 2.167, 7.042,
                    10.791, 5.313, 7.997, 3.1], dtype=np.float32)

y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573,
                    3.366, 2.596, 2.53, 1.221, 2.827,
                    3.465, 1.65, 2.904, 1.3], dtype=np.float32)


@hg.function()
@hg.decl_tweaks(w=hg.Uniform(low=-3, high=3))     # prior distributions for w and b
@hg.decl_tweaks(b=hg.Uniform(low=-1, high=1))
def model_func(x, w, b):
    return x*w + b


@hg.function()
def loss_func(y1, y2):
    return np.mean(np.square(y1 - y2))


@hg.aggregator()
def test1():
    y_hat = model_func(x=hg.input_key('x'))
    return loss_func(y1=y_hat, y2=hg.input_key('y'))  # loss = E((y-y_hat)^2)


def fitness(individual):
    return hg.run(graph1, tweaks=individual, x=x_train, y=y_train)


graph1 = test1()

history = History()
best = hg.optimize(algo='genetic', graph=graph1, fitness=fitness,
                   opt_mode='min', generations=1000, mutation_prob=0.1, lambda_=4,
                   callbacks=[ConsoleLog(), history])   # run the evolutionary algorithm
print("best:" + str(best))  # print the dict with the best configuration of parameters (AKA tweaks)

fig, axs = plt.subplots(nrows=1, ncols=2)

ax = axs[0]
ax.set_title('Loss evolution')
history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
history.plot(x='gen_idx', y=['best_score'], ax=ax)

ax = axs[1]
ax.set_title('Model vs data')
ax.plot(x_train, y_train, 'bo')
x = np.linspace(np.min(x_train), np.max(x_train), 3)
ax.plot(x, x*best['test1.model_func.w']+best['test1.model_func.b'], 'r')

plt.show()
