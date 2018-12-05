import hypergraph as hg
from hypergraph.genetic import MutationOnlyEvoStrategy
from hypergraph.optimizer import History, ConsoleLog
import hypergraph.node_math as hgm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Training data
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


graph1 = hg.Graph(name="g1")
with graph1.as_default():
    w = hg.tweak(hg.Uniform(low=-3, high=3), name='w')  # prior distributions for w and b
    b = hg.tweak(hg.Uniform(low=-1, high=1), name='b')
    y_hat = hgm.add(hgm.mul(w, hg.input_key('x')), b)   # y_hat = w*x + b
    loss = hg.call(lambda y1, y2: np.mean(np.square(y1-y2))) << [y_hat, hg.input_key('y')]   # loss = E((y-y_hat)^2)
    hg.output() << loss


def fitness(individual):
    ctx = hg.ExecutionContext(tweaks=individual)
    try:
        with ctx.as_default():
            return graph1(input={'x': x_train, 'y': y_train})
    except:
        # in case of exception we penalize the maximum
        return np.inf


history = History()
strategy = MutationOnlyEvoStrategy(graph1, fitness=fitness, opt_mode='min',
                                   generations=1000, mutation_prob=0.1, lambda_=4,
                                   callbacks=[ConsoleLog(), history])
strategy()
print()
best = strategy.best
print("best:" + str(best))

history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
history.plot(x='gen_idx', y=['best_score'])
plt.show()

plt.plot(x_train, y_train, 'bo')
x = np.linspace(np.min(x_train), np.max(x_train), 3)
plt.plot(x, x*best['g1.w']+best['g1.b'], 'r')
plt.show()
