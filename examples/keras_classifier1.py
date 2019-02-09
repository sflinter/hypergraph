# An example where we use Hypergraph to tweak a simple Keras model.
# Specifically, we optimize the following hyper-parameters:
# - a global pooling layer (avg vs max)
# - a dropout layer (we optimize the dropout rate)
# - the network optimizer (Adam vs RMS)
# - the learning rate for the optimizer

import keras
from keras.datasets import cifar10
from keras import backend as KBackend
import hypergraph as hg
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hypergraph.optimizer import History, ConsoleLog


# The data, split between train and test sets:
subsample_size = 5000
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
class_count = np.max(y_train) + 1

subsample_idxs = np.random.permutation(len(x_train))[:subsample_size]
x_train = x_train[subsample_idxs]
y_train = y_train[subsample_idxs]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, class_count)
y_test = keras.utils.to_categorical(y_test, class_count)


@hg.function()
# Declare the first tweak as a uniform choice between two types of global pooling
@hg.decl_tweaks(global_pool_op=hg.tweaks.UniformChoice((GlobalAveragePooling2D, GlobalMaxPooling2D)))
# the second tweak is the dropout rate which will be a uniform value between 0 and 0.25
@hg.decl_tweaks(dropout_rate=hg.tweaks.Uniform(0, 0.25))
def classifier_terminal_part(input_layer, class_count, global_pool_op, dropout_rate):
    """
    Create the terminal part of the model. This section is composed by a global pooling layer followed by a dropout
    and finally a dense layer. The type of global pooling is chosen automatically by the hyper-parameters optimization
    algorithm, the same also applies to the dropout rate.
    :param input_layer: The input layer which is the output of the node connected right on the top of this
    :param class_count:
    :param global_pool_op: The type of global pooling, this parameter is handled directly by hypergraph
    :param dropout_rate: The dropout rate, this parameter is handled directly by hypergraph
    :return: The output layer of this section of network
    """
    net = global_pool_op()(input_layer)
    net = Dropout(rate=dropout_rate)(net)
    return Dense(class_count, activation='softmax')(net)


@hg.function()
# Declare the first tweak as a uniform choice between two types of optimizers
@hg.decl_tweaks(optimizer=hg.tweaks.UniformChoice((Adam, RMSprop)))
# the second tweak is the learning rate
@hg.decl_tweaks(lr=hg.tweaks.LogUniform(0.00001, 0.1))
def compile_model(input_layer, output_layer, optimizer, lr):
    """
    Compile the model having the first two parameter as input and output layers respectively. The optimizer and its
    learning rate instead are automatically chosen by the hyper-parameters optimization algorithm
    :param input_layer:
    :param output_layer:
    :param optimizer: The type of optimizer to instantiate, this parameter is handled directly by hypergraph
    :param lr: The learning rate for the optimizer, this parameter is handled directly by hypergraph
    :return: The Keras model compiled together with the optimizer
    """
    output = keras.Model(inputs=input_layer, outputs=output_layer)
    output.compile(optimizer=optimizer(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return output


@hg.function()
def features_extraction_net(input_layer):
    """
    A features extraction network, this is kept simple just for demonstrative purposes.
    :param input_layer:
    :return:
    """

    net = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    net = Conv2D(32, (3, 3), activation='relu')(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Dropout(0.25)(net)

    net = Conv2D(64, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(64, (3, 3), activation='relu')(net)

    return net


@hg.function()
def keras_clear_session():
    KBackend.clear_session()


@hg.function()
def input_layer_():
    return Input(shape=x_train.shape[1:], name='input1')


# Finally we put all together by connecting the various components declared above as nodes of a graph.
@hg.aggregator(on_enter=keras_clear_session)
def model_graph():
    input_layer = input_layer_()
    top_section = features_extraction_net(input_layer=input_layer)
    bottom_section = classifier_terminal_part(input_layer=top_section, class_count=class_count)
    model = compile_model(input_layer=input_layer, output_layer=bottom_section)
    return model


graph1 = model_graph()  # create a graph
hg.tweaks.dump_tweaks(graph1)   # dump the tweaks config for inspection


def objective(individual):
    print(f'Trial, tweaks={individual}')
    model = hg.run(graph1, tweaks=individual)
    history = model.fit(x=x_train, y=y_train, epochs=4, validation_data=(x_test, y_test))
    # The number of epochs here could be handled by resources allocation algorithms such as Hyperband.
    # This feature will be soon available on hypergraph.
    return -np.max(history.history['val_acc'])


history = History()     # the history callback records the evolution of the algorithm
best = hg.optimize(algo='genetic', graph=graph1, objective=objective, callbacks=[ConsoleLog(), history],
                   generations=20, mutation_prob=(0.1, 0.8), lambda_=4)
print("best:" + str(best))     # print a dictionary containing the tweaks that determined the best performance

history = pd.DataFrame(history.generations, columns=['gen_idx', 'best_score', 'population_mean_score'])
history.plot(x='gen_idx', y=['best_score', 'population_mean_score'])
plt.show()
