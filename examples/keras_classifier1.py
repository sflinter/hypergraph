import keras
import hypergraph as hg
from hypergraph import tweaks
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Input
from keras.optimizers import Adam, RMSprop


# Declare the first tweak as a uniform choice between two types of global pooling
@hg.decl_tweaks(global_pool_op=tweaks.UniformChoice((GlobalAveragePooling2D, GlobalMaxPooling2D)))
# the second tweak is the dropout rate which will be a uniform value between 0 and 0.5
@hg.decl_tweaks(dropout_rate=tweaks.Uniform(0, 0.5))
def create_classifier_terminal_part(input_layer, class_count, global_pool_op, dropout_rate):
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


# Declare the first tweak as a uniform choice between two types of optimizers
@hg.decl_tweaks(optimizer=tweaks.UniformChoice((Adam, RMSprop)))
# the second tweak is the learning rate
@hg.decl_tweaks(lr=tweaks.LogUniform(0.00001, 0.1))
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


def create_features_extraction_net(input_layer):
    # TODO
    pass


# Finally we put all together by connection the various components declared above as nodes of a graph.
graph1 = hg.Graph(name="sample_model")
with graph1.as_default():
    input_layer = hg.call(lambda: Input(None, None))
    top_section = hg.call(create_features_extraction_net) << {'input_layer': input_layer}
    bottom_section = hg.call(create_classifier_terminal_part)
    model = hg.call(compile_model) << {
        'input_layer': input_layer,
        'output_layer': bottom_section
    }
    hg.output() << model

# TODO more to come...
