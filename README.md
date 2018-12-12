[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Generic badge](https://img.shields.io/badge/Python-3.6|3.7-yellow.svg)](https://www.python.org/)


# Hypergraph #

__Hypergraph__ is an open source software library for implementing graphs and networks.
 
Hypergraph was original developed to provide a high level of abstraction when developing machine learning 
and deep neural networks. However, the system is general enough to be applicable in a wide variety of other domains.

Hypergraph also provides meta-heurisitic optimisation algorithms (e.g. genetic algorithms) 
to optimise the structure of the graph for a specific task. 
This means that the same graph can be used, and automatically optimised, for a number of different tasks
or for the same task but across different data sets.

## Getting Started  with Hypergraph
To install Hypergraph the following command can be used. Hypergraph is 
compatible with Windows and Linux operating systems. 

```bash
pip install git+https://github.com/aljabr0/hypergraph
```

The core data structure of Hypergraph is a __graph__. Each graph consists of a number of 
__nodes__. Methods of creating nodes and adding them to a graph is demonstrated in the next sections. 

##### Creating Nodes
Hypergraph allows for nodes to be created by using the base class hg.Node. Notably, the __call\__ method is executed at
the runtime of the graph. Custom nodes are very useful when implementing
hyperparameter optimisation. An example of a simple custom node is shown in the following snippet.

```python
import hypergraph as hg
import numpy as np

class MyFirstNode(hg.Node):  # creating a node
    def __init__(self, name=None):
        super().__init__(name)   
    
    def __call__(self, input, hpopt_config={}):
        x = input.get('x')
        return np.multiply(x, 3)
```
Alternatively, nodes can be created 'on-the-fly' by wrapping an existing
function with the hg.call() command. This is useful when no hyperparameter
optimisation is required.

#### Building Your First Graph
The following snippet provides an overview of building graphs with Hypergraph.
The features highlighted include:
- To create a graph, simply call hg.Graph().
- To add nodes to your graph the graph.as_default() method can be used. 
- For flexibility, graphs support dictionary inputs which allows for
multiple inputs to easily be supplied to and specified within a graph.
- The notation __<<__ indicates node inputs, therefore, _a_ __<<__ _b_ indicates
that the node _b_ is input to _a_. 
- Finally, the output of the graph is specified by using the designated
identity node hg.output(). 

```python
my_first_graph = hg.Graph('my_first_graph')  # create your graph
with my_first_graph.as_default():  # add your nodes to the graph
    input = hg.input_key('input')

    func_node_result = (hg.call(np.multiply) << [input, 2])  # calling an existing function to create a node

    custom_node_result = MyFirstNode('custom_node') << func_node_result # Calling the custom built function

    hg.output() << custom_node_result  # setting the output of the graph

y = my_first_graph(input={'input': 2})  # run your graph
print(y)  # see your answer
```

## Main Concepts

#### High Level of Abstraction

Hypergraph allows for structures of arbitrary complexity to be constructed with 
relative ease. This in part is due to the ability to connect nodes.
 
 In the case of deep neural networks, this means that
different building blocks (containing multiple layers) can be readily connected. 

As shown in [this example](examples/keras_exec_opt1.py) it is also possible 
to use Hypergraph to optimise training parameters (such as queue size, batch size and the number of workers) 
to optimise the training time of a Keras model. 

Furthermore, it is possible to create graphs which contain sub-graphs by using the hg.node function as shown in the following snippet.

```python
sub_graph1 = hg.Graph()
with sub_graph1.as_default():
    input = hg.input_key('input1')
    hg.output() << (hg.call(np.remainder) << [input, 6])  # graph outputs the remainder of the inputs when divided by 6

sub_graph2 = hg.Graph()
with sub_graph2.as_default():
    input = hg.input_key('input2')
    hg.output() << (hg.call(np.divide) << [input, 6]) # graph outputs the value of the inputs divided by 6

graph = hg.Graph()
with graph.as_default():
    x = hg.node(sub_graph1) << {'input1':hg.input_all()}
    y = hg.node(sub_graph2) << {'input2':hg.input_all()}
    hg.output() << (hg.call(np.multiply) << [x, y]) # graph outputs the product of the two sub_graphs

print(graph([3, 7, 11]))
```

#### Tweaks
Hypergraph allows for specified values to be automatically modified
via optimisation algorithms. This allows for the value of variables be altered with respect to
user-defined prior distribution. An example of this is the variable _w_ in the snippet below. 
```python
w = hg.tweak(hg.Uniform(low=-3, high=3), name='w')
```
For a full demonstration of the use of tweaks please see [this example](examples/linear_regr1.py).

#### Meta-heuristic Optimisation
Hypergraph offers meta-heuristic optimisation routines to optimise graphs and
networks.

An example of a mutation-only genetic algorithm optimisation routine being applied to a graph
is presented in this simple example for [linear regression](examples/linear_regr1.py)

## Hypergraph in Action
#### Cartesian Genetic Programming (CGP)
In the domain of reinforcement learning, Cartesian Genetic Programming has recently been shown to be a 
competitive, yet simple, alternative approach for learning to play Atari games
[(Wilson et al. 2018)](https://arxiv.org/pdf/1806.05695.pdf). 

Hypergraph is very well suited for evolving such solutions which has motivated its implementation.
Additionally, a gym adapter is provided so that OpenAI's gym environments can be easily assessed.


A full example of a Hypergraph implementation of CGP for the CartPole(v1) gym environment is 
provided [here](examples/cgp-gym1.py). The following video is an example of the results from running this example.

<div align="center">
  <a href="https://www.youtube.com/watch?v=gwb_iDRgi28"><img src="https://img.youtube.com/vi/gwb_iDRgi28/0.jpg" alt="CGP Cartpole"></a>
</div>

It should be noted that both the CGP and optimisation routines are general and can be easily tailored for
a given problem by specifying a suitable set of operations and an appropriate fitness function.