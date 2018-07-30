from . import graph as g
import numpy as np

# TODO class GetBinEncKeySmooth(g.Node): use binary encoding

class GetKeySmooth(g.Node):
    """
    Get an element from a subscriptable, the index is a number in the interval [0.0, 1.0]
    """

    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, input, hpopt_config={}):
        subscriptable = input['subscriptable']
        key = input['key']

        if isinstance(subscriptable, dict):
            subscriptable = subscriptable.values()

        if key < 0.0 or key > 1.0:
            raise ValueError("The key is supposed to be a number in the interval [0.0, 1.0].")
        if len(subscriptable) == 0:
            raise ValueError("Empty vector error")
        key = np.int(np.round(key*(len(subscriptable)-1)))
        return subscriptable[key]


def select(idx_node, nodes) -> g.Node:
    """
    Select a node from a list or dict and jump the execution to it. This node represents a starting point, do not
    link to another input. The idx_node must return a number in the interval [0.0, 1.0], the rescale is performed
    automatically.
    :param idx_node: A node that returns the index
    :param nodes: A list of nodes identifiers
    :return:
    """
    return g.jmp() << g.link(GetKeySmooth(), {'subscriptable': nodes, 'key': idx_node})
