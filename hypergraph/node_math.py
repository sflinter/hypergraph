from . import graph as g
import numpy as np


def def_binary_op(op):
    def f(a, b):
        node = g.call(func=op) << [a, b]
        return node
    return f


add = def_binary_op(np.add)
mul = def_binary_op(np.multiply)
