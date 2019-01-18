import hypergraph as hg


def test1(x, y):
    return x+y


graph1 = hg.Graph(name='g1')
with graph1.as_default():
    hg.mark("x_value") << 5
    hg.output() << (hg.invoke() << [test1, {'x': hg.node_ref("x_value"), 'y': 3}])
    # Note, in a more realistic scenario, the first param of the invoke, that is the function to be invoked,
    # may be the result of another computation or tweak


print(hg.run(graph1))
