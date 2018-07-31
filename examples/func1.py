import hypergraph as hg


@hg.func_node
def test1(x, y):
    return x+y


graph1 = hg.Graph()
with graph1.as_default():
    # Note: the dictionary items are automatically mapped to the arguments
    hg.output() << (hg.node(test1) << {'x': 2, 'y': 3})     # another possibility is passing arg by position: [2, 3]

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1())
