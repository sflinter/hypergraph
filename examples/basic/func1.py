import hypergraph as hg


def test1(x, y):
    return x+y


graph1 = hg.Graph()
with graph1.as_default():
    # Note: the dictionary items are automatically mapped to the arguments
    hg.output() << (hg.call(test1) << {'x': 2, 'y': 3})     # another possibility is passing arg by position: [2, 3]

print(graph1())
print(graph1())
hg.ExecutionContext.reset_default()
