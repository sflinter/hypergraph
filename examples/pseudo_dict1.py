import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    d = hg.input_all().as_dict()
    hg.output() << [d.keys(), d.values(), d.items()]

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1({'a': 1, 'b': 2}))
