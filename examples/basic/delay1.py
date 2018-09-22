import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    hg.output() << (hg.delay(units=2) << hg.input_all())

ctx = hg.ExecutionContext()
with ctx.as_default():
    for i in range(5):
        print(graph1(i))
