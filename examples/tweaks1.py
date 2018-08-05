import hypergraph as hg


@hg.func_node
def test1(x, y):
    return x+y


graph1 = hg.Graph(name="g1")
with graph1.as_default():
    n1 = hg.mark() << {'x': 2, 'y': 3}
    n2 = hg.mark() << {'x': 1, 'y': 7}
    a = hg.node(test1) << (hg.switch(name="sw1") << [n1, n2])
    hg.output() << [a, hg.tweak(hg.UniformInt(high=100)), hg.tweak(hg.LogUniform())]

tweaks = dict([(k, d.sample()) for k, d in graph1.get_hpopt_config_ranges().items()])
print(tweaks)

ctx = hg.ExecutionContext(tweaks=tweaks)
with ctx.as_default():
    print(graph1())
