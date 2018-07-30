import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    hg.mark("abc1") << (hg.dump() << hg.mark("abc2"))
    n = hg.indirect(input_node=hg.node_ref("abc2"), output_node=hg.node_ref("abc1"))
    hg.output() << (n << "test test test")

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1())
