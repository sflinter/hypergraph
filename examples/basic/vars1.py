import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    hg.var('v1', initial_value='test1')
    #hg.output() << (hg.var('v1') << hg.SetVar('test2'))
    hg.output() << (hg.set_var('v1') << 'test2')

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1())

print(ctx.get_var_value(graph1, 'v1'))
