import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    hg.mark("abc") << (hg.dump() << "** abc **")
    hg.mark("abc_indirect") << (hg.dump() << "abc")         # return the name of the the next jump
    hg.output() << (hg.jmp() << hg.jmp("abc_indirect"))     # note, two jmp

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1())
