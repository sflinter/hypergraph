import hypergraph as hg
import numpy as np

graph1 = hg.Graph()
with graph1.as_default():
    hg.mark("abc1") << (hg.dump() << "** abc1 **")
    n = hg.mark("abc2") << (hg.dump() << "** abc2 **")

    idx = hg.node(lambda _: np.random.randint(0, 2))
    hg.output() << hg.select(idx, ["abc1", "abc2"])  # select is a "head" so do not use << on it.

for _ in range(3):
    ctx = hg.ExecutionContext()
    with ctx.as_default():
        print(graph1())
    print("*** end of execution ***")
