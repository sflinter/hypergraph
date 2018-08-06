import hypergraph as hg
from hypergraph.tweaks import GeneticBase

@hg.func_node
def test1(x, y):
    return x+y


graph1 = hg.Graph(name="g1")
with graph1.as_default():
    n1 = hg.mark() << {'x': 2, 'y': 3}
    n2 = hg.mark() << {'x': 1, 'y': 7}
    a = hg.node(test1) << (hg.switch(name="sw1") << [n1, n2])
    hg.output() << [a, hg.tweak(hg.UniformInt(high=100)), hg.tweak(hg.LogUniform())]

# create a population from graph's phenotype
genetic = GeneticBase(graph1)
population = genetic.create_population(3)
print(population[:2])
# crossover two parent to get a new individual
child = genetic.crossover(population[:2])
print(child)
genetic.mutations(child, prob=0.1)
# apply mutations to an individual
print(child)

# use an individual as graph's tweak
ctx = hg.ExecutionContext(tweaks=child)
with ctx.as_default():
    print(graph1())
