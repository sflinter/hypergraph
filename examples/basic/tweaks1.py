import hypergraph as hg
from hypergraph.genetic import GeneticBase


def test1(x, y):
    return x+y


graph1 = hg.Graph(name="g1")
with graph1.as_default():
    n1 = hg.mark() << {'x': 2, 'y': hg.tweak(hg.LogUniform(), name="y1")}
    n2 = hg.mark() << {'x': 1, 'y': hg.tweak(hg.LogUniform(), name="y2")}
    a = hg.call(test1) << (hg.switch(name="sw1") << [n1, n2])
    hg.output() << [a, hg.tweak(hg.QUniform(high=100)), hg.tweak(hg.LogUniform())]

# create a population from graph's phenotype
genetic = GeneticBase(graph1)
print(genetic.phenotype)

population = genetic.create_population(3)
print(population[:2])
# crossover two parent to get a new individual
child = genetic.crossover_uniform_multi_parents(population[:2])
print(child)
genetic.mutations(child, prob=0.5)
# apply mutations to an individual
print(child)

# use an individual as graph's tweak
ctx = hg.ExecutionContext(tweaks=child)
with ctx.as_default():
    print(graph1())
