import hypergraph as hg
from hypergraph.genetic import GeneticBase


@hg.function()
# TODO @hg.decl_tweaks(z=hg.tweak(hg.LogUniform(), name='z'))
def test1(x, y):
    return x+y


@hg.aggregator()
def mygraph1():
    n1 = {'x': 2, 'y': hg.tweak(hg.LogUniform(), name="y1")}
    n2 = {'x': 1, 'y': hg.tweak(hg.LogUniform(), name="y2")}
    a = hg.call(test1) << (hg.switch(name="sw1") << [n1, n2])
    return [a, hg.tweak(hg.QUniform(high=100)), hg.tweak(hg.LogUniform())]


graph1 = mygraph1()

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
print(hg.run(graph1, tweaks=child))
