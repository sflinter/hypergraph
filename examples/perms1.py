import hypergraph as hg
from hypergraph.tweaks import GeneticBase

graph1 = hg.Graph(name="g1")
with graph1.as_default():
    hg.output() << (hg.permutation(size=3) << list('abcdef'))

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
