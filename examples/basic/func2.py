import hypergraph as hg
from hypergraph.genetic import GeneticBase


@hg.decl_tweaks(y=hg.tweaks.Uniform())
@hg.decl_tweaks(z=hg.tweaks.Normal(mean=10))
def test1(x, y, z):
    return x+y+z


graph1 = hg.Graph(name='g1')
with graph1.as_default():
    # Note: the dictionary items are automatically mapped to the arguments
    hg.output() << (hg.call(test1) << {'x': 2})


genetic = GeneticBase(graph1)
print(genetic.phenotype)
tweaks = genetic.create_population(1)[0]
print(tweaks)

ctx = hg.ExecutionContext(tweaks=tweaks)
with ctx.as_default():
    print(graph1())
