import hypergraph as hg
from hypergraph import tweaks
from hypergraph.genetic import GeneticBase


@hg.decl_tweaks(y=tweaks.Uniform())
@hg.decl_tweaks(z=tweaks.Normal(mean=10))
def test1(x, y, z):
    return x+y+z


def tweaks_handler(config):
    genetic = GeneticBase(config)
    print(genetic.phenotype)
    tweaks = genetic.create_population(1)[0]
    print(tweaks)
    return tweaks


print(hg.run(test1, tweaks_handler=tweaks_handler, x=100))
