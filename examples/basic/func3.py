import hypergraph as hg
from hypergraph.genetic import GeneticOperators


@hg.decl_tweaks(y=hg.tweaks.Uniform())
@hg.decl_tweaks(z=hg.tweaks.Normal(mean=10))
def test1(x, y, z):
    return x+y+z


def tweaks_handler(config):
    genetic = GeneticOperators(config)
    print(genetic.phenotype)
    tweaks = genetic.create_population(1)[0]
    print(tweaks)
    return tweaks


print(hg.run(test1, tweaks_handler=tweaks_handler, x=100))
