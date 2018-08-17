from techniques import RandomSearch

objective = 'some_objective'
search_space = 'some_space'

rs = RandomSearch(objective, search_space=search_space, trials=0, random_seed=1)
rs.run()