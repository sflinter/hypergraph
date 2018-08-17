from base import Search, Bandit


class RandomSearch(Search):
    def __init__(self, objective, **kwargs):
        super().__init__(objective, **kwargs)
        self.random_seed = kwargs.get('random_seed')


    def run(self):
        print("running random search with seed %s"%self.random_seed)



class Hyperband(Bandit):
    def run(self):
        pass



