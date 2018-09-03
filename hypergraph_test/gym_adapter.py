import hypergraph as hg
import gym
import abc
import numpy as np
import hypergraph.cgp as cgp
import hypergraph.utils as hg_utils


class ValueAdapter(abc.ABC):
    @abc.abstractmethod
    def to_gym(self, value):
        pass

    @abc.abstractmethod
    def from_gym(self, value):
        pass

    @abc.abstractmethod
    def create_graph_input_range(self):
        pass

    @abc.abstractmethod
    def create_graph_output_factory(self):
        pass

    @staticmethod
    def get(space: gym.Space):
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteAdapter(space)
        if isinstance(space, gym.spaces.Box):
            return BoxAdapter(space)
        raise ValueError()


class DiscreteAdapter(ValueAdapter):
    def __init__(self, space: gym.spaces.Discrete):
        self.space = space

    def from_gym(self, value):
        return 2.*np.float(value)/self.space.n-1.

    def to_gym(self, value):
        value = cgp.TensorOperators.to_scalar(value)
        return int(np.round((self.space.n-1)*(value+1.)/2.))

    def create_graph_input_range(self):
        return None

    def create_graph_output_factory(self):
        return hg_utils.SingleValueStructFactory()


class BoxAdapter(ValueAdapter):
    def __init__(self, space: gym.spaces.Box):
        self.space = space

    def from_gym(self, value):
        space = self.space
        value = (np.array(value, dtype=np.float) - space.low) / (space.high - space.low)
        value = (value * 2.) - 1.
        return value

    def to_gym(self, value):
        raise NotImplemented()

    def create_graph_input_range(self):
        return None

    def create_graph_output_factory(self):
        raise NotImplemented()


class GymManager:
    def __init__(self, env: gym.Env, *, max_steps=100, trials_per_individual=1):
        if not isinstance(env, gym.Env):
            raise ValueError()

        if not isinstance(max_steps, int):
            raise ValueError()
        if max_steps <= 0:
            raise ValueError()

        if not isinstance(trials_per_individual, int):
            raise ValueError()
        if trials_per_individual <= 0:
            raise ValueError()

        self.env = env
        self.max_steps = max_steps
        self.adapters = tuple(map(ValueAdapter.get, [env.observation_space, env.action_space]))
        self.trials_per_individual = trials_per_individual

    # TODO def test(self, individual):

    def get_cgp_net_factory_config(self) -> dict:
        """
        Create some of the parameters necessary to init a CGP network factory
        :return:
        """
        return {
            'input_range': self.adapters[0].create_graph_input_range(),
            'output_factory': self.adapters[1].create_graph_output_factory()
        }

    def create_fitness(self, graph: hg.Graph):
        if not isinstance(graph, hg.Graph):
            raise ValueError()

        def fitness(individual):
            env = self.env
            adapters = self.adapters

            total_reward = 0
            trials = self.trials_per_individual
            for _ in range(trials):
                observation = env.reset()
                ctx = hg.ExecutionContext(tweaks=individual)
                with ctx.as_default():
                    for t in range(self.max_steps):
                        action = graph(input=adapters[0].from_gym(observation))
                        action = adapters[1].to_gym(action)
                        observation, reward, done, info = env.step(action)
                        total_reward += reward
                        if done:
                            break
            return total_reward/trials
        return fitness
