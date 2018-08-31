import hypergraph as hg
import gym
import abc
import numpy as np
import hypergraph.cgp as cgp


class ValueAdapter(abc.ABC):
    @abc.abstractmethod
    def to_gym(self, value):
        pass

    @abc.abstractmethod
    def from_gym(self, value):
        pass

    @staticmethod
    def get(space: gym.Space):
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteAdapter(space)
        raise ValueError()


class DiscreteAdapter(ValueAdapter):
    def __init__(self, space: gym.spaces.Discrete):
        self.space = space

    def from_gym(self, value):
        return 2.*np.float(value)/self.space.n-1.

    def to_gym(self, value):
        value = cgp.TensorOperators.to_scalar(value)
        return int(np.round((self.space.n-1)*(value+1.)/2.))


class GymManager:
    def __init__(self, env: gym.Env, graph: hg.Graph, *, max_steps=100):
        # TODO validate params
        if not isinstance(env, gym.Env):
            raise ValueError()
        if not isinstance(graph, hg.Graph):
            raise ValueError()

        self.env = env
        self.graph = graph
        self.max_steps = max_steps
        self.adapters = map(ValueAdapter.get, [env.observation_space, env.action_space])

    # TODO def test(self, individual):

    def fitness(self, individual):
        env = self.env
        graph = self.graph
        observation_adapter = self.observation_adapter
        action_adapter = self.action_adapter

        observation = env.reset()
        total_reward = 0
        for t in range(self.max_steps):
            ctx = hg.ExecutionContext(tweaks=individual)
            with ctx.as_default():
                action = graph(input=observation_adapter.from_gym(observation))
            action = action_adapter.to_gym(action)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
