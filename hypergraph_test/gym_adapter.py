import hypergraph as hg
import gym
import abc
import numpy as np
import hypergraph.cgp as cgp
import hypergraph.utils as hg_utils
import time


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
    def get_graph_output_size(self):
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
        print("gym space:" + str(space))
        self.space = space

    def from_gym(self, value):
        return 2.*np.float(value)/self.space.n-1.

    def to_gym(self, value):
        value = cgp.TensorOperators.to_scalar(value)
        return int(np.round((self.space.n-1)*(value+1.)/2.))

    def create_graph_input_range(self):
        return None

    def get_graph_output_size(self):
        return None


class BoxAdapter(ValueAdapter):
    def __init__(self, space: gym.spaces.Box):
        print("gym space:" + str(space) + ", low=" + str(space.low) + ", high=" + str(space.high))
        self.space = space

    def from_gym(self, value):
        value = np.nan_to_num(value)
        # TODO provide multiple non-linearities eg: linear+tanh, linear+clip, ...
        return cgp.Tensor2Inputs.transform(np.tanh(value))

        #space = self.space
        #value = (np.array(value, dtype=np.float) - space.low) / (space.high - space.low)
        #value = (value * 2.) - 1.
        #return value

    def to_gym(self, value):
        space = self.space
        value = list(map(cgp.TensorOperators.to_scalar, value))
        value = np.array(value, dtype=np.float)
        value = (value + 1.)*(space.high - space.low)/2. + space.low
        return value.reshape(space.shape)

    def create_graph_input_range(self):
        return cgp.Tensor2Inputs.range(self.space.shape)

    def get_graph_output_size(self):
        return int(np.array(self.space.shape).prod())


class GymManager:
    def __init__(self, env: gym.Env, *, max_steps=100, trials_per_individual=1, action_prob=1):
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

        # TODO validate action_prob, max_reward

        self.env = env
        self.max_steps = max_steps
        self.adapters = tuple(map(ValueAdapter.get, [env.observation_space, env.action_space]))
        self.trials_per_individual = trials_per_individual
        self.action_prob = action_prob

    def test(self, graph: hg.Graph, individual, *, speed=1.0, single_render_invocation=False):
        env = self.env
        adapters = self.adapters

        fps = env.metadata['video.frames_per_second']
        frame_time = 1.0/(fps*speed)

        if single_render_invocation:
            env.render()

        total_reward = 0
        observation = env.reset()
        ctx = hg.ExecutionContext(tweaks=individual)
        action_prob = self.action_prob
        action_valid = False
        action = None
        with ctx.as_default():
            while True:
                if not single_render_invocation:
                    env.render()
                    time.sleep(frame_time)

                if (not action_valid) or action_prob == 1 or np.random.uniform() <= action_prob:
                    action = graph(input=adapters[0].from_gym(observation))
                    action = adapters[1].to_gym(action)
                    action_valid = True
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        print('Test episode concluded, total_reward={}'.format(total_reward))

    def get_cgp_net_factory_config(self) -> dict:
        """
        Create some of the parameters necessary to init a CGP network factory
        :return:
        """
        return {
            'input_range': self.adapters[0].create_graph_input_range(),
            'output_size': self.adapters[1].get_graph_output_size()
        }

    def create_fitness(self, graph: hg.Graph):
        if not isinstance(graph, hg.Graph):
            raise ValueError()

        def fitness(individual):
            env = self.env
            adapters = self.adapters

            total_reward = 0
            trials = self.trials_per_individual
            action_prob = self.action_prob
            action_valid = False
            action = None
            for _ in range(trials):
                observation = env.reset()
                ctx = hg.ExecutionContext(tweaks=individual)
                with ctx.as_default():
                    for t in range(self.max_steps):
                        if (not action_valid) or action_prob == 1 or np.random.uniform() <= action_prob:
                            action = graph(input=adapters[0].from_gym(observation))
                            action = adapters[1].to_gym(action)
                            action_valid = True
                        observation, reward, done, info = env.step(action)
                        total_reward += reward
                        if done:
                            break
            return total_reward/trials
        return fitness
