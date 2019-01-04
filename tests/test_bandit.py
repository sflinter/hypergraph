import pytest
import numpy as np
from hypergraph import tweaks
from hypergraph.bandits import HyperBand


class BogusModel:
    def __init__(self):
        self.min = np.array([0.75, 0.75])

    @staticmethod
    def get_config_ranges():
        return {
            'x1': tweaks.Uniform(),
            'x2': tweaks.Uniform()
        }

    def check_output_err(self, config):
        config = np.array([config['x1'], config['x2']], dtype=np.float32)
        v = np.max(np.abs(config-self.min))
        assert v < 0.1

    def eval(self, config, steps):
        config = np.array([config['x1'], config['x2']], dtype=np.float32)
        if np.any(config < 0) or np.any(config > 1):
            raise ValueError()
        config *= 2*np.pi
        mu = 100.*(np.sin(config[0]) + np.sin(config[1]))
        sigma = 10./steps
        return np.random.normal(mu, sigma)


def test_hyperband():
    model = BogusModel()

    def loss(config, resources):
        return model.eval(config=config, steps=resources), None

    obj = HyperBand(config_ranges=model.get_config_ranges(), loss=loss, max_resources_per_conf=100)
    output = obj()
    model.check_output_err(output)


if __name__ == '__main__':
    pytest.main([__file__])
