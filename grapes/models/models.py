# pylint: disable=missing-docstring,invalid-name

from abc import ABC, abstractmethod

import numpy as np

import grapes.state.seed as seed


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, node):
        pass


class Network(Model):
    def __init__(self, network):
        super().__init__()

        self.network = network

    def eval(self, node):
        vine = node.vine

        if vine.complete:
            policy = np.zeros(vine.data.size, dtype=float)
            value = vine.result * seed.inverse(vine.seed)

            return policy, value

        state = vine.state.astype(float)[None, ...]
        state = state.reshape(1, vine.size, vine.size, -1)
        policy, value = self.network.eval(state)

        return np.squeeze(policy, axis=0), np.squeeze(value)
