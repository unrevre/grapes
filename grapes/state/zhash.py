# pylint: disable=missing-docstring,invalid-name

import numpy as np


MAXINT = 1 << 32


class ZHash:
    def __init__(self, size):
        rng = np.random.default_rng(0)

        self.base = rng.integers(MAXINT, dtype=np.uint32)
        self.data = rng.integers(MAXINT, size=(3, size), dtype=np.uint32)

        self.hash = self.base
        self.history = []

    def __copy__(self):
        result = self.__new__(self.__class__)

        result.base = self.base
        result.data = self.data

        result.hash = self.hash
        result.history = self.history[:]

        return result

    def next(self):
        self.history.append(self.hash)

    def update(self, seed, p):
        self.hash ^= self.data[seed][p]

    def legal(self):
        return self.hash not in self.history
