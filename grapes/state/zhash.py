# pylint: disable=missing-docstring,invalid-name

import numpy as np


MAXINT = 1 << 32


class ZHash:
    def __init__(self, size):
        rng = np.random.default_rng(0)

        self.base = rng.integers(MAXINT, dtype=np.uint32)
        self.data = rng.integers(MAXINT, size=(3, size, size), dtype=np.uint32)

        self.hash = self.base

    def update(self, seed, x, y):
        self.hash ^= self.data[seed][x][y]
