# pylint: disable=missing-docstring,invalid-name

import numpy as np

import grapes.state.errors as errors
import grapes.state.seed as seed


class Vine:
    def __init__(self, size):
        self.size = size
        self.seed = seed.BLACK
        self.data = np.zeros((size, size), dtype=int)

    def __str__(self):
        return '\n'.join(
            '|{}|'.format(r)
            for r in [
                ' '.join(seed.abbr(c.item()) for c in r)
                for r in zip(*[np.nditer(self.data)] * self.size)
            ]
        )

    def next(self):
        self.seed = seed.inverse(self.seed)

    def insert(self, x, y):
        if min(x, y) < 0 or max(x, y) >= self.size:
            raise errors.InvalidPoint((x, y), self.size)

        if self.data[x][y] != seed.EMPTY:
            raise errors.FilledPoint((x, y), self.data[x][y])

        self.data[x][y] = self.seed

    def move(self, x, y):
        self.insert(x, y)

        self.next()
