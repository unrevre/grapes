# pylint: disable=missing-docstring,invalid-name

from enum import IntEnum

import numpy as np

import grapes.state.errors as errors


class Seed(IntEnum):
    empty = 0
    black = 1
    white = 2

    def __init__(self, colour):
        super().__init__()
        self.colour = colour

    def invert(self):
        inverse = {self.black: self.white, self.white: self.black}

        try:
            return inverse[self.colour]
        except KeyError:
            print('Empty points cannot be inverted')
            raise


class Vine:
    def __init__(self, size):
        self.size = size
        self.points = np.empty((self.size, self.size), dtype=np.int)
        self.seed = Seed.black

        self.clear()

    def __str__(self):
        return '{}'.format(self.points.__str__())

    def clear(self):
        self.points.fill(Seed.empty)

    def next(self):
        self.seed = self.seed.invert()

    def place(self, x, y):
        if min(x, y) < 0 or max(x, y) >= self.size:
            raise errors.InvalidPoint((x, y), self.size)

        if self.points[x][y] != Seed.empty:
            raise errors.FilledPoint((x, y), self.points[x][y])

        self.points[x][y] = self.seed

    def move(self, x, y):
        self.place(x, y)
        self.next()
