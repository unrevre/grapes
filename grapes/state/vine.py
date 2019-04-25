# pylint: disable=missing-docstring,invalid-name

from enum import IntEnum

import copy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

    def adjacent(self, x, y):
        points = []
        compass = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for (r, s) in compass:
            p, q = x + r, y + s
            if not (min(p, q) < 0 or max(p, q) >= self.size):
                points.append((p, q))

        return points

    def bunch(self, x, y):
        if self.points[x][y] == Seed.empty:
            raise errors.EmptyPoint((x, y))

        colour = self.points[x][y]

        bunch = set()
        queue = [(x, y)]
        while queue:
            (r, s) = queue.pop()
            bunch.add((r, s))

            for (p, q) in self.adjacent(r, s):
                if (p, q) not in bunch and self.points[p][q] == colour:
                    queue.append((p, q))

        return bunch

    def next(self):
        self.seed = self.seed.invert()

    def place(self, x, y):
        if min(x, y) < 0 or max(x, y) >= self.size:
            raise errors.InvalidPoint((x, y), self.size)

        if self.points[x][y] != Seed.empty:
            raise errors.FilledPoint((x, y), self.points[x][y])

        self.points[x][y] = self.seed

    def remove(self, x, y):
        if self.points[x][y] == Seed.empty:
            raise errors.EmptyPoint((x, y))

        self.points[x][y] = Seed.empty

    def pluck(self, x, y):
        for (r, s) in self.bunch(x, y):
            self.remove(r, s)

    def move(self, x, y):
        self.place(x, y)
        self.next()

    def draw(self):
        board = plt.figure(figsize=(4, 4), facecolor='w')
        grid = board.add_subplot(
            111,
            xticks=range(self.size),
            yticks=range(self.size),
            position=[0.1, 0.1, 0.8, 0.8],
        )
        grid.grid(color='k', linestyle='-', linewidth=1)
        grid.xaxis.set_tick_params(bottom=False, top=False, labelbottom=False)
        grid.yaxis.set_tick_params(left=False, right=False, labelleft=False)

        black = patches.Circle(
            (0, 0),
            0.4,
            facecolor='k',
            edgecolor='k',
            linewidth=1,
            clip_on=False,
            zorder=4,
        )
        white = patches.Circle(
            (0, 0),
            0.4,
            facecolor='w',
            edgecolor='k',
            linewidth=1,
            clip_on=False,
            zorder=4,
        )

        for index, point in np.ndenumerate(self.points):
            if point == Seed.empty:
                continue

            seed = black if point == Seed.black else white
            s = copy.copy(seed)
            s.center = index
            grid.add_patch(s)

        board.show()
