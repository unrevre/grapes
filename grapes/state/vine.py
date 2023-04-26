# pylint: disable=missing-docstring,invalid-name

import copy
import wine

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import grapes.state.errors as errors
import grapes.state.seed as seed
import grapes.state.zhash as zhash


class Vine:
    def __init__(self, size):
        self.size = size
        self.seed = seed.BLACK
        self.hash = zhash.ZHash(size * size)
        self.data = np.zeros(size * size, dtype=int)

    def __str__(self):
        return '\n'.join(
            '|{}|'.format(r)
            for r in [
                ' '.join(seed.abbr(c.item()) for c in r)
                for r in zip(*[np.nditer(self.data)] * self.size)
            ]
        )

    def __copy__(self):
        result = self.__new__(self.__class__)

        result.size = self.size
        result.seed = self.seed
        result.hash = copy.copy(self.hash)
        result.data = self.data.copy()

        return result

    def adjacent(self, p):
        return wine.adjacent(p, self.size)

    def group(self, p):
        return wine.group(p, self.size, self.data)

    def buds(self):
        return np.arange(self.data.size)[self.data == seed.EMPTY]

    def next(self):
        self.hash.next()
        self.seed = seed.inverse(self.seed)

    def insert(self, p):
        self.hash.update(self.seed, p)
        self.data[p] = self.seed

    def remove(self, p):
        self.hash.update(self.data[p], p)
        self.data[p] = seed.EMPTY

    def move(self, p):
        self.insert(p)

        for q in wine.capture(
            p, self.size, seed.inverse(self.seed), self.data
        ):
            self.remove(q)

        if wine.illegal(p, self.size, self.seed, self.data):
            raise errors.IllegalMove(p)

        if not self.hash.legal():
            raise errors.IllegalMove(p)

        self.next()

    def draw(self):
        game = plt.figure(figsize=(4, 4), facecolor='w')
        grid = game.add_subplot(
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

        for (p,), point in np.ndenumerate(self.data):
            if point == seed.EMPTY:
                continue

            x = p % self.size
            y = p // self.size

            patch = black if point == seed.BLACK else white
            stone = copy.copy(patch)
            stone.center = (x, y)
            grid.add_patch(stone)

        game.show()
