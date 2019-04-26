# pylint: disable=missing-docstring,invalid-name

import copy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

    def adjacent(self, x, y):
        adj = []
        compass = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for (r, s) in compass:
            p, q = x + r, y + s
            if not (min(p, q) < 0 or max(p, q) >= self.size):
                adj.append((p, q))

        return adj

    def group(self, x, y):
        if self.data[x][y] == seed.EMPTY:
            raise errors.EmptyPoint((x, y))

        colour = self.data[x][y]

        group = set()
        space = set()
        queue = [(x, y)]
        while queue:
            (r, s) = queue.pop()

            group.add((r, s))

            for (p, q) in self.adjacent(r, s):
                if (p, q) not in group:
                    if self.data[p][q] == seed.EMPTY:
                        space.add((p, q))
                    if self.data[p][q] == colour:
                        queue.append((p, q))

        return group, space

    def buds(self):
        for index, point in np.ndenumerate(self.data):
            if point == seed.EMPTY:
                yield index

    def next(self):
        self.seed = seed.inverse(self.seed)

    def insert(self, x, y):
        if min(x, y) < 0 or max(x, y) >= self.size:
            raise errors.InvalidPoint((x, y), self.size)

        if self.data[x][y] != seed.EMPTY:
            raise errors.FilledPoint((x, y), self.data[x][y])

        self.data[x][y] = self.seed

    def remove(self, x, y):
        if self.data[x][y] == seed.EMPTY:
            raise errors.EmptyPoint((x, y))

        self.data[x][y] = seed.EMPTY

    def move(self, x, y):
        self.insert(x, y)

        for (r, s) in self.adjacent(x, y):
            if self.data[r][s] == seed.inverse(self.seed):
                group, space = self.group(r, s)
                if not space:
                    for (p, q) in group:
                        self.remove(p, q)

        if all(
            self.data[r][s] != seed.EMPTY for (r, s) in self.adjacent(x, y)
        ):
            if not self.group(x, y)[1]:
                raise errors.IllegalMove((x, y))

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

        for index, point in np.ndenumerate(self.data):
            if point == seed.EMPTY:
                continue

            patch = black if point == seed.BLACK else white
            stone = copy.copy(patch)
            stone.center = index
            grid.add_patch(stone)

        game.show()
