# pylint: disable=missing-docstring,invalid-name

import copy
import random

import numpy as np

import grapes.state.errors as errors
import grapes.state.vine as vine


class GrapeTree:
    C = 1.0

    def __init__(self, parent, x, y):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

        self.parent = parent
        self.children = None
        self.vine = None

        self.connect(x, y)

    def action(self):
        return self.q + GrapeTree.C * self.p / (1 + self.n)

    def connect(self, x, y):
        if isinstance(self.parent, vine.Vine):
            self.vine = self.parent
        else:
            self.vine = copy.deepcopy(self.parent.vine)
            self.vine.move(x, y)

    def descend(self):
        node = self
        while node.children is not None:
            u = np.fromiter(
                (child.action() for child in node.children), dtype=float
            )
            node = node.children[np.argmax(u)]

        return node

    def expand(self, policy):
        self.children = []

        for x, y in self.vine.buds():
            try:
                self.children.append(GrapeTree(self, x, y))
                self.children[-1].p = policy.assign(self.vine, x, y)
            except errors.IllegalMove:
                pass

        if not self.children:
            return self

        norm = sum(child.p for child in self.children)
        for child in self.children:
            child.p /= norm

        p = np.fromiter((child.p for child in self.children), dtype=float)
        return self.children[np.argmax(p)]

    def value(self):
        return random.choice((1, -1))

    def update(self, root, value):
        node = self
        while node != root:
            node.n += 1
            node.w += value
            node.q = node.w / node.n

            node = node.parent
            value = -value
