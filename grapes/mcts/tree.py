# pylint: disable=missing-docstring,invalid-name

import copy

import numpy as np

import grapes.state.errors as errors


class Tree:
    C = 5.0

    def __init__(self, parent, action, vine):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

        self.vine = vine
        self.action = action
        self.parent = parent
        self.children = []

    @property
    def next(self):
        q, p, n = zip(*((node.q, node.p, node.n) for node in self.children))
        q, p, n = np.asarray(q), np.asarray(p), np.asarray(n)

        index = np.argmax(q + Tree.C * p * np.sqrt(self.n) / (1 + n))

        return self.children[index]

    def descend(self):
        node = self

        while node.children:
            node = node.next

        return node

    def expand(self, policy):
        self.children = []

        for x, y in self.vine.buds():
            try:
                vine = copy.copy(self.vine)
                vine.move(x, y)

                node = Tree(self, (x, y), vine)
                node.p = policy[x][y]

                self.children.append(node)
            except errors.IllegalMove:
                pass

        if not self.children:
            return

        norm = sum(node.p for node in self.children)

        for node in self.children:
            node.p = node.p / norm

    def update(self, value):
        node = self

        while node is not None:
            node.n = node.n + 1
            node.w = node.w + value
            node.q = node.w / node.n

            node = node.parent

            value = -value
